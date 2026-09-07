# 第二阶段：FluxVLABackend

## 当前完成范围

`FluxVLABackend` 已实现 FluxVLA 推理边界的适配：

```text
RoboVLA Observation
  -> FluxVLA dataset/preprocessor
  -> vla.predict_action(**batch)
  -> action decoder / LIBERO denormalizer
  -> timed ActionChunk
```

运行时组合入口现在是 `run_runtime(config, backend)`。`run_simulation(config)` 仍构建 `MockBackend`，用于第一阶段的确定性基准与回归。Mock 不再是 runtime 的硬编码依赖。

核心包导入时不会导入 FluxVLA、PyTorch 或 NumPy。只有实际创建并调用 FluxVLA 模型时才需要这些依赖。

## 通用使用方式

```python
import torch

from robovla_runtime.backends import FluxVLABackend, decode_first_batch
from robovla_runtime.core import ActionSpec, Observation

spec = ActionSpec(
    dimension=32,
    period_s=0.02,
    representation="absolute_position",
    unit="normalized",
)

backend = FluxVLABackend(
    vla=initialized_fluxvla_model,
    preprocess=lambda observation: build_fluxvla_batch(observation.payload),
    decode_actions=lambda output: decode_first_batch(output, 32),
    action_spec=spec,
    synchronize=torch.cuda.synchronize,
)

pending = backend.request(
    Observation(
        episode_id="episode-0",
        observation_id=0,
        captured_at=0.0,
        state=(0.0,) * 32,
        payload=raw_observation,
    ),
    requested_at=0.0,
)
```

`pending.timings_s` 分别包含 `preprocess`、`inference`、`postprocess` 和 `total`。传入 `torch.cuda.synchronize` 后，预处理后的异步 H2D 工作和模型 CUDA 工作会在阶段边界完成，因此适合诊断分阶段延迟。同步会扰动正常吞吐，后续需要另设无强制同步的吞吐模式。

## 从 FluxVLA LIBERO runner 创建

已经完成 `run_setup()` 的 `LiberoEvalRunner` 可以提供模型、dataset、统计 key 和反归一化 transform：

```python
import torch

from robovla_runtime.backends import FluxVLABackend
from robovla_runtime.core import ActionSpec

spec = ActionSpec(
    dimension=7,
    period_s=0.1,  # 必须替换为 checkpoint/环境的真实动作周期
    representation="absolute_position",  # 动作语义确认前不能照搬
    unit="libero_action",
)

backend = FluxVLABackend.from_libero_runner(
    runner=libero_runner,
    action_spec=spec,
    task_description=task_description,
    synchronize=torch.cuda.synchronize,
)
```

如果把该后端传给 `run_runtime`，`RuntimeConfig.action_unit` 必须与
`ActionSpec.unit` 一致。运行时会在 episode 开始前拒绝维度、动作周期、
表示或单位不一致的后端，避免把动作解释错误后再送入执行器。

该工厂复用了 FluxVLA 的：

- `runner.dataset(raw_obs)`；
- `runner.vla.predict_action(**batch)`；
- `runner.denormalize_action`；
- `runner.norm_stats_key`；
- `runner.eval_chunk_size`。
- `runner.num_inference_steps` 与 `runner.inference_seed`；
- `runner.enable_mixed_precision_training` 与 `runner.mixed_precision_dtype`。

`FluxVLALiberoPreprocessor` 会根据 `episode_id` 设置 `is_new_episode`，从而触发 FluxVLA dataset 的历史 buffer reset，并向 batch 写入 `unnorm_key`。LIBERO decoder 对三维输出执行 batch 0 的 chunk；对二维 `[B, D]` 输出只执行 batch 0 的单个动作，并把 `task_suite_name` 与 `norm_stats_key` 一起传给 `DenormalizeLiberoAction`。这与 `libero_eval_runner.py` 的处理一致。

## 时间语义

`request()` 使用 monotonic wall-clock timer 测得完整调用延迟，再映射为：

```text
ready_at = runtime requested_at + measured total latency
start_at = source observation captured_at
```

因此现有 `ActionBuffer` 会按动作周期丢弃返回时已经过期的前缀，并保留 request、observation 与 action index 的关联。

## Worker 异步执行

`FluxVLABackend.request()` 仍然保持一个易测试的阻塞推理调用。真实运行时通过
`ThreadedPolicyBackend` 把它放入专用 worker，控制线程只调用非阻塞的
`submit()` 和 `poll()`：

```python
from robovla_runtime import RuntimeConfig, run_realtime_runtime
from robovla_runtime.backends import ThreadedPolicyBackend

blocking_backend = FluxVLABackend.from_libero_runner(
    runner=libero_runner,
    action_spec=spec,
    task_description=task_description,
    synchronize=torch.cuda.synchronize,
)
backend = ThreadedPolicyBackend(blocking_backend)

config = RuntimeConfig(
    scheduler="worker_async",
    period_s=0.1,
    duration_s=30.0,
    chunk_size=10,
    execute_horizon=10,
    prefetch_threshold_s=0.3,
    initial_position=(0.0,) * 7,
    target_position=(0.0,) * 7,
    action_unit="libero_action",
    deadline_tolerance_s=0.005,
)

result = run_realtime_runtime(
    config,
    backend,
    observation_payload_factory=lambda environment: read_libero_observation(),
)
```

这里的 `run_realtime_runtime` 仍使用项目自带的 `ToyJoint`，用于验证墙钟调度、
buffer 和并发行为；`read_libero_observation()` 是需要由调用方提供的采集函数。
它不能代替 `LiberoEvalRunner` 的 `env.step()`，也不能把 LIBERO 的末端增量动作
解释成 toy joint 的绝对位置。接入真实 LIBERO/机器人闭环前还需要单独的环境执行
适配器和匹配的 ActionSpec。

worker 同时最多持有一个正在执行的请求和一个等待请求。控制循环提交更近的
observation 时，会原子替换等待请求，避免按 FIFO 累积陈旧 observation。
`mailbox_request_replaced` 事件和 `mailbox_replacement_count` 指标记录替换次数。

每个异步结果包含 `queue`、原后端各阶段和 `end_to_end` 延迟。worker 异常不会
静默退出，而会转换为 `prediction_failed` 事件。episode 结束时通过 generation
取消等待及在途结果，因此即使复用相同 episode ID，旧 chunk 也无法进入新 episode。

`run_realtime_runtime` 使用 monotonic wall clock 按固定周期唤醒控制线程，并记录
`control_tick_lateness_s` 和 deadline miss。结束时会取消当前 episode 并关闭 worker；
如果需要跨多个 episode 复用 worker，可传 `close_backend=False` 并由调用方最终调用
`backend.close()`。

线程方案让 CUDA 推理与 CPU 控制循环重叠，并避免跨进程复制模型。Python 预处理
仍可能与控制线程竞争 GIL；若后续画像表明预处理成为主要瓶颈，再将纯 CPU 图像处理
拆到进程池。

## 尚未完成的边界

目前尚未在本机安装 FluxVLA、下载 SmolVLA checkpoint 或运行 LIBERO。测试使用接口等价的 fake VLA，验证参数转发、三维 action 解码、逐动作反归一化、episode reset、分阶段计时、非阻塞提交、latest-observation 替换、异常传播、episode cancellation 和墙钟控制循环。真实集成必须另外验证图像/state shape、动作周期、动作表示、统计文件和 CUDA 计时。
