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

## 尚未完成的边界

当前 `FluxVLABackend.request()` 是阻塞调用。它可以完成真实模型推理和延迟画像，也能将测得的延迟投影到虚拟时钟实验，但不能让真实 executor 与 GPU 推理同时运行。真正的异步执行需要下一步增加 worker thread/process、非阻塞 submit/poll、容量为一的 latest-observation mailbox 和 episode cancellation。

目前尚未在本机安装 FluxVLA、下载 SmolVLA checkpoint 或运行 LIBERO。测试使用接口等价的 fake VLA，验证参数转发、三维 action 解码、逐动作反归一化、episode reset、分阶段计时和 runtime backend 注入。真实集成必须另外验证图像/state shape、动作周期、动作表示、统计文件和 CUDA 计时。
