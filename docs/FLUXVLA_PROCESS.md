# FluxVLA 进程工厂、启动预热与重启重置

第六轮新增 [离线进程推理 CLI](FLUXVLA_CLI.md)，提供 TOML 配置校验、JSON/NPY/NPZ 观测加载及启动/预热/推理/重启诊断，可直接使用 `infer-fluxvla-process` 验证推理链路。

第五轮将已有 FluxVLA 推理适配器接入进程后端。模型加载、runner setup、预处理、预热和正式推理都在子进程中进行。父进程仅持有可序列化的构建配置，不初始化 PyTorch、CUDA 或模型实例。

## 本轮可运行的验证

在仓库根目录运行，无需安装 FluxVLA、torch、LIBERO 或下载 checkpoint：

```powershell
$env:PYTHONPATH = "src"
python -m benchmarks.run_fluxvla_process --output-dir runs/fluxvla_process_rerun
```

每次使用新的输出目录。默认读取 `configs/fluxvla_process_mock.toml`，可用 `--config` 更换 RuntimeConfig。摘要写入 `results/fluxvla_process.md` / `.json`，可用 `--summary-prefix` 更换位置。

模拟替换的是 torch/mmengine、上游 runner 与模型依赖。实际执行的本项目代码包含 FluxVLALiberoFactory、共享配置加载器、`from_libero_runner`、LIBERO 预处理、`predict_action` 调用边界、逐动作反归一化和进程生命周期。模拟配置不表示真实 LIBERO 动作语义。

五组场景覆盖正常运行、正式推理卡死、预热卡死、预热异常、预热后的模型 reset 卡死。每组 200 个控制 tick；验证旧进程回收、重新构建与预热、模型和 dataset 历史清零、预热动作不执行及超时请求不执行。原始产物包括 runtime config、factory 配置、故障计划、事件、summary 和模拟调用审计。

## 两层工厂

`FluxVLALiberoFactory` 是模型构建配方，保存普通配置和 ActionSpec：

| 字段 | 用途 |
| --- | --- |
| `config_path` / `checkpoint_path` | 上游配置与权重路径；创建配方时转为绝对路径，子进程加载时校验 |
| `action_spec` | 调用方确认过的动作契约；会与父进程契约核对 |
| `task_description` | LIBERO 预处理的任务文本 |
| `fluxvla_root` | 可选的上游 checkout 根目录 |
| `task_suite_name` | 可选覆盖；否则使用 eval 配置中的第一个 suite |
| `num_inference_steps` / `inference_seed` | 传入 runner 并继续转发给模型 |
| `timing_mode` | 默认 throughput；也可使用 diagnostic |
| `reset_model_method` | 可选的模型方法名或属性路径，例如 `reset` 或 `policy.reset`；必须为无参数 callable |

它与原 profiler 共享 `load_libero_runner()`：加载配置、准备 checkpoint/processor 路径、构建 runner 并调用 run_setup。工厂不创建 LIBERO 环境，不运行 eval episode。适配接口基于仓库内 `.upstream/FluxVLA` 的 runner 实现，真实依赖版本仍需集成验证。

`FluxVLAProcessFactory` 包装模型配方，负责启动条件：

可传入独立的 `observation_spec=ObservationSpec(state_dimension=3, required_payload_keys=("camera",))`；预热和正式请求均在进入模型前校验该契约，状态维度无需等于动作维度。离线 CLI 始终传入已解析的观测契约。旧 SDK 调用未传契约时，从首次预热观测（零预热时为首次正式请求）推断状态维度，并在该 worker 生命周期内保持一致；推荐调用方显式传入契约。

`FluxVLALiberoFactory(input_spec=FluxVLAModelInputSpec(...))` 可显式约束相机与原始状态的 shape/dtype，并通过 `StatePayloadMapping(key, indices)` 将 Observation.state 分量转换为 payload 数组。契约类均可从 robovla_runtime.integrations 导入，标准 pickle 可序列化；父进程不导入 torch/FluxVLA。已初始化 runner 可给 `FluxVLABackend.from_libero_runner` 或 `FluxVLALiberoPreprocessor` 传相同 input_spec。校验发生于 dataset 变换前，覆盖预热、正式请求和重启后的新工厂；仅创建 SDK 配方不会提前校验观测，CLI 会校验全部离线样本。详见[模型输入契约](FLUXVLA_CLI.md#显式模型输入契约)。

- `backend_factory`：在子进程构建 FluxVLABackend 的可序列化 callable。可用上述 LIBERO 工厂，也可提供其他 FluxVLA 模型的自定义构建函数。
- `observation_factory(index)`：返回有效预热 Observation 的模块顶层 callable；读取 CPU 图像/state 样本等工作也在子进程中进行。
- `warmup_steps`：默认 2；0 表示不运行预热推理，但仍进行状态重置。
- `model_is_stateless`：默认 false。没有模型 reset 回调时必须显式声明 true，否则启动失败。

普通 FluxVLABackend 仍保留先前“无 reset 表示无状态”的兼容行为；进程启动工厂要求显式声明，避免把预热历史带入正式推理。无参数 `vla.reset()` 会被自动识别；更复杂的清理逻辑可由自定义 backend_factory 提供 `reset_model` 回调。

## Ready 的条件

```text
spawn → 构建模型 / runner setup → 校验 ActionSpec
      → reset 预热 episode
      → N 次完整预处理 / 推理 / 解码（结果丢弃）
      → synchronize → reset 模型与预处理状态 → synchronize
      → ready → 实际 episode reset → 正式请求
```

每个预热样本都会深拷贝，并使用独立的预热 episode 标识。预热输出不进入父进程结果队列，不计入 prediction_count、自适应延迟样本或动作 buffer。预热后重置失败时，即使已经报告 warmup_completed，也不能发送 ready。

模型与预处理在预热开始和结束时都 reset。正式 episode 首条请求前，进程后端再次 reset。LIBERO preprocessor 清除 episode/generation 跟踪，首条正式观测将 `is_new_episode=True` 传给 dataset，触发其历史 buffer 清理。模型自身历史由 reset 回调负责；声明无状态不能替代实际需要的清理。通用工厂不自动恢复所有库的全局 RNG；需要固定随机行为时使用模型支持的 inference_seed 或自定义 reset 策略。

重启后重新执行完整构建、预热、状态清理和 ready 握手；不会复用旧模型缓存。显式 reset 即使使用相同 episode ID 也会通过进程后端重建模型，因此其开销包含再次加载与预热。

## 超时和诊断

`ProcessBackendConfig.startup_timeout_s` 包含子进程导入、模型构建、预热输入生成、完整预热以及预热后 reset/synchronize 的总时间。启动进度消息不会延长该绝对 deadline。正式请求仍受 execution timeout 和请求剩余 deadline 限制。

启动期间控制循环继续运行；排队请求过期会被取消或在派发前丢弃，不能提前进入正在预热的模型。预热卡死、异常或 reset 卡死沿用上一轮的终止、回收、重启预算和 fallback 行为。

新增 `process_startup_progress` 事件，stage 包括：

- `model_loaded`：工厂已返回模型，记录构建耗时。
- `warmup_started` / `warmup_completed`：预热次数与完成耗时。
- `warmup_state_reset`：预热后的 reset 和同步都完成。

这些事件带 process generation；事件时间是父进程收集时间。summary 的 `process_worker.warmups_completed` 和 `warmup_state_resets` 区分预热完成与正式可用前的清理完成。模型加载异常可能在 model_loaded 之前失败，由 process_worker_fault 报告。

## 真实模型接入方式

下例的 `my_adapter` 由调用方提供有效观测工厂与已确认的 ActionSpec；工厂定义应位于可导入模块，而不是 lambda 或局部函数：

```python
from robovla_runtime.backends import ProcessBackendConfig
from robovla_runtime.integrations import FluxVLALiberoFactory, FluxVLAProcessFactory
from my_adapter import action_spec, make_warmup_observation

if __name__ == "__main__":
    model = FluxVLALiberoFactory(
        config_path="/path/to/model_config.py",
        checkpoint_path="/path/to/checkpoint",
        fluxvla_root="/path/to/FluxVLA",
        action_spec=action_spec,
        task_description="pick up the block",
        task_suite_name="libero_spatial",
        # Use the verified reset method for this model, if present.
        reset_model_method="reset",
    )
    factory = FluxVLAProcessFactory(model, make_warmup_observation, warmup_steps=5)
    backend = factory.create_backend(action_spec, config=ProcessBackendConfig(
        startup_timeout_s=120, execution_timeout_s=2, max_restarts=2,
    ))
    try:
        # Pass backend to the runtime with a matching environment adapter,
        # or use submit/poll for inference-only validation.
        ...
    finally:
        backend.close(timeout_s=2)
```

模型没有上述 reset 方法时，不能照搬示例；需提供正确回调，或在确认无跨请求历史后显式选择 model_is_stateless。预热输入必须符合当前模型的相机键、尺寸、state 和任务配置，不能用模拟图像替代真实输入契约校验。

当前 Runtime 执行层仍只支持绝对关节位置。真实 LIBERO 的末端增量与 gripper 动作不能改标签后送入 ToyJoint；本轮完成的是推理进程接入，未扩展真实机器人执行语义。模型重新加载时长、CUDA kernel 预热、真实设备资源释放与任务成功率均尚未实测。
