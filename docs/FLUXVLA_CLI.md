# FluxVLA 离线进程推理 CLI

第六轮提供 `infer-fluxvla-process`，从 TOML 配置和离线观测清单加载样本，调用第五轮的 FluxVLA 进程工厂。命令按清单顺序推理并保存预测，不创建环境、不调用 Executor、不执行机器人动作；真实末端增量动作也可用于推理接口校验。

## 先运行模拟模型

在仓库根目录运行：

```powershell
$env:PYTHONPATH = "src"
python -m robovla_runtime.cli infer-fluxvla-process --config configs/fluxvla_offline_mock.toml --validate-only
python -m robovla_runtime.cli infer-fluxvla-process --config configs/fluxvla_offline_mock.toml --output-root runs/offline
```

示例故意让第一个模型进程在正式推理时卡死，随后重新加载、预热并重试同一条样本。两条样本最终均成功时返回 0，同时保留首次失败的诊断。将 `fault_plan = []` 可运行无故障版本。

默认 stdout 输出 JSONL 事件及最后一行 summary/产物路径。`--quiet` 仅打印最后一行，文件中的事件不减少。模型进程中的 Python stdout/stderr 写入独立 worker 日志；底层原生库直接写 OS 文件描述符的输出不保证被此 Python 重定向捕获。

独立 CLI 验证覆盖正常、推理卡死、预热异常、预热卡死、重启耗尽和非法配置：

```powershell
python -m benchmarks.run_fluxvla_cli --output-dir runs/fluxvla_cli_rerun
```

该验证目录必须不存在。摘要默认为 `results/fluxvla_cli.md` / `.json`，也可用 `--summary-prefix` 指定。每组实际启动 CLI 子进程，而不是仅调用内部函数。

## 配置校验

这是一套独立的版本化配置，不能直接使用旧的 `[runtime]` 配置。根节点为 `version = 1`，支持 `[action]`、`[model]`、`[observations]`、`[process]`、`[inference]`，未知字段直接报错。

| 配置段 | 关键字段 |
| --- | --- |
| `action` | 必填 dimension、period_s、representation、unit、action_space、coordinate_frame；其他 ActionSpec 字段可选 |
| `model` | 必填 kind（mock / fluxvla）、task_description；warmup_steps 默认 2，model_is_stateless 默认 false |
| `observations` | 必填 manifest 和非空 required_payload_keys 列表 |
| `process` | startup_timeout_s、execution_timeout_s、terminate_grace_s、max_restarts；沿用 ProcessBackendConfig 默认值 |
| `inference` | repeats 默认 1、max_attempts_per_sample 默认 2、poll_interval_s 默认 0.005、close_timeout_s 默认 2.0 |

时间必须是有限数值，超时、轮询与关闭等待必须大于 0；计数必须是真正的整数，不接受布尔值冒充整数。warmup_steps、max_restarts 允许 0。动作枚举、维度、单位和 component_units 沿用 ActionSpec 校验。当前 Observation 的 state 元数据长度须与 action.dimension 一致；payload 中供模型消费的原始 state 数组可以另外声明。

TOML 中的 manifest、模型配置、checkpoint 和 fluxvla_root 均相对 **TOML 所在目录** 解析。数组文件则相对 **观测 JSON 所在目录** 解析。支持绝对路径，不依赖启动命令时的工作目录。输出目录仍按 CLI 调用时的工作目录解析。

`--validate-only` 校验配置结构、文件存在性、全部观测、数组 shape/dtype 与 required_payload_keys，不创建输出目录或模型进程。它不导入 FluxVLA/torch，不检查权重内部格式、模型实际输入契约或 CUDA 可用性。这些问题由实际启动诊断报告。

## 离线观测格式

JSON 清单版本为 1，必须包含非空 samples。sample_id 在清单内唯一，payload 必须为对象；不接受未知字段、重复 JSON key 或 NaN/Infinity。

```json
{
  "version": 1,
  "samples": [
    {
      "sample_id": "frame-001",
      "episode_id": "recorded-episode",
      "captured_at": 12.3,
      "state": [0.0, 0.0],
      "payload": {"instruction_metadata": "optional metadata"},
      "arrays": {
        "camera": {
          "path": "observation.npz",
          "key": "image",
          "shape": [256, 256, 3],
          "dtype": "uint8"
        }
      }
    }
  ]
}
```

上述 camera 键和二维 state 只是格式示例。真实模型需要 checkpoint 对应的相机键、图像尺寸、原始 state、动作契约与任务配置；不能用这个示例替代模型输入契约。

- `episode_id` 可省略，默认为 offline；`__fluxvla_` 前缀保留给启动阶段。
- `captured_at` 可省略，默认为 0，必须有限且非负。它作为源时间保存，命令不会按源时间间隔 sleep。
- `payload` 可包含普通 JSON 值。示例模拟相机使用小型嵌套列表，无第三方依赖。
- `arrays` 可省略；每项必须声明 path、shape、dtype。NPZ 必须另给 key，NPY 不接受 key。数组与 inline payload 不得使用相同键。
- NPY/NPZ 始终使用 `allow_pickle=False`，只允许有限的实数或布尔数组；拒绝 object、字符串、复数、非有限值和 shape/dtype 不匹配。

读取数组需要 NumPy。它是可选依赖，普通 JSON 模拟仍不需要第三方包：

```powershell
python -m pip install -e ".[offline]"
```

全部样本会在 preflight 中加载到内存，每次调用获得独立副本；预热源也使用这份已验证数据，之后不会重新读取样本文件。当前不提供流式数据集或共享内存图像传输，跨进程仍有复制开销。

## 正式模型配置

保留已确认的 action / observations / process / inference 配置，将 model 段替换为：

```toml
[model]
kind = "fluxvla"
config_path = "../FluxVLA/configs/your_model.py"
checkpoint_path = "../checkpoints/your_checkpoint"
fluxvla_root = "../FluxVLA"
task_description = "your verified task description"
task_suite_name = "libero_spatial"
warmup_steps = 5
timing_mode = "diagnostic"
# Only specify the reset method actually supported by this model:
reset_model_method = "reset"
```

路径必须存在。`num_inference_steps` 和 `inference_seed` 可选；参数继续通过 runner/preprocessor 传到模型。reset_model_method 支持模型上的点分方法路径。没有 reset 回调时，只有显式确认 `model_is_stateless = true` 才允许进入 ready，详见 [进程工厂与预热](FLUXVLA_PROCESS.md)。示例中的 reset 方法不能假定所有 FluxVLA 模型都具备。

mock 模式不允许填写真实模型路径；真实模式不允许 fault_plan。模拟故障支持 setup_error、warmup_error、warmup_hang、post_warmup_reset_hang、request_hang，按模型进程创建次序依次消费。超出故障列表后模型恢复正常。

## 重试、时间与状态

CLI 同时只提交一条逻辑请求，不替换离线样本。失败后最多尝试 max_attempts_per_sample 次；进程后端另有连续重启预算，预算耗尽会停止处理并报告未处理样本。每次失败都保留记录；不能把已经失败的 request ID 的结果保存为成功预测。

启动/预热由 startup timeout 限制，正式执行由 execution timeout 限制。父进程还有 startup + execution + 2 × terminate_grace + 1 秒的收集等待上限，防止监督器失效后无限等待。运行结束始终尝试有限关闭，关闭未完成也返回失败。

同一 episode 的连续样本保留推理历史。切换 episode 或开始新一轮 repeats 时显式 reset，导致重建、重新预热和状态清理。故障重试不主动重置重启预算，恢复遵循进程后端的策略。

每次尝试的 Observation 时间会重设为本次提交时的 monotonic 相对时间，原始 captured_at 保存在 inputs 和 predictions 的 source_captured_at 中。输出不应解释为实时机器人观测新鲜度。

首条结果的 end_to_end / latency 包含排队、启动和预热；模型调用的 total 以及 diagnostic 模式下的 preprocess/inference/postprocess 单独记录，不能将首条完整等待时间当作纯推理耗时。

## 输出和退出码

每次在 output-root 下创建带时间戳的新目录：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 解析后的配置与绝对路径 |
| `inputs.json` | 配置/清单/数组文件 SHA-256、源样本元数据与 payload 键 |
| `events.jsonl` | 增量写入并 flush 的启动、预热、提交、失败、重试、重启、终止事件 |
| `attempts.jsonl` | 每次尝试的 request ID、结果、错误类型和计时 |
| `predictions.jsonl` | 仅成功的动作 chunk、源样本关联、process generation 和分阶段耗时 |
| `diagnostics.json` | 按 process generation 汇总事件及请求尝试 |
| `summary.json` | 样本完成度、失败尝试、启动/重启/预热次数、资源回收结果 |
| `worker-<pid>-<id>.log` | 该模型进程的 Python stdout/stderr |

mock 模式还保存构建配置、模拟权重标记与调用审计。文件指纹用于追踪本次读取的输入，不代替模型权重完整性或语义检查。

退出码：0 表示所有样本最终成功且 worker 已回收；1 表示推理/启动/关闭/产物 I/O 失败；2 表示配置或输入 preflight 失败；130 表示运行中被 Ctrl+C 中断。运行失败和中断尽可能保存已有事件与 summary，退出 0 也可能包含被成功恢复的失败尝试。

本轮验证使用模拟依赖。真实 checkpoint、CUDA 预热和资源释放、真实观测输入契约仍需单独验证。
