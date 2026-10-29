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

独立观测契约示例使用 **3 维状态、2 维动作**，同样验证预热和卡死后重启：

```powershell
python -m robovla_runtime.cli infer-fluxvla-process --config configs/fluxvla_offline_state3_action2.toml --output-root runs/independent-observation
```

默认 stdout 输出 JSONL 事件及最后一行 summary/产物路径。`--quiet` 仅打印最后一行，文件中的事件不减少。模型进程中的 Python stdout/stderr 写入独立 worker 日志；底层原生库直接写 OS 文件描述符的输出不保证被此 Python 重定向捕获。

独立 CLI 验证覆盖正常、推理卡死、预热异常、预热卡死、重启耗尽和非法配置：

```powershell
python -m benchmarks.run_fluxvla_cli --output-dir runs/fluxvla_cli_rerun
```

该验证目录必须不存在。摘要默认为 `results/fluxvla_cli.md` / `.json`，也可用 `--summary-prefix` 指定。每组实际启动 CLI 子进程，而不是仅调用内部函数。

## 配置校验

这是一套独立的版本化配置，不能直接使用旧的 `[runtime]` 配置。根节点为 `version = 1`，支持 `[action]`、`[model]`、`[observations]`、`[process]`、`[inference]` 和可选 `[model_input]`，未知字段直接报错。

| 配置段 | 关键字段 |
| --- | --- |
| `action` | 必填 dimension、period_s、representation、unit、action_space、coordinate_frame；其他 ActionSpec 字段可选 |
| `model` | 必填 kind（mock / fluxvla）、task_description；warmup_steps 默认 2，model_is_stateless 默认 false |
| `observations` | 必填 manifest 和非空 required_payload_keys 列表；推荐显式填写正整数 state_dimension |
| `model_input` | 可选；非空 arrays 约束模型输入 shape/dtype，state_mapping 显式选择 state 分量写入 payload |
| `process` | startup_timeout_s、execution_timeout_s、terminate_grace_s、max_restarts；沿用 ProcessBackendConfig 默认值 |
| `inference` | repeats 默认 1、max_attempts_per_sample 默认 2、poll_interval_s 默认 0.005、close_timeout_s 默认 2.0 |

时间必须是有限数值，超时、轮询与关闭等待必须大于 0；计数必须是真正的整数，不接受布尔值冒充整数。warmup_steps、max_restarts 允许 0。动作枚举、维度、单位和 component_units 沿用 ActionSpec 校验。

`observations.state_dimension` 与 `action.dimension` 相互独立。`ObservationSpec` 约束 state 长度、有限实数值和必需 payload 键；`ActionSpec` 约束输出动作。state 不接受布尔值或数字字符串，也不会为匹配动作维度自动截断或补零。所有样本（包括不同 episode）须符合同一观测契约，预热、正式请求和重启后也使用这一契约。

兼容既有 version 1 配置：省略 state_dimension 时，从首条样本的非空 state 数组推断，再校验全部样本的一致性；不使用 action.dimension 作为默认值。显式配置错误或推断后的样本不一致均在启动前返回退出码 2。`--validate-only` 输出独立的 observation_spec 和 action_dimension，`config.json` 总是保存解析后的明确 state_dimension。

TOML 中的 manifest、模型配置、checkpoint 和 fluxvla_root 均相对 **TOML 所在目录** 解析。数组文件则相对 **观测 JSON 所在目录** 解析。支持绝对路径，不依赖启动命令时的工作目录。输出目录仍按 CLI 调用时的工作目录解析。

`--validate-only` 校验配置结构、文件存在性、全部观测、数组 shape/dtype 与 required_payload_keys，以及显式声明的 model_input，不创建输出目录或模型进程。它不导入 FluxVLA/torch，不检查权重内部格式、声明的契约与 checkpoint 是否匹配或 CUDA 可用性。这些问题仍需真实模型验证。

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
- `state` 是非空的一维有限实数数组，长度由独立观测契约约束。仅在显式配置 model_input.state_mapping 时，将所选分量注入模型 payload；省略该配置保持旧行为。文件描述符验证文件内容，model_input 另行验证它是否符合模型所需形状和类型。
- `payload` 可包含普通 JSON 值。示例模拟相机使用小型嵌套列表，无第三方依赖。
- `arrays` 可省略；每项必须声明 path、shape、dtype。NPZ 必须另给 key，NPY 不接受 key。数组与 inline payload 不得使用相同键。
- NPY/NPZ 始终使用 `allow_pickle=False`，只允许有限的实数或布尔数组；拒绝 object、字符串、复数、非有限值和 shape/dtype 不匹配。

读取数组需要 NumPy。它是可选依赖，普通 JSON 模拟仍不需要第三方包：

```powershell
python -m pip install -e ".[offline]"
```

全部样本会在 preflight 中加载到内存，每次调用获得独立副本；预热源也使用这份已验证数据，之后不会重新读取样本文件。当前不提供流式数据集或共享内存图像传输，跨进程仍有复制开销。

## 显式模型输入契约

上游 LIBERO 的 `LiberoProprioFromInputs` 从 payload 的位置、四元数和夹爪字段构造 proprio；运行时不推断 `Observation.state` 的物理含义。以下配置假定 state 顺序为位置 3 维、四元数 4 维、夹爪 2 维，必须与实际采样约定一致。相机尺寸也必须按实际 checkpoint 配置填写。

```toml
[[model_input.arrays]]
key = "agentview_image"
kind = "camera"
shape = [256, 256, 3]
dtype = "uint8"

[[model_input.arrays]]
key = "robot0_eef_pos"
kind = "state"
shape = [3]
dtype = "float32"

[[model_input.arrays]]
key = "robot0_eef_quat"
kind = "state"
shape = [4]
dtype = "float32"

[[model_input.arrays]]
key = "robot0_gripper_qpos"
kind = "state"
shape = [2]
dtype = "float32"

[[model_input.state_mapping]]
key = "robot0_eef_pos"
indices = [0, 1, 2]

[[model_input.state_mapping]]
key = "robot0_eef_quat"
indices = [3, 4, 5, 6]

[[model_input.state_mapping]]
key = "robot0_gripper_qpos"
indices = [7, 8]
```

配套 `[observations]` 设置 `state_dimension = 9`、`required_payload_keys = ["agentview_image"]`。required_payload_keys 校验映射前的原始 payload，因此不能包含仅由映射生成的字段。省略某个字段的 state_mapping 时，该字段必须直接来自 payload（离线使用 NPY/NPZ）；直接提供的数组和映射产物使用同一份 arrays 契约。

- indices 是从 0 开始的有序索引，可重排但单个映射内不可重复，不能越过 state_dimension；每个映射必须对应一个 kind=state 的数组声明，长度必须一致。
- camera 使用 HWC RGB 三通道，dtype 可为 uint8、float32 或 float64；原始 state 使用一维 float32/float64。shape 必须为正整数，且与输入精确匹配，不支持通配维度。
- 输入必须是 NumPy 数组；不将普通 JSON 列表、不同 dtype 或布局静默转换为模型输入。仅显式 state_mapping 将有限状态数值转换到声明的浮点 dtype，并拒绝转换后溢出的非有限值。所有声明的数组也须有限。
- 映射不会覆盖 payload 中已有字段；重复目标、保留的 task_description/is_new_episode 键以及未知配置字段均报错。模型预处理得到独立副本，变换不能污染源观测。
- 所有样本在 CLI preflight 校验，失败返回 2 并带样本 ID；预热、正式推理和重启后的新模型会在 dataset 调用前再次准备和校验输入。`--validate-only` 与 `config.json` 输出解析后的 model_input。
- 此契约声明形状和类型，不改变单位、坐标系、四元数顺序或图像数值范围，也不替代上游归一化。未声明的 payload 元数据仍透传；省略整个 model_input 的旧配置行为保持兼容。

可复跑的模拟验证会生成 NPY 样本、TOML 配置及报告，覆盖正常推理、卡死重启、相机类型错误和原始状态类型错误，并核对两个进程中预热/推理实际收到的映射值：

```powershell
python -m benchmarks.run_fluxvla_inputs --output-dir runs/fluxvla_inputs_smoke
```

该目录须不存在，运行需要 NumPy；生成的 healthy.toml / restart.toml 可直接传给 infer-fluxvla-process。模拟模型使用 camera 键和 32×32 合成图像，不代表真实 checkpoint 的输入配置。

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
