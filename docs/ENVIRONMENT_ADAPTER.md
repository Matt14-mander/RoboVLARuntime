# 环境适配、动作契约与 episode reset

本轮实现将 Runtime/Executor 从 ToyJoint 解耦。核心包仍只依赖标准库；真实 FluxVLA、LIBERO、机器人和硬实时执行尚未在本轮验证。

## EnvironmentAdapter

`run_runtime(..., environment=adapter)` 与 `run_realtime_runtime(..., environment=adapter)` 接收同一接口：

| 成员 | 责任 |
| --- | --- |
| `action_spec` | 环境实际接受的动作契约，必须与后端完整一致 |
| `state` | 当前状态快照；状态维度可以不同于动作维度 |
| `reset(episode_id)` | 恢复初始状态并清除待执行命令、历史；同名 episode 也必须重置 |
| `observe(observation_id, captured_at)` | 返回当前 episode 的 Observation，可携带图像等 payload |
| `fallback_command(reason)` | 返回环境定义的 fallback，使用同一动作契约 |
| `apply_action(command)` | 接收一个经过维度和有限值检查的命令 |
| `advance(period_s)` | 推进一个逻辑步长并返回状态；等待推理和 fallback 时同样调用 |
| `metrics()` | 环境自有指标，如成功率所需的任务状态或累计时间 |

Runtime 在启动前校验完整 ActionSpec，并核对配置中的 `period_s`、`action_unit`。传入外部环境后，`initial_position`、`target_position` 和 `max_velocity` 不用于构造该环境。未传 `environment` 时通过默认工厂创建 ToyJoint，原有 TOML/CLI 用法保留。

`observation_payload_factory` 作为兼容入口保留；提供它时会覆盖 `observe()` 返回的 payload。观测身份由适配器正确填写，buffer generation 由 Runtime 附加。

ToyJoint 实现了上述接口，并保留旧的 `step()` 和 `tracking_error()` 方法。Executor 不再直接调用这两个方法。环境指标写入 `summary.environment_metrics`；没有跟踪目标的环境，其兼容字段 `final_tracking_error` 为 null。

### 最小可运行示例

```python
from robovla_runtime import RuntimeConfig, run_runtime
from robovla_runtime.backends import MockBackend
from robovla_runtime.core import ActionSpec
from robovla_runtime.environments import ToyJoint

spec = ActionSpec(
    dimension=1,
    period_s=0.02,
    unit="radian",
    normalization="unnormalized",
    component_units=("radian",),
)
environment = ToyJoint((0.0,), (1.0,), 2.0, action_spec=spec)
backend = MockBackend(spec, 10, (0.05, 0.15), (1.0,))
config = RuntimeConfig(scheduler="fixed_async", action_unit="radian")
result = run_runtime(config, backend, environment=environment)

# 同名 episode：环境初始状态和延迟序列重置，请求 ID 保持递增。
again = run_runtime(config, backend, environment=environment)
assert result.summary == again.summary
```

## 描述能力与执行能力

ActionSpec 新增 `action_space`、`coordinate_frame`、`rotation_representation`、`normalization`、`component_units` 和 `gripper_semantics`。表示类型可声明 absolute/relative/delta position 或 velocity。

旧构造方式的 normalization 默认为 `unspecified`，用于兼容已有 toy 配置；它不代表自动检测或转换。新适配器应明确填写归一化状态，混合单位动作应逐分量描述单位。任何字段不匹配都不会被隐式转换。

当前时间对齐执行器只支持 **joint 坐标中的绝对关节位置，无旋转或独立夹爪通道**。其他语义可以被描述或用于模型 profiling，但 Runtime、ToyJoint 和 ActionBuffer 会拒绝执行。尤其不能将 LIBERO delta 动作改名为 absolute_position 来绕过检查。

buffer 在接收时验证动作契约，拒绝不兼容 chunk，并保留原来的有效尾部；Executor 对正常动作和 fallback 都检查维度及 NaN/Inf，然后才调用适配器。

## Reset 与并发边界

每次 Runtime 调用对应一个 episode。启动时执行环境与后端 reset，创建新的 scheduler/executor，并 reset buffer。重复调用可以复用环境和后端，但不能并发共享同一个有状态实例。

- `PolicyBackend.reset(episode_id)` 现在是显式协议；无状态后端也需提供 no-op 实现。
- 自定义后端生成 ActionChunk 时必须复制输入 Observation 的 `episode_generation`。请求 ID 在后端生命周期内保持唯一。
- `ActionBuffer.reset()` 清空动作、已见请求、已消费索引，并递增 generation。同名旧 chunk 以 `wrong_generation` 拒绝。独立使用 buffer 时，后续 Observation 要带上它当前的 generation。
- `ThreadedPolicyBackend.reset()` 非阻塞地清除等待请求和已完成结果，使所有旧在途结果失效。真正的模型 reset 在 worker 上、下一条新请求之前串行执行，因此不会与旧推理同时修改模型历史。没有新请求时不会立即调用模型 reset。
- worker 的内部取消 generation 与 Observation 的 buffer generation 是两个作用域：前者保护异步结果生命周期，后者保护 buffer 的复用。模型返回的 chunk 保留输入中的 buffer generation。
- `cancel_episode()` 作废指定 episode 的结果；同名 episode 再次提交也会触发 worker 的模型 reset。reset 失败会成为 `PredictionFailure`，该请求不会调用模型推理。
- `FluxVLABackend.reset()` 调用预处理器的 `reset(episode_id)`，并调用模型的无参数 `reset()`；也可以显式传入 `reset_model=callback`（工厂方法同样支持）。没有模型 reset 或 callback 意味着调用方声明模型本身无状态；有额外缓存的模型必须提供清理回调。
- LIBERO 预处理器 reset 后，即便 episode ID 不变，下一条输入也携带 `is_new_episode=True`，使 dataset 在使用新观测前重置历史。

跨 episode 复用异步后端时，传 `close_backend=False`，最后由调用方 `close()`。后续已加入请求 timeout 和有限 shutdown 等待，详见 [Worker 自适应与超时](WORKER_ADAPTIVE.md)。线程方案不强制终止正在执行的推理，也不提供进程重启。

## 时间语义与验证范围

`advance(period_s)` 仍使用固定逻辑步长。墙钟循环报告实际唤醒延迟，但不会自动把超时期间所有物理演化补算进模拟器。适配器接入不意味着已实现真实机器人的时钟、反馈或停止控制；本轮保证的是每个控制 tick（含 fallback）都会推进环境。

验证覆盖：独立于 ToyJoint 的外部环境、不同状态/动作维度、两种确定性调度器与墙钟路径、fallback 推进、动作契约拒绝、同名 episode 重置、worker 在途 reset 的串行顺序，以及模型与预处理历史清空。另将 8 组 mock 矩阵逐字段与 `results/phase1_matrix.json` 比较，结果保持一致。
