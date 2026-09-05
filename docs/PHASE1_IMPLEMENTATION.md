# 第一阶段开发规格：确定性 Runtime Core

目标：在不安装 PyTorch、FluxVLA、LIBERO 或 ROS 的情况下，构建可重复的 action-chunk 时序实验。第一阶段回答“给定动作周期、chunk 长度和推理延迟，执行器何时连续运行、何时耗尽、执行动作来自多旧的观测”。

第一阶段不评价 VLA 任务成功率，不实现 RTC、MPC、Ruckig、网络传输或 Web dashboard。

## 1. 最小目录

```text
pyproject.toml
src/robovla_runtime/
  __init__.py
  core/
    types.py
    clock.py
    events.py
  backends/
    base.py
    mock.py
  execution/
    action_buffer.py
    executor.py
  scheduling/
    sync.py
    fixed_async.py
  environments/
    toy_joint.py
  metrics/
    summary.py
  cli.py
configs/
  phase1_sync.toml
  phase1_async.toml
tests/
  test_action_buffer.py
  test_runtime_scenarios.py
```

只使用 Python 标准库作为运行时依赖。测试可使用 pytest；如果不希望增加依赖，也可先用 `unittest`。

## 2. 第一批数据对象

在 `core/types.py` 写冻结 dataclass。所有时间使用秒，来自同一个 `Clock`。

```python
@dataclass(frozen=True)
class Observation:
    episode_id: str
    observation_id: int
    captured_at: float
    state: tuple[float, ...]

@dataclass(frozen=True)
class ActionSpec:
    dimension: int
    period_s: float
    representation: Literal['absolute_position']
    unit: str

@dataclass(frozen=True)
class ActionChunk:
    episode_id: str
    request_id: int
    source_observation_id: int
    source_observation_time: float
    ready_at: float
    start_at: float
    spec: ActionSpec
    actions: tuple[tuple[float, ...], ...]

@dataclass(frozen=True)
class ExecutionRecord:
    tick_id: int
    scheduled_at: float
    executed_at: float
    request_id: int | None
    action_index: int | None
    source_observation_id: int | None
    source_observation_time: float | None
    command: tuple[float, ...]
    fallback_reason: str | None
```

第一阶段只支持 `absolute_position`，避免对 delta/velocity 动作错误地跳过、保持或融合。构造时验证维度、正周期、动作长度以及同 episode 关系。

## 3. 时钟

`core/clock.py` 定义：

```python
class Clock(Protocol):
    def now(self) -> float: ...
    def advance_to(self, timestamp: float) -> None: ...
```

实现 `VirtualClock`。它只能单调前进，倒退时报错。第一阶段的主实验全部使用虚拟时间，不使用 `sleep()`，保证测试快速且可重复。`MonotonicClock` 留到后续 wall-clock smoke test。

## 4. Mock policy backend

`backends/base.py` 定义最小协议：

```python
class PolicyBackend(Protocol):
    def request(
        self,
        observation: Observation,
        requested_at: float,
    ) -> PendingPrediction: ...
```

`PendingPrediction` 至少暴露 `request_id`、`ready_at` 和完成后得到的 `ActionChunk`。第一版不用线程，使用离散事件：`ready_at = requested_at + latency_model.next()`。

`MockBackend` 参数：

- chunk 长度 H；
- ActionSpec；
- 固定或序列化 latency trace；
- 确定性动作生成函数。

默认动作可以是朝 toy joint 目标位置的线性序列。相同 seed/config 必须生成相同 trace。

## 5. ActionBuffer

`execution/action_buffer.py` 负责：

- 有界保存 ready chunk；
- 按动作的计划时间返回下一条有效动作；
- 拒绝上一 episode 的 chunk；
- 拒绝重复 request；
- 丢弃已过期的绝对位置动作前缀；
- 返回连续有效动作的 `coverage_end - now`；
- 清晰记录 `expired`、`wrong_episode`、`duplicate`、`capacity` 原因。

buffer 不负责请求模型，也不执行动作。不要把 scheduler、backend 和 buffer 写成一个大类。

## 6. Toy environment 与 executor

`environments/toy_joint.py` 先实现一个一维关节：

```text
x_next = x + clamp(command - x, -v_max * dt, v_max * dt)
```

它接收绝对位置命令，提供当前 state 和跟踪目标。它不是机器人动力学模型，只用于检查时序、跟踪误差和 fallback 后果。

`execution/executor.py` 按固定 `ActionSpec.period_s` 产生 tick：

1. 从 buffer 取当前动作；
2. 无动作时执行 `hold_position`；
3. 将命令交给 toy environment；
4. 写入 `ExecutionRecord` 和事件；
5. 推进虚拟时钟。

fallback 必须单独计数。不能因为持续下发 hold command 就认为 buffer 没有耗尽。

## 7. 两个 scheduler

`SyncScheduler`：buffer 为空时采集观测并请求 chunk；在 chunk ready 前环境仍按 tick 执行 fallback。这样模拟真实时间流逝，而不是像普通同步仿真一样冻结环境。

`FixedAsyncScheduler`：当 `buffer_coverage <= prefetch_threshold_s` 且没有在途请求时，采集最新观测并发起请求。第一版只允许一个在途请求，避免并发返回顺序问题掩盖基本语义。

两个 scheduler 使用同一个 backend、buffer、executor 和 latency trace，确保比较只改变调度策略。

## 8. 事件与指标

`core/events.py` 写 JSON Lines 兼容的结构化事件，至少包括：

```text
observation_captured
prediction_requested
prediction_ready
chunk_accepted
chunk_rejected
action_executed
fallback_executed
episode_started
episode_finished
```

`metrics/summary.py` 从事件计算：

- prediction latency p50/p95/max；
- observation age：动作执行时刻减来源观测采集时刻；
- buffer underrun tick 数与比例；
- fallback 总时长；
- deadline miss；
- 动作边界跳变；
- toy joint 跟踪误差。

样本少于能支持稳定 p95 估计时仍输出数值，但报告样本数，不作尾延迟强结论。

## 9. CLI 和第一组配置

CLI 只需支持：

```bash
python -m robovla_runtime.cli run --config configs/phase1_sync.toml
python -m robovla_runtime.cli run --config configs/phase1_async.toml
```

输出：

```text
runs/<run-id>/config.toml
runs/<run-id>/events.jsonl
runs/<run-id>/summary.json
```

配置至少包含：动作周期 20 ms、chunk 长度 10、执行长度 10、episode 时长、固定延迟、异步预取阈值和 seed。第一批延迟取 0、50、150、250 ms，覆盖低于、接近和超过 chunk coverage 的情况。

## 10. 有意义的测试

`test_action_buffer.py`：

- 每个动作最多消费一次；
- 过期前缀按时间丢弃；
- 上一 episode 的迟到 chunk 被拒绝；
- 重复 request 被拒绝；
- coverage 计算正确。

`test_runtime_scenarios.py`：

- 零延迟且供给充足时，warm-up 后无 underrun；
- 推理延迟持续大于 chunk coverage 时，两种 scheduler 都出现 underrun；
- 延迟低于 coverage 且预取足够早时，fixed async 比 sync 少 fallback；
- episode reset 后不执行旧 episode action；
- 相同配置重复运行产生相同事件序列和 summary。

这些测试验证的是时序不变量，不是照抄具体实现。

## 11. 开发顺序和完成定义

建议按以下顺序提交本项目内部 commit：

1. `core/types.py` + 校验；
2. `VirtualClock` + event schema；
3. `MockBackend`；
4. `ActionBuffer`；
5. toy environment + executor；
6. sync scheduler；
7. fixed async scheduler；
8. metrics + CLI + 两个配置；
9. 场景测试与第一份结果说明。

第一阶段完成的判断标准：运行同一条延迟 trace 时，报告能够解释每个动作来自哪个 observation/chunk、为什么发生 fallback，以及固定预取异步在什么延迟范围内改善连续执行。达到这一点后再实现 FluxVLA adapter 和真实 profiler。

