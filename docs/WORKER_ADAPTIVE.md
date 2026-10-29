# Worker 自适应预取、请求超时与迟到结果

墙钟循环现在可在 `scheduler="worker_async"` 下选择 `worker_prefetch="adaptive"`。确定性和墙钟路径共用 `AdaptivePrefetchController`，包含滚动分位数、冷启动、余量、迟滞和覆盖上限。默认 `worker_prefetch="fixed"` 保留原固定阈值 mailbox 行为。

## 使用方式

运行真实线程和模拟推理等待，无需安装模型依赖：

```powershell
$env:PYTHONPATH = "src"
python -m robovla_runtime.cli run-worker-mock --config configs/worker_adaptive.toml
python benchmarks/run_worker_smoke.py
```

`run-worker-mock` 同样支持 `--latency-trace <JSON>` 和 `--output-root`。没有 trace 时按实际开始处理的请求序号循环使用 `latency_s`；有 trace 时 MockBackend 在 worker 开始调用它时取样，传入时间包含本地排队延迟。已被 mailbox 替换或取消的请求不会消耗序列。

这里使用 `SleepingMockBackend` 阻塞 worker 线程，控制线程继续 tick。sleep 模拟经过的时间，不代表模型计算、GPU 争用或 GIL 密集预处理。不要将不 sleep 的原 MockBackend 直接包装成墙钟推理基准，因为 ThreadedPolicyBackend 会按实际执行耗时重建 ready_at。

已有模型可以继续使用 `ThreadedPolicyBackend(FluxVLABackend(...))` 和 `run_realtime_runtime()`，通过 RuntimeConfig 打开新选项；本轮没有在真实模型上验证。

| 配置 | 默认值 | 语义 |
| --- | --- | --- |
| `worker_prefetch` | `fixed` | `fixed` 或 `adaptive` |
| `request_timeout_s` | 0.5 | 提交到控制循环取回结果的 deadline |
| `worker_close_timeout_s` | 0.1 | episode 结束时最多等待 worker 退出的秒数；0 表示不等待 |
| `adaptive_*` | 同虚拟时间路径 | 窗口、分位数、余量、迟滞、最小请求间隔 |

请求 timeout 必须有限且大于零。close timeout 必须有限且非负。这些字段不会改变已有虚拟时间 mock 调度行为。

## 自适应请求方式

自适应 worker 正常情况下最多保留一个有效逻辑请求，完成或超时后才考虑发起下一条，并遵守最小请求间隔。固定 worker 保留原来的 latest-only mailbox 刷新机制：一个实际在途请求和一个可被替换的等待请求。

超时不会停止实际推理，因此自适应模式在超时后也可能出现“旧推理还在运行、新请求等待”的状态。底层 ThreadedPolicyBackend 仍只有一个 worker，不会并发调用同一个模型，也不会创建额外 worker 绕开过载。

两种墙钟模式的排队方式不同，smoke 结果用于验证生命周期，不能把差异全部归因于预取阈值。供给缺口公式沿用虚拟路径的稳定延迟近似，在实际排队和超时恢复时仅作诊断；fallback 和实际控制 tick 指标才反映执行情况。

## Deadline 与结果归属

deadline 从 scheduler 调用 submit 时开始，覆盖排队、模型 reset/预处理/推理/后处理及控制线程取回等待。每轮 poll 后重新读取墙钟，避免 poll 耗时或并发完成绕过 deadline。观测采集时间不计入请求 timeout，但仍计入 observation age。

- 在 deadline 当次 poll 已取到的结果可以接受；到该 tick 仍没有结果的请求立即标记超时。过了 deadline 才取回的结果一律丢弃，即使模型先前已经完成。
- 超时只记录一次 `prediction_timed_out`。ThreadedPolicyBackend 提供 `cancel_queued(request_id)`，原子移除尚未开始的等待请求；已开始的推理继续运行。其他后端可选实现此非阻塞能力。
- 迟到结果不会调用 ActionBuffer.accept，也不会清空仍然有效的旧动作。无可用动作时继续由环境的 fallback_command 决定命令。
- 结果必须匹配已提交的 request ID、episode、buffer generation、源 observation ID 和时间；未知结果、身份不匹配、已替换请求、重复结果及比已接收观测更旧的结果都拒绝。
- 已退休请求保留最多 256 条诊断元数据，不保留图像 payload。过久后才返回、已无法归属的结果以 `unknown_request` 丢弃；因此 late_result_count 不是所有物理迟到结果的无限历史统计。

正常完成和能够确认归属的迟到完成都可以更新自适应分位数，样本为 `collection_time - submitted_at`，包含排队与取回延迟。迟到样本有 `sample_kind="late_completion"`，仅影响估计，不影响动作接收。重复结果不会重复计入估计。

超时发生时并不知道完整延迟，所以不把 timeout 时长伪装成完整延迟样本。没有结果返回的请求通过超时计数体现。推理异常单独报告 `prediction_failed`，不作为正常 chunk 生成延迟；迟到异常同时计入超时结果丢弃。

## 结束、reset 与卡住的模型

线程 worker 的 reset、推理或结果封装抛异常后，会将模型状态标记为无效，并返回 `PredictionFailure`。下一条请求（包括已排队的同 episode 请求）必须先在同一 worker 线程成功执行 reset，才会进入推理。如果 reset 再次失败，该请求只返回失败结果，后续请求仍须先尝试 reset；健康请求之间正常保留 episode 历史。恢复依赖模型与预处理器提供正确的 reset 实现。

episode 结束时先回收当前可见结果，再取消 episode 的等待请求和在途结果归属。backend.close 使用有限等待，超过上限产生 `worker_shutdown_timeout`，运行函数可以返回。即使最后一次 poll 抛异常，也会尝试取消 episode 和关闭 worker。

Python 线程方案不能强杀卡住的 CUDA/模型调用。close 超时后底层 daemon 线程仍可能运行、持有模型资源，但不会再发布结果；已关闭的 ThreadedPolicyBackend 不能复用。第四轮已提供可选的 [进程隔离后端](PROCESS_BACKEND.md)，支持终止、回收和重建进程；线程模式仍保留上述限制。

使用 `close_backend=False` 时，取消仍会执行，但线程生命周期由调用方负责。下一次 Runtime 调用会建立新的 scheduler 并 reset 后端，沿用已有 generation 隔离和模型 reset 串行执行规则。真实有状态模型仍需提供正确的 reset 回调。

## 事件、指标与验证

新增事件与指标：

- `prediction_timed_out` / `prediction_timeout_count`：请求 deadline 到期，记录是否取消了等待请求。
- `prediction_discarded` / `discarded_result_count`：按 reason 区分 timeout、duplicate、mailbox replacement、身份错误、未知请求或旧观测。
- `late_result_count`：能够归属到超时请求的迟到结果数量；不要求与超时数相同。
- `adaptive_prefetch.late_completion_samples`：迟到完成贡献的估计样本数。
- `worker_shutdown_timeout` / `worker_shutdown_timeout_count`：关闭等待达到上限。

`prediction_count` 统计进入 prediction_ready 路径的结果；自适应 completed_samples 还可包含迟到完成，因此两者可能不同。固定模式被 mailbox 替换的等待请求不会再被算作 timeout。

测试使用显式控制完成时刻的模拟后端验证边界，再用 Event gate 验证真实线程阻塞时控制循环持续推进和有限关闭。sleep smoke 覆盖固定延迟、尖峰和持续过载，固定/自适应各运行一次，每次 60 个 tick，检查所有超时请求均未被执行。

结果保存于 `results/worker_smoke.md` / `.json`，原始事件在 `runs/worker_smoke/`。重复 smoke 需传新的 `--output-dir`，数值受操作系统调度影响，不以逐次性能数值相同作为验收。固定步长环境推进、Python 控制循环和原 fallback 时长统计仍沿用既有语义，不构成硬实时或真实物理执行保证。
