# 自适应预取与合成延迟故障

本轮实现位于虚拟时间 `run_runtime` / `run_simulation` 路径，选择 `scheduler="adaptive_async"`。实际线程循环继续使用 `worker_async`；本轮实验没有测量真实模型、GPU 或网络性能。

## 调度规则

与固定异步共用同一个 BaseScheduler、ActionBuffer、Executor 和 fallback。每个策略最多一个在途请求，每个 tick 最多发起一个请求；chunk 仍从源观测时刻开始计时，仍按执行 horizon 截断并移除过期前缀。

1. 冷启动使用 `prefetch_threshold_s`，默认 80 ms。
2. 只有请求完成并被控制循环取回后，才加入 `now - requested_at` 样本。该值包含虚拟请求延迟与控制 tick 的收取延迟；即使整个 chunk 已过期，样本仍参与估计。
3. 最近 W 个样本的线性插值分位数为 L，候选阈值为 `min(L + margin, E × dt_action)`。
4. 首个实测样本立即替换冷启动值；后续候选值相对当前阈值变化超过 hysteresis 才更新，升高和降低均适用。
5. 无在途请求、剩余覆盖不大于阈值、且距离上次请求达到最小间隔时，才捕获新观测并发起请求。

默认 W=20、分位数=0.95、margin=20 ms、hysteresis=10 ms、最小请求间隔=20 ms。配置见 `configs/adaptive_synthetic.toml`。最小间隔必须为正；阈值受执行覆盖预算限制，不会增大 action buffer。零延迟也不会在一个 tick 内形成请求自旋。

调度器不读取尚未完成请求的未来 ready_at 来更新估计，不接收故障轨迹对象。合成后端知道未来完成时间是离散事件模拟的需要，不构成调度器的先验。

## 供给不足的判定

当前单在途、单 chunk 替换策略中，观测锚定的 chunk 返回时只剩 `E×dt - L`。下一次请求完成至少需要 `max(L, min_request_interval)` 的起始间隔。因此在稳定延迟近似下，持续无断供至少要求：

```text
L + max(L, min_request_interval) <= E × dt_action
```

`supply_shortfall_s` 报告左侧超过右侧的量，状态变化产生 `supply_state_changed`。这是基于滚动估计的诊断，不能推广成所有 RTC、队列或控制策略的通用极限，也不是实际耗尽时长。实际供给仍以 fallback 指标为准。

超过整个 chunk 时长的延迟会使 chunk 全部过期；即使没达到这个极端，剩余覆盖也可能不足。检测到供给不足后仍以有界阈值请求，由已有 Executor 在无动作时执行环境定义的 fallback。这里没有通过扩充队列或执行陈旧动作隐藏失败。

## 合成故障轨迹

`LatencyTrace` 是按模拟时间定义的分段常量延迟场：请求在提交时选择一个时间槽，该槽决定此请求的总虚拟延迟。时间槽切换不会追溯修改已在途请求。超出文件末尾时保持最后一个槽的值。

`make_latency_trace()` 提供：

| profile | 含义 |
| --- | --- |
| `fixed` | 固定延迟 |
| `long_tail` | 基础延迟，加独立槽位尖峰；幅值使用有上限的 Pareto 分布 |
| `overload` | 从指定时间之后的首个槽起持续使用过载延迟 |

生成过程使用局部 seeded RNG；完整采样值保存在 JSON 中。两个调度器共用同一轨迹，在同一时刻请求时一定得到相同延迟，但由于请求时机不同，实际遇到的槽位与样本数量可以不同。这表示相同外部条件，不表示相同请求序列或总计算预算。

传入 `latency_trace` 时它覆盖原 `latency_s` 序列。未传时旧行为保持不变：按每个 episode 的请求序号循环读取 `latency_s`。两种模式都不 sleep，不注入真实 CUDA 计算。

```python
from robovla_runtime import RuntimeConfig, run_simulation
from robovla_runtime.backends import make_latency_trace

trace = make_latency_trace("long_tail", duration_s=20.0, seed=7)
config = RuntimeConfig(scheduler="adaptive_async", duration_s=20.0)
result = run_simulation(config, latency_trace=trace)
```

## 运行与重放

```powershell
$env:PYTHONPATH = "src"
python -m robovla_runtime.cli run --config configs/adaptive_synthetic.toml
python benchmarks/run_adaptive_matrix.py
```

矩阵默认执行 20 s 的 6 个场景：50 ms / 100 ms 固定延迟，3 个预先固定种子的长尾轨迹，以及第 3 s 起持续 260 ms 过载。每个场景都运行固定异步与自适应，共 12 次模拟。参数在实验前固定，不针对结果单独优化。

输出：

- `results/adaptive_matrix.md`：全部结果表及解释边界。
- `results/adaptive_matrix.json`：配置、完整输入轨迹、Python 版本、源码 SHA256 指纹和全部结果。
- `runs/adaptive_matrix/<scenario>/<scheduler>/`：每次运行的配置、输入轨迹、events.jsonl 与 summary.json。

原始运行目录必须不存在。重复执行时使用新的 `--output-dir`；`--summary-prefix` 决定摘要文件名，同名摘要可被重新生成。源码指纹包含 src 下的 Python 文件及矩阵脚本，忽略 CRLF/LF 差异。

复用矩阵输入进行单独实验：

```powershell
python -m robovla_runtime.cli run `
  --config configs/adaptive_synthetic.toml `
  --latency-trace runs/adaptive_matrix/long_tail_seed_7/adaptive_async/latency_trace.json
```

CLI 会复制输入 trace 到新 run 目录。`config.json` 的 `latency_s` 此时是被覆盖的默认参数，实际注入以同目录 `latency_trace.json` 为准。

## 日志与指标

新增事件：`adaptive_prefetch_started`、`adaptive_prefetch_triggered`、`prefetch_estimate_updated`、`supply_state_changed`。每次估计记录实测延迟、分位数、窗口样本数、候选/生效阈值、阈值是否被限制和供给缺口。`prediction_requested` 仍保留源观测与请求身份。

summary 增加请求数、按原因统计的 chunk 拒绝数量、自适应阈值与供给诊断统计。原有 fallback、observation age、deadline miss 和跟踪误差继续保留。request_count 包括 episode 结束时仍未完成的请求；延迟估计只使用已完成样本。所有 fallback 比例包含冷启动，没有筛掉失败时段。

结果必须同时检查请求数和 observation age。提高阈值可能增加推理需求；短窗口的 p95 不能预测突然尖峰，迟滞和窗口也可能延缓恢复。在合成过载下两种策略都失败，是需要保留的结果。确定性时钟的 deadline miss 为零不代表动作可用，也不能推出真机实时性。

回归测试保留原 8 组 mock 矩阵逐字段比较，新增覆盖不偷看未来、收取延迟、窗口淘汰、迟滞、最小请求间隔、过载/恢复、轨迹重放和 CLI 产物保存。
