# 第一阶段基线结果

生成日期：2026-09-05

运行命令：

```powershell
$env:PYTHONPATH = "src"
python benchmarks/run_phase1_matrix.py
```

条件：一维 toy joint、2 秒 episode、20 ms 动作周期、10 步生成与执行 horizon（200 ms coverage）、80 ms 固定预取阈值。每种条件运行 100 个控制 tick。

| 推理延迟 | 调度器 | fallback ticks | fallback 比例 | observation age p95 | 最终跟踪误差 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 ms | sync | 0 | 0% | 180 ms | 0 |
| 0 ms | fixed async | 0 | 0% | 100 ms | 0.0000011 |
| 50 ms | sync | 30 | 30% | 180 ms | 0 |
| 50 ms | fixed async | 3 | 3% | 160 ms | 0.0000016 |
| 150 ms | sync | 80 | 80% | 180 ms | 0.20 |
| 150 ms | fixed async | 76 | 76% | 180 ms | 0.04 |
| 250 ms | sync | 100 | 100% | 无有效动作 | 1.0 |
| 250 ms | fixed async | 100 | 100% | 无有效动作 | 1.0 |

本实验说明当前实现具备预期的基本行为：可覆盖的 50 ms 延迟下，固定预取减少了 buffer underrun；150 ms 延迟超过 80 ms 预取阈值，收益很小；250 ms 延迟超过整个 200 ms chunk coverage，所有动作在到达时均已过期，两种策略都无法供给有效动作。

这些数值来自确定性 mock policy 和简化 plant，只验证软件时序与指标计算。它们不能用于推断 FluxVLA、SmolVLA、LIBERO 或真实机器人的速度、平滑性和任务成功率。最终跟踪误差也受到 toy policy 和速度上限影响，不适合单独用于评价调度器。

机器可读结果见 [phase1_matrix.json](phase1_matrix.json)。
