# 自适应预取：合成延迟对照

同一场景共用按模拟时间定义的延迟轨迹；延迟在请求提交时取样。每个策略只有一个在途请求，使用相同的动作 buffer、fallback 和 toy plant。

控制周期 20 ms，生成/执行 horizon 都为 10（200 ms）；固定阈值及自适应冷启动阈值均为 80 ms。自适应使用最近 20 个已完成请求的 p95，加 20 ms 余量、10 ms 迟滞，最小请求间隔 20 ms。

| 场景 | 调度器 | 请求数 | fallback ticks | fallback 比例 | 观测 age p95 (ms) | 过期 chunks |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fixed_50ms | fixed_async | 167 | 3 | 0.3% | 160.0 | 0 |
| fixed_50ms | adaptive_async | 167 | 3 | 0.3% | 160.0 | 0 |
| fixed_100ms | fixed_async | 167 | 170 | 17.0% | 180.0 | 0 |
| fixed_100ms | adaptive_async | 200 | 5 | 0.5% | 180.0 | 0 |
| long_tail_seed_7 | fixed_async | 158 | 146 | 14.6% | 160.0 | 2 |
| long_tail_seed_7 | adaptive_async | 205 | 143 | 14.3% | 160.0 | 2 |
| long_tail_seed_17 | fixed_async | 155 | 135 | 13.5% | 140.0 | 8 |
| long_tail_seed_17 | adaptive_async | 199 | 144 | 14.4% | 160.0 | 8 |
| long_tail_seed_29 | fixed_async | 162 | 64 | 6.4% | 140.0 | 2 |
| long_tail_seed_29 | adaptive_async | 158 | 63 | 6.3% | 160.0 | 2 |
| sustained_overload | fixed_async | 91 | 848 | 84.8% | 140.0 | 65 |
| sustained_overload | adaptive_async | 88 | 845 | 84.5% | 160.0 | 65 |

固定延迟为 50/100 ms。长尾场景每 0.5 s 一个时间槽，基础延迟 40 ms，15% 概率进入尖峰，尖峰为 min(300 ms, 120 ms × Pareto(alpha=2))；种子预先固定为 7、17、29。持续过载从第 3 s 起切换到 260 ms。

结果包含冷启动和全部失败 tick；请求数不同意味着采样到的槽位及完成样本数可能不同。这是同一外部延迟场下的闭环比较，不是强制每个请求经历相同延迟序列。

本策略的 chunk 从观测时刻起算，返回时只剩 E×dt−L 的覆盖。单在途、单 chunk 替换情况下，持续无断供至少要求 max(L, 最小请求间隔) ≤ E×dt−L。自适应只能调整请求时机，无法恢复已过期动作或消除持续供给不足。

固定 80 ms 不是针对每个场景调参后的最优固定阈值；本表不声称自适应在所有负载下更优。p95 也不是尾延迟保证，突发尖峰只能在完成后影响估计。

自适应可能增加请求数，因而这里并未强制两种策略消耗相同总计算量。持续过载后的结果差异也可能来自切换时刻前的请求相位，不能解释为解决了过载。

所有延迟均为虚拟时间注入，不使用 sleep，不代表真实 GPU 争用、网络队列或 VLA 成功率。deadline miss 在该确定性循环中为零也不表示动作供给充足。
