# Worker 墙钟循环 smoke 验证

真实线程 + 模拟推理 sleep；每轮 60 个控制 tick。数值受操作系统调度影响，不是性能基准。

| 场景 | 预取 | 请求 | 超时 | 迟到结果 | fallback ticks | shutdown 超时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fixed | fixed | 6 | 0 | 0 | 10 | 0 |
| fixed | adaptive | 3 | 0 | 0 | 7 | 0 |
| spike | fixed | 16 | 1 | 1 | 16 | 0 |
| spike | adaptive | 5 | 2 | 2 | 18 | 0 |
| overload | fixed | 60 | 4 | 3 | 60 | 0 |
| overload | adaptive | 10 | 9 | 3 | 60 | 0 |

检查通过：所有运行完成控制 tick；所有已判超时的请求均未进入动作执行。

固定模式保留 latest-only mailbox；自适应模式通常只保留一个有效逻辑请求。因此这张表用于验证生命周期与故障处理，不隔离预取阈值这一单独因素。

deadline 为提交到控制循环取回的 60 ms；尖峰 120 ms，持续过载 150 ms。超时不会强杀推理线程。shutdown 等待上限为 100 ms；超过上限会留下显式事件。
