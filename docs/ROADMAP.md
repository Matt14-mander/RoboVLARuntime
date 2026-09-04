# RoboVLA-Runtime 开发路线

规划日期：2026-09-04。当前仓库只有 README，以下均为待开发内容，不代表已有实现或实验结果。

## 1. 项目定位与边界

目标：构建可插拔的 VLA 执行与评测框架，量化推理延迟、抖动、动作时效、执行连续性和闭环任务表现之间的关系，并验证延迟感知调度的适用边界。

第一版交付：CPU 可运行的确定性调度实验 + 一个真实 VLA 的性能记录 + 一组可复现的基线对比。闭环仿真是下一道独立门槛，真机和 MPC 是后续扩展。

核心研究问题：

- 在相同模型、任务和资源预算下，自适应预取能否降低缓冲耗尽和 observation age？
- 对齐、丢弃或融合 action chunk，分别如何影响动作跳变、跟踪误差和任务成功率？
- 约束层改善平滑性时，是否因为增加执行滞后而损害任务表现？

这些是假设，允许实验得到“无收益”或“仅在部分负载下有效”的结论。工程完整性与算法新颖性分别评价。

### 已有工作与差异化

| 已核对的已有能力 | 本项目应采用的做法 |
| --- | --- |
| LeRobot 已有异步推理、预取阈值和重叠动作聚合 | 将官方实现作为参照；简单 async queue 仅作为基线。[官方说明](https://github.com/huggingface/lerobot/blob/main/docs/source/async.mdx) |
| LeRobot 已集成 RTC，通过生成过程中的引导改善 chunk 衔接 | 与普通队列调度区分；在后端支持时增加 RTC 对照，不能将末端插值称为 RTC。[RTC 文档](https://huggingface.co/docs/lerobot/rtc) |
| FluxVLA 已支持远程推理和部分 RTC 能力；Franka 文档包含 Ruckig 控制器 | 优先贡献测量、故障复现或独立适配器，先评估差距再决定 PR。[仓库](https://github.com/FluxVLA/FluxVLA)、[Franka 文档](https://github.com/FluxVLA/FluxVLA/blob/main/docs/franka.md) |
| SmolVLA 提供面向消费级硬件的模型；base checkpoint 的用途包括任务微调 | 可用于性能测量；闭环评测另选机器人、任务和归一化统计匹配的 checkpoint。[模型卡](https://huggingface.co/lerobot/smolvla_base) |

不以社区的单张显卡训练案例推导本机可达到的性能；不把 PR 合并或新颖性视为既定结果。以上是定向核对，不是穷尽的相关工作综述。

## 2. 先固定时间与动作语义

### 三个频率、两个 horizon

- `f_obs`：观测更新频率。
- `f_action = 1 / dt_action`：模型训练数据对应的动作时间网格。
- `f_servo`：底层控制器更新频率；可以高于动作频率，但必须由明确的插值或控制接口连接。
- `H`：一次生成的动作数量，索引为 `0 ... H-1`。
- `E`：计划执行后再替换/重规划的动作数量，`E <= H`。

不能任意把 10 Hz 训练动作改为 50 Hz 逐点执行。改变 H、截取前 E 步、改变采样迭代数是不同实验变量；截取输出通常不减少已经发生的生成计算。

200 ms 在 50 Hz 执行循环中对应约 10 个更新周期；只有动作网格也为 50 Hz 且时间原点明确时，才对应 10 个模型动作点。异步隐藏等待，不自动消除状态过时。

### 每条观测、chunk 和执行记录都有身份与时钟

| 对象 | 最小字段 |
| --- | --- |
| Observation | episode_id、obs_id、capture_time、receive_time、clock_domain、图像、机器人状态、指令 |
| ActionSpec | joint/Cartesian、absolute/delta、单位、坐标系、维度、dt_action、夹爪语义 |
| ActionChunk | request_id、obs_id、动作数组、生成起止时间、目标时间网格、有效期、ActionSpec |
| ExecutionRecord | tick_id、目标/实际下发时间、来源 chunk 与动作索引、原始/过滤命令、实测状态、fallback 原因 |

同机使用 monotonic clock；多机同时记录各自本地耗时和客户端往返耗时。没有时钟同步和偏移误差界时，不直接相减两台机器的 monotonic 时间。没有硬件反馈时，指标只能叫“下发”，不能叫“物理执行”。

初版每个策略只允许一个在途请求，观测队列容量为 1，新观测覆盖尚未处理的旧观测。动作缓冲有上限；episode 重置清空历史并拒收上一 episode 的返回。接口仍保留 request_id，便于处理网络重试和未来并发。

### 延迟与调度规则

串行路径可分解为采集、预处理、推理、后处理、通信和排队耗时；异步运行存在重叠，端到端延迟必须由事件时间戳计算，不能将所有 span 直接求和。

定义：

```
observation_age = command_send_time - source_observation_capture_time
buffer_coverage = 最后一个连续有效动作覆盖的结束时刻 - now
prefetch: buffer_coverage <= rolling_quantile(request_to_buffer_ready) + margin
```

从滚动 p95 开始，记录估计误差；p95 不是绝不超时的保证。设置最小请求间隔、迟滞和冷启动估计，避免频繁触发。统计当前请求排队、通信和推理的完整耗时。

当延迟分位数超过可用动作覆盖时长，必须报告超出当前配置的供给能力并触发降级；异步无法解决持续性的供给不足。不可通过无限增加缓冲掩盖 observation age。

对按观测时刻锚定、具有固定 dt 的绝对目标序列，可研究按目标时间移除过期前缀。delta 动作依赖累积状态，不能默认跳过前缀或线性融合；不支持的语义显式拒绝。时间对齐也不等于状态补偿，后者需要实测状态、模型或生成时条件化。

## 3. 建议架构

```text
Observation source -> latest observation -> Policy worker
                                             |
                                      timestamped chunk
                                             |
                          Scheduler -> bounded action buffer
                                             |
                        Execution loop -> constraint filter
                                             |
                                simulator / robot adapter
                                             |
                                       state feedback

All components -> event trace -> metrics -> reproducible report
```

Python 负责策略编排、调度实验和日志；底层伺服通过已有控制器接入，不对通用操作系统上的 Python 循环宣称硬实时保证。先做进程内/本机通信，真实网络在后续加入。

建议目录（尚未创建实现）：

```text
src/robovla_runtime/
  core/          # 数据契约、ActionSpec、Clock、配置
  backends/      # mock、replay、smolvla，后续 fluxvla
  scheduling/    # sync、fixed_async、time_aligned、adaptive
  execution/     # buffer、executor、watchdog
  constraints/   # identity、joint filter，后续 Ruckig/QP
  profiling/     # event trace、metrics、报告
  environments/  # toy plant，后续 LIBERO/robot adapter
configs/
benchmarks/
tests/
docs/
```

核心包不强制依赖 PyTorch、ROS 或 MuJoCo；真实模型和仿真作为可选依赖。MockBackend 返回可验证轨迹，ReplayBackend 复用记录用于时序实验；replay 不能替代真实闭环 policy。

## 4. 开发阶段与验收门槛

工期假设：一人每周约 15–20 小时，约 10–12 周完成含仿真的版本；这是容量估算，模型适配、设备可用性可能改变排期。当前尚未确认 GPU、内存、机器人和时间预算。

| 阶段 | 参考时间 | 开发内容 | 进入下一阶段的门槛 |
| --- | --- | --- | --- |
| M0 契约与环境 | 第 1 周 | 明确 ActionSpec、时钟、指标；检查本机环境；建立轻量包与 CI | CPU mock 可导入；配置能拒绝未知动作语义；记录环境信息 |
| M1 CPU 最小闭环 | 第 2–3 周 | VirtualClock、toy plant、可控延迟后端、sync/fixed async、事件报告 | 同一 seed 事件顺序可重现；延迟跨越 buffer coverage 时能复现耗尽；零延迟不凭空耗尽 |
| M2 真实性能画像 | 第 4 周 | SmolVLA adapter、有效样本输入、计时与资源记录 | 真实生成 chunk；区分模型生成与内部缓存取动作；输出分位数、显存和完整配置 |
| M3 延迟感知调度 | 第 5–6 周 | 时间对齐、过期丢弃、自适应预取、fallback、消融 | 相同 traces 比较全部基线；覆盖尖峰、乱序、超时、重置；明确收益和失败区间 |
| M4 任务闭环 | 第 7–9 周 | 单一仿真平台、匹配 checkpoint、少量任务、时延注入 | 无人工延迟的基线先具备可测成功率；仿真推理期间时间继续推进；输出任务级统计 |
| M5 约束与发布 | 第 10–12 周 | joint 限制层或 Ruckig、干预反馈、技术报告和复现命令 | 同时报执行质量、滞后和成功率；完成 scheduler × filter 消融；发布本地可复现产物 |

### M1：第一份有价值的结果，不等真实模型

toy plant 使用简单关节动力学和目标跟踪策略；它验证系统行为，不代表 VLA 的任务能力。先实现虚拟时间离散事件模拟，再加入 wall-clock 模式验证实际循环抖动。

必须具备：动作不重复消费；时间索引单调；过期/上一 episode chunk 不执行；队列有界；后台超时不阻塞执行循环；丢弃有原因码；warm-up 和 shutdown 的 fallback 单独统计。

### M2：真实性能测量

从 batch=1、固定相机数/分辨率、固定精度和固定推理步数开始。记录 warm-up 与稳态，建议稳态至少 200 次用于初步画像；尾延迟结论需更多样本，样本不足时明确标注。

CPU 墙钟计时必须覆盖 GPU 工作完成；细粒度 GPU span 使用 CUDA event 等后端计时。逐阶段强制同步会扰动吞吐，因此分别运行“细粒度诊断”和“正常执行”两种测量。无法观测 encoder/action expert 时只报模型整体，不伪造拆分。

注意部分 policy 的 `select_action` 会从内部缓存返回动作。适配器应调用真实 chunk 生成入口，或标记是否发生实际推理；不能将缓存读取时间视作模型延迟。

先做硬件探测再决定推理环境；Windows 可承担核心开发，模型/仿真如有依赖障碍再选择 WSL2 或 Linux。没有 GPU 时完成 M1/M3，并用短时远程 GPU 获取真实 traces；这只能支持时序回放结论。

### M3：先提高可解释性，再做自适应

实现顺序：同步 → 固定预取异步 → 显式时间对齐 → 滚动延迟预取。一次只改变一个机制。

fallback 由机器人适配器定义：位置保持、受限减速或停止请求。不能把“重复上一条命令”作为所有动作空间的默认行为，尤其是速度和 delta 命令。无新鲜动作、无新鲜状态、约束不可行都有独立路径。

### M4：不被 checkpoint 拖住

LIBERO 是候选起点，不是必须绑定的第一环境。先核对 checkpoint 的任务集、动作维度/单位、图像处理、控制器和归一化统计，再跑原生同步基线。若两到三个工作日仍不能获得有意义的闭环行为，暂停 scheduler 接入排错，换已验证的匹配组合；不直接启动大规模训练。

仿真必须区分逻辑时间和墙钟时间。推理期间如果环境冻结，测不到真实 stale action 后果。时延模拟应让旧动作或 fallback 持续驱动环境，并将模拟延迟与真实运行速度分别报告。

### M5：控制贡献从小范围开始

首先仅支持一种明确的关节位置命令。基于实际 dt、前次命令和可用实测状态联合施加位置/速度/加速度限制；逐项独立 clip 可能产生彼此冲突的结果，需要可行性检查和明确 fallback。

如加入 Ruckig，它是运动生成工具，需要提供当前/目标状态及限制；不能把它当作通用碰撞安全证明。[Ruckig 官方仓库](https://github.com/pantor/ruckig)

Cartesian 或 delta 输出必须有匹配的 IK/控制适配器，不直接套关节限幅。分别记录下发命令与实际关节轨迹；命令满足约束不代表真实机械臂一定满足。被过滤后的已执行轨迹也需反馈给后续拼接/RTC，而非继续假设原始 chunk 已被完整执行。

暂不做完整 MPC、碰撞规划、多机器人和分布式集群。只有存在可解释的跟踪/动态约束问题，且简单过滤确实不够时，再定义 MPC 状态模型、代价和计算预算。

## 5. 实验方案

### 基线

| ID | 方案 | 用途 |
| --- | --- | --- |
| B0 | 同步推理与执行，等待时使用相同 fallback | 显示推理阻塞的影响 |
| B1 | 固定阈值 async FIFO | 隔离异步的收益与动作过时问题 |
| B2 | B1 + 时间对齐/过期处理 | 隔离时间语义处理的收益 |
| B3 | B2 + 自适应预取 | 检验本项目调度假设 |
| B4 | 官方 async / RTC，兼容时接入 | 避免只和弱基线比较；注明模型适用范围和计算差异 |

B1 是本项目定义的朴素基线，不声称等价于官方 LeRobot。M5 对 B2/B3 比较无过滤、有过滤；预算允许再扩大。所有方案共用底层执行器、任务超时和 fallback。

### 先筛选，后扩大

1. CPU 诊断：围绕 `L / (E × dt_action)` 取 0、0.25、0.5、1、1.5，覆盖固定延迟、长尾尖峰、周期拥塞；这些是合成压力条件。
2. 真实后端：先固定 H，扫描合法 E 和预取参数；记录真实推理开销，再叠加可解释的网络/排队时延。避免重复计算同一延迟。
3. 闭环：先选 2–3 个任务，每个配置 10 个配对 episode 排错；最终重点条件建议每任务至少 30 个 episode，并按置信区间宽度决定是否增加。小样本不声称细小优势。
4. 资源：先测单一资源受限因素，如 GPU 争用或通信延迟，再做混合；合成 sleep 不能代表显存压力、热降频或真实 GPU 争用。

禁止从大矩阵里只挑好结果。使用独立调参集，固定主评测配置；按任务报告成功率和区间，配对 seed 比较，保留失败与超时。完成时间同时给出成功样本统计和失败/超时信息，避免幸存者偏差。

### 必须输出的指标

| 类别 | 指标 |
| --- | --- |
| 推理/资源 | 实际 chunk 生成延迟 p50/p95/p99、峰值 allocated/reserved 显存、模型配置、资源争用配置 |
| 时序 | observation age、请求排队时间、动作 buffer 驻留时间、deadline miss rate、实际 tick 间隔及抖动 |
| 供给 | 有效 buffer coverage、耗尽时长占比、fallback 次数/时长、丢弃原因 |
| 动作/控制 | chunk 边界跳变、过滤干预比例/幅度、跟踪误差、速度/加速度/jerk、约束违反 |
| 任务 | 每任务成功率与置信区间、任务完成时间、固定时间预算内完成数量 |

deadline miss 定义为实际下发时间超过计划下发时间加预先固定容差的 tick 占比；fallback 单独记录，不能因持续发送保持命令就声称供给正常。jerk 使用实际 dt，并声明差分/滤波方法与动作单位。不同动作空间不混合比较数值。

主要图表：延迟—耗尽率曲线、延迟—任务成功率曲线、observation age 分布、chunk 衔接轨迹、平滑性—跟踪滞后权衡图。每次运行保存配置、版本、seed、原始 trace 与摘要；以脚本生成静态图和报告即可，早期不做 Web dashboard。

## 6. 第一个开发迭代的任务清单

建议只安排 M0/M1，预计 10–15 个工作时段：

- [ ] 记录 CPU/GPU/显存/内存、Python、操作系统和可用开发时间。
- [ ] 确定第一种 ActionSpec：toy plant 的绝对关节位置 + 显式 dt。
- [ ] 建立 pyproject、src layout、配置加载和 CPU CI。
- [ ] 实现数据对象、VirtualClock 与事件 schema。
- [ ] 实现 MockBackend、简单 plant 与同步 runner。
- [ ] 实现有界 buffer、固定预取 async runner 与 fallback。
- [ ] 加入固定延迟、尖峰和超时注入。
- [ ] 验证时间顺序、重复消费、过期返回、episode reset 和 buffer 耗尽。
- [ ] 输出 sync/async 同 trace 对照报告；明确 mock 结论范围。
- [ ] 再决定 SmolVLA 依赖版本与运行环境。

第一迭代验收演示：在 CPU 上运行同一个跟踪任务，改变推理延迟后，报告能解释何时同步等待、何时异步覆盖延迟、何时两者都需要 fallback，并能追溯每次动作来自哪个观测和 chunk。

## 7. 发布与上游贡献

- v0.1：确定性 mock benchmark、profiler、基线与本地报告。
- v0.2：真实 SmolVLA adapter、时间对齐、自适应预取与消融。
- v0.3：一个匹配的闭环仿真组合、约束层和完整技术报告。
- 后续：真机验证、FluxVLA adapter 或上游 PR；先用明确的 issue/设计说明和维护者对齐。当前规划不发送消息或提交 PR。

有真机之后，才回答“该硬件与任务组合是否达到目标频率”；只有仿真/回放时，结论限定于相应环境。简历表述基于实际完成的模块和测得数据，不提前写成功率提升或硬实时能力。
