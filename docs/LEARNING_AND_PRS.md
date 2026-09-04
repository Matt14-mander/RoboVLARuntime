# 学习与 FluxVLA 贡献路线

日期：2026-09-04。配套 [开发路线](ROADMAP.md)。本文件是执行计划，PR 候选不是已确认缺陷，也不承诺提交数量或合并结果。

## 1. 从一条推理链路进入

学习主线：FluxVLA 的 SmolVLA 评测链路 → 性能测量 → 远程推理 → LeRobot 异步执行 → FluxVLA RTC → 控制约束。

每个阶段都完成“读代码 → 做最小实验 → 记录问题 → 验证上游是否需要 → 独立补丁”。不必读完整仓库再开始实验，也不必等 RoboVLA-Runtime 完工才贡献。

两个代码库分工：

- FluxVLA 的独立 checkout/fork：复现原生行为、制作符合其接口的补丁。
- RoboVLA-Runtime：放学习笔记、时序模拟、实验矩阵、跨策略对比和报告。

初期上游补丁不要求 FluxVLA 安装 RoboVLA-Runtime。将经验证的通用小模块或修复上游化，避免把研究框架整体塞进上游。阅读第三方代码时保留来源；复制实现时遵循其许可证与版权声明。

## 2. 分阶段任务

### A. 读懂并跑通一个配置（3–5 个学习时段）

材料：FluxVLA README 的安装与评测部分、`docs/CONTRIBUTING.md`、`scripts/eval.py`，随后在 checkout 中定位 SmolVLA 配置。

本次核对的 `scripts/eval.py` 路径为：配置加载/覆盖 → `build_runner_from_cfg` → `run_setup()` → `run()` → summary → cleanup。以这个入口向下追踪，不从训练器或所有模型类开始读。

在 FluxVLA 根目录执行定位搜索（以下是阅读命令，不会训练）：

```bash
rg --files configs fluxvla test | rg -i 'smol|libero|registry|builder'
rg -n 'build_runner_from_cfg|class LiberoEvalRunner|predict_action' scripts fluxvla
```

必须回答：配置的 runner/model 类型是什么？权重在哪里加载？图像如何变换？状态如何归一化？输出 chunk 的形状和单位是什么？哪一层选出实际执行动作？一次推理对应多少环境步？

实验：按实际环境安装最小必要依赖，选匹配的 SmolVLA LIBERO checkpoint 和配置，只运行一个任务的少量 episode。无 GPU 时先追踪配置、导入和张量契约；记录未执行部分，不声称完成模型复现。

交付：`notes/01_fluxvla_inference_path.md`（开发时创建），包含 commit SHA、文件/函数、数据形状表、命令、环境和结果。

PR 候选：经实际复现的配置说明错误、缺失前置条件、错误提示不明确。只有确认主分支仍存在且没有重叠 PR 时才提交。不要把个人安装日志直接作为上游文档。

### B. 理解 SmolVLA 的 chunk 生成（3–4 个时段）

只在这一阶段补读 LeRobot SmolVLA 的模型说明与实现，对照 FluxVLA 的输入适配和输出接口。学习 flow matching 的推理过程、采样步数、H 与执行步数 E 的区别，以及内部 action cache。

实验：固定同一合法观测，区分一次真正生成 chunk 和从缓存取出一个动作；打印 shape、dtype、设备和动作语义。追踪 episode reset 对缓存的影响。只有模型配置支持时才改变 H；优先保持 H 固定扫描 E。

交付：一页模型接口说明和一次真实推理记录。能够解释“输出多步动作”为什么不等于“每一步都重新看图”。

PR 候选：已有接口的边界错误或参数校验，必须先有最小复现。此阶段也可能没有值得提交的 PR，学习照常推进。

### C. 做第一份可信 profiler（约 1 周）

材料：FluxVLA 评测链路与远程推理文档中的现有 profiling；PyTorch 的 CUDA 计时与 profiler 官方资料。重点是分清 GPU 完成时间、端到端耗时、缓存命中和 warm-up。

实验：固定模型/输入/精度/采样步数；分别测预处理、模型生成、后处理与客户端总耗时。先做完整请求墙钟测量，再做细粒度诊断。同步计时会改变运行行为，诊断与正常运行结果分开。

交付：延迟分布、资源记录和计时边界说明，报告样本数与 instrumentation 开销。没有 GPU 时先用有确定延迟的 mock 验证统计，不给出 GPU 性能结论。

PR 候选：对现有统计补充可选的结构化输出、分位数、样本数或计时边界说明。先确认当前实现缺什么；这不是“新增 FluxVLA profiling”，因为上游已有相关能力。新增功能先讨论范围。

### D. 用远程推理学习系统可靠性（约 1 周）

阅读顺序：`docs/remote_inference_serving.md` → `fluxvla/engines/runners/serving/serve.py` → 同目录 `zmq_server.py`、`serializers.py` → 搜索客户端和 `BaseInferenceRunner`。

```bash
rg -n 'enable_profiling|infer_time|timeout_s|class BaseInferenceRunner' fluxvla
rg -n 'zmq.REQ|zmq.REP|predict_action|reset' fluxvla/engines/runners
```

学习：请求/响应协议、序列化成本、超时、异常传播、连接清理，以及“远程推理”与“异步执行”的区别。远程服务不必然让执行循环非阻塞。

实验：优先给现有通用服务层接 fake handler，检查是否能脱离模型启动；若导入依赖阻碍则如实记录。设置 50/200/500 ms 响应延迟、一次超时、断连和重连；检查之后的请求能否恢复、资源是否释放。模拟动作消费循环不需要真机器人。

交付：带 request_id 的事件 trace、一份正常/失败时序对照和确定性回归用例。

PR 候选：只有复现到实际问题后，提交超时恢复、错误信息、连接生命周期或协议测试的独立补丁。不要预先断定 ZMQ 代码有 bug。

### E. 学习异步调度并建立自己的基线（1–2 周）

材料：LeRobot async 文档，跟随文档定位当前版本的 policy server、robot client、action queue。重点阅读预取阈值、重叠融合、过期动作和 episode 生命周期。

实验：在 RoboVLA-Runtime 实现 sync、fixed async、time-aligned async；使用相同延迟 trace 比较耗尽时间、动作来源观测年龄和边界跳变。使用单在途请求、有界队列和虚拟时钟，先确保结果可复现。

交付：一张时间轴、一组同 trace 对比，以及动作时间语义说明。不要把 LeRobot 的参数或动作融合规则不加检查地搬到 FluxVLA。

PR 候选：若上游实际缺少且维护者认可，可添加局部诊断或无需机器人硬件的执行回归测试。自适应调度先留在独立项目验证；此时不提交通用 runtime 重构。

### F. 阅读 RTC，并提出有证据的调度改进（1–2 周）

材料：FluxVLA `docs/rtc.md`、`scripts/test_rtc.py`，搜索 `AlohaRTCInferenceRunner`；对照 LeRobot RTC 与 Physical Intelligence 的参考实现。

```bash
rg -n 'class AlohaRTCInferenceRunner|execute_horizon|async_execution|prefix_len' fluxvla configs
```

学习：已承诺执行的动作前缀如何与新预测对齐、推理延迟如何换算成动作步数、guidance 与 prefix 模式的假设。不同项目的 RTC 支持矩阵分别核对，不能因为 LeRobot 支持 SmolVLA RTC 就推断 FluxVLA 同样支持。

实验：先复现官方 RTC 可视化，再用真实已执行轨迹作为前缀进行时延实验。官方数据样本可视化与闭环表现分别评价；固定模型、延迟和资源预算后再比较自适应预取。

交付：小型消融与可解释失败案例。若新策略只减少耗尽却加剧动作过时，也必须记录。

PR 候选：经过验证的 delay/prefix 边界修复，或已讨论的可选预取策略。涉及 wire protocol、线程生命周期、公共接口的变更要先写设计说明。保持默认行为与兼容性可验证。

### G. 最后接入控制约束（约 2 周，依赖前述结果）

材料：FluxVLA `docs/franka.md`、轨迹后处理 PR #20、Ruckig 的当前/目标状态与速度/加速度/jerk 约束接口。

实验：先用 toy joint plant 比较无过滤和受限轨迹，测边界跳变、跟踪误差、滞后、不可行情形。之后才对接与动作空间匹配的仿真控制器。

PR 候选：补充既有实现的边界处理、测试或与 RTC 的已执行轨迹接口；先查看 #20 的最新状态并协调，避免重复添加 MPC/Ruckig 后处理包。

## 3. 第一周的具体安排

| 时段 | 做什么 | 完成证据 |
| --- | --- | --- |
| 第 1 次，约 2 小时 | 获取独立 FluxVLA checkout，记录 SHA，读贡献规范和 eval 入口 | 一张初版调用图 |
| 第 2 次，约 2 小时 | 找到一份 SmolVLA 配置，沿 model/runner 类型追踪 | 配置到类的对应表 |
| 第 3 次，约 2 小时 | 追踪图像、state、action 的 shape 和归一化 | 输入输出契约表 |
| 第 4 次，约 3 小时 | 在可用环境运行最小推理/少量评测；记录阻碍 | 原始日志或明确未通过的门槛 |
| 第 5 次，约 2 小时 | 阅读 serving 和已有 profiling | 一次请求的计时边界图 |
| 第 6 次，约 2 小时 | 整理问题，搜索 open/closed issue 与 PR，选一个最小复现 | 一个有证据的候选，或明确暂无候选 |

第一周最重要的验收：能指出动作从哪个函数产生、在哪个函数被消费，以及等待推理期间执行端在做什么。PR 是理解代码和解决问题后的产物。

## 4. 如何形成连续、有价值的 PR

建议顺序：复现相关的文档/错误提示 → 单点 bug 与回归 → profiling 增强 → 调度改进 → 控制接口扩展。不是每阶段必须一篇 PR；围绕一个问题合并相关修改，不为数量拆碎补丁。

每个候选保留一张记录卡：上游 SHA、用户可见问题、最小复现、预期/实际、相关 issue/PR、变更范围、验证方式、维护者反馈。

提交条件：

1. 最新主分支仍能复现，已检查 open 与 closed PR 的重复工作。
2. 修复有用户价值；功能有明确用例和已讨论的范围。
3. 一个 PR 解决一个问题，描述触发条件与前后行为。
4. 验证与风险匹配：文档检查实际命令，bug 提供回归，性能变更报告测量条件与开销。
5. 遵循上游格式、pre-commit 和 CI，持续回应 review。

上游贡献规范要求新功能先开 issue 讨论；公开 issue 页面本次显示创建受限，实际权限需登录后核对。若确实无法创建，可准备好复现与设计说明再按维护者提供的渠道沟通。当前仅规划，不发送消息、开 issue 或提交 PR。

## 5. 阅读入口与核对记录

- [FluxVLA 贡献规范](https://github.com/FluxVLA/FluxVLA/blob/main/docs/CONTRIBUTING.md)
- [FluxVLA 评测入口](https://github.com/FluxVLA/FluxVLA/blob/main/scripts/eval.py)
- [FluxVLA 远程推理](https://github.com/FluxVLA/FluxVLA/blob/main/docs/remote_inference_serving.md)
- [FluxVLA RTC](https://github.com/FluxVLA/FluxVLA/blob/main/docs/rtc.md)
- [FluxVLA Franka](https://github.com/FluxVLA/FluxVLA/blob/main/docs/franka.md)
- [FluxVLA PR #20：MPC/Ruckig 轨迹后处理](https://github.com/FluxVLA/FluxVLA/pull/20)
- [LeRobot SmolVLA](https://huggingface.co/docs/lerobot/smolvla)
- [LeRobot async](https://github.com/huggingface/lerobot/blob/main/docs/source/async.mdx)
- [LeRobot RTC](https://huggingface.co/docs/lerobot/rtc)
- [Physical Intelligence RTC 参考实现](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)
- [Ruckig](https://github.com/pantor/ruckig)

本次通过上游网页核对了入口、文档、贡献规范和相关 PR；未完成完整源码 checkout 或运行复现。网站抓取有缓存时间差，开始改代码前必须以本地最新 SHA 再核对。具体 SmolVLA/runner 文件通过上述搜索定位，避免凭记忆猜路径。
