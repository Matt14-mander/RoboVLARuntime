# 进程隔离推理与故障恢复

FluxVLA 的可序列化模型工厂、启动预热和重启后重置已在第五轮接入，见 [FluxVLA 进程启动说明](FLUXVLA_PROCESS.md)。可选 backend.startup(report) 在 ready 前执行，其进度消息不会延长启动超时；未提供该钩子的后端保持原行为。

`ProcessPolicyBackend` 实现与线程后端相同的 AsyncPolicyBackend 接口，可直接交给 `run_realtime_runtime()`。控制线程仍使用 `worker_async`，支持 fixed 和 adaptive 预取。默认线程模式保持不变。

## 运行模拟后端

在仓库根目录运行：

```powershell
$env:PYTHONPATH = "src"
python -m robovla_runtime.cli run-worker-mock --config configs/process_adaptive.toml
python -m benchmarks.run_process_recovery --output-dir runs/process_recovery_rerun
```

恢复演示的输出目录必须不存在；再次运行时换一个 `--output-dir`。摘要默认为 `results/process_recovery.md` / `.json`，可用 `--summary-prefix` 更换位置。CLI 保存 config、events 和 summary；恢复演示另外保存故障计划和进程创建计数。所有模拟均不依赖模型或 GPU。

| 配置 | 默认 | 含义 |
| --- | ---: | --- |
| `worker_isolation` | `thread` | `run_realtime_mock` 选择 thread / process |
| `process_startup_timeout_s` | 10.0 | 子进程启动后，等待导入、工厂构建与 ready 握手的上限 |
| `process_execution_timeout_s` | 1.0 | 派发后 IPC、模型 reset、推理与返回结果的 watchdog 上限 |
| `process_terminate_grace_s` | 0.1 | terminate 后等待回收的上限；必要时 kill 并再等待一次 |
| `process_max_restarts` | 3 | 连续故障允许的自动重启次数；初始启动不占用次数 |
| `request_timeout_s` | 0.5 | 保留上一轮语义：提交到控制循环收集结果的 deadline |
| `worker_close_timeout_s` | 0.1 | 控制线程在退出时等待整个监督线程结束的上限 |

示例配置将 close 等待设为 1 秒，为终止、进程回收和 IPC 退出留出时间。实际模型的构建时间可能远超模拟，需要单独设置 startup timeout。工厂的父进程序列化及 OS 的 `Process.start()` 位于监督线程，不占控制线程，但不在子进程启动 watchdog 的计时区间内。

## 生命周期与超时

模型只在 `spawn` 子进程中创建和使用；不会从父进程 fork 已初始化的 CUDA 状态。父进程有一个监督线程和一个 IPC 线程：监督线程负责 watchdog、终止、回收及重启；IPC 线程负责序列化和管道读写，避免部分消息读取阻塞 watchdog。控制线程的 submit、poll、expire、reset 都只操作短临界区中的 mailbox、结果和信号。

正常时只有一个实际执行请求和一个 latest-only 等待槽。adaptive 调度仍限制一个有效逻辑请求。启动期间可以替换或取消等待请求，不会仅因排队请求过期而杀掉正在正常加载模型的进程。scheduler 向进程后端传递 deadline，派发前再次检查，已过期的请求不进入模型。

请求超时通过 `expire_request` 撤销在途结果资格并通知监督线程。若请求正在执行，监督线程终止子进程、join 回收，必要时升级到 kill；新请求等待旧进程完全退出。若请求已经完成，只按已有规则丢弃迟到结果，不为迟到收集而重启空闲模型。独立 watchdog 取 execution timeout 与已传入请求 deadline 剩余时间的较小值，即使调用方没有继续 poll，也能终止卡住的 reset 或推理。

启动超时、异常退出、IPC 错误以及模型异常同样使当前进程失效。模型异常可能已修改内部状态，因此也会销毁进程。失败由 PredictionFailure 和诊断事件报告，不阻塞控制循环；旧 buffer 中仍有效的动作可以继续执行，缺少动作时走环境 fallback。

重启采用按需方式：有下一条等待请求时才创建新进程，不在空闲时反复拉起。每次使用新 pipe、新 IPC 队列和新的 process generation，旧进程回包无法进入新进程的通道。request ID 在后端整个生命周期内递增，scheduler 继续核对 episode、观测和 buffer generation。

连续故障超过 `process_max_restarts` 后，后端进入不可用状态，后续请求立即返回 ProcessUnavailable，控制循环继续 fallback，不会无限启动子进程。正常收到一次推理结果后连续故障计数归零；显式 reset 也会重新启用预算。该“成功”指后端正常完成，不保证结果在 scheduler 的收集 deadline 内被执行。

## 状态重置与退出

新进程首次处理观测前必须调用模型 `reset(episode_id)`。同一进程中 episode / observation generation 改变也触发 reset。显式 backend.reset 会使所有旧请求失效并终止旧进程，即使再次使用相同 episode ID，也不会继承原模型缓存。新请求必须等旧进程回收后才能执行。

cancel_episode 撤销等待请求和结果，并终止该 episode 的在途推理；close 进一步关闭整个监督器。close 返回 true 表示子进程和 IPC 线程已经回收。若调用方给出的等待太短，会返回 false，监督线程仍继续清理；运行时记录 `worker_shutdown_timeout`，调用方可再次 close 等待清理完成。

如果 terminate/kill 后仍不能确认旧进程退出，后端停止接收实际推理，不会启动第二个模型进程；记录 `process_termination_failed`。监督线程不可恢复地退出时，应建立新的 backend，不能用 reset 假装完成恢复。

## 接入真实模型的接口

使用可通过标准 pickle 序列化的模块顶层函数或 callable dataclass 作为工厂，携带模型路径、配置等普通数据。不要传模型实例、CUDA context、lambda、局部函数或只能在 multiprocessing spawn 阶段传递的同步句柄。工厂在父进程先序列化，再将字节串交给子进程解码构建，避免 Windows 在序列化失败后留下半启动进程。

```python
from robovla_runtime.backends import ProcessBackendConfig, ProcessPolicyBackend
from robovla_runtime.runtime import run_realtime_runtime

# Put build_policy in an importable module; load the model inside this function.
def build_policy():
    return MyBlockingPolicy(model_path="...")

if __name__ == "__main__":
    backend = ProcessPolicyBackend(
        build_policy, action_spec,
        config=ProcessBackendConfig(startup_timeout_s=120, execution_timeout_s=2),
    )
    try:
        result = run_realtime_runtime(runtime_config, backend)
    finally:
        backend.close(timeout_s=2)
```

此处 MyBlockingPolicy、action_spec 和 runtime_config 由集成方提供。手工构建后端时，其 ProcessBackendConfig 才是实际配置；RuntimeConfig 中 process 字段由 `run_realtime_mock` 工厂消费。子进程会校验实际 ActionSpec 与父进程声明一致。

观测 payload 与返回结果也必须可序列化；当前实现会复制数据，不提供共享内存图像传输。pickle 通道仅用于本机可信父子进程。子进程是 daemon，暂不支持模型内部再启动 multiprocessing worker。

## 事件与验收

事件包括 `process_worker_started`、`process_worker_ready`、`process_request_started`、`process_worker_fault`、`process_worker_terminated`、`process_worker_restarted`、`process_restart_exhausted`，以及派发前过期、无法终止、IPC 卡住和监督器异常事件。包含 process generation，相关事件还包含 PID 或 request ID。日志时间是控制线程收集诊断的时间，不是系统终止发生的精确时刻。summary 的 `process_worker` 汇总启动、重启、终止、故障和预算耗尽等计数。

测试覆盖真实 spawn、历史状态重置、推理/reset/启动卡死、异常退出、失败后的恢复、序列化失败、排队替换和过期、episode 取消、重启预算及关闭回收。恢复演示包含七组场景，每组 200 个 tick，检查超时请求未被执行、所有子进程被回收、单次故障后恢复动作、连续故障后停止重启。状态计数按新进程中的调用次数校验，允许先前完成的动作因收集过晚而被丢弃。

本轮验证的是 OS 进程生命周期与合成模型状态。真实 FluxVLA / CUDA 模型加载、驱动卡死、设备资源回收和真实机器人安全停止尚未验证；进程隔离也不构成硬实时保证。
