# RoboVLARuntime
A resource-aware runtime and evaluation framework for real-time Vision-Language-Action model deployment, featuring latency-aware scheduling, action chunk execution, profiling, and pluggable VLA backends.

Status: phase-one deterministic runtime core is implemented. It provides a
virtual clock, mock chunked policy, time-aligned action buffer, synchronous and
fixed-prefetch schedulers, a toy joint plant, structured event traces, and
summary metrics. Phase two adds a dependency-injected `FluxVLABackend` and a
worker-backed real-time loop so GPU inference can overlap control execution;
real checkpoint and LIBERO validation remain pending.

Runtime and Executor now accept an `EnvironmentAdapter`, with ToyJoint as the
default implementation. Action contracts explicitly describe representation,
frame, units, normalization, and gripper semantics. Execution currently supports
absolute joint positions only. Episode reset propagates through the environment,
buffer, worker, preprocessing, and model reset hooks, including reused episode IDs.

## Quick start

Python 3.11 or newer is required. The phase-one runtime has no third-party
runtime dependencies.

```powershell
$env:PYTHONPATH = "src"
python -m robovla_runtime.cli run --config configs/phase1_sync.toml
python -m robovla_runtime.cli run --config configs/phase1_async.toml
```

Each run writes its resolved configuration, `events.jsonl`, and `summary.json`
under `runs/`. Run the standard-library test suite with:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Run the phase-one latency matrix with:

```powershell
$env:PYTHONPATH = "src"
python benchmarks/run_phase1_matrix.py
```

The included results are deterministic timing experiments with a toy plant;
they are not VLA task-success or real-robot performance claims.

- [开发路线与验收标准（中文）](docs/ROADMAP.md)
- [学习顺序与 FluxVLA PR 路线（中文）](docs/LEARNING_AND_PRS.md)
- [第一份源码分析笔记：FluxVLA SmolVLA LIBERO 调用链](notes/01_fluxvla_inference_path.md)
- [第一阶段开发规格：确定性 Runtime Core](docs/PHASE1_IMPLEMENTATION.md)
- [第一阶段基线结果](results/phase1_matrix.md)
- [第二阶段 FluxVLABackend 接入说明](docs/PHASE2_FLUXVLA_BACKEND.md)
- [FluxVLA Profiler 使用与计时语义](docs/FLUXVLA_PROFILER.md)
- [环境适配、动作契约与 reset](docs/ENVIRONMENT_ADAPTER.md)
- [自适应预取与合成延迟故障](docs/ADAPTIVE_PREFETCH.md)
- [自适应 / 固定异步对照结果](results/adaptive_matrix.md)
- [Worker 墙钟自适应、超时与迟到结果](docs/WORKER_ADAPTIVE.md)
- [Worker 模拟验证结果](results/worker_smoke.md)
- [进程隔离、卡死终止与状态恢复](docs/PROCESS_BACKEND.md)
- [进程恢复模拟验证结果](results/process_recovery.md)

Synthetic adaptive-prefetch experiments (virtual time):

```powershell
$env:PYTHONPATH = "src"
python -m robovla_runtime.cli run --config configs/adaptive_synthetic.toml
python benchmarks/run_adaptive_matrix.py
```

The matrix compares fixed and adaptive scheduling on shared time-indexed fixed,
long-tail, and overload traces. It retains unfavorable cases and reports request
counts alongside fallback and observation age; these are not real-model results.

Run the wall-clock worker with simulated blocking inference:

```powershell
python -m robovla_runtime.cli run-worker-mock --config configs/worker_adaptive.toml
python benchmarks/run_worker_smoke.py
```

Worker prefetch can be fixed or adaptive. Request deadlines reject late results,
and shutdown waits are bounded. Thread mode cannot forcibly terminate inference.
Process mode owns the model in a spawned child and can terminate, reap, and
recreate it after a timeout, with fresh model state and a bounded restart budget.

```powershell
python -m robovla_runtime.cli run-worker-mock --config configs/process_adaptive.toml
python -m benchmarks.run_process_recovery --output-dir runs/process_recovery_rerun
```
