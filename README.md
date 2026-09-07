# RoboVLARuntime
A resource-aware runtime and evaluation framework for real-time Vision-Language-Action model deployment, featuring latency-aware scheduling, action chunk execution, profiling, and pluggable VLA backends.

Status: phase-one deterministic runtime core is implemented. It provides a
virtual clock, mock chunked policy, time-aligned action buffer, synchronous and
fixed-prefetch schedulers, a toy joint plant, structured event traces, and
summary metrics. Phase two adds a dependency-injected `FluxVLABackend` and a
worker-backed real-time loop so GPU inference can overlap control execution;
real checkpoint and LIBERO validation remain pending.

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
