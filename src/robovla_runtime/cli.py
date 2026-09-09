"""Command-line interface for deterministic runtime experiments."""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from robovla_runtime.core.events import EventLog
from robovla_runtime.integrations.fluxvla_libero import (
    create_libero_profile_session,
)
from robovla_runtime.profiling import (
    CudaMemorySampler,
    ProfileConfig,
    profile_backend,
    write_profile,
)
from robovla_runtime.runtime import RuntimeConfig, run_simulation


def load_config(path: str | Path) -> RuntimeConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    runtime = raw.get("runtime", raw)
    for key in ("latency_s", "initial_position", "target_position"):
        if key in runtime:
            runtime[key] = tuple(runtime[key])
    return RuntimeConfig(**runtime)


def write_result(result, output_root: str | Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = Path(output_root) / f"{stamp}-{result.config.scheduler}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(asdict(result.config), stream, indent=2, sort_keys=True)
    EventLog.from_events(result.events).write_jsonl(run_dir / "events.jsonl")
    with (run_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(result.summary, stream, indent=2, sort_keys=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robovla-runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a deterministic simulation")
    run.add_argument("--config", required=True)
    run.add_argument("--output-root", default="runs")
    profile = subparsers.add_parser(
        "profile-fluxvla", help="profile FluxVLA on one valid LIBERO observation"
    )
    profile.add_argument("--config", required=True, help="FluxVLA model config")
    profile.add_argument("--checkpoint", required=True)
    profile.add_argument("--fluxvla-root")
    profile.add_argument("--output", required=True)
    profile.add_argument("--warmup", type=int, default=10)
    profile.add_argument("--samples", type=int, default=100)
    profile.add_argument(
        "--mode", choices=("diagnostic", "throughput"), default="diagnostic"
    )
    profile.add_argument("--task-suite")
    profile.add_argument("--task-id", type=int, default=0)
    profile.add_argument("--trial-id", type=int, default=0)
    profile.add_argument("--action-period", type=float, default=0.1)
    profile.add_argument("--action-dimension", type=int, default=7)
    profile.add_argument("--num-inference-steps", type=int)
    profile.add_argument("--inference-seed", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    # read TOML config, run simulation, write results to disk, and print summary
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config = load_config(args.config)
        result = run_simulation(config)
        run_dir = write_result(result, args.output_root)
        print(json.dumps(result.summary, indent=2, sort_keys=True))
        print(f"artifacts: {run_dir}")
        return 0

    profile_config = ProfileConfig(args.warmup, args.samples, args.mode)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"profile output already exists: {output_path}")
    with create_libero_profile_session(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        fluxvla_root=args.fluxvla_root,
        task_suite_name=args.task_suite,
        task_id=args.task_id,
        trial_id=args.trial_id,
        action_period_s=args.action_period,
        action_dimension=args.action_dimension,
        timing_mode=args.mode,
        num_inference_steps=args.num_inference_steps,
        inference_seed=args.inference_seed,
    ) as session:
        result = profile_backend(
            session.backend,
            session.observation_factory,
            profile_config,
            memory_sampler=CudaMemorySampler(),
            metadata=session.metadata,
        )
    output = write_profile(result, output_path)
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print(f"artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
