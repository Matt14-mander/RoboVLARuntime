"""Command-line interface for deterministic runtime experiments."""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from robovla_runtime.core.events import EventLog
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    result = run_simulation(config)
    run_dir = write_result(result, args.output_root)
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print(f"artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
