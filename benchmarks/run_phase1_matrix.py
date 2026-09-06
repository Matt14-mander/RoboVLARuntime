"""Generate the phase-one sync versus fixed-prefetch comparison."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from robovla_runtime.runtime import RuntimeConfig, run_simulation


LATENCIES_S = (0.0, 0.05, 0.15, 0.25)
SCHEDULERS = ("sync", "fixed_async")


def run_matrix() -> list[dict[str, object]]:
    base = RuntimeConfig(
        period_s=0.02,
        duration_s=2.0,
        chunk_size=10,
        execute_horizon=10,
        prefetch_threshold_s=0.08,
    )
    rows = []
    for latency_s in LATENCIES_S:
        for scheduler in SCHEDULERS:
            config = replace(
                base, scheduler=scheduler, latency_s=(latency_s,)
            )
            result = run_simulation(config)
            rows.append(
                {
                    "scheduler": scheduler,
                    "latency_s": latency_s,
                    "prefetch_threshold_s": config.prefetch_threshold_s,
                    "ticks": result.summary["ticks"],
                    "buffer_underrun_ticks": result.summary[
                        "buffer_underrun_ticks"
                    ],
                    "buffer_underrun_ratio": result.summary[
                        "buffer_underrun_ratio"
                    ],
                    "observation_age_p95_s": result.summary[
                        "observation_age_s"
                    ]["p95"],
                    "final_tracking_error": result.summary[
                        "final_tracking_error"
                    ],
                }
            )
    return rows


def write_json(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            {
                "experiment": "phase1_latency_matrix",
                "scope": "deterministic toy plant; not VLA performance",
                "rows": rows,
            },
            stream,
            indent=2,
            sort_keys=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="results/phase1_matrix.json", type=Path
    )
    args = parser.parse_args()
    rows = run_matrix()
    write_json(rows, args.output)
    print(json.dumps(rows, indent=2))
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
