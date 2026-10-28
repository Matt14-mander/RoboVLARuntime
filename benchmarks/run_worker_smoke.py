"""Wall-clock worker smoke checks with real threads and simulated model sleeps."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path

from robovla_runtime.backends import LatencyTrace
from robovla_runtime.cli import write_result
from robovla_runtime.runtime import RuntimeConfig, run_realtime_mock


def run_smoke(output_dir: Path, summary_prefix: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    config = RuntimeConfig(
        scheduler="worker_async", period_s=0.01, duration_s=0.6,
        chunk_size=30, execute_horizon=30, prefetch_threshold_s=0.08,
        request_timeout_s=0.06, worker_close_timeout_s=0.1,
        adaptive_min_request_interval_s=0.01,
    )
    # Every pair samples the same episode-time latency field. Actual timing
    # depends on OS scheduling; this is a lifecycle check, not a performance claim.
    traces = {
        "fixed": LatencyTrace("fixed", 0.1, (0.015,) * 6),
        "spike": LatencyTrace("spike", 0.1, (0.015, 0.015, 0.12, 0.015, 0.015, 0.015)),
        "overload": LatencyTrace("overload", 0.1, (0.15,) * 6),
    }
    rows = []
    for name, trace in traces.items():
        for prefetch in ("fixed", "adaptive"):
            result = run_realtime_mock(replace(config, worker_prefetch=prefetch), latency_trace=trace)
            directory = write_result(result, output_dir / name / prefetch)
            trace.write(directory / "latency_trace.json")
            timed_out = {
                event.data["request_id"] for event in result.events
                if event.type == "prediction_timed_out"
            }
            executed = {record.request_id for record in result.records if record.request_id is not None}
            assert not timed_out & executed, "timed-out prediction reached execution"
            assert len(result.records) == 60, "control loop did not complete"
            summary = result.summary
            rows.append({
                "scenario": name, "prefetch": prefetch,
                "requests": summary["prediction_request_count"],
                "timeouts": summary["prediction_timeout_count"],
                "late_results": summary["late_result_count"],
                "fallback_ticks": summary["buffer_underrun_ticks"],
                "ticks": summary["ticks"],
                "latency_samples": summary["adaptive_prefetch"]["completed_samples"],
                "shutdown_timeouts": summary["worker_shutdown_timeout_count"],
                "control_lateness_p95_s": summary["control_tick_lateness_s"]["p95"],
                "artifacts": directory.as_posix(),
            })
    report = {
        "scope": "wall-clock sleeping mock; OS-dependent smoke check, no GPU or VLA",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "base_config": asdict(config),
        "traces": {name: trace.to_dict() for name, trace in traces.items()},
        "rows": rows,
    }
    summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Worker 墙钟循环 smoke 验证", "",
        "真实线程 + 模拟推理 sleep；每轮 60 个控制 tick。数值受操作系统调度影响，不是性能基准。", "",
        "| 场景 | 预取 | 请求 | 超时 | 迟到结果 | fallback ticks | shutdown 超时 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['prefetch']} | {row['requests']} | {row['timeouts']} | "
            f"{row['late_results']} | {row['fallback_ticks']} | {row['shutdown_timeouts']} |"
        )
    lines.extend([
        "", "检查通过：所有运行完成控制 tick；所有已判超时的请求均未进入动作执行。",
        "", "固定模式保留 latest-only mailbox；自适应模式通常只保留一个有效逻辑请求。"
        "因此这张表用于验证生命周期与故障处理，不隔离预取阈值这一单独因素。",
        "", "deadline 为提交到控制循环取回的 60 ms；尖峰 120 ms，持续过载 150 ms。"
        "超时不会强杀推理线程。shutdown 等待上限为 100 ms；超过上限会留下显式事件。",
    ])
    summary_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/worker_smoke"))
    parser.add_argument("--summary-prefix", type=Path, default=Path("results/worker_smoke"))
    args = parser.parse_args()
    report = run_smoke(args.output_dir, args.summary_prefix)
    print(json.dumps(report["rows"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
