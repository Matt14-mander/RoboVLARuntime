"""Compare fixed and adaptive prefetch on identical synthetic time fields."""

from __future__ import annotations

import argparse
import json
import hashlib
import platform
from dataclasses import asdict, replace
from pathlib import Path

from robovla_runtime.backends.latency import LatencyTrace, make_latency_trace
from robovla_runtime.core.events import EventLog
from robovla_runtime.runtime import RuntimeConfig, run_simulation


def scenarios(duration_s: float = 20.0) -> dict[str, LatencyTrace]:
    traces = {
        "fixed_50ms": make_latency_trace("fixed", duration_s=duration_s, base_latency_s=0.05),
        "fixed_100ms": make_latency_trace("fixed", duration_s=duration_s, base_latency_s=0.10),
    }
    for seed in (7, 17, 29):
        traces[f"long_tail_seed_{seed}"] = make_latency_trace(
            "long_tail", duration_s=duration_s, seed=seed
        )
    traces["sustained_overload"] = make_latency_trace("overload", duration_s=duration_s)
    return traces


def run_comparison(duration_s: float = 20.0):
    """Return all cases, including failures; no scheduler tuning on the traces."""
    base = RuntimeConfig(duration_s=duration_s)
    runs = []
    for name, trace in scenarios(duration_s).items():
        for scheduler in ("fixed_async", "adaptive_async"):
            config = replace(base, scheduler=scheduler)
            result = run_simulation(config, latency_trace=trace)
            summary = result.summary
            runs.append((name, trace, result, {
                "scenario": name,
                "scheduler": scheduler,
                "ticks": summary["ticks"],
                "requests": summary["prediction_request_count"],
                "completed_requests": summary["prediction_count"],
                "fallback_ticks": summary["buffer_underrun_ticks"],
                "fallback_ratio": summary["buffer_underrun_ratio"],
                "fallback_duration_s": summary["fallback_duration_s"],
                "observation_age_p95_s": summary["observation_age_s"]["p95"],
                "expired_chunks": summary["chunk_rejection_counts"].get("expired", 0),
                "deadline_miss_count": summary["deadline_miss_count"],
                "max_command_jump": summary["max_command_jump"],
                "final_tracking_error": summary["final_tracking_error"],
                "adaptive_prefetch": summary["adaptive_prefetch"],
            }))
    return runs


def markdown_report(rows) -> str:
    lines = [
        "# 自适应预取：合成延迟对照",
        "",
        "同一场景共用按模拟时间定义的延迟轨迹；延迟在请求提交时取样。"
        "每个策略只有一个在途请求，使用相同的动作 buffer、fallback 和 toy plant。",
        "",
        "控制周期 20 ms，生成/执行 horizon 都为 10（200 ms）；固定阈值及自适应冷启动阈值均为 80 ms。"
        "自适应使用最近 20 个已完成请求的 p95，加 20 ms 余量、10 ms 迟滞，最小请求间隔 20 ms。",
        "",
        "| 场景 | 调度器 | 请求数 | fallback ticks | fallback 比例 | 观测 age p95 (ms) | 过期 chunks |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        age = row["observation_age_p95_s"]
        age_text = "无有效动作" if age is None else f"{age * 1000:.1f}"
        lines.append(
            f"| {row['scenario']} | {row['scheduler']} | {row['requests']} | "
            f"{row['fallback_ticks']} | {row['fallback_ratio']:.1%} | {age_text} | "
            f"{row['expired_chunks']} |"
        )
    lines.extend([
        "",
        "固定延迟为 50/100 ms。长尾场景每 0.5 s 一个时间槽，基础延迟 40 ms，"
        "15% 概率进入尖峰，尖峰为 min(300 ms, 120 ms × Pareto(alpha=2))；"
        "种子预先固定为 7、17、29。持续过载从第 3 s 起切换到 260 ms。",
        "",
        "结果包含冷启动和全部失败 tick；请求数不同意味着采样到的槽位及完成样本数可能不同。"
        "这是同一外部延迟场下的闭环比较，不是强制每个请求经历相同延迟序列。",
        "",
        "本策略的 chunk 从观测时刻起算，返回时只剩 E×dt−L 的覆盖。"
        "单在途、单 chunk 替换情况下，持续无断供至少要求 max(L, 最小请求间隔) ≤ E×dt−L。"
        "自适应只能调整请求时机，无法恢复已过期动作或消除持续供给不足。",
        "",
        "固定 80 ms 不是针对每个场景调参后的最优固定阈值；本表不声称自适应在所有负载下更优。"
        "p95 也不是尾延迟保证，突发尖峰只能在完成后影响估计。",
        "",
        "自适应可能增加请求数，因而这里并未强制两种策略消耗相同总计算量。"
        "持续过载后的结果差异也可能来自切换时刻前的请求相位，不能解释为解决了过载。",
        "",
        "所有延迟均为虚拟时间注入，不使用 sleep，不代表真实 GPU 争用、网络队列或 VLA 成功率。"
        "deadline miss 在该确定性循环中为零也不表示动作供给充足。",
    ])
    return "\n".join(lines) + "\n"


def write_comparison(runs, output_dir: Path, summary_prefix: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    trace_metadata = {}
    configs = {}
    for name, trace, result, row in runs:
        directory = output_dir / name / result.config.scheduler
        directory.mkdir(parents=True)
        trace.write(directory / "latency_trace.json")
        EventLog.from_events(result.events).write_jsonl(directory / "events.jsonl")
        for filename, value in (
            ("config.json", asdict(result.config)), ("summary.json", result.summary)
        ):
            (directory / filename).write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        rows.append(row)
        trace_metadata[name] = trace.to_dict()
        configs[result.config.scheduler] = asdict(result.config)
    report = {
        "schema_version": 1,
        "experiment": "adaptive_prefetch_synthetic_latency",
        "scope": "deterministic toy plant; not VLA or real hardware performance",
        "fault_semantics": "total latency sampled at request time; final time slot held",
        "python_version": platform.python_version(),
        "source_sha256": source_fingerprint(),
        "configs": configs,
        "traces": trace_metadata,
        "rows": rows,
    }
    summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_prefix.with_suffix(".md").write_text(markdown_report(rows), encoding="utf-8")


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src").rglob("*.py")) + [Path(__file__).resolve()]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/adaptive_matrix"))
    parser.add_argument("--summary-prefix", type=Path, default=Path("results/adaptive_matrix"))
    args = parser.parse_args()
    runs = run_comparison()
    write_comparison(runs, args.output_dir, args.summary_prefix)
    print(markdown_report([run[3] for run in runs]))
    print(f"traces: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
