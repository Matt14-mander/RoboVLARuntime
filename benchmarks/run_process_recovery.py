"""Real spawn/kill/restart recovery smoke check using stateful synthetic models."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
from dataclasses import asdict
from pathlib import Path

from benchmarks.process_fixtures import AttemptCounter, RecoveryMockFactory
from robovla_runtime.backends import ProcessPolicyBackend
from robovla_runtime.cli import write_result
from robovla_runtime.core import ActionSpec
from robovla_runtime.runtime import RuntimeConfig, run_realtime_runtime


def run_recovery(output_dir: Path, summary_prefix: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    config = RuntimeConfig(
        scheduler="worker_async", worker_prefetch="adaptive", worker_isolation="process",
        period_s=0.02, duration_s=4.0, chunk_size=100, execute_horizon=20,
        request_timeout_s=0.3, worker_close_timeout_s=1.0,
        process_startup_timeout_s=2.0, process_execution_timeout_s=1.0,
        process_max_restarts=1,
    )
    rows = []
    scenarios = {
        "healthy": (), "inference_hang": ("request_hang",),
        "reset_hang": ("reset_hang",), "startup_hang": ("startup_hang",),
        "crash": ("crash",), "inference_error": ("request_error",),
        "restart_exhausted": ("request_hang", "request_hang"),
    }
    for name, faults in scenarios.items():
        scenario_dir = output_dir / name
        scenario_dir.mkdir()
        attempts = AttemptCounter(scenario_dir.resolve() / "attempts.txt")
        spec = ActionSpec(1, config.period_s)
        backend = ProcessPolicyBackend(
            RecoveryMockFactory(spec, attempts, faults), spec, config=config.process_config(),
        )
        try:
            result = run_realtime_runtime(config, backend)
        finally:
            assert backend.close(2.0), "child/IPC resources were not reaped"
        directory = write_result(result, output_dir / name)
        (directory / "fault_plan.json").write_text(json.dumps(faults) + "\n", encoding="utf-8")
        expired = {e.data["request_id"] for e in result.events if e.type == "prediction_timed_out"}
        executed = {r.request_id for r in result.records if r.request_id is not None}
        actions = [r for r in result.records if r.request_id is not None]
        # Earlier healthy completions may be discarded at the collection deadline.
        # Check history against calls in this child, not against visible action count.
        calls_by_generation = {}
        expected_history = {}
        for event in result.events:
            if event.type == "process_request_started":
                generation = event.data["process_generation"]
                calls_by_generation[generation] = calls_by_generation.get(generation, 0) + 1
                expected_history[event.data["request_id"]] = calls_by_generation[generation]
        started = [e.data["pid"] for e in result.events if e.type == "process_worker_started"]
        stopped = [e.data["pid"] for e in result.events if e.type == "process_worker_terminated"]
        assert len(result.records) == 200
        assert not expired & executed, "expired request reached execution"
        assert sorted(started) == sorted(stopped), "not all child exits were observed"
        assert not set(started) & {child.pid for child in mp.active_children()}
        if name == "restart_exhausted":
            assert not actions
            assert result.summary["process_worker"]["restart_exhaustions"] == 1
            assert attempts.value == 2
        else:
            assert actions, "no action after recovery"
            assert all(r.command == (float(expected_history[r.request_id]),) for r in actions), "model history was not reset"
            assert attempts.value == (1 if name == "healthy" else 2)
        rows.append({
            "scenario": name, "ticks": len(result.records),
            "starts": len(started), "terminations": len(stopped),
            "restarts": result.summary["process_worker"]["restarts"],
            "restart_exhaustions": result.summary["process_worker"]["restart_exhaustions"],
            "request_timeouts": result.summary["prediction_timeout_count"],
            "fallback_ticks": result.summary["buffer_underrun_ticks"],
            "executed_ticks": len(actions),
            "first_command": list(actions[0].command) if actions else None,
            "first_execution_s": actions[0].executed_at if actions else None,
            "timed_out_actions_executed": len(expired & executed),
            "artifacts": directory.as_posix(),
        })
    report = {
        "scope": "spawn/termination/restart smoke with synthetic state; no real model or GPU",
        "platform": platform.platform(), "python_version": platform.python_version(),
        "config": asdict(config), "rows": rows,
    }
    summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_prefix.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    lines = [
        "# 进程隔离恢复验证", "",
        "真实 spawn 子进程 + 合成有状态模型；每组 200 个控制 tick。计时受 OS 调度影响，不是性能基准。", "",
        "| 场景 | 启动 | 终止 | 重启 | 超时请求 | fallback ticks | 执行动作 ticks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['starts']} | {row['terminations']} | {row['restarts']} | "
            f"{row['request_timeouts']} | {row['fallback_ticks']} | {row['executed_ticks']} |"
        )
    lines.extend([
        "", "检查通过：所有控制 tick 完成；超时请求的动作未执行；启动的子进程全部退出并回收。",
        "正常与单次故障场景最终均执行新动作；所有命令中的状态计数均等于当前子进程从 reset 后开始的调用次数。",
        "连续卡死场景在一次自动重启后耗尽预算，持续 fallback，没有无限重启。",
        "", "故障按进程创建次数注入：首进程卡死/崩溃，下一进程健康；连续故障场景前两个进程均卡死。",
        "模拟未覆盖真实 CUDA 驱动、设备资源释放或模型重新加载成本。原始事件和故障计划路径见同名 JSON。",
    ])
    summary_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/process_recovery"))
    parser.add_argument("--summary-prefix", type=Path, default=Path("results/process_recovery"))
    args = parser.parse_args()
    print(json.dumps(run_recovery(args.output_dir, args.summary_prefix)["rows"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
