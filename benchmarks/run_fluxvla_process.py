"""Exercise the real FluxVLA adapter/factory path with simulated dependencies."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
from dataclasses import asdict
from pathlib import Path

from benchmarks.fluxvla_fixtures import MockObservationFactory, make_simulated_factory, runtime_payload
from robovla_runtime.cli import load_config, write_result
from robovla_runtime.core import ActionSpec
from robovla_runtime.integrations import FluxVLAProcessFactory
from robovla_runtime.runtime import run_realtime_runtime


def run_smoke(config_path: Path, output_dir: Path, summary_prefix: Path) -> dict:
    config = load_config(config_path)
    if (config.period_s, len(config.initial_position), config.chunk_size) != (0.02, 2, 30):
        raise ValueError("the synthetic FluxVLA fixture requires period 0.02, dimension 2 and chunk size 30")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, fault in (
        ("healthy", None), ("inference_hang", "request_hang"),
        ("warmup_hang", "warmup_hang"), ("warmup_error", "warmup_error"),
        ("post_warmup_reset_hang", "post_warmup_reset_hang"),
    ):
        mock = make_simulated_factory(output_dir / name, (fault,) if fault else ())
        recipe = FluxVLAProcessFactory(mock, MockObservationFactory(), warmup_steps=2)
        backend = recipe.create_backend(ActionSpec(2, 0.02), config=config.process_config())
        try:
            result = run_realtime_runtime(config, backend, observation_payload_factory=runtime_payload)
        finally:
            assert backend.close(2.0), "process/transport not reaped"
        directory = write_result(result, output_dir / name)
        (directory / "factory.json").write_text(json.dumps({
            "model_recipe": asdict(mock.recipe), "warmup_steps": recipe.warmup_steps,
            "model_is_stateless": recipe.model_is_stateless, "faults": mock.faults,
            "audit_path": mock.audit_path.as_posix(),
        }, indent=2), encoding="utf-8")
        audit = [json.loads(line) for line in mock.audit_path.read_text(encoding="utf-8").splitlines()]
        events = result.events
        executed = [r for r in result.records if r.request_id is not None]
        timed_out = {e.data["request_id"] for e in events if e.type == "prediction_timed_out"}
        starts = [e.data["pid"] for e in events if e.type == "process_worker_started"]
        stops = [e.data["pid"] for e in events if e.type == "process_worker_terminated"]
        reset_generations = {
            e.data["process_generation"] for e in events
            if e.type == "process_startup_progress" and e.data["stage"] == "warmup_state_reset"
        }
        request_generations = {
            e.data["request_id"]: e.data["process_generation"] for e in events if e.type == "process_request_started"
        }
        assert len(result.records) == round(config.duration_s / config.period_s)
        assert executed and not timed_out & {r.request_id for r in executed}
        assert all(request_generations[r.request_id] in reset_generations for r in executed)
        assert sorted(starts) == sorted(stops)
        assert not set(starts) & {p.pid for p in mp.active_children()}
        assert mock.attempts.value == (2 if fault else 1)
        for generation in range(1, mock.attempts.value + 1):
            live = [e for e in audit if e["generation"] == generation and e["kind"] == "predict" and e["phase"] == "inference"]
            if live:
                assert live[0]["model_history"] == live[0]["dataset_history"] == 1
        warmups = [e for e in audit if e["kind"] == "predict" and e["phase"] == "warmup"]
        rows.append({
            "scenario": name, "ticks": len(result.records), "model_builds": mock.attempts.value,
            "warmup_predict_calls": len(warmups),
            "warmups_completed": result.summary["process_worker"]["warmups_completed"],
            "warmup_state_resets": len(reset_generations), "restarts": result.summary["process_worker"]["restarts"],
            "terminated": len(stops), "prediction_timeouts": result.summary["prediction_timeout_count"],
            "executed_ticks": len(executed), "fallback_ticks": result.summary["buffer_underrun_ticks"],
            "expired_actions_executed": 0, "artifacts": directory.as_posix(),
            "audit": mock.audit_path.as_posix(),
        })
    report = {"scope": "real FluxVLA factory/adapter; simulated torch, upstream runner and model; no GPU",
              "python": platform.python_version(), "platform": platform.platform(), "rows": rows}
    summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# FluxVLA 进程工厂、预热与恢复验证", "",
        "使用真实 FluxVLA 工厂、配置加载、预处理及解码路径，上游 runner/model 与 torch 由模拟依赖替代。", "",
        "| 场景 | tick | 模型构建 | 预热调用 | 预热完成 | 预热后重置 | 重启 | 执行动作 tick |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['scenario']} | {row['ticks']} | {row['model_builds']} | {row['warmup_predict_calls']} | "
                     f"{row['warmups_completed']} | {row['warmup_state_resets']} | {row['restarts']} | {row['executed_ticks']} |")
    lines.extend([
        "", "全部通过：控制循环完成；预热及超时动作未进入执行；恢复后的模型与 dataset 首次正式推理历史计数均为 1；所有子进程退出并回收。",
        "预热后 reset 卡死时，可以出现预热完成但状态重置未完成；该进程不能发送 ready。",
        "", "本结果不代表真实模型成功率、CUDA 预热效果或 GPU 资源回收保证。OS 调度会改变超时和 fallback 数值。",
    ])
    summary_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/fluxvla_process_mock.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/fluxvla_process"))
    parser.add_argument("--summary-prefix", type=Path, default=Path("results/fluxvla_process"))
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.config, args.output_dir, args.summary_prefix)["rows"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
