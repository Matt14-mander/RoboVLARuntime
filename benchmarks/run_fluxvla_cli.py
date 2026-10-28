"""Run the installed CLI boundary in subprocesses, including expected failures."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def run_smoke(output_dir: Path, summary_prefix: Path):
    root = Path(__file__).resolve().parents[1]
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "samples.json").write_bytes((root / "samples/fluxvla_offline_mock.json").read_bytes())
    template = (root / "configs/fluxvla_offline_mock.toml").read_text(encoding="utf-8").replace(
        "../samples/fluxvla_offline_mock.json", "samples.json",
    ).replace("startup_timeout_s = 5.0", "startup_timeout_s = 2.0")
    environment = dict(os.environ, PYTHONPATH=str(root / "src"))
    rows = []
    for name, faults, expected in (
        ("healthy", [], 0), ("inference_hang", ["request_hang"], 0),
        ("warmup_error", ["warmup_error"], 0), ("warmup_hang", ["warmup_hang"], 0),
        ("restart_exhausted", ["request_hang", "request_hang"], 1), ("invalid_config", [], 2),
    ):
        source = template.replace('fault_plan = ["request_hang"]', f"fault_plan = {json.dumps(faults)}")
        if name == "restart_exhausted":
            source = source.replace("max_restarts = 2", "max_restarts = 1")
        if name == "invalid_config":
            source = source.replace("dimension = 2", "dimension = true")
        config = output_dir / f"{name}.toml"
        config.write_text(source, encoding="utf-8")
        case_output = output_dir / name
        result = subprocess.run([
            sys.executable, "-m", "robovla_runtime.cli", "infer-fluxvla-process", "--config", str(config),
            "--output-root", str(case_output), "--quiet",
        ], cwd=root, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=30)
        (output_dir / f"{name}.stdout.txt").write_text(result.stdout, encoding="utf-8")
        (output_dir / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
        assert result.returncode == expected, (name, result.stdout, result.stderr)
        if expected == 2:
            assert not case_output.exists()
            rows.append({"scenario": name, "exit_code": 2, "validation_before_spawn": True, "artifacts": None})
            continue
        output = json.loads(result.stdout)
        summary = output["summary"]
        directory = Path(output["artifacts"])
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        predictions = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        failed = {e["data"]["request_id"] for e in events if e["type"] == "prediction_failed"}
        assert not failed & {p["request_id"] for p in predictions}
        assert summary["closed"] and summary["model_starts"] == summary["terminations"]
        assert not any(e["type"] in {"action_executed", "fallback_executed"} for e in events)
        if expected == 0:
            assert len(predictions) == 2
            assert predictions[0]["chunk"]["actions"][0] == [0.1, 0.1]
        else:
            assert not predictions and summary["restart_exhaustions"] == 1
        rows.append({"scenario": name, "exit_code": result.returncode,
                     "succeeded_samples": summary["succeeded_samples"], "attempts": summary["attempt_count"],
                     "failed_attempts": summary["failed_attempts"], "restarts": summary["restarts"],
                     "warmups_completed": summary["warmups_completed"], "terminations": summary["terminations"],
                     "artifacts": str(directory)})
    summary_prefix.parent.mkdir(parents=True, exist_ok=True)
    report = {"scope": "actual subprocess CLI with simulated FluxVLA; inference only, no robot/GPU",
              "python": platform.python_version(), "platform": platform.platform(), "rows": rows}
    summary_prefix.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# FluxVLA 离线进程 CLI 验证", "",
             "每组从独立 CLI 子进程读取 TOML 与两条 JSON 离线观测；模型使用模拟依赖。", "",
             "| 场景 | 退出码 | 成功样本 | 尝试 | 失败尝试 | 重启 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['scenario']} | {row['exit_code']} | {row.get('succeeded_samples', '—')} | "
                     f"{row.get('attempts', '—')} | {row.get('failed_attempts', '—')} | {row.get('restarts', '—')} |")
    lines.extend(["", "全部符合预期：单次故障后恢复推理，重启预算耗尽返回 1，非法配置在启动前返回 2。",
                  "有效运行均保存启动、预热、推理与终止诊断；未发布失败请求的预测，所有模型子进程已回收。",
                  "本命令不执行机器人动作。真实 checkpoint、CUDA、观测形状契约与任务成功率仍待验证。"])
    summary_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/fluxvla_cli_smoke"))
    parser.add_argument("--summary-prefix", type=Path, default=Path("results/fluxvla_cli"))
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output_dir, args.summary_prefix)["rows"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
