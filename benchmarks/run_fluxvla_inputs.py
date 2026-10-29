"""Generate typed synthetic observations and verify the real subprocess CLI boundary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_smoke(output_dir: Path):
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    np.save(output_dir / "camera.npy", np.zeros((32, 32, 3), dtype="uint8"))
    np.save(output_dir / "raw.npy", np.array([0.25, 0.5], dtype="float64"))
    samples = [{"sample_id": f"frame-{i}", "state": [float(i), 0.25, 0.5, 0., 0., 0., 1.],
                "payload": {}, "arrays": {
                    "camera": {"path": "camera.npy", "shape": [32, 32, 3], "dtype": "uint8"},
                    "robot0_gripper_qpos": {"path": "raw.npy", "shape": [2], "dtype": "float64"},
                }} for i in range(2)]
    (output_dir / "samples.json").write_text(json.dumps({"version": 1, "samples": samples}), encoding="utf-8")
    template = (root / "configs/fluxvla_offline_mock.toml").read_text(encoding="utf-8").replace(
        "../samples/fluxvla_offline_mock.json", "samples.json",
    ).replace("state_dimension = 2", "state_dimension = 7")
    for key, kind, shape, dtype in (
        ("camera", "camera", [32, 32, 3], "uint8"),
        ("robot0_eef_pos", "state", [3], "float32"),
        ("robot0_eef_quat", "state", [4], "float32"),
        ("robot0_gripper_qpos", "state", [2], "float64"),
    ):
        template += (f'\n[[model_input.arrays]]\nkey = "{key}"\nkind = "{kind}"\n'
                     f'shape = {shape}\ndtype = "{dtype}"\n')
    for key, indices in (("robot0_eef_pos", [0, 1, 2]), ("robot0_eef_quat", [3, 4, 5, 6])):
        template += f'\n[[model_input.state_mapping]]\nkey = "{key}"\nindices = {indices}\n'
    rows = []
    for name, text, expected in (
        ("healthy", template.replace('["request_hang"]', '[]'), 0),
        ("restart", template, 0),
        ("bad_camera", template.replace('dtype = "uint8"', 'dtype = "float32"'), 2),
        ("bad_raw_state", template.replace('dtype = "float64"', 'dtype = "float32"'), 2),
    ):
        config = output_dir / f"{name}.toml"
        config.write_text(text, encoding="utf-8")
        destination = output_dir / name
        result = subprocess.run([
            sys.executable, "-m", "robovla_runtime.cli", "infer-fluxvla-process", "--config", str(config),
            "--output-root", str(destination), "--quiet",
        ], cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")},
            capture_output=True, text=True, encoding="utf-8", timeout=40)
        assert result.returncode == expected, (name, result.stdout, result.stderr)
        if expected == 2:
            assert not destination.exists()
            assert json.loads(result.stderr)["error"] == "validation_failed"
        else:
            output = json.loads(result.stdout)
            summary = output["summary"]
            assert summary["closed"] and summary["succeeded_samples"] == 2
            assert summary["restarts"] == (1 if name == "restart" else 0)
            assert summary["warmups_completed"] == (2 if name == "restart" else 1)
            audit = Path(output["artifacts"]) / "mock/audit.jsonl"
            calls = [row for row in map(json.loads, audit.read_text().splitlines()) if row["kind"] == "predict"]
            assert {row["generation"] for row in calls} == ({1, 2} if name == "restart" else {1})
            assert {row["phase"] for row in calls} == {"warmup", "inference"}
            for call in calls:
                inputs = call["model_inputs"]
                assert inputs["robot0_eef_pos"]["values"] == call["observation_state"][:3]
                assert inputs["robot0_eef_quat"]["values"] == call["observation_state"][3:]
                assert inputs["robot0_eef_pos"]["dtype"] == "float32"
                assert inputs["robot0_gripper_qpos"]["values"] == [0.25, 0.5]
                assert inputs["camera"]["shape"] == [32, 32, 3]
        rows.append({"scenario": name, "exit_code": result.returncode, "passed": True})
    (output_dir / "report.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/fluxvla_inputs_smoke"))
    print(json.dumps(run_smoke(parser.parse_args().output_dir), indent=2))
