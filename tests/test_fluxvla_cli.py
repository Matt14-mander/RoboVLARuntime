from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robovla_runtime.cli import main
from robovla_runtime.integrations.fluxvla_cli_config import load_inference_config
from robovla_runtime.integrations.offline_inference import run_offline_inference

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]


class FluxVLACliTests(unittest.TestCase):
    def typed_inputs(self):
        raw = json.loads(self.manifest.read_text(encoding="utf-8"))
        np.save(self.root / "camera.npy", np.zeros((2, 3, 3), dtype="uint8"))
        np.save(self.root / "raw.npy", np.array([4.5], dtype="float64"))
        for sample in raw["samples"]:
            sample["state"] = [1.25, 2.5, 3.75]
            sample["payload"] = {}
            sample["arrays"] = {
                "camera": {"path": "camera.npy", "shape": [2, 3, 3], "dtype": "uint8"},
                "raw_state": {"path": "raw.npy", "shape": [1], "dtype": "float64"},
            }
        self.manifest.write_text(json.dumps(raw), encoding="utf-8")
        return self.template.replace("state_dimension = 2", "state_dimension = 3") + '''
[[model_input.arrays]]
key = "camera"
kind = "camera"
shape = [2, 3, 3]
dtype = "uint8"
[[model_input.arrays]]
key = "proprio"
kind = "state"
shape = [2]
dtype = "float32"
[[model_input.arrays]]
key = "raw_state"
kind = "state"
shape = [1]
dtype = "float64"
[[model_input.state_mapping]]
key = "proprio"
indices = [2, 0]
'''

    @unittest.skipIf(np is None, "numpy is optional")
    def test_model_input_mapping_survives_warmup_and_process_restart(self):
        text = self.typed_inputs()
        code, stdout, stderr = self.invoke(text, extra=("--validate-only",))
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["model_input"]["state_mapping"][0]["indices"], [2, 0])
        self.assertFalse(self.output.exists())
        code, _, stderr = self.invoke()
        self.assertEqual(code, 0, stderr)
        directory, summary, _, predictions = self.artifacts()
        self.assertEqual(summary["succeeded_samples"], 2)
        self.assertEqual(summary["restarts"], 1)
        self.assertEqual(summary["warmups_completed"], 2)
        self.assertTrue(summary["closed"])
        self.assertTrue(all(len(action) == 2 for p in predictions for action in p["chunk"]["actions"]))
        resolved = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(resolved["model_input"], json.loads(stdout)["model_input"])
        audit = [json.loads(line) for line in (directory / "mock/audit.jsonl").read_text().splitlines()]
        calls = [e for e in audit if e["kind"] == "predict"]
        self.assertEqual({(e["generation"], e["phase"]) for e in calls},
                         {(1, "warmup"), (1, "inference"), (2, "warmup"), (2, "inference")})
        for call in calls:
            self.assertEqual(call["model_inputs"]["proprio"],
                             {"shape": [2], "dtype": "float32", "values": [3.75, 1.25]})
            self.assertEqual(call["model_inputs"]["raw_state"]["values"], [4.5])
            self.assertEqual(call["model_inputs"]["camera"]["dtype"], "uint8")

    @unittest.skipIf(np is None, "numpy is optional")
    def test_model_input_config_errors_are_preflight_failures(self):
        template = self.typed_inputs()
        cases = [
            template.replace("indices = [2, 0]", "indices = [3, 0]"),
            template.replace("indices = [2, 0]", "indices = [true, 0]"),
            template.replace("indices = [2, 0]", "indices = [0]"),
            template.replace("indices = [2, 0]", 'indices = "2,0"'),
            template.replace("indices = [2, 0]", "indices = [2, 0]\nunknown = 1"),
            template.replace('dtype = "float32"', 'dtype = "int32"'),
            template.replace("shape = [2, 3, 3]", "shape = [3, 2, 3]"),
            template.replace('dtype = "uint8"', 'dtype = "float32"'),
            template.replace('dtype = "float64"', 'dtype = "float32"'),
            template.replace("shape = [1]", "shape = [2]"),
            template.replace('key = "proprio"', 'key = "raw_state"'),
            template + '\n[[model_input.arrays]]\nkey = "missing"\nkind = "state"\nshape = [1]\ndtype = "float32"\n',
        ]
        for text in cases:
            with self.subTest(text=text), patch("robovla_runtime.integrations.offline_inference._factory") as factory:
                code, _, stderr = self.invoke(text)
                self.assertEqual(code, 2, stderr)
                self.assertEqual(json.loads(stderr)["error"], "validation_failed")
                factory.assert_not_called()
                self.assertFalse(self.output.exists())

    @unittest.skipIf(np is None, "numpy is optional")
    def test_input_collision_and_overflow_fail_before_process_creation(self):
        template = self.typed_inputs()
        original = self.manifest.read_text(encoding="utf-8")
        for collision in (True, False):
            raw = json.loads(original)
            # An invalid later sample must also fail the complete preflight.
            if collision:
                raw["samples"][1]["payload"]["proprio"] = [0.0, 0.0]
            else:
                raw["samples"][1]["state"][0] = 1e300
            self.manifest.write_text(json.dumps(raw), encoding="utf-8")
            with patch("robovla_runtime.integrations.offline_inference._factory") as factory:
                code, _, stderr = self.invoke(template)
                self.assertEqual(code, 2, stderr)
                self.assertIn("frame-1", json.loads(stderr)["message"])
                factory.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_independent_dimensions_survive_warmup_and_process_restart(self):
        raw = json.loads(self.manifest.read_text(encoding="utf-8"))
        for sample in raw["samples"]:
            sample["state"] = [0.1, 0.2, 0.3]
        self.manifest.write_text(json.dumps(raw), encoding="utf-8")
        text = self.template.replace("state_dimension = 2", "state_dimension = 3")
        code, stdout, stderr = self.invoke(text, extra=("--validate-only",))
        self.assertEqual(code, 0, stderr)
        validation = json.loads(stdout)
        self.assertEqual(validation["observation_spec"]["state_dimension"], 3)
        self.assertEqual(validation["action_dimension"], 2)
        code, _, stderr = self.invoke()
        self.assertEqual(code, 0, stderr)
        directory, summary, _, predictions = self.artifacts()
        self.assertEqual(summary["succeeded_samples"], 2)
        self.assertEqual(summary["restarts"], 1)
        self.assertEqual(summary["warmups_completed"], 2)
        self.assertTrue(all(len(action) == 2 for p in predictions for action in p["chunk"]["actions"]))
        resolved = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(resolved["observations"]["state_dimension"], 3)
        audit = [json.loads(line) for line in (directory / "mock/audit.jsonl").read_text().splitlines()]
        calls = [e for e in audit if e["kind"] == "predict"]
        self.assertEqual({e["phase"] for e in calls}, {"warmup", "inference"})
        self.assertEqual({e["generation"] for e in calls}, {1, 2})
        self.assertTrue(all(e["observation_state"] == [0.1, 0.2, 0.3] for e in calls))

    def test_legacy_config_infers_state_dimension_independently_of_action(self):
        text = self.template.replace("state_dimension = 2\n", "")
        text = text.replace("[action]\ndimension = 2", "[action]\ndimension = 7")
        code, stdout, stderr = self.invoke(text, extra=("--validate-only",))
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["observation_spec"]["state_dimension"], 2)
        self.assertEqual(json.loads(stdout)["action_dimension"], 7)
        self.assertFalse(self.output.exists())

    def test_state_contract_config_errors_fail_before_backend_start(self):
        for dimension in ("true", "0", "-1", "1.5", '"2"', "3"):
            text = self.template.replace("state_dimension = 2", f"state_dimension = {dimension}")
            with self.subTest(dimension=dimension), patch("robovla_runtime.integrations.offline_inference._factory") as factory:
                code, stdout, stderr = self.invoke(text)
                self.assertEqual(code, 2, stderr)
                self.assertEqual(json.loads(stderr)["error"], "validation_failed")
                factory.assert_not_called()
                self.assertFalse(self.output.exists())

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "config.toml"
        self.manifest = self.root / "samples.json"
        self.manifest.write_bytes((ROOT / "samples/fluxvla_offline_mock.json").read_bytes())
        self.template = (ROOT / "configs/fluxvla_offline_mock.toml").read_text(encoding="utf-8").replace(
            "../samples/fluxvla_offline_mock.json", "samples.json",
        )
        self.config.write_text(self.template, encoding="utf-8")
        self.output = self.root / "runs"

    def invoke(self, text=None, extra=()):
        if text is not None:
            self.config.write_text(text, encoding="utf-8")
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["infer-fluxvla-process", "--config", str(self.config),
                         "--output-root", str(self.output), "--quiet", *extra])
        return code, stdout.getvalue(), stderr.getvalue()

    def artifacts(self):
        directory = next(self.output.iterdir())
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        predictions = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        return directory, summary, events, predictions

    def test_validate_only_does_not_start_backend_or_create_artifacts(self):
        with patch("robovla_runtime.integrations.fluxvla_process.ProcessPolicyBackend") as backend:
            code, stdout, stderr = self.invoke(extra=("--validate-only",))
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["samples"], 2)
        backend.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_config_errors_have_exit_code_two_before_process_or_output(self):
        cases = [
            "unknown = 1\n" + self.template,
            self.template.replace('kind = "mock"', 'kind = "unknown"'),
            self.template.replace("dimension = 2", "dimension = true"),
            self.template.replace("warmup_steps = 2", "warmup_steps = -1"),
            self.template.replace("num_inference_steps = 4", "num_inference_steps = true"),
            self.template.replace("execution_timeout_s = 0.2", "execution_timeout_s = inf"),
            self.template.replace("poll_interval_s = 0.005", "poll_interval_s = 0"),
            self.template.replace('fault_plan = ["request_hang"]', 'fault_plan = ["typo"]'),
            self.template.replace('manifest = "samples.json"', 'manifest = "missing.json"'),
            self.template.replace('kind = "mock"', 'kind = "fluxvla"').replace('fault_plan = ["request_hang"]', ''),
        ]
        for text in cases:
            with self.subTest(text=text), patch("robovla_runtime.integrations.fluxvla_process.ProcessPolicyBackend") as backend:
                code, stdout, stderr = self.invoke(text)
                self.assertEqual(code, 2, (stdout, stderr))
                self.assertEqual(json.loads(stderr)["error"], "validation_failed")
                self.assertFalse(self.output.exists())
                backend.assert_not_called()

    def test_actual_cli_from_another_directory_loads_relative_samples_and_recovers(self):
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        result = subprocess.run([
            sys.executable, "-m", "robovla_runtime.cli", "infer-fluxvla-process",
            "--config", str(self.config), "--output-root", str(self.output), "--quiet",
        ], cwd=self.root, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=25)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        directory, summary, events, predictions = self.artifacts()
        self.assertEqual(summary["failed_attempts"], 1)
        self.assertEqual(summary["succeeded_samples"], 2)
        self.assertEqual(summary["restarts"], 1)
        self.assertEqual(summary["warmups_completed"], 2)
        self.assertEqual(summary["terminations"], 2)
        self.assertEqual(len(summary["worker_logs"]), 2)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[0]["chunk"]["actions"][0], [0.1, 0.1])
        self.assertEqual(predictions[1]["chunk"]["actions"][0], [0.2, 0.2])
        self.assertEqual(predictions[1]["source_captured_at"], 0.02)
        failures = {e["data"]["request_id"] for e in events if e["type"] == "prediction_failed"}
        self.assertFalse(failures & {p["request_id"] for p in predictions})
        self.assertFalse(any(e["type"] in {"action_executed", "fallback_executed"} for e in events))
        diagnostics = json.loads((directory / "diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(set(diagnostics["processes"]), {"1", "2"})
        self.assertIn("inference", diagnostics["attempts"][-1]["timings_s"])
        inputs = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
        self.assertEqual(len(inputs["files"]), 2)

    def test_exhausted_restart_budget_returns_one_and_preserves_failure_diagnostics(self):
        text = self.template.replace("max_restarts = 2", "max_restarts = 0")
        code, _, stderr = self.invoke(text)
        self.assertEqual(code, 1, stderr)
        _, summary, events, predictions = self.artifacts()
        self.assertEqual(summary["restart_exhaustions"], 1)
        self.assertEqual(summary["model_starts"], summary["terminations"])
        self.assertTrue(summary["closed"])
        self.assertEqual(summary["unprocessed_samples"], 1)
        self.assertEqual(predictions, [])
        self.assertTrue(any(e["type"] == "process_worker_fault" for e in events))

    def test_warmup_failure_recovers_without_publishing_warmup_actions(self):
        code, _, stderr = self.invoke(self.template.replace('"request_hang"', '"warmup_error"'))
        self.assertEqual(code, 0, stderr)
        _, summary, _, predictions = self.artifacts()
        self.assertEqual(summary["warmups_completed"], 1)
        self.assertEqual(summary["failed_attempts"], 1)
        self.assertEqual(len(predictions), 2)

    def test_zero_warmup_repeats_reset_history_and_allow_inference_only_action_semantics(self):
        text = self.template.replace('fault_plan = ["request_hang"]', 'fault_plan = []')
        text = text.replace("warmup_steps = 2", "warmup_steps = 0").replace("repeats = 1", "repeats = 2")
        text = text.replace('representation = "absolute_position"', 'representation = "delta_position"')
        text = text.replace('action_space = "joint"', 'action_space = "cartesian"').replace('coordinate_frame = "joint"', 'coordinate_frame = "base"')
        code, _, stderr = self.invoke(text)
        self.assertEqual(code, 0, stderr)
        _, summary, events, predictions = self.artifacts()
        self.assertEqual(summary["succeeded_samples"], 4)
        self.assertEqual([p["chunk"]["actions"][0] for p in predictions], [[0.1, 0.1], [0.2, 0.2]] * 2)
        self.assertTrue(all(e["data"]["steps"] == 0 for e in events
                            if e["type"] == "process_startup_progress" and e["data"]["stage"] == "warmup_completed"))

    def test_interrupt_closes_backend_and_writes_summary(self):
        config, observations, provenance = load_inference_config(self.config)
        class InterruptedBackend:
            closed = False
            def reset(self, episode_id):
                pass
            def submit(self, *args):
                raise KeyboardInterrupt
            def close(self, timeout_s):
                self.closed = True
                return True
            def poll_diagnostics(self):
                return ()
        backend = InterruptedBackend()
        with patch("robovla_runtime.integrations.offline_inference._factory") as factory:
            factory.return_value.create_backend.return_value = backend
            code, directory, summary = run_offline_inference(config, observations, provenance, self.output, emit=None)
        self.assertEqual(code, 130)
        self.assertTrue(backend.closed)
        self.assertEqual(summary["unprocessed_samples"], 2)
        self.assertTrue((directory / "summary.json").is_file())

    @unittest.skipIf(np is None, "optional numpy is not installed")
    def test_empty_and_truncated_npy_cli_returns_structured_validation_failure(self):
        stream = io.BytesIO()
        np.save(stream, np.zeros((2, 2, 3), dtype=np.uint8))
        complete = stream.getvalue()
        raw = json.loads(self.manifest.read_text(encoding="utf-8"))
        raw["samples"][0]["payload"] = {}
        raw["samples"][0]["arrays"] = {
            "camera": {"path": "camera.npy", "shape": [2, 2, 3], "dtype": "uint8"},
        }
        self.manifest.write_text(json.dumps(raw), encoding="utf-8")
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        for name, data in (("empty", b""), ("truncated_header", complete[:12]),
                           ("truncated_data", complete[:-1])):
            (self.root / "camera.npy").write_bytes(data)
            for extra in ([], ["--validate-only"]):
                with self.subTest(case=name, validate_only=bool(extra)):
                    result = subprocess.run([
                        sys.executable, "-m", "robovla_runtime.cli", "infer-fluxvla-process",
                        "--config", str(self.config), "--output-root", str(self.output),
                        "--quiet", *extra,
                    ], cwd=self.root, env=environment, capture_output=True,
                        text=True, encoding="utf-8", timeout=25)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertEqual(result.stdout, "")
                    error = json.loads(result.stderr)
                    self.assertEqual(error["error"], "validation_failed")
                    self.assertIn("invalid array camera in", error["message"])
                    self.assertIn(str(self.root / "camera.npy"), error["message"])
                    self.assertFalse(self.output.exists())

    @unittest.skipIf(np is None, "optional numpy is not installed")
    def test_npz_samples_reach_spawned_model_via_cli(self):
        np.savez(self.root / "camera.npz", image=np.ones((2, 2, 3), dtype=np.uint8))
        raw = json.loads(self.manifest.read_text(encoding="utf-8"))
        for sample in raw["samples"]:
            sample["payload"] = {}
            sample["arrays"] = {"camera": {"path": "camera.npz", "key": "image", "shape": [2, 2, 3], "dtype": "uint8"}}
        self.manifest.write_text(json.dumps(raw), encoding="utf-8")
        code, _, stderr = self.invoke(self.template.replace('fault_plan = ["request_hang"]', 'fault_plan = []'))
        self.assertEqual(code, 0, stderr)
        directory, summary, _, _ = self.artifacts()
        self.assertEqual(summary["succeeded_samples"], 2)
        self.assertEqual(len(json.loads((directory / "inputs.json").read_text(encoding="utf-8"))["files"]), 3)


if __name__ == "__main__":
    unittest.main()
