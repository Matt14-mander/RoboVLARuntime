from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import tempfile
import time
import unittest
from contextlib import nullcontext
from pathlib import Path

from benchmarks.fluxvla_fixtures import MockObservationFactory, make_simulated_factory, runtime_payload
from robovla_runtime.backends import FluxVLABackend, PredictionFailure, ProcessBackendConfig, decode_first_batch
from robovla_runtime.core import ActionSpec, Observation
from robovla_runtime.integrations import FluxVLALiberoFactory, FluxVLAProcessFactory
from robovla_runtime.runtime import RuntimeConfig, run_realtime_runtime
from tests.test_fluxvla_backend import FakeVLA


class FluxVLAProcessTests(unittest.TestCase):
    def factory(self, faults=(), steps=2):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        mock = make_simulated_factory(Path(temporary.name), faults)
        recipe = FluxVLAProcessFactory(mock, MockObservationFactory(), steps)
        return mock, recipe

    def backend(self, recipe):
        backend = recipe.create_backend(ActionSpec(2, 0.02), config=ProcessBackendConfig(
            startup_timeout_s=2.0, execution_timeout_s=0.2,
        ))
        self.addCleanup(lambda: self.assertTrue(backend.close(2.0)))
        backend.reset("episode")
        return backend

    def result(self, backend, timeout=5.0):
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            results = backend.poll()
            if results:
                self.assertEqual(len(results), 1, "warmup must never publish predictions")
                return results[0]
            time.sleep(0.005)
        self.fail("no result before test deadline")

    def success(self, backend, index=0):
        receipt = backend.submit(MockObservationFactory("inference")(index), index * 0.02)
        result = self.result(backend)
        self.assertNotIsInstance(result, PredictionFailure, result)
        self.assertEqual(result.request_id, receipt.request_id)
        self.assertEqual(result.chunk.source_observation_id, index)
        return result

    def audit(self, mock):
        return [json.loads(line) for line in mock.audit_path.read_text(encoding="utf-8").splitlines()]

    def test_serializable_recipe_does_not_load_model_in_parent(self):
        mock, recipe = self.factory()
        restored = pickle.loads(pickle.dumps(recipe))
        self.assertEqual(restored, recipe)
        self.assertFalse(mock.audit_path.exists())
        self.assertEqual(mock.attempts.value, 0)

    def test_complete_runner_preprocess_warmup_decode_chain_and_ready_order(self):
        mock, recipe = self.factory()
        backend = self.backend(recipe)
        first = self.success(backend)
        second = self.success(backend, 1)
        self.assertEqual(first.chunk.actions, ((0.1, 0.1),) * 30)
        self.assertEqual(second.chunk.actions[0], (0.2, 0.2))
        self.assertEqual(first.chunk.source_observation_time, 0.0)
        audit = self.audit(mock)
        predictions = [e for e in audit if e["kind"] == "predict"]
        self.assertEqual([e["phase"] for e in predictions], ["warmup", "warmup", "inference", "inference"])
        self.assertEqual([e["model_history"] for e in predictions], [1, 2, 1, 2])
        self.assertEqual([e["dataset_history"] for e in predictions], [1, 2, 1, 2])
        for event in predictions:
            self.assertEqual(event["seed"], 11)
            self.assertEqual(event["num_inference_steps"], 4)
            self.assertEqual(event["unnorm_key"], "mock_suite")
        config = next(e for e in audit if e["kind"] == "runner_config")
        self.assertFalse(config["has_feishu"])
        self.assertEqual(config["checkpoint"], mock.recipe.checkpoint_path)
        self.assertEqual(config["processor_path"], mock.recipe.checkpoint_path)
        self.assertNotEqual(predictions[0]["pid"], os.getpid())
        events = backend.poll_diagnostics()
        stages = [data["stage"] for kind, data in events if kind == "process_startup_progress"]
        self.assertEqual(stages, ["model_loaded", "warmup_started", "warmup_completed", "warmup_state_reset"])
        ready = next(i for i, (kind, _) in enumerate(events) if kind == "process_worker_ready")
        reset = next(i for i, (_, data) in enumerate(events) if data.get("stage") == "warmup_state_reset")
        self.assertLess(reset, ready)

    def test_same_episode_restart_reloads_rewarms_and_resets_both_histories(self):
        mock, recipe = self.factory()
        backend = self.backend(recipe)
        self.success(backend)
        self.success(backend, 1)
        backend.reset("episode")
        result = self.success(backend)
        self.assertEqual(result.chunk.actions[0], (0.1, 0.1))
        self.assertEqual(mock.attempts.value, 2)
        audit = self.audit(mock)
        for generation in (1, 2):
            warmups = [e for e in audit if e["generation"] == generation and e["kind"] == "predict" and e["phase"] == "warmup"]
            self.assertEqual(len(warmups), 2)
        self.assertTrue(backend.close(2.0))
        pids = {e["pid"] for e in audit}
        self.assertFalse(pids & {p.pid for p in mp.active_children()})

    def test_warmup_failure_hang_and_cleanup_hang_never_send_ready_then_recover(self):
        for fault in ("setup_error", "warmup_error", "warmup_hang", "post_warmup_reset_hang"):
            with self.subTest(fault=fault):
                mock, recipe = self.factory((fault,))
                backend = self.backend(recipe)
                backend.submit(MockObservationFactory("inference")(0), 0.0)
                failed = self.result(backend)
                self.assertIsInstance(failed, PredictionFailure)
                if "hang" in fault:
                    self.assertEqual(failed.error_type, "ProcessStartupTimeout")
                events = backend.poll_diagnostics()
                self.assertFalse(any(kind == "process_worker_ready" for kind, _ in events))
                if fault == "setup_error":
                    self.assertTrue(any(e["kind"] == "cleanup" for e in self.audit(mock)))
                self.assertEqual(self.success(backend, 1).chunk.actions[0], (0.1, 0.1))
                self.assertEqual(mock.attempts.value, 2)
                self.assertTrue(backend.close(2.0))

    def test_runtime_recovers_inference_hang_with_freshly_warmed_model(self):
        mock, recipe = self.factory(("request_hang",))
        backend = self.backend(recipe)
        result = run_realtime_runtime(RuntimeConfig(
            scheduler="worker_async", worker_prefetch="adaptive", worker_isolation="process",
            period_s=0.02, duration_s=2.5, chunk_size=30, execute_horizon=20,
            initial_position=(0.0, 0.0), target_position=(1.0, 1.0),
            request_timeout_s=0.4, worker_close_timeout_s=1.0,
        ), backend, observation_payload_factory=runtime_payload)
        self.assertEqual(len(result.records), 125)
        actions = [r for r in result.records if r.request_id is not None]
        self.assertTrue(actions)
        self.assertTrue(all(r.command[0] == r.command[1] for r in actions))
        expired = {e.data["request_id"] for e in result.events if e.type == "prediction_timed_out"}
        self.assertFalse(expired & {r.request_id for r in actions})
        self.assertEqual(result.summary["process_worker"]["warmups_completed"], 2)
        self.assertEqual(result.summary["process_worker"]["warmup_state_resets"], 2)
        self.assertEqual(mock.attempts.value, 2)
        self.assertEqual(result.summary["process_worker"]["terminations"], 2)

    def test_missing_reset_requires_explicit_stateless_declaration(self):
        def factory():
            return FluxVLABackend(FakeVLA(), lambda obs: {}, decode_first_batch,
                                  ActionSpec(1, 0.02), inference_context=nullcontext)
        with self.assertRaisesRegex(ValueError, "reset hook"):
            FluxVLAProcessFactory(factory, warmup_steps=0)()
        backend = FluxVLAProcessFactory(factory, warmup_steps=0, model_is_stateless=True)()
        reports = []
        backend.startup(lambda stage, **data: reports.append(stage))
        self.assertEqual(reports[-1], "warmup_state_reset")

    def test_zero_warmup_still_resets_model_before_live_request(self):
        mock, recipe = self.factory(steps=0)
        backend = self.backend(recipe)
        self.assertEqual(self.success(backend).chunk.actions[0], (0.1, 0.1))
        self.assertFalse(any(e.get("phase") == "warmup" for e in self.audit(mock)))

    def test_warmup_does_not_mutate_source_sample(self):
        sample = Observation("source", 9, 0.0, (0.0,), {"frame": [1, 2]})
        def preprocess(observation):
            observation.payload["frame"][0] = 999
            return {}
        def build():
            return FluxVLABackend(FakeVLA(), preprocess, decode_first_batch, ActionSpec(1, 0.02),
                                  inference_context=nullcontext)
        backend = FluxVLAProcessFactory(build, lambda index: sample, model_is_stateless=True)()
        backend.startup(lambda *args, **kwargs: None)
        self.assertEqual(sample.payload, {"frame": [1, 2]})
        self.assertEqual(sample.observation_id, 9)

    def test_invalid_recipes_fail_early(self):
        for steps in (-1, True, 1.5):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                FluxVLAProcessFactory(FakeVLA, warmup_steps=steps)
        with self.assertRaisesRegex(ValueError, "observation_factory"):
            FluxVLAProcessFactory(FakeVLA)
        with self.assertRaisesRegex(ValueError, "task_description"):
            FluxVLALiberoFactory("model", "weights", ActionSpec(2, 0.02), "")


if __name__ == "__main__":
    unittest.main()
