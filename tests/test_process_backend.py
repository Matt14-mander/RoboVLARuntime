from __future__ import annotations

import multiprocessing as mp
import time
import tempfile
import unittest
from pathlib import Path

from benchmarks.process_fixtures import AttemptCounter, RecoveryMockFactory
from robovla_runtime.backends import ProcessBackendConfig, ProcessPolicyBackend, PredictionFailure
from robovla_runtime.core import ActionSpec, Observation
from robovla_runtime.runtime import RuntimeConfig, run_realtime_runtime


class ProcessBackendTests(unittest.TestCase):
    def backend(self, faults=(), **settings):
        spec = ActionSpec(1, 0.02)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        attempts = AttemptCounter(Path(temporary.name) / "attempts.txt")
        factory = RecoveryMockFactory(spec, attempts, tuple(faults))
        backend = ProcessPolicyBackend(factory, spec, config=ProcessBackendConfig(
            **{"startup_timeout_s": 2.0, "execution_timeout_s": 0.2, **settings},
        ))
        self.addCleanup(lambda: self.assertTrue(backend.close(2.0)))
        backend.reset("ep")
        return backend, attempts

    def observation(self, index=0):
        return Observation("ep", index, index * 0.02, (0.0,), episode_generation=1)

    def wait_result(self, backend, timeout=5.0):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            results = backend.poll()
            if results:
                return results[0]
            time.sleep(0.005)
        self.fail("no process result before test deadline")

    def wait_event(self, backend, kind, timeout=5.0):
        deadline = time.perf_counter() + timeout
        seen = []
        while time.perf_counter() < deadline:
            seen.extend(backend.poll_diagnostics())
            for event_kind, data in seen:
                if event_kind == kind:
                    return data
            time.sleep(0.005)
        self.fail(f"missing {kind}; events={seen}")

    def test_spawn_owner_identity_state_and_reused_episode_reset(self):
        backend, attempts = self.backend()
        for index in range(2):
            receipt = backend.submit(self.observation(index), index * 0.02)
            result = self.wait_result(backend)
            self.assertNotIsInstance(result, PredictionFailure, result)
            self.assertEqual(result.request_id, receipt.request_id)
            self.assertEqual(result.chunk.source_observation_id, index)
            self.assertEqual(result.chunk.episode_generation, 1)
            self.assertEqual(result.chunk.actions[0], (float(index + 1),))
        backend.reset("ep")
        backend.submit(self.observation(0), 0.0)
        result = self.wait_result(backend)
        self.assertEqual(result.chunk.actions[0], (1.0,))
        self.assertEqual(attempts.value, 2)

    def test_scheduler_expiry_kills_active_call_and_preserves_new_request(self):
        backend, attempts = self.backend(("request_hang",), execution_timeout_s=5.0)
        first = backend.submit(self.observation(), 0.0)
        old = self.wait_event(backend, "process_request_started")
        queued = backend.submit(self.observation(1), 0.02)
        self.assertEqual(backend.expire_request(queued.request_id), "queued")
        self.assertEqual(backend.expire_request(first.request_id), "active")
        healthy = backend.submit(self.observation(2), 0.04)
        result = self.wait_result(backend)
        self.assertEqual(result.request_id, healthy.request_id)
        self.assertEqual(result.chunk.actions[0], (1.0,))
        events = backend.poll_diagnostics()
        stopped = [data for kind, data in events if kind == "process_worker_terminated"]
        self.assertEqual(stopped[0]["pid"], old["pid"])
        self.assertEqual(stopped[0]["reason"], "request_timeout")
        self.assertEqual(attempts.value, 2)
        self.assertNotIn(old["pid"], [child.pid for child in mp.active_children()])

    def test_watchdog_recovers_reset_hang_crash_and_mutating_failure(self):
        for fault in ("reset_hang", "crash", "request_error"):
            with self.subTest(fault=fault):
                backend, attempts = self.backend((fault,))
                first = backend.submit(self.observation(), 0.0)
                failed = self.wait_result(backend)
                self.assertIsInstance(failed, PredictionFailure)
                self.assertEqual(failed.request_id, first.request_id)
                second = backend.submit(self.observation(1), 0.02)
                recovered = self.wait_result(backend)
                self.assertEqual(recovered.request_id, second.request_id)
                self.assertEqual(recovered.chunk.actions[0], (1.0,))
                self.assertEqual(attempts.value, 2)
                self.assertTrue(backend.close(2.0))

    def test_startup_timeout_recovers_on_next_request(self):
        backend, attempts = self.backend(("startup_hang",), startup_timeout_s=2.0)
        backend.submit(self.observation(), 0.0)
        failed = self.wait_result(backend)
        self.assertEqual(failed.error_type, "ProcessStartupTimeout")
        backend.submit(self.observation(1), 0.02)
        self.assertEqual(self.wait_result(backend).chunk.actions[0], (1.0,))
        self.assertEqual(attempts.value, 2)

    def test_latest_mailbox_during_startup_and_reset_invalidate_old_generation(self):
        backend, attempts = self.backend(("startup_hang",), startup_timeout_s=5.0)
        first = backend.submit(self.observation(), 0.0)
        self.wait_event(backend, "process_worker_started")
        # Wait for the factory to select its first fault, then reset during startup.
        deadline = time.perf_counter() + 2.0
        while attempts.value == 0 and time.perf_counter() < deadline:
            time.sleep(0.005)
        self.assertEqual(attempts.value, 1)
        replacement = backend.submit(self.observation(1), 0.02)
        self.assertEqual(replacement.replaced_request_id, first.request_id)
        backend.reset("ep")
        current = backend.submit(self.observation(2), 0.04)
        result = self.wait_result(backend)
        self.assertEqual(result.request_id, current.request_id)
        self.assertEqual(result.chunk.actions[0], (1.0,))
        self.assertEqual(backend.poll(), ())

    def test_restart_budget_stops_repeated_faults_and_explicit_reset_rearms(self):
        backend, attempts = self.backend(("request_hang",) * 4, max_restarts=1, execution_timeout_s=0.05)
        for index in range(2):
            backend.submit(self.observation(index), index * 0.02)
            self.assertEqual(self.wait_result(backend).error_type, "ProcessExecutionTimeout")
        backend.submit(self.observation(2), 0.04)
        self.assertEqual(self.wait_result(backend).error_type, "ProcessUnavailable")
        self.assertEqual(attempts.value, 2)
        backend.reset("ep")
        backend.submit(self.observation(), 0.0)
        self.assertEqual(self.wait_result(backend).error_type, "ProcessExecutionTimeout")
        self.assertEqual(attempts.value, 3)

    def test_unserializable_observation_is_reported_then_recovers(self):
        backend, attempts = self.backend()
        backend.submit(Observation("ep", 0, 0.0, (0.0,), payload=lambda: None), 0.0)
        self.assertIsInstance(self.wait_result(backend), PredictionFailure)
        backend.submit(self.observation(1), 0.02)
        self.assertEqual(self.wait_result(backend).chunk.actions[0], (1.0,))
        self.assertEqual(attempts.value, 2)

    def test_expired_startup_queue_does_not_enter_model_or_restart_healthy_child(self):
        backend, attempts = self.backend()
        backend.submit_with_deadline(self.observation(), 0.0, 0.000001)
        failed = self.wait_result(backend)
        self.assertEqual(failed.error_type, "ProcessQueueTimeout")
        backend.submit(self.observation(1), 0.02)
        self.assertEqual(self.wait_result(backend).chunk.actions[0], (1.0,))
        self.assertEqual(attempts.value, 1)

    def test_cancel_episode_kills_active_and_suppresses_old_result(self):
        backend, attempts = self.backend(("request_hang",), execution_timeout_s=5.0)
        first = backend.submit(self.observation(), 0.0)
        self.wait_event(backend, "process_request_started")
        self.assertIn(first.request_id, backend.cancel_episode("ep"))
        backend.reset("ep")
        next_request = backend.submit(self.observation(1), 0.02)
        self.assertEqual(self.wait_result(backend).request_id, next_request.request_id)
        self.assertEqual(backend.poll(), ())
        self.assertEqual(attempts.value, 2)

    def test_non_picklable_factory_fails_without_leaking_child(self):
        backend = ProcessPolicyBackend(lambda: None, ActionSpec(1, 0.02))
        try:
            backend.submit(self.observation(), 0.0)
            self.assertIsInstance(self.wait_result(backend), PredictionFailure)
        finally:
            self.assertTrue(backend.close(2.0))

    def test_deadline_watchdog_terminates_without_scheduler_polling(self):
        backend, _ = self.backend(("request_hang",), execution_timeout_s=30.0)
        backend.submit_with_deadline(self.observation(), 0.0, 1.0)
        self.wait_event(backend, "process_request_started")
        stopped = self.wait_event(backend, "process_worker_terminated", timeout=2.0)
        self.assertEqual(stopped["reason"], "ProcessExecutionTimeout")
        self.assertEqual(self.wait_result(backend).error_type, "ProcessExecutionTimeout")

    def test_close_terminates_hung_process_and_reaps_it(self):
        backend, _ = self.backend(("request_hang",), execution_timeout_s=5.0)
        backend.submit(self.observation(), 0.0)
        event = self.wait_event(backend, "process_request_started")
        backend.close(0.0)
        self.assertTrue(backend.close(2.0))
        self.assertNotIn(event["pid"], [child.pid for child in mp.active_children()])
        with self.assertRaises(RuntimeError):
            backend.submit(self.observation(), 0.0)

    def test_runtime_keeps_ticking_then_executes_fresh_actions_after_timeout(self):
        backend, _ = self.backend(("request_hang",), execution_timeout_s=5.0)
        result = run_realtime_runtime(RuntimeConfig(
            scheduler="worker_async", worker_prefetch="adaptive", worker_isolation="process",
            period_s=0.02, duration_s=1.4, request_timeout_s=0.2,
            chunk_size=100, execute_horizon=20, worker_close_timeout_s=1.0,
        ), backend)
        self.assertEqual(len(result.records), 70)
        timed_out = {e.data["request_id"] for e in result.events if e.type == "prediction_timed_out"}
        executed = {r.request_id for r in result.records if r.request_id is not None}
        self.assertTrue(timed_out)
        self.assertTrue(executed)
        self.assertFalse(timed_out & executed)
        self.assertEqual(result.summary["process_worker"]["restarts"], 1)
        self.assertEqual(result.summary["process_worker"]["terminations"], 2)
        self.assertEqual(result.summary["worker_shutdown_timeout_count"], 0)
        accepted = [e for e in result.events if e.type == "chunk_accepted"]
        stopped = [e for e in result.events if e.type == "process_worker_terminated"]
        self.assertLess(stopped[0].timestamp, accepted[0].timestamp)

    def test_invalid_process_settings(self):
        for settings in ({"execution_timeout_s": 0}, {"startup_timeout_s": float("inf")},
                         {"terminate_grace_s": -1}, {"max_restarts": -1}, {"max_restarts": 1.5}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ProcessBackendConfig(**settings)
        with self.assertRaises(ValueError):
            RuntimeConfig(worker_isolation="invalid")


if __name__ == "__main__":
    unittest.main()
