from __future__ import annotations

from dataclasses import replace
from threading import Event
import unittest

from robovla_runtime.backends import (
    PendingPrediction, PredictionFailure, SubmissionReceipt, ThreadedPolicyBackend,
)
from robovla_runtime.core import ActionChunk, ActionSpec, EventLog, Observation
from robovla_runtime.execution.action_buffer import ActionBuffer
from robovla_runtime.runtime import RuntimeConfig, run_realtime_runtime
from robovla_runtime.scheduling.adaptive import AdaptivePrefetchConfig
from robovla_runtime.scheduling.worker_async import WorkerAsyncScheduler
from tests.test_environment_adapter import ExternalEnvironment
from tests.test_threaded_backend import GateBackend


class ControlledBackend:
    """Explicit completions keep deadline tests independent of OS scheduling."""
    action_spec = ActionSpec(1, 0.01)

    def __init__(self):
        self.submissions = {}
        self.results = []
        self.replace_next = None
        self.queued_cancellations = []
        self.cancelled = []
        self.close_timeout = None

    def reset(self, episode_id):
        self.results.clear()

    def submit(self, observation, requested_at):
        request_id = len(self.submissions)
        self.submissions[request_id] = (observation, requested_at)
        receipt = SubmissionReceipt(request_id, self.replace_next)
        self.replace_next = None
        return receipt

    def success(self, request_id, ready_at):
        obs, requested_at = self.submissions[request_id]
        result = PendingPrediction(
            request_id, requested_at, ready_at,
            ActionChunk(
                obs.episode_id, request_id, obs.observation_id, obs.captured_at,
                ready_at, obs.captured_at, self.action_spec,
                ((float(request_id + 1),),) * 100,
                episode_generation=obs.episode_generation,
            ),
            (("queue", 0.02), ("inference", 0.02)),
        )
        self.results.append(result)
        return result

    def poll(self):
        results, self.results = tuple(self.results), []
        return results

    def cancel_queued(self, request_id):
        self.queued_cancellations.append(request_id)
        return False

    def cancel_episode(self, episode_id):
        self.cancelled.append(episode_id)
        return ()

    def close(self, timeout_s=None):
        self.close_timeout = timeout_s
        return True


class WorkerSchedulerTests(unittest.TestCase):
    def harness(self, *, adaptive=True, timeout=0.05, clock=None):
        backend = ControlledBackend()
        log = EventLog()
        buffer = ActionBuffer("ep", action_spec=backend.action_spec)
        scheduler = WorkerAsyncScheduler(
            backend, log, 0.08, request_timeout_s=timeout,
            adaptive_config=AdaptivePrefetchConfig() if adaptive else None,
            coverage_budget_s=1.0, clock=clock,
        )
        def capture(now):
            return Observation("ep", len(backend.submissions), now, (0.0,))
        scheduler.tick(0.0, buffer, capture)
        return backend, log, buffer, scheduler, capture

    def events(self, log, kind):
        return [event for event in log.events if event.type == kind]

    def test_adaptive_sample_includes_queue_inference_and_collection(self):
        backend, log, buffer, scheduler, _ = self.harness(timeout=0.1)
        backend.success(0, 0.04)
        scheduler.poll(0.05, buffer)
        self.assertAlmostEqual(scheduler.prefetch.samples[0], 0.05)
        self.assertAlmostEqual(scheduler.prefetch.threshold_s, 0.07)
        self.assertEqual(len(self.events(log, "chunk_accepted")), 1)

    def test_timeout_is_counted_once_and_late_chunk_is_never_executed(self):
        backend, log, buffer, scheduler, _ = self.harness()
        scheduler.poll(0.05, buffer)
        scheduler.poll(0.06, buffer)
        self.assertEqual(len(self.events(log, "prediction_timed_out")), 1)
        self.assertEqual(backend.queued_cancellations, [0])
        self.assertEqual(len(scheduler.prefetch.samples), 0)  # Timeout is censored.
        result = backend.success(0, 0.07)
        scheduler.poll(0.08, buffer)
        self.assertIsNone(buffer.pop(0.08))
        self.assertAlmostEqual(scheduler.prefetch.samples[0], 0.08)
        self.assertEqual(self.events(log, "prediction_discarded")[-1].data["reason"], "request_timeout")
        backend.results.append(result)
        scheduler.poll(0.09, buffer)
        self.assertEqual(len(scheduler.prefetch.samples), 1)
        self.assertEqual(self.events(log, "prediction_discarded")[-1].data["reason"], "duplicate_result")

    def test_completion_at_deadline_is_accepted_but_collection_after_it_is_not(self):
        for collected, accepted in ((0.05, True), (0.06, False)):
            with self.subTest(collected=collected):
                backend, log, buffer, scheduler, _ = self.harness()
                backend.success(0, 0.03)
                scheduler.poll(collected, buffer)
                self.assertEqual(buffer.pop(collected) is not None, accepted)
                self.assertEqual(len(self.events(log, "prediction_timed_out")), int(not accepted))

    def test_poll_clock_refresh_prevents_slow_poll_from_bypassing_deadline(self):
        clock = [0.0]
        backend, log, buffer, scheduler, _ = self.harness(clock=lambda: clock[0])
        backend.success(0, 0.03)
        original_poll = backend.poll
        def slow_poll():
            clock[0] = 0.06
            return original_poll()
        backend.poll = slow_poll
        scheduler.poll(0.04, buffer)
        self.assertIsNone(buffer.pop(0.06))
        self.assertEqual(len(self.events(log, "prediction_timed_out")), 1)

    def test_late_failure_is_reported_without_action_or_latency_sample(self):
        backend, log, buffer, scheduler, _ = self.harness()
        scheduler.poll(0.05, buffer)
        backend.results.append(PredictionFailure(0, 0.0, "ep", "RuntimeError", "failed"))
        scheduler.poll(0.08, buffer)
        self.assertTrue(self.events(log, "prediction_failed")[0].data["late"])
        self.assertEqual(len(scheduler.prefetch.samples), 0)
        self.assertIsNone(buffer.pop(0.08))

    def test_out_of_order_result_cannot_replace_newer_observation(self):
        backend, log, buffer, scheduler, capture = self.harness(adaptive=False)
        scheduler.tick(0.01, buffer, capture)
        backend.success(1, 0.02)
        backend.success(0, 0.02)
        scheduler.poll(0.02, buffer)
        self.assertEqual(buffer.pop(0.02).request_id, 1)
        self.assertEqual(self.events(log, "prediction_discarded")[0].data["reason"], "superseded_observation")

    def test_replaced_mailbox_result_is_rejected(self):
        backend, log, buffer, scheduler, capture = self.harness(adaptive=False)
        backend.replace_next = 0
        scheduler.tick(0.01, buffer, capture)
        backend.success(0, 0.02)
        scheduler.poll(0.02, buffer)
        self.assertIsNone(buffer.pop(0.02))
        self.assertEqual(self.events(log, "prediction_discarded")[0].data["reason"], "mailbox_replaced")

    def test_wrong_episode_generation_or_observation_is_rejected(self):
        for changes in ({"episode_id": "old"}, {"episode_generation": 3}, {"source_observation_id": 99}):
            with self.subTest(changes=changes):
                backend, log, buffer, scheduler, _ = self.harness()
                result = backend.success(0, 0.02)
                backend.results[:] = [replace(result, chunk=replace(result.chunk, **changes))]
                scheduler.poll(0.02, buffer)
                self.assertIsNone(buffer.pop(0.02))
                self.assertEqual(self.events(log, "prediction_discarded")[0].data["reason"], "identity_mismatch")

    def test_adaptive_keeps_one_live_request_and_recovers_after_late_result(self):
        backend, log, buffer, scheduler, capture = self.harness()
        for now in (0.01, 0.02, 0.03, 0.04):
            scheduler.tick(now, buffer, capture)
        self.assertEqual(len(backend.submissions), 1)
        scheduler.tick(0.05, buffer, capture)  # Timeout, then new request.
        self.assertEqual(len(backend.submissions), 2)
        backend.success(0, 0.06)
        backend.success(1, 0.06)
        scheduler.poll(0.07, buffer)
        self.assertEqual(buffer.pop(0.07).request_id, 1)
        self.assertEqual(len(self.events(log, "prediction_timed_out")), 1)
        self.assertEqual(len(self.events(log, "chunk_accepted")), 1)

    def test_expired_diagnostic_history_is_bounded_and_unknown_results_stay_rejected(self):
        backend, log, buffer, scheduler, capture = self.harness(timeout=0.001)
        for tick in range(1, 300):
            scheduler.tick(tick * 0.02, buffer, capture)
        self.assertLessEqual(len(scheduler._retired), 256)
        self.assertLessEqual(len(scheduler._requests), 1)
        backend.success(0, 6.0)
        scheduler.poll(6.0, buffer)
        self.assertEqual(self.events(log, "prediction_discarded")[-1].data["reason"], "unknown_request")

    def test_runtime_continues_ticks_and_returns_with_blocked_worker(self):
        gate = Event()
        blocking = GateBackend(gate)
        backend = ThreadedPolicyBackend(blocking)
        env = ExternalEnvironment(blocking.action_spec)
        clock = [0.0]
        def sleep(duration):
            self.assertTrue(blocking.started.wait(0.5))
            clock[0] += duration
        try:
            result = run_realtime_runtime(
                RuntimeConfig(
                    scheduler="worker_async", worker_prefetch="adaptive", period_s=0.01,
                    duration_s=0.05, request_timeout_s=0.02, worker_close_timeout_s=0.0,
                ), backend, environment=env, timer=lambda: clock[0], sleeper=sleep,
            )
            self.assertFalse(gate.is_set())
            self.assertEqual(len(result.records), 5)
            self.assertAlmostEqual(env.elapsed, 0.05)
            self.assertGreater(result.summary["prediction_timeout_count"], 0)
            self.assertEqual(result.summary["worker_shutdown_timeout_count"], 1)
            self.assertEqual(result.summary["buffer_underrun_ticks"], 5)
        finally:
            gate.set()
            self.assertTrue(backend.close(1.0))

    def test_cleanup_still_cancels_and_joins_when_poll_raises(self):
        backend = ControlledBackend()
        def fail():
            raise RuntimeError("transport failed")
        backend.poll = fail
        with self.assertRaisesRegex(RuntimeError, "transport failed"):
            run_realtime_runtime(
                RuntimeConfig(scheduler="worker_async", period_s=0.01, duration_s=0.02,
                              worker_close_timeout_s=0.03), backend,
            )
        self.assertEqual(backend.cancelled, ["episode-seed-7"])
        self.assertEqual(backend.close_timeout, 0.03)

    def test_invalid_worker_settings_fail_before_startup(self):
        for settings in ({"worker_prefetch": "unknown"}, {"request_timeout_s": 0},
                         {"request_timeout_s": float("inf")}, {"worker_close_timeout_s": -1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                RuntimeConfig(**settings)


if __name__ == "__main__":
    unittest.main()
