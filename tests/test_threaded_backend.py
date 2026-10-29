from __future__ import annotations

import time
import unittest
from dataclasses import replace
from threading import Event, get_ident

from robovla_runtime.backends.base import PendingPrediction
from robovla_runtime.backends.threaded import (
    PredictionFailure,
    ThreadedPolicyBackend,
)
from robovla_runtime.core.types import ActionChunk, ActionSpec, Observation
from robovla_runtime.runtime import RuntimeConfig, run_realtime_runtime


class GateBackend:
    def __init__(self, gate: Event | None = None, fail: bool = False) -> None:
        self.action_spec = ActionSpec(1, 0.01)
        self.gate = gate
        self.fail = fail
        self.started = Event()
        self.observation_ids: list[int] = []

    def reset(self, episode_id: str) -> None:
        pass  # This fixture has no model history.

    def request(self, observation: Observation, requested_at: float):
        self.observation_ids.append(observation.observation_id)
        self.started.set()
        if self.gate is not None:
            self.gate.wait(1.0)
        if self.fail:
            raise RuntimeError("inference exploded")
        chunk = ActionChunk(
            episode_id=observation.episode_id,
            request_id=999,
            source_observation_id=observation.observation_id,
            source_observation_time=observation.captured_at,
            ready_at=requested_at,
            start_at=observation.captured_at,
            spec=self.action_spec,
            actions=((1.0,),) * 10,
            episode_generation=observation.episode_generation,
        )
        return PendingPrediction(999, requested_at, requested_at, chunk)


def collect_results(backend, count: int, timeout_s: float = 1.0):
    results = []
    deadline = time.perf_counter() + timeout_s
    while len(results) < count and time.perf_counter() < deadline:
        results.extend(backend.poll())
        if len(results) < count:
            time.sleep(0.005)
    return results


class ThreadedPolicyBackendTests(unittest.TestCase):
    def test_mutating_failure_requires_successful_reset_before_next_inference(self):
        for recovery_reset_failures in (0, 2):
            with self.subTest(recovery_reset_failures=recovery_reset_failures):
                gate = Event()

                class StatefulBackend(GateBackend):
                    def __init__(self):
                        super().__init__(gate)
                        self.history = []
                        self.calls = []
                        self.reset_attempts = 0

                    def reset(self, episode_id):
                        self.calls.append(("reset", get_ident()))
                        self.reset_attempts += 1
                        if 1 < self.reset_attempts <= 1 + recovery_reset_failures:
                            raise RuntimeError("recovery reset failed")
                        self.history.clear()

                    def request(self, observation, requested_at):
                        self.calls.append(("request", get_ident()))
                        self.history.append(observation.observation_id)
                        result = super().request(observation, requested_at)
                        if observation.observation_id == 0:
                            raise RuntimeError("inference failed after mutating history")
                        return replace(result, chunk=replace(
                            result.chunk, actions=((float(len(self.history)),),),
                        ))

                blocking = StatefulBackend()
                backend = ThreadedPolicyBackend(blocking)
                try:
                    failed = backend.submit(Observation("ep", 0, 0.0, (0.0,)), 0.0)
                    self.assertTrue(blocking.started.wait(0.5))
                    # The next request is already waiting when inference fails.
                    queued = backend.submit(Observation("ep", 1, 0.01, (0.0,)), 0.01)
                    self.assertEqual(blocking.reset_attempts, 1)
                    gate.set()
                    results = collect_results(backend, 2)
                    self.assertEqual([r.request_id for r in results], [failed.request_id, queued.request_id])
                    self.assertIsInstance(results[0], PredictionFailure)
                    self.assertEqual(results[0].message, "inference failed after mutating history")
                    recovery = results[1]
                    for index in range(recovery_reset_failures):
                        self.assertIsInstance(recovery, PredictionFailure)
                        self.assertEqual(recovery.message, "recovery reset failed")
                        # Failed resets must never enter inference with dirty state.
                        self.assertEqual(blocking.observation_ids, [0])
                        observation_id = index + 2
                        receipt = backend.submit(Observation("ep", observation_id, 0.02, (0.0,)), 0.02)
                        recovered = collect_results(backend, 1)
                        self.assertEqual([r.request_id for r in recovered], [receipt.request_id])
                        recovery = recovered[0]
                    self.assertIsInstance(recovery, PendingPrediction)
                    self.assertEqual(recovery.chunk.actions, ((1.0,),))
                    self.assertEqual(blocking.reset_attempts, 2 + recovery_reset_failures)
                    # Successful requests retain normal same-episode history.
                    backend.submit(Observation("ep", 10, 0.03, (0.0,)), 0.03)
                    healthy = collect_results(backend, 1)
                    self.assertEqual(len(healthy), 1)
                    self.assertEqual(healthy[0].chunk.actions, ((2.0,),))
                    self.assertEqual(blocking.reset_attempts, 2 + recovery_reset_failures)
                    owner_threads = {thread_id for _, thread_id in blocking.calls}
                    self.assertEqual(len(owner_threads), 1)
                    self.assertNotIn(get_ident(), owner_threads)
                finally:
                    gate.set()
                    self.assertTrue(backend.close(1.0))

    def test_cancel_queued_does_not_interrupt_inflight_model(self):
        gate = Event()
        blocking = GateBackend(gate)
        backend = ThreadedPolicyBackend(blocking)
        try:
            first = backend.submit(Observation("ep", 0, 0.0, (0.0,)), 0.0)
            self.assertTrue(blocking.started.wait(0.5))
            queued = backend.submit(Observation("ep", 1, 0.01, (0.0,)), 0.01)
            self.assertFalse(backend.cancel_queued(first.request_id))
            self.assertTrue(backend.cancel_queued(queued.request_id))
            gate.set()
            results = collect_results(backend, 1)
            self.assertEqual([result.request_id for result in results], [first.request_id])
            self.assertEqual(blocking.observation_ids, [0])
        finally:
            gate.set()
            self.assertTrue(backend.close(1.0))

    def test_reset_is_serialized_after_inflight_request_and_clears_model_history(self):
        gate = Event()

        class StatefulBackend(GateBackend):
            def __init__(self):
                super().__init__(gate)
                self.history = []
                self.history_before_request = []
                self.reset_threads = []
                self.request_threads = []

            def reset(self, episode_id):
                self.reset_threads.append(get_ident())
                self.history.clear()

            def request(self, observation, requested_at):
                self.request_threads.append(get_ident())
                self.history_before_request.append(list(self.history))
                result = super().request(observation, requested_at)
                self.history.append(observation.observation_id)
                return result

        blocking = StatefulBackend()
        backend = ThreadedPolicyBackend(blocking)
        try:
            backend.submit(Observation("same", 0, 0.0, (0.0,)), 0.0)
            self.assertTrue(blocking.started.wait(0.5))
            backend.submit(Observation("same", 1, 0.01, (0.0,)), 0.01)
            backend.reset("same")
            # Reset never touches model state concurrently with the old request.
            self.assertEqual(len(blocking.reset_threads), 1)
            receipt = backend.submit(Observation("same", 2, 0.02, (0.0,)), 0.02)
            gate.set()
            results = collect_results(backend, 1)
            self.assertEqual([item.request_id for item in results], [receipt.request_id])
            self.assertEqual(blocking.observation_ids, [0, 2])
            self.assertEqual(blocking.history_before_request, [[], []])
            self.assertEqual(blocking.history, [2])
            self.assertEqual(blocking.reset_threads, blocking.request_threads)
            self.assertNotEqual(blocking.reset_threads[0], get_ident())
        finally:
            gate.set()
            self.assertTrue(backend.close(1.0))

    def test_reset_failure_is_reported_without_running_inference(self):
        class BadResetBackend(GateBackend):
            def reset(self, episode_id):
                raise RuntimeError("reset failed")

        blocking = BadResetBackend()
        backend = ThreadedPolicyBackend(blocking)
        try:
            backend.submit(Observation("ep", 0, 0.0, (0.0,)), 0.0)
            results = collect_results(backend, 1)
            self.assertEqual(len(results), 1)
            self.assertIsInstance(results[0], PredictionFailure)
            self.assertEqual(results[0].message, "reset failed")
            self.assertFalse(blocking.started.is_set())
        finally:
            self.assertTrue(backend.close(1.0))

    def test_submit_is_non_blocking_and_mailbox_keeps_latest(self) -> None:
        gate = Event()
        blocking = GateBackend(gate)
        backend = ThreadedPolicyBackend(blocking)
        try:
            first = backend.submit(Observation("ep", 0, 0.0, (0.0,)), 0.0)
            self.assertTrue(blocking.started.wait(0.5))

            second = backend.submit(Observation("ep", 1, 0.01, (0.0,)), 0.01)
            third = backend.submit(Observation("ep", 2, 0.02, (0.0,)), 0.02)

            self.assertIsNone(first.replaced_request_id)
            self.assertIsNone(second.replaced_request_id)
            self.assertEqual(third.replaced_request_id, second.request_id)
            self.assertEqual(backend.poll(), ())

            gate.set()
            results = collect_results(backend, 2)
            self.assertEqual([result.request_id for result in results], [0, 2])
            self.assertEqual(blocking.observation_ids, [0, 2])
            self.assertEqual(results[1].chunk.source_observation_id, 2)
        finally:
            gate.set()
            self.assertTrue(backend.close(1.0))

    def test_worker_exception_is_returned_to_control_thread(self) -> None:
        backend = ThreadedPolicyBackend(GateBackend(fail=True))
        try:
            receipt = backend.submit(
                Observation("ep", 0, 0.0, (0.0,)), requested_at=0.0
            )
            results = collect_results(backend, 1)

            self.assertEqual(len(results), 1)
            self.assertIsInstance(results[0], PredictionFailure)
            self.assertEqual(results[0].request_id, receipt.request_id)
            self.assertEqual(results[0].error_type, "RuntimeError")
        finally:
            self.assertTrue(backend.close(1.0))

    def test_cancelled_inflight_result_cannot_leak_into_reused_episode(self) -> None:
        gate = Event()
        blocking = GateBackend(gate)
        backend = ThreadedPolicyBackend(blocking)
        try:
            backend.submit(Observation("ep", 0, 0.0, (0.0,)), 0.0)
            self.assertTrue(blocking.started.wait(0.5))
            backend.cancel_episode("ep")
            replacement = backend.submit(
                Observation("ep", 1, 0.01, (0.0,)), 0.01
            )
            gate.set()

            results = collect_results(backend, 1)

            self.assertEqual([result.request_id for result in results], [replacement.request_id])
            self.assertEqual(results[0].chunk.source_observation_id, 1)
        finally:
            gate.set()
            self.assertTrue(backend.close(1.0))

    def test_realtime_control_ticks_while_inference_is_running(self) -> None:
        class SlowBackend(GateBackend):
            def request(self, observation, requested_at):
                time.sleep(0.06)
                return super().request(observation, requested_at)

        blocking = SlowBackend()
        backend = ThreadedPolicyBackend(blocking)
        config = RuntimeConfig(
            scheduler="worker_async",
            period_s=0.01,
            duration_s=0.04,
            chunk_size=10,
            execute_horizon=10,
            prefetch_threshold_s=0.02,
        )

        result = run_realtime_runtime(config, backend)

        self.assertEqual(len(result.records), 4)
        self.assertTrue(blocking.started.is_set())
        self.assertTrue(
            any(event.type == "mailbox_request_replaced" for event in result.events)
        )
        self.assertLess(result.records[-1].executed_at, 0.055)


if __name__ == "__main__":
    unittest.main()
