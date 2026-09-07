from __future__ import annotations

import time
import unittest
from threading import Event

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
