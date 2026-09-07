"""Worker-thread adapter for blocking policy backends."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, replace
from threading import Condition, Thread
from typing import Protocol

from robovla_runtime.core.types import ActionSpec, Observation

from .base import PendingPrediction, PolicyBackend


@dataclass(frozen=True, slots=True)
class SubmissionReceipt:
    request_id: int
    replaced_request_id: int | None = None


@dataclass(frozen=True, slots=True)
class PredictionFailure:
    request_id: int
    requested_at: float
    episode_id: str
    error_type: str
    message: str


AsyncResult = PendingPrediction | PredictionFailure


class AsyncPolicyBackend(Protocol):
    action_spec: ActionSpec

    def submit(
        self, observation: Observation, requested_at: float
    ) -> SubmissionReceipt: ...

    def poll(self) -> tuple[AsyncResult, ...]: ...

    def cancel_episode(self, episode_id: str) -> tuple[int, ...]: ...

    def close(self, timeout_s: float | None = None) -> bool: ...


@dataclass(frozen=True, slots=True)
class _Submission:
    request_id: int
    observation: Observation
    requested_at: float
    enqueued_at: float
    episode_generation: int


class ThreadedPolicyBackend:
    """Run a blocking backend on one worker with a latest-only mailbox.

    One request may be executing and one observation may wait. A newer submit
    atomically replaces the waiting observation, preventing an unbounded stale
    queue. CUDA model ownership remains in one thread.
    """

    def __init__(
        self,
        backend: PolicyBackend,
        *,
        timer=time.perf_counter,
        thread_name: str = "robovla-policy-worker",
    ) -> None:
        self.backend = backend
        self.action_spec = backend.action_spec
        self.timer = timer
        self._condition = Condition()
        self._queued: _Submission | None = None
        self._results: deque[AsyncResult] = deque()
        self._episode_generations: dict[str, int] = {}
        self._next_request_id = 0
        self._closed = False
        self._worker = Thread(target=self._run, name=thread_name, daemon=True)
        self._worker.start()

    def submit(
        self, observation: Observation, requested_at: float
    ) -> SubmissionReceipt:
        with self._condition:
            if self._closed:
                raise RuntimeError("threaded backend is closed")
            request_id = self._next_request_id
            self._next_request_id += 1
            replaced = self._queued.request_id if self._queued is not None else None
            generation = self._episode_generations.setdefault(
                observation.episode_id, 0
            )
            self._queued = _Submission(
                request_id,
                observation,
                requested_at,
                self.timer(),
                generation,
            )
            self._condition.notify()
            return SubmissionReceipt(request_id, replaced)

    def poll(self) -> tuple[AsyncResult, ...]:
        with self._condition:
            results = tuple(self._results)
            self._results.clear()
            return results

    def cancel_episode(self, episode_id: str) -> tuple[int, ...]:
        """Drop queued/results and suppress an in-flight result for an episode."""
        with self._condition:
            self._episode_generations[episode_id] = (
                self._episode_generations.get(episode_id, 0) + 1
            )
            dropped: list[int] = []
            if self._queued is not None and (
                self._queued.observation.episode_id == episode_id
            ):
                dropped.append(self._queued.request_id)
                self._queued = None
            retained: deque[AsyncResult] = deque()
            for result in self._results:
                result_episode = (
                    result.chunk.episode_id
                    if isinstance(result, PendingPrediction)
                    else result.episode_id
                )
                if result_episode == episode_id:
                    dropped.append(result.request_id)
                else:
                    retained.append(result)
            self._results = retained
            return tuple(dropped)

    def close(self, timeout_s: float | None = None) -> bool:
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        with self._condition:
            self._closed = True
            self._queued = None
            self._condition.notify_all()
        self._worker.join(timeout_s)
        return not self._worker.is_alive()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._queued is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                submission = self._queued
                self._queued = None

            worker_started = self.timer()
            queue_delay_s = max(0.0, worker_started - submission.enqueued_at)
            try:
                prediction = self.backend.request(
                    submission.observation,
                    submission.requested_at + queue_delay_s,
                )
                completed_at = self.timer()
                end_to_end_s = max(0.0, completed_at - submission.enqueued_at)
                ready_at = submission.requested_at + end_to_end_s
                chunk = replace(
                    prediction.chunk,
                    request_id=submission.request_id,
                    ready_at=ready_at,
                )
                result: AsyncResult = PendingPrediction(
                    request_id=submission.request_id,
                    requested_at=submission.requested_at,
                    ready_at=ready_at,
                    chunk=chunk,
                    timings_s=(
                        ("queue", queue_delay_s),
                        *prediction.timings_s,
                        ("end_to_end", end_to_end_s),
                    ),
                )
            except Exception as error:  # surfaced to the control thread by poll()
                result = PredictionFailure(
                    submission.request_id,
                    submission.requested_at,
                    submission.observation.episode_id,
                    type(error).__name__,
                    str(error),
                )

            with self._condition:
                current_generation = self._episode_generations.get(
                    submission.observation.episode_id, 0
                )
                if submission.episode_generation == current_generation:
                    self._results.append(result)
