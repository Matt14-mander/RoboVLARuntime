"""Blocking mock for testing real worker/control overlap without model libraries."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

from .base import PendingPrediction
from .mock import MockBackend
from robovla_runtime.core.types import Observation


class SleepingMockBackend:
    """Sleep for MockBackend's synthetic latency, then return its action chunk.

    Use only behind a worker; the sleep intentionally blocks its calling thread.
    This emulates elapsed inference time, not CUDA work, GIL pressure, or a GPU.
    """

    def __init__(
        self, backend: MockBackend, *, timer: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.backend = backend
        self.action_spec = backend.action_spec
        self.timer = timer
        self.sleeper = sleeper

    def reset(self, episode_id: str) -> None:
        self.backend.reset(episode_id)

    def request(self, observation: Observation, requested_at: float) -> PendingPrediction:
        started_at = self.timer()
        prediction = self.backend.request(observation, requested_at)
        delay = max(0.0, prediction.ready_at - requested_at)
        self.sleeper(delay)
        elapsed = max(0.0, self.timer() - started_at)
        ready_at = requested_at + elapsed
        return replace(
            prediction, ready_at=ready_at,
            chunk=replace(prediction.chunk, ready_at=ready_at),
            timings_s=(("simulated_inference", elapsed), ("requested_sleep", delay)),
        )
