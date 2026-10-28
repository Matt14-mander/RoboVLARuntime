"""Blocking mock for testing real worker/control overlap without model libraries."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from .base import PendingPrediction
from .mock import MockBackend
from .latency import LatencyTrace
from robovla_runtime.core.types import ActionSpec, Observation


@dataclass(frozen=True, slots=True)
class SleepingMockFactory:
    """Serializable construction recipe; the model is built in its owner."""
    action_spec: ActionSpec
    chunk_size: int
    latency_s: tuple[float, ...]
    target: tuple[float, ...]
    latency_trace: LatencyTrace | None = None

    def __call__(self) -> SleepingMockBackend:
        return SleepingMockBackend(MockBackend(
            self.action_spec, self.chunk_size, self.latency_s, self.target,
            latency_trace=self.latency_trace,
        ))


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
