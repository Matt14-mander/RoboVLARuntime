"""Prefetch a chunk when remaining action coverage crosses a threshold."""

from __future__ import annotations

from collections.abc import Callable

from robovla_runtime.core.types import Observation
from robovla_runtime.execution.action_buffer import ActionBuffer

from .base import BaseScheduler


class FixedAsyncScheduler(BaseScheduler):
    def __init__(self, backend, event_log, prefetch_threshold_s: float) -> None:
        super().__init__(backend, event_log)
        if prefetch_threshold_s < 0:
            raise ValueError("prefetch threshold must be non-negative")
        self.prefetch_threshold_s = prefetch_threshold_s

    def tick(
        self,
        now: float,
        action_buffer: ActionBuffer,
        capture_observation: Callable[[float], Observation],
    ) -> None:
        self._poll(now, action_buffer)
        coverage = action_buffer.coverage_s(now)
        if self.pending is None and coverage <= self.prefetch_threshold_s + 1e-12:
            self._request(now, capture_observation)
            self._poll(now, action_buffer)

