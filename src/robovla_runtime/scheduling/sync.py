"""Request a prediction only after executable coverage is exhausted."""

from __future__ import annotations

from collections.abc import Callable

from robovla_runtime.core.types import Observation
from robovla_runtime.execution.action_buffer import ActionBuffer

from .base import BaseScheduler


class SyncScheduler(BaseScheduler):
    def tick(
        self,
        now: float,
        action_buffer: ActionBuffer,
        capture_observation: Callable[[float], Observation],
    ) -> None:
        self._poll(now, action_buffer)
        if self.pending is None and action_buffer.is_empty(now):
            self._request(now, capture_observation)
            self._poll(now, action_buffer)

