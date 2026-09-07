"""Policy backend protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from robovla_runtime.core.types import ActionChunk, ActionSpec, Observation


@dataclass(frozen=True, slots=True)
class PendingPrediction:
    request_id: int
    requested_at: float
    ready_at: float
    chunk: ActionChunk
    timings_s: tuple[tuple[str, float], ...] = ()

    def is_ready(self, now: float) -> bool:
        return now + 1e-12 >= self.ready_at


class PolicyBackend(Protocol):
    action_spec: ActionSpec

    def request(
        self, observation: Observation, requested_at: float
    ) -> PendingPrediction: ...
