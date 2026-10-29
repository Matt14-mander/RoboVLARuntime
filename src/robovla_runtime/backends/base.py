"""Policy backend protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from robovla_runtime.core.types import ActionChunk, ActionSpec, Observation, validate_timestamp


@dataclass(frozen=True, slots=True)
class PendingPrediction:
    request_id: int
    requested_at: float
    ready_at: float
    chunk: ActionChunk
    timings_s: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        self.validate_timestamps()

    def validate_timestamps(self) -> None:
        validate_timestamp(self.requested_at, "requested_at")
        validate_timestamp(self.ready_at, "ready_at")
        self.chunk.validate_timestamps()

    def is_ready(self, now: float) -> bool:
        return now + 1e-12 >= self.ready_at


class PolicyBackend(Protocol):
    action_spec: ActionSpec

    def reset(self, episode_id: str) -> None:
        """Clear episode history; stateless policies implement a no-op."""
        ...

    def request(
        self, observation: Observation, requested_at: float
    ) -> PendingPrediction: ...
