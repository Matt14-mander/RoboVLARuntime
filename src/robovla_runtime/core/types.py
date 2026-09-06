"""Data contracts shared by runtime components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Observation:
    episode_id: str
    observation_id: int
    captured_at: float
    state: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if self.observation_id < 0:
            raise ValueError("observation_id must be non-negative")
        if self.captured_at < 0:
            raise ValueError("captured_at must be non-negative")
        if not self.state:
            raise ValueError("state must not be empty")


@dataclass(frozen=True, slots=True)
class ActionSpec:
    dimension: int
    period_s: float
    representation: Literal["absolute_position"] = "absolute_position"
    unit: str = "normalized"

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.period_s <= 0:
            raise ValueError("period_s must be positive")
        if self.representation != "absolute_position":
            raise ValueError("phase 1 only supports absolute_position actions")
        if not self.unit:
            raise ValueError("unit must not be empty")


@dataclass(frozen=True, slots=True)
class ActionChunk:
    episode_id: str
    request_id: int
    source_observation_id: int
    source_observation_time: float
    ready_at: float
    start_at: float
    spec: ActionSpec
    actions: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if self.request_id < 0 or self.source_observation_id < 0:
            raise ValueError("request and observation ids must be non-negative")
        if self.source_observation_time < 0 or self.ready_at < 0:
            raise ValueError("timestamps must be non-negative")
        if self.start_at < self.source_observation_time:
            raise ValueError("start_at cannot precede the source observation")
        if not self.actions:
            raise ValueError("an action chunk must contain at least one action")
        if any(len(action) != self.spec.dimension for action in self.actions):
            raise ValueError("every action must match ActionSpec.dimension")

    @property
    def end_at(self) -> float:
        """Exclusive end of the time interval covered by this chunk."""
        return self.start_at + len(self.actions) * self.spec.period_s

    def action_time(self, index: int) -> float:
        return self.start_at + index * self.spec.period_s


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    tick_id: int
    scheduled_at: float
    executed_at: float
    request_id: int | None
    action_index: int | None
    source_observation_id: int | None
    source_observation_time: float | None
    command: tuple[float, ...]
    fallback_reason: str | None

    @property
    def observation_age_s(self) -> float | None:
        if self.source_observation_time is None:
            return None
        return self.executed_at - self.source_observation_time

