"""Clock abstractions used by deterministic simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...

    def advance_to(self, timestamp: float) -> None: ...


@dataclass(slots=True)
class VirtualClock:
    _time: float = 0.0

    def __post_init__(self) -> None:
        if self._time < 0:
            raise ValueError("initial time must be non-negative")

    def now(self) -> float:
        return self._time

    def advance_to(self, timestamp: float) -> None:
        if timestamp + 1e-12 < self._time:
            raise ValueError("virtual clock cannot move backwards")
        self._time = timestamp

    def advance_by(self, duration_s: float) -> None:
        if duration_s < 0:
            raise ValueError("duration must be non-negative")
        self._time += duration_s

