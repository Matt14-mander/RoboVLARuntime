"""Small deterministic plant for runtime timing experiments."""

from __future__ import annotations


class ToyJoint:
    def __init__(
        self,
        position: tuple[float, ...],
        target: tuple[float, ...],
        max_velocity: float,
    ) -> None:
        if not position or len(position) != len(target):
            raise ValueError("position and target need the same positive dimension")
        if max_velocity <= 0:
            raise ValueError("max_velocity must be positive")
        self.position = tuple(float(value) for value in position)
        self.target = tuple(float(value) for value in target)
        self.max_velocity = float(max_velocity)

    @property
    def state(self) -> tuple[float, ...]:
        return self.position

    def step(self, command: tuple[float, ...], period_s: float) -> tuple[float, ...]:
        if len(command) != len(self.position):
            raise ValueError("command dimension does not match plant")
        max_delta = self.max_velocity * period_s
        self.position = tuple(
            current + max(-max_delta, min(max_delta, desired - current))
            for current, desired in zip(self.position, command)
        )
        return self.position

    def tracking_error(self) -> float:
        return sum(
            (goal - current) ** 2
            for current, goal in zip(self.position, self.target)
        ) ** 0.5

