"""Small deterministic plant for runtime timing experiments."""

from __future__ import annotations

from robovla_runtime.core.types import ActionSpec, Observation


class ToyJoint:
    def __init__(
        self,
        position: tuple[float, ...],
        target: tuple[float, ...],
        max_velocity: float,
        *,
        action_spec: ActionSpec | None = None,
    ) -> None:
        if not position or len(position) != len(target):
            raise ValueError("position and target need the same positive dimension")
        if max_velocity <= 0:
            raise ValueError("max_velocity must be positive")
        self.position = tuple(float(value) for value in position)
        self.target = tuple(float(value) for value in target)
        self.max_velocity = float(max_velocity)
        self.action_spec = action_spec or ActionSpec(len(position), 0.02)
        self.action_spec.require_time_aligned_execution()
        if self.action_spec.dimension != len(position):
            raise ValueError("ActionSpec dimension does not match plant")
        self._initial_position = self.position
        self._command = self.position
        self._episode_id = "toy"

    def reset(self, episode_id: str) -> None:
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        self._episode_id = episode_id
        self.position = self._initial_position
        self._command = self.position

    def observe(self, observation_id: int, captured_at: float) -> Observation:
        return Observation(self._episode_id, observation_id, captured_at, self.state)

    def fallback_command(self, reason: str) -> tuple[float, ...]:
        return self.state

    def apply_action(self, command: tuple[float, ...]) -> None:
        self.action_spec.validate_command(command)
        self._command = tuple(command)

    def advance(self, period_s: float) -> tuple[float, ...]:
        return self.step(self._command, period_s)

    def metrics(self) -> dict[str, float]:
        return {"final_tracking_error": self.tracking_error()}

    @property
    def state(self) -> tuple[float, ...]:
        return self.position

    def step(self, command: tuple[float, ...], period_s: float) -> tuple[float, ...]:
        self.action_spec.validate_command(command)
        if period_s <= 0:
            raise ValueError("period_s must be positive")
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

