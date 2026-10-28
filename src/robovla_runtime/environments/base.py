"""Environment boundary with no simulator or model dependencies."""

from typing import Protocol

from robovla_runtime.core.types import ActionSpec, Observation


class EnvironmentAdapter(Protocol):
    action_spec: ActionSpec

    @property
    def state(self) -> tuple[float, ...]: ...

    def reset(self, episode_id: str) -> None:
        """Restore initial state and clear commands/history, even for the same ID."""
        ...

    def observe(self, observation_id: int, captured_at: float) -> Observation:
        """Return a snapshot for the current episode, including optional payload."""
        ...

    def fallback_command(self, reason: str) -> tuple[float, ...]:
        """Return an embodiment-specific fallback in action_spec's units."""
        ...

    def apply_action(self, command: tuple[float, ...]) -> None: ...

    def advance(self, period_s: float) -> tuple[float, ...]:
        """Advance one logical interval, including fallback ticks.

        Wall-clock mode currently uses the same fixed logical step. Real robot
        adapters must define feedback behavior; this is not a hard-real-time or
        physical-execution guarantee.
        """
        ...

    def metrics(self) -> dict[str, float]:
        """Environment-specific metrics; no toy tracking objective is required."""
        ...
