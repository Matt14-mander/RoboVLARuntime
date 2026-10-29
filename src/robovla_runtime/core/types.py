"""Data contracts shared by runtime components."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Literal


def validate_timestamp(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class Observation:
    episode_id: str
    observation_id: int
    captured_at: float
    state: tuple[float, ...]
    payload: object | None = None
    episode_generation: int = 0

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if self.observation_id < 0:
            raise ValueError("observation_id must be non-negative")
        validate_timestamp(self.captured_at, "captured_at")
        if not self.state:
            raise ValueError("state must not be empty")
        if self.episode_generation < 0:
            raise ValueError("episode_generation must be non-negative")


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """Input contract independent of the policy's output ActionSpec."""
    state_dimension: int
    required_payload_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.state_dimension) is not int or self.state_dimension <= 0:
            raise ValueError("state_dimension must be a positive integer")
        keys = self.required_payload_keys
        if not isinstance(keys, (list, tuple)) or any(
            not isinstance(key, str) or not key.strip() for key in keys
        ):
            raise ValueError("required_payload_keys must contain non-empty strings")
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate required payload key")
        object.__setattr__(self, "required_payload_keys", tuple(keys))

    def validate_state(self, state) -> None:
        if not isinstance(state, (list, tuple)) or len(state) != self.state_dimension:
            raise ValueError(f"state must have {self.state_dimension} components")
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(value) for value in state):
            raise ValueError("state must contain finite real numbers, not booleans")

    def validate(self, observation: Observation) -> None:
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        validate_timestamp(observation.captured_at, "captured_at")
        self.validate_state(observation.state)
        if self.required_payload_keys:
            if not isinstance(observation.payload, Mapping):
                raise ValueError("required payload keys need a mapping payload")
            missing = set(self.required_payload_keys) - set(observation.payload)
            if missing:
                raise ValueError(f"missing required payload keys {sorted(missing)}")


@dataclass(frozen=True, slots=True)
class ActionSpec:
    dimension: int
    period_s: float
    representation: Literal[
        "absolute_position", "relative_position", "delta_position", "velocity"
    ] = "absolute_position"
    unit: str = "normalized"
    action_space: Literal["joint", "cartesian"] = "joint"
    coordinate_frame: str = "joint"
    rotation_representation: Literal[
        "none", "axis_angle", "quaternion_xyzw", "quaternion_wxyz", "euler_xyz"
    ] = "none"
    normalization: Literal["normalized", "unnormalized", "unspecified"] = "unspecified"
    component_units: tuple[str, ...] = ()
    gripper_semantics: str = "none"

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if not math.isfinite(self.period_s) or self.period_s <= 0:
            raise ValueError("period_s must be positive")
        if self.representation not in {
            "absolute_position", "relative_position", "delta_position", "velocity"
        }:
            raise ValueError("unknown action representation")
        if not self.unit:
            raise ValueError("unit must not be empty")
        if self.action_space not in {"joint", "cartesian"}:
            raise ValueError("unknown action space")
        if not self.coordinate_frame or not self.gripper_semantics:
            raise ValueError("coordinate_frame and gripper_semantics are required")
        if self.rotation_representation not in {
            "none", "axis_angle", "quaternion_xyzw", "quaternion_wxyz", "euler_xyz"
        }:
            raise ValueError("unknown rotation representation")
        if self.normalization not in {"normalized", "unnormalized", "unspecified"}:
            raise ValueError("unknown normalization state")
        if self.component_units and (
            len(self.component_units) != self.dimension
            or any(not unit for unit in self.component_units)
        ):
            raise ValueError("component_units must describe every action component")

    def require_time_aligned_execution(self) -> None:
        """Describing actions does not imply support for skipping their prefix."""
        if (
            self.action_space != "joint"
            or self.representation != "absolute_position"
            or self.coordinate_frame != "joint"
            or self.rotation_representation != "none"
            or self.gripper_semantics != "none"
        ):
            raise ValueError(
                "ActionSpec is unsupported by time-aligned execution; "
                "only absolute joint positions without gripper channels are supported"
            )

    def validate_command(self, command: tuple[float, ...]) -> None:
        if len(command) != self.dimension:
            raise ValueError("command must match ActionSpec.dimension")
        if any(not math.isfinite(value) for value in command):
            raise ValueError("action values must be finite")


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
    episode_generation: int = 0

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if self.request_id < 0 or self.source_observation_id < 0:
            raise ValueError("request and observation ids must be non-negative")
        self.validate_timestamps()
        if not self.actions:
            raise ValueError("an action chunk must contain at least one action")
        if any(len(action) != self.spec.dimension for action in self.actions):
            raise ValueError("every action must match ActionSpec.dimension")
        for action in self.actions:
            self.spec.validate_command(action)
        if self.episode_generation < 0:
            raise ValueError("episode_generation must be non-negative")

    def validate_timestamps(self) -> None:
        """Also callable at receive boundaries; unpickling skips __post_init__."""
        for name in ("source_observation_time", "ready_at", "start_at"):
            validate_timestamp(getattr(self, name), name)
        if self.start_at < self.source_observation_time:
            raise ValueError("start_at cannot precede the source observation")
        validate_timestamp(self.end_at, "end_at")

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
