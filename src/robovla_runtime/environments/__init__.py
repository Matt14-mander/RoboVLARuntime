from robovla_runtime.core.types import ActionSpec

from .base import EnvironmentAdapter
from .toy_joint import ToyJoint


def make_toy_environment(
    position: tuple[float, ...],
    target: tuple[float, ...],
    max_velocity: float,
    action_spec: ActionSpec,
) -> EnvironmentAdapter:
    """Compatibility factory for the default runtime experiment."""
    return ToyJoint(position, target, max_velocity, action_spec=action_spec)


__all__ = ["EnvironmentAdapter", "ToyJoint", "make_toy_environment"]

