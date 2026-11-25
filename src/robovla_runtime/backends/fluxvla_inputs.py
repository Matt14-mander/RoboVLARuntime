"""Explicit CPU input contracts, independent of FluxVLA/torch and action dimensions."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass

from robovla_runtime.core import Observation, ObservationSpec


def _key(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model input key must be a non-empty string")
    if value in {"task_description", "is_new_episode"}:
        raise ValueError(f"model input key {value} is reserved by the preprocessor")


@dataclass(frozen=True, slots=True)
class PayloadArraySpec:
    key: str
    kind: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self):
        _key(self.key)
        if not isinstance(self.shape, (list, tuple)) or not self.shape or any(
            type(n) is not int or n <= 0 for n in self.shape
        ):
            raise ValueError(f"{self.key}: shape must contain positive integers")
        object.__setattr__(self, "shape", tuple(self.shape))
        if self.kind == "camera":
            if len(self.shape) != 3 or self.shape[-1] != 3:
                raise ValueError(f"{self.key}: camera shape must be HWC RGB")
            allowed = ("uint8", "float32", "float64")
        elif self.kind == "state":
            if len(self.shape) != 1:
                raise ValueError(f"{self.key}: raw state shape must be one-dimensional")
            allowed = ("float32", "float64")
        else:
            raise ValueError(f"{self.key}: kind must be camera or state")
        if self.dtype not in allowed:
            raise ValueError(f"{self.key}: dtype must be one of {allowed}")


@dataclass(frozen=True, slots=True)
class StatePayloadMapping:
    key: str
    indices: tuple[int, ...]

    def __post_init__(self):
        _key(self.key)
        if not isinstance(self.indices, (list, tuple)) or not self.indices or any(
            type(n) is not int or n < 0 for n in self.indices
        ):
            raise ValueError(f"{self.key}: indices must contain non-negative integers")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError(f"{self.key}: duplicate state indices")
        object.__setattr__(self, "indices", tuple(self.indices))


@dataclass(frozen=True, slots=True)
class FluxVLAModelInputSpec:
    arrays: tuple[PayloadArraySpec, ...]
    state_mapping: tuple[StatePayloadMapping, ...] = ()

    def __post_init__(self):
        for name, cls in (("arrays", PayloadArraySpec), ("state_mapping", StatePayloadMapping)):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or any(not isinstance(v, cls) for v in values):
                raise ValueError(f"model_input.{name} has invalid entries")
            if len({v.key for v in values}) != len(values):
                raise ValueError(f"model_input.{name} has duplicate keys")
            object.__setattr__(self, name, tuple(values))
        if not self.arrays:
            raise ValueError("model_input.arrays must not be empty")
        specs = {a.key: a for a in self.arrays}
        for mapping in self.state_mapping:
            spec = specs.get(mapping.key)
            if spec is None or spec.kind != "state" or spec.shape != (len(mapping.indices),):
                raise ValueError(f"{mapping.key}: mapping requires a matching raw state array specification")

    def validate_state_dimension(self, dimension: int):
        ObservationSpec(dimension)
        for mapping in self.state_mapping:
            if max(mapping.indices) >= dimension:
                raise ValueError(f"{mapping.key}: state index exceeds state_dimension {dimension}")

    def prepare(self, observation: Observation) -> dict:
        """Build owned raw inputs before dataset transforms; never coerce supplied arrays.

        Mapped state is explicitly converted to the declared floating dtype. Shape,
        dtype and finiteness are checked again after conversion (including overflow).
        """
        ObservationSpec(len(observation.state)).validate(observation)
        self.validate_state_dimension(len(observation.state))
        if not isinstance(observation.payload, Mapping):
            raise ValueError("model input requires a mapping payload")
        try:
            import numpy as np
        except ImportError as error:
            raise ValueError("model_input requires numpy in this Python environment") from error
        raw = dict(observation.payload)
        specs = {a.key: a for a in self.arrays}
        for mapping in self.state_mapping:
            if mapping.key in raw:
                raise ValueError(f"{mapping.key}: mapped state would overwrite an existing payload key")
            with np.errstate(over="ignore", invalid="ignore"):
                raw[mapping.key] = np.array([observation.state[i] for i in mapping.indices],
                                            dtype=specs[mapping.key].dtype)
        for spec in self.arrays:
            value = raw.get(spec.key)
            if not isinstance(value, np.ndarray):
                raise ValueError(f"{spec.key}: model input must be a numpy array (use NPY/NPZ offline)")
            if value.shape != spec.shape or value.dtype != np.dtype(spec.dtype):
                raise ValueError(f"{spec.key}: expected shape={spec.shape}, dtype={spec.dtype}; "
                                 f"got shape={value.shape}, dtype={value.dtype}")
            if not np.isfinite(value).all():
                raise ValueError(f"{spec.key}: model input must contain finite values")
        return copy.deepcopy(raw)
