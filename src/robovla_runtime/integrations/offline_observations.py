"""Versioned offline observations: JSON metadata and optional numeric NPY/NPZ arrays."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from robovla_runtime.core import Observation


def fields(value, allowed, required, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object/table")
    unknown, missing = set(value) - set(allowed), set(required) - set(value)
    if unknown or missing:
        raise ValueError(f"{label}: unknown fields {sorted(unknown)}, missing fields {sorted(missing)}")


def number(value, label, *, minimum=0.0, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if value < minimum or (positive and value <= 0):
        raise ValueError(f"{label} is outside its allowed range")
    return value


def integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"non-finite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_payload(value, label):
    if isinstance(value, dict):
        for key, item in value.items():
            nonempty(key, label)
            _json_payload(item, f"{label}.{key}")
    elif isinstance(value, list):
        for item in value:
            _json_payload(item, label)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError(f"{label} is not JSON data")


@dataclass(frozen=True)
class OfflineSample:
    sample_id: str
    episode_id: str
    captured_at: float
    state: tuple[float, ...]
    payload: dict


@dataclass(frozen=True)
class OfflineObservations:
    samples: tuple[OfflineSample, ...]
    provenance: tuple[dict, ...]

    def observation(self, index: int, *, observation_id: int, captured_at: float, generation: int) -> Observation:
        sample = self.samples[index]
        return Observation(sample.episode_id, observation_id, captured_at, sample.state,
                           copy.deepcopy(sample.payload), generation)

    def __call__(self, index: int) -> Observation:
        """A picklable warmup source with no file reread after validation."""
        sample_index = index % len(self.samples)
        return self.observation(sample_index, observation_id=index, captured_at=0.0, generation=0)


def load_observations(path: str | Path, *, state_dimension: int, required_payload_keys=()) -> OfflineObservations:
    manifest = Path(path).resolve()
    raw = read_json(manifest)
    fields(raw, {"version", "samples"}, {"version", "samples"}, "observation manifest")
    if type(raw["version"]) is not int or raw["version"] != 1:
        raise ValueError("observation manifest version must be 1")
    if not isinstance(raw["samples"], list) or not raw["samples"]:
        raise ValueError("samples must be a non-empty array")
    provenance = {str(manifest): {"path": str(manifest), "sha256": sha256(manifest)}}
    samples, identifiers = [], set()
    for index, sample in enumerate(raw["samples"]):
        label = f"samples[{index}]"
        fields(sample, {"sample_id", "episode_id", "captured_at", "state", "payload", "arrays"},
               {"sample_id", "state", "payload"}, label)
        sample_id = nonempty(sample["sample_id"], f"{label}.sample_id")
        if sample_id in identifiers:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        identifiers.add(sample_id)
        state = sample["state"]
        if not isinstance(state, list) or len(state) != state_dimension:
            raise ValueError(f"{label}.state must have {state_dimension} components")
        state = tuple(float(number(x, f"{label}.state", minimum=-math.inf)) for x in state)
        payload = sample["payload"]
        if not isinstance(payload, dict):
            raise ValueError(f"{label}.payload must be an object")
        _json_payload(payload, f"{label}.payload")
        arrays = sample.get("arrays", {})
        if not isinstance(arrays, dict):
            raise ValueError(f"{label}.arrays must be an object")
        for key, descriptor in arrays.items():
            nonempty(key, "array payload key")
            if key in payload:
                raise ValueError(f"duplicate inline/array payload key: {key}")
            fields(descriptor, {"path", "key", "shape", "dtype"}, {"path", "shape", "dtype"}, f"array {key}")
            array_path = (manifest.parent / nonempty(descriptor["path"], "array path")).resolve()
            if array_path.suffix.lower() not in {".npy", ".npz"}:
                raise ValueError("array files must be .npy or .npz")
            if array_path.suffix.lower() == ".npz":
                nonempty(descriptor.get("key"), "NPZ key")
            elif "key" in descriptor:
                raise ValueError("NPY arrays do not take a key")
            if not array_path.is_file():
                raise ValueError(f"array file not found: {array_path}")
            shape = descriptor["shape"]
            if not isinstance(shape, list) or not shape:
                raise ValueError("array shape must be a non-empty list")
            for size in shape:
                integer(size, "array shape dimension", 1)
            nonempty(descriptor["dtype"], "array dtype")
            try:
                import numpy as np
            except ImportError as error:
                raise ValueError("NPY/NPZ observations require numpy in this Python environment") from error
            try:
                if array_path.suffix.lower() == ".npz":
                    with np.load(array_path, allow_pickle=False) as archive:
                        array = archive[descriptor["key"]].copy()
                else:
                    array = np.load(array_path, allow_pickle=False)
                    if not isinstance(array, np.ndarray):
                        array.close()
                        raise ValueError("NPY path contains an archive instead of an array")
                if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                    raise ValueError("arrays must contain finite real numeric or boolean values")
                if list(array.shape) != shape or array.dtype != np.dtype(descriptor["dtype"]):
                    raise ValueError(f"array {key} shape/dtype differs from its manifest")
            except (ValueError, KeyError, TypeError, OSError, EOFError) as error:
                raise ValueError(f"invalid array {key} in {array_path}: {error}") from error
            payload[key] = array
            provenance[str(array_path)] = {"path": str(array_path), "sha256": sha256(array_path)}
        missing = set(required_payload_keys) - set(payload)
        if missing:
            raise ValueError(f"{label}: missing required payload keys {sorted(missing)}")
        episode_id = nonempty(sample.get("episode_id", "offline"), f"{label}.episode_id")
        if episode_id.startswith("__fluxvla_"):
            raise ValueError("episode_id uses the reserved __fluxvla_ prefix")
        samples.append(OfflineSample(sample_id, episode_id,
            float(number(sample.get("captured_at", 0.0), f"{label}.captured_at")), state, payload))
    return OfflineObservations(tuple(samples), tuple(provenance.values()))
