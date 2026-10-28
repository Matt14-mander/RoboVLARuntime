"""Reproducible, time-indexed synthetic latency faults (virtual time only)."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LatencyTrace:
    """A piecewise-constant latency field sampled at request submission time.

    Values are total synthetic request latency, not additive sleeps. The last
    slot is held past the trace end. Sampling does not consume RNG state, so
    policies sharing a trace see the same external conditions at the same time.
    """

    label: str
    interval_s: float
    latencies_s: tuple[float, ...]
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("latency trace label must not be empty")
        if not math.isfinite(self.interval_s) or self.interval_s <= 0:
            raise ValueError("latency trace interval must be finite and positive")
        if not self.latencies_s or any(
            not math.isfinite(value) or value < 0 for value in self.latencies_s
        ):
            raise ValueError("latency trace values must be finite and non-negative")
        object.__setattr__(self, "latencies_s", tuple(self.latencies_s))

    def at(self, requested_at: float) -> float:
        if not math.isfinite(requested_at) or requested_at < 0:
            raise ValueError("request time must be finite and non-negative")
        index = math.floor(requested_at / self.interval_s + 1e-9)
        return self.latencies_s[min(index, len(self.latencies_s) - 1)]

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": 1, **asdict(self)}

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: str | Path) -> "LatencyTrace":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.pop("schema_version", None) != 1:
            raise ValueError("unsupported latency trace schema version")
        return cls(**data)


def make_latency_trace(
    profile: str,
    *,
    duration_s: float = 20.0,
    interval_s: float = 0.5,
    base_latency_s: float = 0.04,
    spike_probability: float = 0.15,
    spike_min_s: float = 0.12,
    spike_max_s: float = 0.30,
    overload_latency_s: float = 0.26,
    overload_start_s: float = 3.0,
    seed: int = 7,
) -> LatencyTrace:
    """Create fixed, bounded Pareto spike, or sustained overload conditions."""
    if profile not in {"fixed", "long_tail", "overload"}:
        raise ValueError("profile must be fixed, long_tail, or overload")
    if any(not math.isfinite(value) or value <= 0 for value in (duration_s, interval_s)):
        raise ValueError("duration and interval must be finite and positive")
    if not math.isfinite(spike_probability) or not 0 <= spike_probability <= 1:
        raise ValueError("spike_probability must be between zero and one")
    if any(not math.isfinite(value) or value < 0 for value in (
        base_latency_s, spike_min_s, spike_max_s,
        overload_latency_s, overload_start_s,
    )):
        raise ValueError("fault timing values must be finite and non-negative")
    if profile == "long_tail" and (
        spike_max_s < spike_min_s or spike_min_s < base_latency_s
    ):
        raise ValueError("spike range must be ordered and no smaller than base latency")
    rng = random.Random(seed)
    latencies = []
    for index in range(math.ceil(duration_s / interval_s)):
        value = base_latency_s
        if profile == "long_tail" and rng.random() < spike_probability:
            value = min(spike_max_s, spike_min_s * rng.paretovariate(2.0))
        elif profile == "overload" and index * interval_s + 1e-12 >= overload_start_s:
            value = overload_latency_s
        latencies.append(value)
    return LatencyTrace(profile, interval_s, tuple(latencies), seed)
