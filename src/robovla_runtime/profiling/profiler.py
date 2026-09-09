"""Repeatable per-request profiling for chunked policy backends."""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from robovla_runtime.backends.base import PolicyBackend
from robovla_runtime.core.types import Observation

from .resources import MemorySampler, NullMemorySampler


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    warmup: int = 10
    samples: int = 100
    mode: Literal["diagnostic", "throughput"] = "diagnostic"

    def __post_init__(self) -> None:
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.samples <= 0:
            raise ValueError("samples must be positive")
        if self.mode not in {"diagnostic", "throughput"}:
            raise ValueError("mode must be diagnostic or throughput")


@dataclass(frozen=True, slots=True)
class ProfileRecord:
    sequence: int
    measured_index: int | None
    warmup: bool
    observation_id: int
    request_id: int
    wall_time_s: float
    backend_timings_s: dict[str, float]
    chunk_horizon: int
    action_dimension: int
    resources: dict[str, int | float | str | bool | None]


@dataclass(frozen=True, slots=True)
class ProfileResult:
    config: ProfileConfig
    records: tuple[ProfileRecord, ...]
    summary: dict[str, Any]
    environment: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProfileSession:
    backend: PolicyBackend
    observation_factory: Callable[[int], Observation]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    cleanup: Callable[[], None] = lambda: None

    def close(self) -> None:
        self.cleanup()

    def __enter__(self) -> "ProfileSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def collect_environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import torch
    except ImportError:
        environment["torch_available"] = False
        return environment
    environment["torch_available"] = True
    environment["torch_version"] = torch.__version__
    environment["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        environment.update(
            {
                "cuda_device_index": int(device),
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_device_capability": list(
                    torch.cuda.get_device_capability(device)
                ),
                "cuda_version": torch.version.cuda,
            }
        )
    return environment


def profile_backend(
    backend: PolicyBackend,
    observation_factory: Callable[[int], Observation],
    config: ProfileConfig,
    *,
    memory_sampler: MemorySampler | None = None,
    timer: Callable[[], float] = time.perf_counter,
    environment: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ProfileResult:
    sampler = memory_sampler or NullMemorySampler()
    records: list[ProfileRecord] = []
    total_requests = config.warmup + config.samples
    for sequence in range(total_requests):
        warmup = sequence < config.warmup
        observation = observation_factory(sequence)
        sampler.begin_sample()
        started_at = timer()
        prediction = backend.request(observation, requested_at=started_at)
        wall_time_s = timer() - started_at
        if wall_time_s < 0:
            raise ValueError("timer must be monotonic")
        records.append(
            ProfileRecord(
                sequence=sequence,
                measured_index=None if warmup else sequence - config.warmup,
                warmup=warmup,
                observation_id=observation.observation_id,
                request_id=prediction.request_id,
                wall_time_s=wall_time_s,
                backend_timings_s=dict(prediction.timings_s),
                chunk_horizon=len(prediction.chunk.actions),
                action_dimension=prediction.chunk.spec.dimension,
                resources=dict(sampler.snapshot()),
            )
        )

    measured = [record for record in records if not record.warmup]
    wall_times = [record.wall_time_s for record in measured]
    measured_wall_time_s = sum(wall_times)
    stage_names = sorted(
        {name for record in measured for name in record.backend_timings_s}
    )
    numeric_resource_names = sorted(
        {
            name
            for record in measured
            for name, value in record.resources.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    summary = {
        "mode": config.mode,
        "warmup_requests": config.warmup,
        "measured_requests": len(measured),
        "wall_time_s": _distribution(wall_times),
        "sequential_request_throughput_hz": (
            len(measured) / measured_wall_time_s
            if measured_wall_time_s > 0
            else None
        ),
        "generated_action_steps_per_s": (
            sum(record.chunk_horizon for record in measured) / measured_wall_time_s
            if measured_wall_time_s > 0
            else None
        ),
        "backend_timings_s": {
            name: _distribution(
                [
                    record.backend_timings_s[name]
                    for record in measured
                    if name in record.backend_timings_s
                ]
            )
            for name in stage_names
        },
        "resources": {
            name: _distribution(
                [
                    float(record.resources[name])
                    for record in measured
                    if isinstance(record.resources.get(name), (int, float))
                    and not isinstance(record.resources.get(name), bool)
                ]
            )
            for name in numeric_resource_names
        },
        "chunk_horizon": sorted({record.chunk_horizon for record in measured}),
        "action_dimension": sorted(
            {record.action_dimension for record in measured}
        ),
    }
    return ProfileResult(
        config=config,
        records=tuple(records),
        summary=summary,
        environment=dict(environment or collect_environment()),
        metadata=dict(metadata or {}),
    )


def write_profile(result: ProfileResult, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {"profile": asdict(result.config), "metadata": result.metadata},
            stream,
            indent=2,
            sort_keys=True,
        )
    with (output / "requests.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for record in result.records:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(result.summary, stream, indent=2, sort_keys=True)
    with (output / "environment.json").open("w", encoding="utf-8") as stream:
        json.dump(result.environment, stream, indent=2, sort_keys=True)
    with (output / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(result.metadata, stream, indent=2, sort_keys=True)
    return output
