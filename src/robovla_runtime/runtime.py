"""Composition root for phase-one deterministic simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from robovla_runtime.backends.mock import MockBackend
from robovla_runtime.core.clock import VirtualClock
from robovla_runtime.core.events import Event, EventLog
from robovla_runtime.core.types import ActionSpec, ExecutionRecord, Observation
from robovla_runtime.environments.toy_joint import ToyJoint
from robovla_runtime.execution.action_buffer import ActionBuffer
from robovla_runtime.execution.executor import Executor
from robovla_runtime.metrics.summary import summarize
from robovla_runtime.scheduling.fixed_async import FixedAsyncScheduler
from robovla_runtime.scheduling.sync import SyncScheduler


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    scheduler: str = "sync"
    period_s: float = 0.02
    duration_s: float = 2.0
    chunk_size: int = 10
    execute_horizon: int = 10
    latency_s: tuple[float, ...] = (0.05,)
    prefetch_threshold_s: float = 0.08
    initial_position: tuple[float, ...] = (0.0,)
    target_position: tuple[float, ...] = (1.0,)
    max_velocity: float = 2.0
    seed: int = 7

    def __post_init__(self) -> None:
        if self.scheduler not in {"sync", "fixed_async"}:
            raise ValueError("scheduler must be sync or fixed_async")
        if self.period_s <= 0 or self.duration_s <= 0:
            raise ValueError("period and duration must be positive")
        if self.chunk_size <= 0 or self.execute_horizon <= 0:
            raise ValueError("chunk sizes must be positive")
        if self.execute_horizon > self.chunk_size:
            raise ValueError("execute_horizon cannot exceed chunk_size")
        if not self.latency_s or any(value < 0 for value in self.latency_s):
            raise ValueError("latency trace must be non-empty and non-negative")
        if len(self.initial_position) != len(self.target_position):
            raise ValueError("initial and target positions must have equal dimensions")


@dataclass(frozen=True, slots=True)
class SimulationResult:
    config: RuntimeConfig
    records: tuple[ExecutionRecord, ...]
    events: tuple[Event, ...]
    summary: dict[str, object]

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


def run_simulation(config: RuntimeConfig) -> SimulationResult:
    clock = VirtualClock()
    log = EventLog()
    episode_id = f"episode-seed-{config.seed}"
    action_spec = ActionSpec(
        dimension=len(config.initial_position),
        period_s=config.period_s,
        representation="absolute_position",
        unit="normalized",
    )
    environment = ToyJoint(
        config.initial_position, config.target_position, config.max_velocity
    )
    backend = MockBackend(
        action_spec,
        config.chunk_size,
        config.latency_s,
        config.target_position,
    )
    action_buffer = ActionBuffer(episode_id, config.execute_horizon)
    executor = Executor(action_spec, environment, log)
    if config.scheduler == "sync":
        scheduler = SyncScheduler(backend, log)
    else:
        scheduler = FixedAsyncScheduler(
            backend, log, config.prefetch_threshold_s
        )

    observation_id = 0

    def capture(timestamp: float) -> Observation:
        nonlocal observation_id
        observation = Observation(
            episode_id, observation_id, timestamp, environment.state
        )
        observation_id += 1
        return observation

    log.emit(clock.now(), "episode_started", episode_id=episode_id)
    tick_count = math_ceil_ratio(config.duration_s, config.period_s)
    for tick_id in range(tick_count):
        now = clock.now()
        scheduler.tick(now, action_buffer, capture)
        executor.tick(tick_id, now, now, action_buffer)
        clock.advance_by(config.period_s)
    log.emit(
        clock.now(),
        "episode_finished",
        episode_id=episode_id,
        final_state=list(environment.state),
        final_tracking_error=environment.tracking_error(),
    )
    summary = summarize(
        executor.records,
        log.events,
        config.period_s,
        environment.tracking_error(),
    )
    return SimulationResult(
        config,
        tuple(executor.records),
        log.events,
        summary,
    )


def math_ceil_ratio(numerator: float, denominator: float) -> int:
    """Ceiling division with tolerance for decimal period values."""
    quotient = numerator / denominator
    rounded = round(quotient)
    if abs(quotient - rounded) < 1e-9:
        return int(rounded)
    return int(quotient) + 1
