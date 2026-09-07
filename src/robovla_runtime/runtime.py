"""Composition root for phase-one deterministic simulations."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from robovla_runtime.backends.base import PolicyBackend
from robovla_runtime.backends.mock import MockBackend
from robovla_runtime.backends.threaded import AsyncPolicyBackend
from robovla_runtime.core.clock import VirtualClock
from robovla_runtime.core.events import Event, EventLog
from robovla_runtime.core.types import ActionSpec, ExecutionRecord, Observation
from robovla_runtime.environments.toy_joint import ToyJoint
from robovla_runtime.execution.action_buffer import ActionBuffer
from robovla_runtime.execution.executor import Executor
from robovla_runtime.metrics.summary import summarize
from robovla_runtime.scheduling.fixed_async import FixedAsyncScheduler
from robovla_runtime.scheduling.sync import SyncScheduler
from robovla_runtime.scheduling.worker_async import WorkerAsyncScheduler


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
    deadline_tolerance_s: float = 0.001
    action_unit: str = "normalized"
    seed: int = 7

    def __post_init__(self) -> None:
        if self.scheduler not in {"sync", "fixed_async", "worker_async"}:
            raise ValueError("scheduler must be sync, fixed_async, or worker_async")
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
        if not self.action_unit:
            raise ValueError("action_unit must not be empty")
        if self.deadline_tolerance_s < 0:
            raise ValueError("deadline_tolerance_s must be non-negative")


@dataclass(frozen=True, slots=True)
class SimulationResult:
    config: RuntimeConfig
    records: tuple[ExecutionRecord, ...]
    events: tuple[Event, ...]
    summary: dict[str, object]

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)


def run_runtime(
    config: RuntimeConfig,
    backend: PolicyBackend,
    *,
    observation_payload_factory=None,
) -> SimulationResult:
    if config.scheduler == "worker_async":
        raise ValueError("worker_async requires run_realtime_runtime")
    clock = VirtualClock()
    log = EventLog()
    episode_id = f"episode-seed-{config.seed}"
    action_spec = ActionSpec(
        dimension=len(config.initial_position),
        period_s=config.period_s,
        representation="absolute_position",
        unit=config.action_unit,
    )
    environment = ToyJoint(
        config.initial_position, config.target_position, config.max_velocity
    )
    if backend.action_spec != action_spec:
        raise ValueError("backend ActionSpec does not match runtime configuration")
    action_buffer = ActionBuffer(episode_id, config.execute_horizon)
    executor = Executor(
        action_spec, environment, log, config.deadline_tolerance_s
    )
    if config.scheduler == "sync":
        scheduler = SyncScheduler(backend, log)
    else:
        scheduler = FixedAsyncScheduler(
            backend, log, config.prefetch_threshold_s
        )

    observation_id = 0

    def capture(timestamp: float) -> Observation:
        nonlocal observation_id
        payload = (
            observation_payload_factory(environment)
            if observation_payload_factory is not None
            else None
        )
        observation = Observation(
            episode_id,
            observation_id,
            timestamp,
            environment.state,
            payload,
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


def run_realtime_runtime(
    config: RuntimeConfig,
    backend: AsyncPolicyBackend,
    *,
    observation_payload_factory: Callable[[ToyJoint], object] | None = None,
    timer: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
    close_backend: bool = True,
) -> SimulationResult:
    """Run a wall-clock control loop while policy inference runs on a worker."""
    if config.scheduler != "worker_async":
        raise ValueError("run_realtime_runtime requires scheduler='worker_async'")
    action_spec = ActionSpec(
        dimension=len(config.initial_position),
        period_s=config.period_s,
        representation="absolute_position",
        unit=config.action_unit,
    )
    if backend.action_spec != action_spec:
        raise ValueError("backend ActionSpec does not match runtime configuration")

    log = EventLog()
    episode_id = f"episode-seed-{config.seed}"
    environment = ToyJoint(
        config.initial_position, config.target_position, config.max_velocity
    )
    action_buffer = ActionBuffer(episode_id, config.execute_horizon)
    executor = Executor(
        action_spec, environment, log, config.deadline_tolerance_s
    )
    scheduler = WorkerAsyncScheduler(
        backend, log, config.prefetch_threshold_s
    )
    observation_id = 0

    def capture(timestamp: float) -> Observation:
        nonlocal observation_id
        payload = (
            observation_payload_factory(environment)
            if observation_payload_factory is not None
            else None
        )
        observation = Observation(
            episode_id,
            observation_id,
            timestamp,
            environment.state,
            payload,
        )
        observation_id += 1
        return observation

    started_at = timer()
    log.emit(0.0, "episode_started", episode_id=episode_id)
    tick_count = math_ceil_ratio(config.duration_s, config.period_s)
    try:
        for tick_id in range(tick_count):
            scheduled_at = tick_id * config.period_s
            delay_s = started_at + scheduled_at - timer()
            if delay_s > 0:
                sleeper(delay_s)
            executed_at = max(0.0, timer() - started_at)
            scheduler.tick(executed_at, action_buffer, capture)
            executor.tick(
                tick_id,
                scheduled_at,
                max(0.0, timer() - started_at),
                action_buffer,
            )
    finally:
        finished_at = max(0.0, timer() - started_at)
        scheduler.poll(finished_at, action_buffer)
        dropped = backend.cancel_episode(episode_id)
        for request_id in dropped:
            log.emit(
                finished_at,
                "prediction_cancelled",
                request_id=request_id,
                episode_id=episode_id,
            )
        if close_backend:
            backend.close()

    log.emit(
        finished_at,
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


def run_simulation(config: RuntimeConfig) -> SimulationResult:
    """Run the built-in deterministic mock experiment."""
    action_spec = ActionSpec(
        dimension=len(config.initial_position),
        period_s=config.period_s,
        representation="absolute_position",
        unit=config.action_unit,
    )
    backend = MockBackend(
        action_spec,
        config.chunk_size,
        config.latency_s,
        config.target_position,
    )
    return run_runtime(config, backend)


def math_ceil_ratio(numerator: float, denominator: float) -> int:
    """Ceiling division with tolerance for decimal period values."""
    quotient = numerator / denominator
    rounded = round(quotient)
    if abs(quotient - rounded) < 1e-9:
        return int(rounded)
    return int(quotient) + 1
