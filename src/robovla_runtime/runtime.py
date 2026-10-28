"""Deterministic and wall-clock loops composed with environment adapters."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

from robovla_runtime.backends.base import PolicyBackend
from robovla_runtime.backends.mock import MockBackend
from robovla_runtime.backends.latency import LatencyTrace
from robovla_runtime.backends.sleeping_mock import SleepingMockBackend
from robovla_runtime.backends.threaded import AsyncPolicyBackend, ThreadedPolicyBackend
from robovla_runtime.core.clock import VirtualClock
from robovla_runtime.core.events import Event, EventLog
from robovla_runtime.core.types import ActionSpec, ExecutionRecord, Observation
from robovla_runtime.environments import EnvironmentAdapter, make_toy_environment
from robovla_runtime.execution.action_buffer import ActionBuffer
from robovla_runtime.execution.executor import Executor
from robovla_runtime.metrics.summary import summarize
from robovla_runtime.scheduling.fixed_async import FixedAsyncScheduler
from robovla_runtime.scheduling.adaptive import (
    AdaptiveAsyncScheduler,
    AdaptivePrefetchConfig,
)
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
    adaptive_window_size: int = 20
    adaptive_quantile: float = 0.95
    adaptive_margin_s: float = 0.02
    adaptive_hysteresis_s: float = 0.01
    adaptive_min_request_interval_s: float = 0.02
    worker_prefetch: str = "fixed"
    request_timeout_s: float = 0.5
    worker_close_timeout_s: float = 0.1

    def adaptive_config(self) -> AdaptivePrefetchConfig:
        return AdaptivePrefetchConfig(
            self.adaptive_window_size, self.adaptive_quantile,
            self.adaptive_margin_s, self.adaptive_hysteresis_s,
            self.adaptive_min_request_interval_s,
        )

    def __post_init__(self) -> None:
        if self.scheduler not in {"sync", "fixed_async", "adaptive_async", "worker_async"}:
            raise ValueError(
                "scheduler must be sync, fixed_async, adaptive_async, or worker_async"
            )
        self.adaptive_config()
        if self.worker_prefetch not in {"fixed", "adaptive"}:
            raise ValueError("worker_prefetch must be fixed or adaptive")
        if not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be finite and positive")
        if not math.isfinite(self.worker_close_timeout_s) or self.worker_close_timeout_s < 0:
            raise ValueError("worker_close_timeout_s must be finite and non-negative")
        if any(not math.isfinite(value) or value < 0 for value in (
            self.period_s, self.duration_s, self.prefetch_threshold_s,
        )):
            raise ValueError("runtime timing must be finite and non-negative")
        if self.period_s <= 0 or self.duration_s <= 0:
            raise ValueError("period and duration must be positive")
        if self.chunk_size <= 0 or self.execute_horizon <= 0:
            raise ValueError("chunk sizes must be positive")
        if self.execute_horizon > self.chunk_size:
            raise ValueError("execute_horizon cannot exceed chunk_size")
        if not self.latency_s or any(
            not math.isfinite(value) or value < 0 for value in self.latency_s
        ):
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


def _prepare_environment(
    config: RuntimeConfig,
    backend: PolicyBackend | AsyncPolicyBackend,
    environment: EnvironmentAdapter | None,
    episode_id: str,
) -> EnvironmentAdapter:
    if environment is None:
        spec = ActionSpec(
            len(config.initial_position), config.period_s, unit=config.action_unit
        )
        environment = make_toy_environment(
            config.initial_position, config.target_position, config.max_velocity, spec
        )
    spec = environment.action_spec
    spec.require_time_aligned_execution()
    if (
        backend.action_spec != spec
        or spec.period_s != config.period_s
        or spec.unit != config.action_unit
    ):
        raise ValueError(
            "backend/environment ActionSpec does not match runtime configuration"
        )
    if not callable(getattr(backend, "reset", None)):
        raise TypeError("backend must implement reset(episode_id), even if stateless")
    environment.reset(episode_id)
    backend.reset(episode_id)
    return environment


def _observation_source(
    environment: EnvironmentAdapter,
    action_buffer: ActionBuffer,
    payload_factory: Callable[[EnvironmentAdapter], object] | None,
) -> Callable[[float], Observation]:
    observation_id = 0

    def capture(timestamp: float) -> Observation:
        nonlocal observation_id
        observation = environment.observe(observation_id, timestamp)
        if (
            observation.episode_id != action_buffer.episode_id
            or observation.observation_id != observation_id
        ):
            raise ValueError("environment returned an observation with incorrect identity")
        if payload_factory is not None:
            observation = replace(observation, payload=payload_factory(environment))
        observation = replace(
            observation, episode_generation=action_buffer.episode_generation
        )
        observation_id += 1
        return observation

    return capture


def run_runtime(
    config: RuntimeConfig,
    backend: PolicyBackend,
    *,
    environment: EnvironmentAdapter | None = None,
    observation_payload_factory: Callable[[EnvironmentAdapter], object] | None = None,
) -> SimulationResult:
    if config.scheduler == "worker_async":
        raise ValueError("worker_async requires run_realtime_runtime")
    clock = VirtualClock()
    log = EventLog()
    episode_id = f"episode-seed-{config.seed}"
    environment = _prepare_environment(
        config, backend, environment, episode_id
    )
    action_spec = environment.action_spec
    action_buffer = ActionBuffer(
        episode_id, config.execute_horizon, action_spec=action_spec
    )
    action_buffer.reset(episode_id)
    executor = Executor(
        action_spec, environment, log, config.deadline_tolerance_s
    )
    if config.scheduler == "sync":
        scheduler = SyncScheduler(backend, log)
    elif config.scheduler == "adaptive_async":
        scheduler = AdaptiveAsyncScheduler(
            backend, log, config.prefetch_threshold_s,
            config.execute_horizon * action_spec.period_s, config.adaptive_config(),
        )
    else:
        scheduler = FixedAsyncScheduler(
            backend, log, config.prefetch_threshold_s
        )

    capture = _observation_source(
        environment, action_buffer, observation_payload_factory
    )

    log.emit(clock.now(), "episode_started", episode_id=episode_id)
    tick_count = math_ceil_ratio(config.duration_s, config.period_s)
    for tick_id in range(tick_count):
        now = clock.now()
        scheduler.tick(now, action_buffer, capture)
        executor.tick(tick_id, now, now, action_buffer)
        clock.advance_by(config.period_s)
    environment_metrics = environment.metrics()
    log.emit(
        clock.now(),
        "episode_finished",
        episode_id=episode_id,
        final_state=list(environment.state),
        final_tracking_error=environment_metrics.get("final_tracking_error"),
    )
    summary = summarize(
        executor.records,
        log.events,
        config.period_s,
        environment_metrics.get("final_tracking_error"),
    )
    summary["environment_metrics"] = environment_metrics
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
    environment: EnvironmentAdapter | None = None,
    observation_payload_factory: Callable[[EnvironmentAdapter], object] | None = None,
    timer: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
    close_backend: bool = True,
) -> SimulationResult:
    """Run a wall-clock control loop while policy inference runs on a worker."""
    if config.scheduler != "worker_async":
        raise ValueError("run_realtime_runtime requires scheduler='worker_async'")
    log = EventLog()
    episode_id = f"episode-seed-{config.seed}"
    environment = _prepare_environment(
        config, backend, environment, episode_id
    )
    action_spec = environment.action_spec
    action_buffer = ActionBuffer(
        episode_id, config.execute_horizon, action_spec=action_spec
    )
    action_buffer.reset(episode_id)
    executor = Executor(
        action_spec, environment, log, config.deadline_tolerance_s
    )
    scheduler = WorkerAsyncScheduler(
        backend, log, config.prefetch_threshold_s,
        request_timeout_s=config.request_timeout_s,
        adaptive_config=config.adaptive_config() if config.worker_prefetch == "adaptive" else None,
        coverage_budget_s=config.execute_horizon * action_spec.period_s,
        clock=lambda: max(0.0, timer() - started_at),
    )
    capture = _observation_source(
        environment, action_buffer, observation_payload_factory
    )

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
        try:
            scheduler.poll(finished_at, action_buffer)
        finally:
            try:
                scheduler.cancel_episode(max(0.0, timer() - started_at), episode_id)
            finally:
                if close_backend:
                    closed = backend.close(timeout_s=config.worker_close_timeout_s)
                    if not closed:
                        log.emit(
                            max(0.0, timer() - started_at), "worker_shutdown_timeout",
                            timeout_s=config.worker_close_timeout_s,
                        )
        finished_at = max(0.0, timer() - started_at)

    environment_metrics = environment.metrics()
    log.emit(
        finished_at,
        "episode_finished",
        episode_id=episode_id,
        final_state=list(environment.state),
        final_tracking_error=environment_metrics.get("final_tracking_error"),
    )
    summary = summarize(
        executor.records,
        log.events,
        config.period_s,
        environment_metrics.get("final_tracking_error"),
    )
    summary["environment_metrics"] = environment_metrics
    return SimulationResult(
        config,
        tuple(executor.records),
        log.events,
        summary,
    )


def run_simulation(
    config: RuntimeConfig, *, latency_trace: LatencyTrace | None = None
) -> SimulationResult:
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
        latency_trace=latency_trace,
    )
    return run_runtime(config, backend)


def run_realtime_mock(
    config: RuntimeConfig, *, latency_trace: LatencyTrace | None = None
) -> SimulationResult:
    """Exercise the wall-clock worker path with sleeping synthetic inference."""
    if config.scheduler != "worker_async":
        raise ValueError("run_realtime_mock requires scheduler='worker_async'")
    spec = ActionSpec(len(config.initial_position), config.period_s, unit=config.action_unit)
    blocking = SleepingMockBackend(MockBackend(
        spec, config.chunk_size, config.latency_s, config.target_position,
        latency_trace=latency_trace,
    ))
    backend = ThreadedPolicyBackend(blocking)
    try:
        return run_realtime_runtime(config, backend)
    finally:
        # Also release an idle worker if setup/validation failed before the loop.
        # The normal runtime already performed its configured bounded join.
        backend.close(timeout_s=0.0)


def math_ceil_ratio(numerator: float, denominator: float) -> int:
    """Ceiling division with tolerance for decimal period values."""
    quotient = numerator / denominator
    rounded = round(quotient)
    if abs(quotient - rounded) < 1e-9:
        return int(rounded)
    return int(quotient) + 1
