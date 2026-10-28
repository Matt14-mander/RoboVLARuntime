"""Single-in-flight adaptive prefetch for virtual-time experiments."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from robovla_runtime.backends.base import PolicyBackend
from robovla_runtime.core.events import EventLog
from robovla_runtime.core.types import Observation
from robovla_runtime.execution.action_buffer import ActionBuffer

from .base import BaseScheduler


@dataclass(frozen=True, slots=True)
class AdaptivePrefetchConfig:
    window_size: int = 20
    quantile: float = 0.95
    margin_s: float = 0.02
    hysteresis_s: float = 0.01
    min_request_interval_s: float = 0.02

    def __post_init__(self) -> None:
        if (
            not isinstance(self.window_size, int)
            or isinstance(self.window_size, bool)
            or self.window_size < 1
        ):
            raise ValueError("adaptive window_size must be a positive integer")
        if not math.isfinite(self.quantile) or not 0 < self.quantile <= 1:
            raise ValueError("adaptive quantile must be in (0, 1]")
        for value in (self.margin_s, self.hysteresis_s, self.min_request_interval_s):
            if not math.isfinite(value) or value < 0:
                raise ValueError("adaptive timing settings must be finite and non-negative")
        if self.min_request_interval_s == 0:
            raise ValueError("min_request_interval_s must be positive")


class AdaptiveAsyncScheduler(BaseScheduler):
    def __init__(
        self,
        backend: PolicyBackend,
        event_log: EventLog,
        initial_threshold_s: float,
        coverage_budget_s: float,
        config: AdaptivePrefetchConfig,
    ) -> None:
        super().__init__(backend, event_log)
        if not math.isfinite(initial_threshold_s) or initial_threshold_s < 0:
            raise ValueError("initial threshold must be finite and non-negative")
        if not math.isfinite(coverage_budget_s) or coverage_budget_s <= 0:
            raise ValueError("coverage budget must be finite and positive")
        self.config: AdaptivePrefetchConfig = config
        self.coverage_budget_s = coverage_budget_s
        self.threshold_s = min(initial_threshold_s, coverage_budget_s)
        self.samples: deque[float] = deque(maxlen=config.window_size)
        self.completed_requests = 0
        self._last_requested_at: float | None = None
        self._supply_limited = False
        self._started = False

    def _collect(self, now: float, action_buffer: ActionBuffer) -> None:
        pending = self._poll(now, action_buffer)
        if pending is None:
            return
        # Include polling/collection delay. Never read the pending request's
        # advertised future ready_at before it has completed.
        observed_s = max(0.0, now - pending.requested_at)
        self.samples.append(observed_s)
        self.completed_requests += 1
        values = sorted(self.samples)
        position = (len(values) - 1) * self.config.quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        estimate_s = values[lower] + (values[upper] - values[lower]) * (position - lower)
        requested_threshold_s = estimate_s + self.config.margin_s
        candidate_s = min(requested_threshold_s, self.coverage_budget_s)
        previous_threshold_s = self.threshold_s
        # Always adopt the first measured estimate; later changes use a deadband.
        if (
            self.completed_requests == 1
            or abs(candidate_s - self.threshold_s) > self.config.hysteresis_s + 1e-12
        ):
            self.threshold_s = candidate_s

        # For this observation-anchored, single-chunk replacement strategy,
        # arriving chunks retain E*dt-L seconds. Requests are spaced by at
        # least max(L, min_interval), so even immediate prefetch can fall short.
        shortfall_s = max(
            0.0, estimate_s + max(estimate_s, self.config.min_request_interval_s)
            - self.coverage_budget_s,
        )
        supply_limited = shortfall_s > 1e-12
        self.event_log.emit(
            now, "prefetch_estimate_updated", request_id=pending.request_id,
            observed_latency_s=observed_s, estimated_latency_s=estimate_s,
            sample_count=len(self.samples), requested_threshold_s=requested_threshold_s,
            threshold_s=self.threshold_s, previous_threshold_s=previous_threshold_s,
            threshold_capped=requested_threshold_s > self.coverage_budget_s + 1e-12,
            supply_shortfall_s=shortfall_s,
        )
        if supply_limited != self._supply_limited:
            self._supply_limited = supply_limited
            self.event_log.emit(
                now, "supply_state_changed", limited=supply_limited,
                estimated_latency_s=estimate_s, coverage_budget_s=self.coverage_budget_s,
                supply_shortfall_s=shortfall_s,
            )

    def tick(
        self,
        now: float,
        action_buffer: ActionBuffer,
        capture_observation: Callable[[float], Observation],
    ) -> None:
        if not self._started:
            self._started = True
            self.event_log.emit(
                now, "adaptive_prefetch_started", threshold_s=self.threshold_s,
                coverage_budget_s=self.coverage_budget_s,
            )
        self._collect(now, action_buffer)
        interval_elapsed = (
            self._last_requested_at is None
            or now - self._last_requested_at + 1e-12 >= self.config.min_request_interval_s
        )
        if (
            self.pending is None and interval_elapsed
            and action_buffer.coverage_s(now) <= self.threshold_s + 1e-12
        ):
            self.event_log.emit(
                now, "adaptive_prefetch_triggered",
                threshold_s=self.threshold_s, coverage_s=action_buffer.coverage_s(now),
            )
            self._request(now, capture_observation)
            self._last_requested_at = now
            self._collect(now, action_buffer)
