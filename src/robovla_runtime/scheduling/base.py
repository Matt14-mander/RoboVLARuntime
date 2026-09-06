"""Shared mechanics for deterministic schedulers."""

from __future__ import annotations

from collections.abc import Callable

from robovla_runtime.backends.base import PendingPrediction, PolicyBackend
from robovla_runtime.core.events import EventLog
from robovla_runtime.core.types import Observation
from robovla_runtime.execution.action_buffer import ActionBuffer


class BaseScheduler:
    def __init__(self, backend: PolicyBackend, event_log: EventLog) -> None:
        self.backend = backend
        self.event_log = event_log
        self.pending: PendingPrediction | None = None

    def _poll(self, now: float, action_buffer: ActionBuffer) -> None:
        if self.pending is None or not self.pending.is_ready(now):
            return
        pending = self.pending
        self.pending = None
        self.event_log.emit(
            now,
            "prediction_ready",
            request_id=pending.request_id,
            requested_at=pending.requested_at,
            ready_at=pending.ready_at,
            latency_s=pending.ready_at - pending.requested_at,
        )
        result = action_buffer.accept(pending.chunk, now)
        self.event_log.emit(
            now,
            "chunk_accepted" if result.accepted else "chunk_rejected",
            request_id=pending.request_id,
            reason=result.reason,
            dropped_prefix_actions=result.dropped_prefix_actions,
            coverage_s=action_buffer.coverage_s(now),
        )

    def _request(
        self,
        now: float,
        capture_observation: Callable[[float], Observation],
    ) -> None:
        observation = capture_observation(now)
        self.event_log.emit(
            now,
            "observation_captured",
            episode_id=observation.episode_id,
            observation_id=observation.observation_id,
            state=list(observation.state),
        )
        self.pending = self.backend.request(observation, now)
        self.event_log.emit(
            now,
            "prediction_requested",
            request_id=self.pending.request_id,
            observation_id=observation.observation_id,
            ready_at=self.pending.ready_at,
        )

    def tick(
        self,
        now: float,
        action_buffer: ActionBuffer,
        capture_observation: Callable[[float], Observation],
    ) -> None:
        raise NotImplementedError

