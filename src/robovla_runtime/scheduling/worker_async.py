"""Non-blocking scheduler for a worker-backed policy."""

from __future__ import annotations

from collections.abc import Callable

from robovla_runtime.backends.base import PendingPrediction
from robovla_runtime.backends.threaded import AsyncPolicyBackend, PredictionFailure
from robovla_runtime.core.events import EventLog
from robovla_runtime.core.types import Observation
from robovla_runtime.execution.action_buffer import ActionBuffer


class WorkerAsyncScheduler:
    def __init__(
        self,
        backend: AsyncPolicyBackend,
        event_log: EventLog,
        prefetch_threshold_s: float,
    ) -> None:
        if prefetch_threshold_s < 0:
            raise ValueError("prefetch threshold must be non-negative")
        self.backend = backend
        self.event_log = event_log
        self.prefetch_threshold_s = prefetch_threshold_s

    def tick(
        self,
        now: float,
        action_buffer: ActionBuffer,
        capture_observation: Callable[[float], Observation],
    ) -> None:
        self.poll(now, action_buffer)
        if action_buffer.coverage_s(now) <= self.prefetch_threshold_s + 1e-12:
            self._submit(now, capture_observation)

    def poll(self, now: float, action_buffer: ActionBuffer) -> None:
        for result in self.backend.poll():
            if isinstance(result, PredictionFailure):
                self.event_log.emit(
                    now,
                    "prediction_failed",
                    request_id=result.request_id,
                    requested_at=result.requested_at,
                    error_type=result.error_type,
                    message=result.message,
                )
                continue
            assert isinstance(result, PendingPrediction)
            self.event_log.emit(
                now,
                "prediction_ready",
                request_id=result.request_id,
                requested_at=result.requested_at,
                ready_at=result.ready_at,
                latency_s=result.ready_at - result.requested_at,
                collection_delay_s=max(0.0, now - result.ready_at),
                timings_s=dict(result.timings_s),
            )
            acceptance = action_buffer.accept(result.chunk, now)
            self.event_log.emit(
                now,
                "chunk_accepted" if acceptance.accepted else "chunk_rejected",
                request_id=result.request_id,
                reason=acceptance.reason,
                dropped_prefix_actions=acceptance.dropped_prefix_actions,
                coverage_s=action_buffer.coverage_s(now),
            )

    def _submit(
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
        receipt = self.backend.submit(observation, now)
        self.event_log.emit(
            now,
            "prediction_submitted",
            request_id=receipt.request_id,
            observation_id=observation.observation_id,
        )
        if receipt.replaced_request_id is not None:
            self.event_log.emit(
                now,
                "mailbox_request_replaced",
                replaced_request_id=receipt.replaced_request_id,
                replacement_request_id=receipt.request_id,
            )
