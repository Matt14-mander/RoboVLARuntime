"""Non-blocking scheduler for a worker-backed policy."""

from __future__ import annotations

from collections.abc import Callable
from collections import OrderedDict
from dataclasses import dataclass
import math

from robovla_runtime.backends.base import PendingPrediction
from robovla_runtime.backends.threaded import AsyncPolicyBackend, PredictionFailure
from robovla_runtime.core.events import EventLog
from robovla_runtime.core.types import Observation
from robovla_runtime.execution.action_buffer import ActionBuffer
from .adaptive import AdaptivePrefetchConfig, AdaptivePrefetchController


@dataclass(frozen=True, slots=True)
class _Request:
    requested_at: float
    deadline_at: float
    episode_id: str
    generation: int
    observation_id: int
    observation_time: float


class WorkerAsyncScheduler:
    def __init__(
        self,
        backend: AsyncPolicyBackend,
        event_log: EventLog,
        prefetch_threshold_s: float,
        *,
        request_timeout_s: float = 0.5,
        adaptive_config: AdaptivePrefetchConfig | None = None,
        coverage_budget_s: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not math.isfinite(prefetch_threshold_s) or prefetch_threshold_s < 0:
            raise ValueError("prefetch threshold must be non-negative")
        if not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be finite and positive")
        self.backend = backend
        self.event_log = event_log
        self.prefetch_threshold_s = prefetch_threshold_s
        self.request_timeout_s = request_timeout_s
        self.clock = clock
        self.prefetch: AdaptivePrefetchController | None = None
        if adaptive_config is not None:
            if coverage_budget_s is None:
                raise ValueError("adaptive worker requires a coverage budget")
            self.prefetch = AdaptivePrefetchController(
                event_log, prefetch_threshold_s, coverage_budget_s, adaptive_config
            )
        self._requests: dict[int, _Request] = {}
        # Bounded diagnostic history. Unknown/evicted IDs are always rejected.
        self._retired: OrderedDict[int, tuple[_Request, str]] = OrderedDict()
        self._last_requested_at: float | None = None
        self._latest_accepted_observation: tuple[float, int] | None = None

    def _retire(self, request_id: int, request: _Request, reason: str) -> None:
        self._requests.pop(request_id, None)
        self._retired[request_id] = (request, reason)
        self._retired.move_to_end(request_id)
        while len(self._retired) > 256:
            self._retired.popitem(last=False)

    def _timeout(self, now: float, request_id: int, request: _Request) -> None:
        self._retire(request_id, request, "request_timeout")
        expire = getattr(self.backend, "expire_request", None)
        recovery = None
        if callable(expire):
            recovery = expire(request_id)
            cancelled = recovery == "queued"
        else:
            cancel_queued = getattr(self.backend, "cancel_queued", None)
            cancelled = bool(cancel_queued(request_id)) if callable(cancel_queued) else False
        self.event_log.emit(
            now, "prediction_timed_out", request_id=request_id,
            requested_at=request.requested_at, deadline_at=request.deadline_at,
            elapsed_s=now - request.requested_at, queued_cancelled=cancelled,
            recovery=recovery,
        )

    def _discard(self, now: float, request_id: int, reason: str) -> None:
        self.event_log.emit(now, "prediction_discarded", request_id=request_id, reason=reason)

    def tick(
        self,
        now: float,
        action_buffer: ActionBuffer,
        capture_observation: Callable[[float], Observation],
    ) -> None:
        if self.prefetch is not None:
            self.prefetch.start(now)
        self.poll(now, action_buffer)
        if self.clock is not None:
            now = max(now, self.clock())
        threshold = self.prefetch_threshold_s
        if self.prefetch is not None:
            threshold = self.prefetch.threshold_s
            # One live logical request, like the deterministic adaptive baseline.
            # A timed-out physical call can still occupy the worker underneath.
            if self._requests or (
                self._last_requested_at is not None
                and now - self._last_requested_at + 1e-12
                < self.prefetch.config.min_request_interval_s
            ):
                return
        if action_buffer.coverage_s(now) <= threshold + 1e-12:
            if self.prefetch is not None:
                self.event_log.emit(
                    now, "adaptive_prefetch_triggered", threshold_s=threshold,
                    coverage_s=action_buffer.coverage_s(now),
                )
            self._submit(now, capture_observation)

    def poll(self, now: float, action_buffer: ActionBuffer) -> None:
        results = self.backend.poll()
        if self.clock is not None:
            now = max(now, self.clock())
        self.poll_diagnostics(now)
        for result in results:
            request_id = result.request_id
            request = self._requests.get(request_id)
            retired_reason = None
            if request is None:
                retired = self._retired.get(request_id)
                if retired is None or retired[1] != "request_timeout":
                    reason = retired[1] if retired else "unknown_request"
                    if reason in {"completed", "late_completion", "failed"}:
                        reason = "duplicate_result"
                    self._discard(now, request_id, reason)
                    continue
                request, retired_reason = retired

            result_episode = (
                result.episode_id if isinstance(result, PredictionFailure)
                else result.chunk.episode_id
            )
            identity_matches = (
                request.episode_id == action_buffer.episode_id == result_episode
                and request.generation == action_buffer.episode_generation
                and math.isclose(result.requested_at, request.requested_at, abs_tol=1e-9, rel_tol=0)
            )
            if isinstance(result, PendingPrediction):
                identity_matches = identity_matches and (
                    result.chunk.request_id == request_id
                    and result.chunk.episode_generation == request.generation
                    and result.chunk.source_observation_id == request.observation_id
                    and result.chunk.source_observation_time == request.observation_time
                )
            if not identity_matches:
                self._retire(request_id, request, "identity_mismatch")
                self._discard(now, request_id, "identity_mismatch")
                continue

            late = retired_reason == "request_timeout" or now > request.deadline_at + 1e-12
            if late and retired_reason is None:
                self._timeout(now, request_id, request)
            if isinstance(result, PredictionFailure):
                self.event_log.emit(
                    now,
                    "prediction_failed",
                    request_id=result.request_id,
                    requested_at=result.requested_at,
                    error_type=result.error_type,
                    message=result.message,
                    late=late,
                )
                self._retire(request_id, request, "failed")
                if late:
                    self._discard(now, request_id, "request_timeout")
                continue
            assert isinstance(result, PendingPrediction)
            if late:
                self._retire(request_id, request, "late_completion")
                self._discard(now, request_id, "request_timeout")
                if self.prefetch is not None:
                    self.prefetch.observe_completion(
                        now, request_id, request.requested_at, late=True
                    )
                continue
            self._retire(request_id, request, "completed")
            observation_key = (request.observation_time, request.observation_id)
            if (
                self._latest_accepted_observation is not None
                and observation_key <= self._latest_accepted_observation
            ):
                self._discard(now, request_id, "superseded_observation")
                continue
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
            if acceptance.accepted:
                self._latest_accepted_observation = observation_key
            self.event_log.emit(
                now,
                "chunk_accepted" if acceptance.accepted else "chunk_rejected",
                request_id=result.request_id,
                reason=acceptance.reason,
                dropped_prefix_actions=acceptance.dropped_prefix_actions,
                coverage_s=action_buffer.coverage_s(now),
            )
            if self.prefetch is not None:
                self.prefetch.observe_completion(now, request_id, request.requested_at)
        # Results collected exactly at the deadline are accepted above; missing
        # results at that tick expire here. Expiry cannot block the executor.
        for request_id, request in tuple(self._requests.items()):
            if now + 1e-12 >= request.deadline_at:
                self._timeout(now, request_id, request)

    def poll_diagnostics(self, now: float) -> None:
        poll = getattr(self.backend, "poll_diagnostics", None)
        if callable(poll):
            for kind, data in poll():
                self.event_log.emit(now, kind, **data)

    def cancel_episode(self, now: float, episode_id: str) -> None:
        dropped = set(self.backend.cancel_episode(episode_id))
        for request_id, request in tuple(self._requests.items()):
            if request.episode_id == episode_id:
                dropped.add(request_id)
                self._retire(request_id, request, "episode_cancelled")
        for request_id in sorted(dropped):
            self.event_log.emit(
                now, "prediction_cancelled", request_id=request_id, episode_id=episode_id
            )
        # Suppress even timed-out outcomes if somebody polls after episode end.
        for request_id, (request, reason) in tuple(self._retired.items()):
            if request.episode_id == episode_id and reason == "request_timeout":
                self._retired[request_id] = (request, "episode_cancelled")

    def _submit(
        self,
        now: float,
        capture_observation: Callable[[float], Observation],
    ) -> None:
        observation = capture_observation(now)
        if self.clock is not None:
            now = max(now, self.clock())
        self.event_log.emit(
            now,
            "observation_captured",
            episode_id=observation.episode_id,
            observation_id=observation.observation_id,
            state=list(observation.state),
        )
        submit_with_deadline = getattr(self.backend, "submit_with_deadline", None)
        receipt = (
            submit_with_deadline(observation, now, now + self.request_timeout_s)
            if callable(submit_with_deadline)
            else self.backend.submit(observation, now)
        )
        self._last_requested_at = now
        self._requests[receipt.request_id] = _Request(
            now, now + self.request_timeout_s, observation.episode_id,
            observation.episode_generation, observation.observation_id,
            observation.captured_at,
        )
        self.event_log.emit(
            now,
            "prediction_submitted",
            request_id=receipt.request_id,
            observation_id=observation.observation_id,
            deadline_at=now + self.request_timeout_s,
        )
        if receipt.replaced_request_id is not None:
            replaced = self._requests.get(receipt.replaced_request_id)
            if replaced is not None:
                self._retire(receipt.replaced_request_id, replaced, "mailbox_replaced")
            self.event_log.emit(
                now,
                "mailbox_request_replaced",
                replaced_request_id=receipt.replaced_request_id,
                replacement_request_id=receipt.request_id,
            )
