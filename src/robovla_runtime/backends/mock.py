"""Deterministic policy backend with configurable inference latency."""

from __future__ import annotations

from collections.abc import Sequence

from robovla_runtime.core.types import ActionChunk, ActionSpec, Observation

from .base import PendingPrediction


class MockBackend:
    def __init__(
        self,
        action_spec: ActionSpec,
        chunk_size: int,
        latency_s: float | Sequence[float],
        target: tuple[float, ...],
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if len(target) != action_spec.dimension:
            raise ValueError("target must match action dimension")
        latencies = (
            (float(latency_s),)
            if isinstance(latency_s, (int, float))
            else tuple(float(value) for value in latency_s)
        )
        if not latencies or any(value < 0 for value in latencies):
            raise ValueError("latency trace must be non-empty and non-negative")
        self.action_spec = action_spec
        self.chunk_size = chunk_size
        self.latencies = latencies
        self.target = target
        self._next_request_id = 0

    def request(
        self, observation: Observation, requested_at: float
    ) -> PendingPrediction:
        request_id = self._next_request_id
        self._next_request_id += 1
        latency = self.latencies[request_id % len(self.latencies)]
        ready_at = requested_at + latency
        actions = tuple(
            tuple(
                start + (goal - start) * ((index + 1) / self.chunk_size)
                for start, goal in zip(observation.state, self.target)
            )
            for index in range(self.chunk_size)
        )
        chunk = ActionChunk(
            episode_id=observation.episode_id,
            request_id=request_id,
            source_observation_id=observation.observation_id,
            source_observation_time=observation.captured_at,
            ready_at=ready_at,
            start_at=observation.captured_at,
            spec=self.action_spec,
            actions=actions,
        )
        return PendingPrediction(request_id, requested_at, ready_at, chunk)

