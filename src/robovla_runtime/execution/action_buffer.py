"""A bounded, time-aligned action chunk buffer."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from robovla_runtime.core.types import ActionChunk


@dataclass(frozen=True, slots=True)
class ChunkAcceptance:
    accepted: bool
    reason: str
    dropped_prefix_actions: int = 0


@dataclass(frozen=True, slots=True)
class BufferedAction:
    command: tuple[float, ...]
    request_id: int
    action_index: int
    source_observation_id: int
    source_observation_time: float


class ActionBuffer:
    """Store one current chunk and align it to execution time.

    Phase 1 permits a single active chunk. Accepting a newer valid chunk
    replaces the remaining tail of the previous one.
    """

    def __init__(
        self, episode_id: str, max_actions_per_chunk: int | None = None
    ) -> None:
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        self.episode_id = episode_id
        if max_actions_per_chunk is not None and max_actions_per_chunk <= 0:
            raise ValueError("max_actions_per_chunk must be positive")
        self.max_actions_per_chunk = max_actions_per_chunk
        self._chunk: ActionChunk | None = None
        self._seen_requests: set[int] = set()
        self._consumed: set[tuple[int, int]] = set()

    def reset(self, episode_id: str) -> None:
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        self.episode_id = episode_id
        self._chunk = None
        self._seen_requests.clear()
        self._consumed.clear()

    def accept(self, chunk: ActionChunk, now: float) -> ChunkAcceptance:
        if chunk.episode_id != self.episode_id:
            return ChunkAcceptance(False, "wrong_episode")
        if chunk.request_id in self._seen_requests:
            return ChunkAcceptance(False, "duplicate")
        self._seen_requests.add(chunk.request_id)
        if self.max_actions_per_chunk is not None:
            chunk = replace(
                chunk, actions=chunk.actions[: self.max_actions_per_chunk]
            )
        first_valid = self._index_at(chunk, now)
        if first_valid >= len(chunk.actions):
            return ChunkAcceptance(False, "expired", len(chunk.actions))
        self._chunk = chunk
        return ChunkAcceptance(True, "accepted", max(first_valid, 0))

    def pop(self, now: float) -> BufferedAction | None:
        chunk = self._chunk
        if chunk is None:
            return None
        index = self._index_at(chunk, now)
        if index < 0 or index >= len(chunk.actions):
            if index >= len(chunk.actions):
                self._chunk = None
            return None
        key = (chunk.request_id, index)
        if key in self._consumed:
            return None
        self._consumed.add(key)
        return BufferedAction(
            command=chunk.actions[index],
            request_id=chunk.request_id,
            action_index=index,
            source_observation_id=chunk.source_observation_id,
            source_observation_time=chunk.source_observation_time,
        )

    def coverage_s(self, now: float) -> float:
        chunk = self._chunk
        if chunk is None or now + 1e-12 >= chunk.end_at:
            return 0.0
        return max(0.0, chunk.end_at - max(now, chunk.start_at))

    def is_empty(self, now: float) -> bool:
        return self.coverage_s(now) <= 1e-12

    @staticmethod
    def _index_at(chunk: ActionChunk, now: float) -> int:
        relative = (now - chunk.start_at) / chunk.spec.period_s
        if relative < -1e-9:
            return -1
        return math.floor(relative + 1e-9)
