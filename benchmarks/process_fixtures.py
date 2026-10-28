"""Spawn-safe synthetic stateful models shared by recovery checks and tests."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

from robovla_runtime.backends.mock import MockBackend
from robovla_runtime.core.types import ActionSpec


def hang() -> None:
    while True:
        time.sleep(60)


@dataclass(frozen=True)
class AttemptCounter:
    """Persistent injection state; only one child may increment it at a time."""
    path: Path

    @property
    def value(self) -> int:
        return self.path.read_text(encoding="utf-8").count("\n") if self.path.exists() else 0

    def next(self) -> int:
        index = self.value
        # Count complete appended lines so the parent can observe initialization
        # without a multiprocessing handle in the construction recipe.
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write("attempt\n")
        return index


@dataclass
class RecoveryMockFactory:
    action_spec: ActionSpec
    attempts: AttemptCounter
    faults: tuple[str, ...] = ()
    repeat_fault: bool = False
    chunk_size: int = 100

    def __call__(self):
        index = self.attempts.next()
        fault = self.faults[index] if index < len(self.faults) else (
            self.faults[-1] if self.repeat_fault and self.faults else ""
        )
        if fault == "startup_hang":
            hang()
        if fault == "startup_error":
            raise RuntimeError("injected constructor failure")
        return StatefulRecoveryMock(self.action_spec, fault, self.chunk_size)


class StatefulRecoveryMock:
    def __init__(self, action_spec, fault, chunk_size):
        self.action_spec = action_spec
        self.fault = fault
        self.calls = 1000  # A missing reset is observable in the output actions.
        self.mock = MockBackend(action_spec, chunk_size, 0.0, (1.0,) * action_spec.dimension)

    def reset(self, episode_id):
        if self.fault == "reset_hang":
            hang()
        self.calls = 0
        self.mock.reset(episode_id)

    def request(self, observation, requested_at):
        self.calls += 1
        if self.fault == "request_hang":
            hang()
        if self.fault == "crash":
            os._exit(17)
        if self.fault == "request_error":
            raise RuntimeError("injected inference failure after state mutation")
        prediction = self.mock.request(observation, requested_at)
        # Expose the episode-local history counter to verify reset after recovery.
        return replace(prediction, chunk=replace(
            prediction.chunk,
            actions=((float(self.calls),) * self.action_spec.dimension,) * self.mock.chunk_size,
        ))
