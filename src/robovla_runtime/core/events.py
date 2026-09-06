"""Structured runtime event recording."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class Event:
    sequence: int
    timestamp: float
    type: str
    data: dict[str, Any]


class EventLog:
    def __init__(self) -> None:
        self._events: list[Event] = []

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def emit(self, timestamp: float, event_type: str, **data: Any) -> Event:
        event = Event(len(self._events), timestamp, event_type, data)
        self._events.append(event)
        return event

    def write_jsonl(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            for event in self._events:
                stream.write(json.dumps(asdict(event), sort_keys=True) + "\n")

    @classmethod
    def from_events(cls, events: Iterable[Event]) -> "EventLog":
        log = cls()
        log._events.extend(events)
        return log

    def __iter__(self) -> Iterable[Event]:
        return iter(self._events)
