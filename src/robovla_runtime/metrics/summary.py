"""Metrics computed from runtime records and events."""

from __future__ import annotations

import math

from robovla_runtime.core.events import Event
from robovla_runtime.core.types import ExecutionRecord


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(
    records: list[ExecutionRecord],
    events: tuple[Event, ...],
    period_s: float,
    final_tracking_error: float,
) -> dict[str, object]:
    latencies = [
        float(event.data["latency_s"])
        for event in events
        if event.type == "prediction_ready"
    ]
    stage_latencies: dict[str, list[float]] = {}
    for event in events:
        if event.type != "prediction_ready":
            continue
        for stage, value in event.data.get("timings_s", {}).items():
            stage_latencies.setdefault(stage, []).append(float(value))
    ages = [
        age
        for record in records
        if (age := record.observation_age_s) is not None
    ]
    tick_lateness = [
        max(0.0, record.executed_at - record.scheduled_at) for record in records
    ]
    fallback_count = sum(record.fallback_reason is not None for record in records)
    deadline_misses = sum(
        event.type in {"action_executed", "fallback_executed"}
        and bool(event.data["deadline_missed"])
        for event in events
    )
    mailbox_replacements = sum(
        event.type == "mailbox_request_replaced" for event in events
    )
    prediction_failures = sum(event.type == "prediction_failed" for event in events)
    jumps = [
        math.sqrt(
            sum((right - left) ** 2 for left, right in zip(a.command, b.command))
        )
        for a, b in zip(records, records[1:])
    ]
    return {
        "ticks": len(records),
        "prediction_count": len(latencies),
        "prediction_failure_count": prediction_failures,
        "mailbox_replacement_count": mailbox_replacements,
        "prediction_latency_s": {
            "samples": len(latencies),
            "p50": _quantile(latencies, 0.50),
            "p95": _quantile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "prediction_stage_latency_s": {
            stage: {
                "samples": len(values),
                "p50": _quantile(values, 0.50),
                "p95": _quantile(values, 0.95),
                "max": max(values),
            }
            for stage, values in sorted(stage_latencies.items())
        },
        "observation_age_s": {
            "samples": len(ages),
            "p50": _quantile(ages, 0.50),
            "p95": _quantile(ages, 0.95),
            "max": max(ages) if ages else None,
        },
        "control_tick_lateness_s": {
            "samples": len(tick_lateness),
            "p50": _quantile(tick_lateness, 0.50),
            "p95": _quantile(tick_lateness, 0.95),
            "max": max(tick_lateness) if tick_lateness else None,
        },
        "buffer_underrun_ticks": fallback_count,
        "buffer_underrun_ratio": fallback_count / len(records) if records else 0.0,
        "fallback_duration_s": fallback_count * period_s,
        "deadline_miss_count": deadline_misses,
        "deadline_miss_ratio": deadline_misses / len(records) if records else 0.0,
        "max_command_jump": max(jumps) if jumps else 0.0,
        "final_tracking_error": final_tracking_error,
    }
