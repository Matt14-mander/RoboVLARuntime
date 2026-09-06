"""Fixed-period command executor."""

from __future__ import annotations

from robovla_runtime.core.events import EventLog
from robovla_runtime.core.types import ActionSpec, ExecutionRecord
from robovla_runtime.environments.toy_joint import ToyJoint

from .action_buffer import ActionBuffer


class Executor:
    def __init__(
        self,
        action_spec: ActionSpec,
        environment: ToyJoint,
        event_log: EventLog,
        deadline_tolerance_s: float = 1e-9,
    ) -> None:
        if action_spec.dimension != len(environment.state):
            raise ValueError("action and environment dimensions differ")
        if deadline_tolerance_s < 0:
            raise ValueError("deadline tolerance must be non-negative")
        self.action_spec = action_spec
        self.environment = environment
        self.event_log = event_log
        self.deadline_tolerance_s = deadline_tolerance_s
        self.records: list[ExecutionRecord] = []

    def tick(
        self,
        tick_id: int,
        scheduled_at: float,
        executed_at: float,
        action_buffer: ActionBuffer,
    ) -> ExecutionRecord:
        buffered = action_buffer.pop(executed_at)
        if buffered is None:
            command = self.environment.state
            record = ExecutionRecord(
                tick_id,
                scheduled_at,
                executed_at,
                None,
                None,
                None,
                None,
                command,
                "buffer_empty",
            )
            event_type = "fallback_executed"
        else:
            command = buffered.command
            record = ExecutionRecord(
                tick_id,
                scheduled_at,
                executed_at,
                buffered.request_id,
                buffered.action_index,
                buffered.source_observation_id,
                buffered.source_observation_time,
                command,
                None,
            )
            event_type = "action_executed"
        state = self.environment.step(command, self.action_spec.period_s)
        self.records.append(record)
        self.event_log.emit(
            executed_at,
            event_type,
            tick_id=tick_id,
            scheduled_at=scheduled_at,
            request_id=record.request_id,
            action_index=record.action_index,
            source_observation_id=record.source_observation_id,
            source_observation_time=record.source_observation_time,
            observation_age_s=record.observation_age_s,
            command=list(command),
            state=list(state),
            fallback_reason=record.fallback_reason,
            deadline_missed=(
                executed_at > scheduled_at + self.deadline_tolerance_s
            ),
        )
        return record

