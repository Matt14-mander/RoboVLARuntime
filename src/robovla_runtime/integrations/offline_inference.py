"""Sequential inference-only replay with durable process lifecycle diagnostics."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from robovla_runtime.backends import PendingPrediction, PredictionFailure, ProcessPolicyBackend
from robovla_runtime.core.events import EventLog
from .fluxvla_process import FluxVLALiberoFactory, FluxVLAProcessFactory


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class _LoggedFactory:
    factory: FluxVLAProcessFactory
    directory: str

    def __call__(self):
        # Python model/runner output stays separate from the CLI's JSON stream.
        # These streams live for the child lifetime, including warmup/inference.
        path = Path(self.directory) / f"worker-{os.getpid()}-{time.monotonic_ns()}.log"
        stream = path.open("w", encoding="utf-8", buffering=1)
        sys.stdout = stream
        sys.stderr = stream
        return self.factory()

    def create_backend(self, action_spec, *, config):
        return ProcessPolicyBackend(self, action_spec, config=config)


def _factory(config, observations, directory):
    model = config.model
    if model["kind"] == "mock":
        from .fluxvla_mock import make_simulated_factory

        mock = make_simulated_factory(directory / "mock", model["fault_plan"])
        overrides = {key: model[key] for key in (
            "task_description", "task_suite_name", "num_inference_steps", "inference_seed", "timing_mode",
        ) if key in model}
        recipe = replace(mock.recipe, action_spec=config.action_spec, input_spec=config.input_spec, **overrides)
        builder = replace(mock, recipe=recipe, offline_mode=True)
    else:
        builder = FluxVLALiberoFactory(action_spec=config.action_spec, input_spec=config.input_spec, **{
            k: v for k, v in model.items() if k not in {"kind", "warmup_steps", "model_is_stateless"}
        })
    return _LoggedFactory(
        FluxVLAProcessFactory(builder, observations, model["warmup_steps"], model["model_is_stateless"],
                              observation_spec=config.observation_spec),
        str(directory),
    )


def _validate_result(result, request_id, observation, requested_at, received_at, spec):
    if not isinstance(result, PendingPrediction):
        raise ValueError("unknown result type")
    result.validate_timestamps()
    chunk = result.chunk
    if not (
        result.request_id == chunk.request_id == request_id
        and chunk.episode_id == observation.episode_id
        and chunk.episode_generation == observation.episode_generation
        and chunk.source_observation_id == observation.observation_id
        and chunk.source_observation_time == observation.captured_at
        and chunk.spec == spec
        and math.isclose(result.requested_at, requested_at, rel_tol=0, abs_tol=1e-9)
        and math.isfinite(result.ready_at) and requested_at <= result.ready_at <= received_at + 1e-6
        and chunk.ready_at == result.ready_at
    ):
        raise ValueError("result identity, action contract or timing mismatch")
    if not chunk.actions:
        raise ValueError("empty action chunk")
    for command in chunk.actions:
        spec.validate_command(command)
    for name, value in result.timings_s:
        if not isinstance(name, str) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid stage timing")


def run_offline_inference(config, observations, provenance, output_root, *, emit=print):
    """Return (exit_code, artifact_directory, summary); never execute robot actions."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = Path(output_root).resolve() / f"{stamp}-fluxvla-process"
    directory.mkdir(parents=True, exist_ok=False)
    _write_json(directory / "config.json", config.resolved())
    _write_json(directory / "inputs.json", {"files": provenance, "samples": [
        {"sample_id": s.sample_id, "episode_id": s.episode_id, "source_captured_at": s.captured_at,
         "state_dimension": len(s.state), "payload_keys": sorted(s.payload)} for s in observations.samples
    ]})
    log = EventLog()
    backend = None
    started = time.perf_counter()
    attempts, outcomes = [], []
    exhausted = False
    request_generations = {}
    interrupted = False
    fatal = None
    closed = True
    total_samples = len(observations.samples) * config.repeats

    with (directory / "events.jsonl").open("w", encoding="utf-8") as event_file, \
         (directory / "predictions.jsonl").open("w", encoding="utf-8") as prediction_file, \
         (directory / "attempts.jsonl").open("w", encoding="utf-8") as attempt_file:
        def now():
            return time.perf_counter() - started

        def record(kind, **data):
            event = log.emit(now(), kind, **data)
            line = json.dumps(asdict(event), ensure_ascii=False, allow_nan=False)
            event_file.write(line + "\n")
            event_file.flush()
            if emit is not None:
                emit(line)

        def drain():
            nonlocal exhausted
            for kind, data in backend.poll_diagnostics():
                record(kind, **data)
                if kind == "process_request_started":
                    request_generations[data["request_id"]] = data["process_generation"]
                if kind in {"process_restart_exhausted", "process_supervisor_failed", "process_termination_failed", "process_transport_stuck"}:
                    exhausted = True

        record("offline_inference_started", model_kind=config.model["kind"], samples=total_samples)
        try:
            backend = _factory(config, observations, directory).create_backend(config.action_spec, config=config.process)
            observation_id, generation = 0, 0
            for repetition in range(config.repeats):
                previous_episode = None
                for index, sample in enumerate(observations.samples):
                    if exhausted:
                        break
                    if previous_episode != sample.episode_id:
                        backend.reset(sample.episode_id)
                        generation += 1
                        record("offline_episode_reset", episode_id=sample.episode_id, episode_generation=generation)
                        previous_episode = sample.episode_id
                    success = False
                    for attempt in range(config.max_attempts_per_sample):
                        requested_at = now()
                        observation = observations.observation(index, observation_id=observation_id,
                                                              captured_at=requested_at, generation=generation)
                        observation_id += 1
                        receipt = backend.submit(observation, requested_at)
                        record("prediction_submitted", request_id=receipt.request_id, sample_id=sample.sample_id,
                               repeat=repetition, attempt=attempt + 1, source_captured_at=sample.captured_at)
                        # Backend watchdogs own startup and execution deadlines. This
                        # parent bound also covers a failed/stalled supervisor.
                        deadline = time.perf_counter() + config.process.startup_timeout_s + config.process.execution_timeout_s + 2 * config.process.terminate_grace_s + 1.0
                        result = None
                        while time.perf_counter() < deadline:
                            drain()
                            results = backend.poll()
                            drain()
                            for candidate in results:
                                if candidate.request_id == receipt.request_id and result is None:
                                    result = candidate
                                else:
                                    record("prediction_discarded", request_id=candidate.request_id, reason="unexpected_or_late_result")
                            if result is not None or exhausted:
                                break
                            time.sleep(min(config.poll_interval_s, max(0.0, deadline - time.perf_counter())))
                        received_at = now()
                        entry = {"sample_id": sample.sample_id, "repeat": repetition, "attempt": attempt + 1,
                                 "request_id": receipt.request_id, "elapsed_s": received_at - requested_at}
                        if result is None:
                            recovery = backend.expire_request(receipt.request_id)
                            entry.update(status="failed", error_type="ProcessUnavailable" if exhausted else "CollectionTimeout",
                                         message="no result within the bounded wait", recovery=recovery)
                        elif isinstance(result, PredictionFailure):
                            entry.update(status="failed", error_type=result.error_type, message=result.message)
                            if result.error_type == "ProcessUnavailable":
                                exhausted = True
                        else:
                            try:
                                _validate_result(result, receipt.request_id, observation, requested_at, received_at, config.action_spec)
                            except (ValueError, TypeError, AttributeError) as error:
                                entry.update(status="failed", error_type="InvalidPrediction", message=str(error))
                                exhausted = True
                            else:
                                success = True
                                entry.update(status="succeeded", latency_s=result.ready_at - requested_at,
                                             collection_delay_s=max(0.0, received_at - result.ready_at),
                                             timings_s=dict(result.timings_s))
                                prediction = {"sample_id": sample.sample_id, "repeat": repetition,
                                    "process_generation": request_generations.get(result.request_id),
                                    "source_captured_at": sample.captured_at, **asdict(result)}
                                prediction_file.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                                prediction_file.flush()
                        attempts.append(entry)
                        attempt_file.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                        attempt_file.flush()
                        record("prediction_ready" if success else "prediction_failed", **entry)
                        if success or exhausted:
                            break
                        if attempt + 1 < config.max_attempts_per_sample:
                            record("prediction_retry", sample_id=sample.sample_id, next_attempt=attempt + 2)
                    outcomes.append({"sample_id": sample.sample_id, "repeat": repetition,
                                     "status": "succeeded" if success else "failed"})
                if exhausted:
                    break
        except KeyboardInterrupt:
            interrupted = True
            record("offline_inference_interrupted")
        except Exception as error:
            fatal = {"error_type": type(error).__name__, "message": str(error)}
            record("offline_inference_error", **fatal)
        finally:
            if backend is not None:
                try:
                    closed = backend.close(config.close_timeout_s)
                    drain()
                    if not closed:
                        record("worker_shutdown_timeout", timeout_s=config.close_timeout_s)
                except Exception as error:
                    closed = False
                    record("worker_shutdown_error", error_type=type(error).__name__, message=str(error))
            succeeded = sum(row["status"] == "succeeded" for row in outcomes)
            exit_code = 130 if interrupted else (0 if succeeded == total_samples and closed and fatal is None else 1)
            record("offline_inference_finished", exit_code=exit_code, succeeded=succeeded, total=total_samples, closed=closed)
    counts = Counter(event.type for event in log.events)
    stages = Counter(e.data["stage"] for e in log.events if e.type == "process_startup_progress")
    summary = {"exit_code": exit_code, "mode": "offline_inference_only", "model_kind": config.model["kind"],
        "requested_samples": total_samples, "succeeded_samples": succeeded,
        "failed_samples": len(outcomes) - succeeded, "unprocessed_samples": total_samples - len(outcomes),
        "attempt_count": len(attempts), "failed_attempts": sum(a["status"] == "failed" for a in attempts),
        "model_starts": counts["process_worker_started"], "restarts": counts["process_worker_restarted"],
        "warmups_completed": stages["warmup_completed"], "warmup_state_resets": stages["warmup_state_reset"],
        "terminations": counts["process_worker_terminated"], "restart_exhaustions": counts["process_restart_exhausted"],
        "closed": closed, "fatal_error": fatal, "outcomes": outcomes, "event_counts": dict(counts),
        "worker_logs": [path.name for path in sorted(directory.glob("worker-*.log"))]}
    processes = {}
    for event in log.events:
        generation = event.data.get("process_generation")
        if generation is not None:
            processes.setdefault(str(generation), []).append(asdict(event))
    _write_json(directory / "diagnostics.json", {"processes": processes, "attempts": attempts})
    _write_json(directory / "summary.json", summary)
    return exit_code, directory, summary
