"""Spawn-isolated policy ownership with a non-blocking control-side mailbox."""

from __future__ import annotations

import math
import multiprocessing as mp
import pickle
import queue
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Condition, Thread

from robovla_runtime.core.types import ActionSpec, Observation
from .base import PendingPrediction, PolicyBackend
from .threaded import AsyncResult, PredictionFailure, SubmissionReceipt


@dataclass(frozen=True, slots=True)
class ProcessBackendConfig:
    startup_timeout_s: float = 10.0
    execution_timeout_s: float = 1.0
    terminate_grace_s: float = 0.1
    max_restarts: int = 3

    def __post_init__(self) -> None:
        for value in (self.startup_timeout_s, self.execution_timeout_s, self.terminate_grace_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("process timeouts must be finite and positive")
        if isinstance(self.max_restarts, bool) or not isinstance(self.max_restarts, int) or self.max_restarts < 0:
            raise ValueError("max_restarts must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class _Request:
    request_id: int
    observation: Observation
    requested_at: float
    enqueued_at: float
    epoch: int
    expires_at: float | None = None


def _child_main(connection, factory_bytes: bytes, action_spec: ActionSpec) -> None:
    """Only the child constructs, resets, and calls the model."""
    try:
        backend = pickle.loads(factory_bytes)()
        if backend.action_spec != action_spec:
            raise ValueError("child backend ActionSpec differs from the parent contract")
        if not callable(getattr(backend, "reset", None)):
            raise TypeError("child backend must implement reset")
        connection.send(("ready", None))
        episode_key = None
        while True:
            request = connection.recv()
            started = time.perf_counter()
            try:
                key = (request.observation.episode_id, request.observation.episode_generation)
                reset_s = 0.0
                if episode_key != key:
                    backend.reset(request.observation.episode_id)
                    reset_s = time.perf_counter() - started
                    episode_key = key
                prediction = backend.request(
                    request.observation,
                    request.requested_at + max(0.0, started - request.enqueued_at),
                )
                if not isinstance(prediction, PendingPrediction):
                    raise TypeError("child request must return PendingPrediction")
                connection.send(("result", (request.request_id, prediction, reset_s)))
            except Exception as error:
                connection.send(("failure", (type(error).__name__, str(error))))
                # A failed call may have partially mutated model state.
                return
    except Exception as error:
        try:
            connection.send(("failure", (type(error).__name__, str(error))))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


def _transport(connection, outgoing: queue.Queue, incoming: queue.Queue) -> None:
    """Keep serialization and even partial pipe reads off the watchdog thread."""
    try:
        incoming.put(connection.recv())
        while True:
            request = outgoing.get()
            if request is None:
                return
            connection.send(request)
            incoming.put(connection.recv())
    except Exception as error:
        incoming.put(("transport_error", (type(error).__name__, str(error))))


class _Fault(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class _Cancelled(Exception):
    pass


class ProcessPolicyBackend:
    """One spawned model process, one active call, and a latest-only mailbox.

    The factory must be spawn-picklable (a module-level function or callable
    dataclass). Model/CUDA objects must be created inside it, never in the parent.
    Restart is lazy: a fresh child is launched when the next request is waiting.
    """

    def __init__(
        self, factory: Callable[[], PolicyBackend], action_spec: ActionSpec, *,
        config: ProcessBackendConfig | None = None,
    ) -> None:
        if not callable(factory):
            raise TypeError("factory must be callable")
        self.factory = factory
        self.action_spec = action_spec
        self.config = config or ProcessBackendConfig()
        self._context = mp.get_context("spawn")
        self._condition = Condition()
        self._queued: _Request | None = None
        self._active: _Request | None = None
        self._abort_reason: str | None = None
        self._results: deque[AsyncResult] = deque()
        self._diagnostics: deque[tuple[str, dict]] = deque()
        self._epoch = 0
        self._next_request_id = 0
        self._closed = False
        self._exhausted = False
        self._failures = 0
        self._generation = 0
        # Process/IPC handles are owned exclusively by the supervisor.
        self._process = None
        self._connection = None
        self._io = None
        self._incoming = None
        self._outgoing = None
        self._process_epoch = -1
        self._supervisor = Thread(target=self._run, name="robovla-process-supervisor", daemon=True)
        self._supervisor.start()

    def _emit(self, kind: str, **data) -> None:
        with self._condition:
            self._diagnostics.append((kind, {"process_generation": self._generation, **data}))

    def poll_diagnostics(self) -> tuple[tuple[str, dict], ...]:
        with self._condition:
            events = tuple(self._diagnostics)
            self._diagnostics.clear()
            return events

    def reset(self, episode_id: str) -> None:
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        with self._condition:
            if self._closed:
                raise RuntimeError("process backend is closed")
            if not self._supervisor.is_alive():
                raise RuntimeError("process supervisor stopped; create a new backend")
            self._epoch += 1
            self._queued = None
            self._results.clear()
            self._abort_reason = "episode_reset"
            self._exhausted = False
            self._failures = 0
            self._condition.notify_all()

    def submit(self, observation: Observation, requested_at: float) -> SubmissionReceipt:
        return self._submit(observation, requested_at, None)

    def submit_with_deadline(
        self, observation: Observation, requested_at: float, deadline_at: float,
    ) -> SubmissionReceipt:
        if not math.isfinite(deadline_at) or deadline_at <= requested_at:
            raise ValueError("deadline must be finite and later than submission")
        return self._submit(observation, requested_at, deadline_at - requested_at)

    def _submit(self, observation: Observation, requested_at: float, timeout_s: float | None) -> SubmissionReceipt:
        with self._condition:
            if self._closed:
                raise RuntimeError("process backend is closed")
            request_id = self._next_request_id
            self._next_request_id += 1
            replaced = self._queued.request_id if self._queued is not None else None
            if self._exhausted:
                self._results.append(PredictionFailure(
                    request_id, requested_at, observation.episode_id,
                    "ProcessUnavailable", "restart budget exhausted; explicit reset required",
                ))
            else:
                enqueued_at = time.perf_counter()
                self._queued = _Request(
                    request_id, observation, requested_at, enqueued_at, self._epoch,
                    enqueued_at + timeout_s if timeout_s is not None else None,
                )
                self._condition.notify_all()
            return SubmissionReceipt(request_id, replaced)

    def poll(self) -> tuple[AsyncResult, ...]:
        with self._condition:
            results = tuple(self._results)
            self._results.clear()
            return results

    def cancel_queued(self, request_id: int) -> bool:
        with self._condition:
            if self._queued is not None and self._queued.request_id == request_id:
                self._queued = None
                return True
            return False

    def expire_request(self, request_id: int) -> str:
        """Signal the supervisor; never terminate/join on the control thread."""
        with self._condition:
            if self.cancel_queued(request_id):
                return "queued"
            if self._active is not None and self._active.request_id == request_id:
                self._abort_reason = "request_timeout"
                self._condition.notify_all()
                return "active"
            return "completed_or_unknown"

    def cancel_episode(self, episode_id: str) -> tuple[int, ...]:
        with self._condition:
            dropped = []
            if self._queued is not None and self._queued.observation.episode_id == episode_id:
                dropped.append(self._queued.request_id)
                self._queued = None
            if self._active is not None and self._active.observation.episode_id == episode_id:
                dropped.append(self._active.request_id)
                self._abort_reason = "episode_cancelled"
            retained = deque()
            for result in self._results:
                episode = result.episode_id if isinstance(result, PredictionFailure) else result.chunk.episode_id
                if episode == episode_id:
                    dropped.append(result.request_id)
                else:
                    retained.append(result)
            self._results = retained
            self._condition.notify_all()
            return tuple(dropped)

    def close(self, timeout_s: float | None = None) -> bool:
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s < 0):
            raise ValueError("timeout_s must be finite and non-negative")
        with self._condition:
            self._closed = True
            self._queued = None
            self._results.clear()
            self._condition.notify_all()
        self._supervisor.join(1.0 if timeout_s is None else timeout_s)
        return (not self._supervisor.is_alive() and self._process is None
                and (self._io is None or not self._io.is_alive()))

    def _check_cancelled(self, epoch: int, active: bool) -> None:
        with self._condition:
            if self._closed:
                raise _Cancelled("closed")
            if epoch != self._epoch:
                raise _Cancelled("episode_reset")
            if active and self._abort_reason is not None:
                raise _Cancelled(self._abort_reason)

    def _receive(self, timeout_s: float, epoch: int, *, active: bool):
        deadline = time.perf_counter() + timeout_s
        while True:
            self._check_cancelled(epoch, active)
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise _Fault("ProcessExecutionTimeout" if active else "ProcessStartupTimeout", "process watchdog expired")
            try:
                kind, payload = self._incoming.get(timeout=min(0.005, remaining))
            except queue.Empty:
                if not self._process.is_alive():
                    # The IPC thread may still be delivering a final message.
                    try:
                        kind, payload = self._incoming.get(timeout=0.01)
                    except queue.Empty:
                        raise _Fault("ProcessExited", f"child exited with code {self._process.exitcode}")
                else:
                    continue
            self._check_cancelled(epoch, active)
            if time.perf_counter() > deadline:
                raise _Fault("ProcessExecutionTimeout" if active else "ProcessStartupTimeout", "process watchdog expired")
            if kind in {"failure", "transport_error"}:
                raise _Fault(payload[0], payload[1])
            return kind, payload

    def _start(self, epoch: int) -> None:
        # Pre-serialize the construction recipe before spawning. On Windows a
        # Process.start pickle failure can otherwise leave a partially booted child.
        try:
            factory_bytes = pickle.dumps(self.factory)
        except Exception as error:
            raise _Fault(type(error).__name__, str(error)) from error
        self._connection, child_connection = self._context.Pipe()
        self._incoming, self._outgoing = queue.Queue(), queue.Queue()
        self._process = self._context.Process(
            target=_child_main, args=(child_connection, factory_bytes, self.action_spec),
            name="robovla-policy-process", daemon=True,
        )
        self._generation += 1
        self._process_epoch = epoch
        try:
            self._process.start()
        except Exception as error:
            self._process.close()
            self._process = None
            raise _Fault(type(error).__name__, str(error)) from error
        finally:
            child_connection.close()
        self._emit("process_worker_started", pid=self._process.pid)
        if self._generation > 1:
            self._emit("process_worker_restarted", pid=self._process.pid)
        self._io = Thread(
            target=_transport, args=(self._connection, self._outgoing, self._incoming),
            name="robovla-process-ipc", daemon=True,
        )
        self._io.start()
        kind, _ = self._receive(self.config.startup_timeout_s, epoch, active=False)
        if kind != "ready":
            raise _Fault("ProcessProtocolError", "missing startup handshake")
        self._emit("process_worker_ready", pid=self._process.pid)

    def _stop(self, reason: str) -> bool:
        process = self._process
        if process is not None:
            pid = process.pid
            if process.is_alive():
                process.terminate()
                process.join(self.config.terminate_grace_s)
                if process.is_alive():
                    process.kill()
                    process.join(self.config.terminate_grace_s)
            else:
                process.join(0)
            if process.is_alive():
                self._emit("process_termination_failed", pid=pid, reason=reason)
                return False  # Never create a second model while the old one lives.
            self._emit("process_worker_terminated", pid=pid, reason=reason, exitcode=process.exitcode)
            process.close()
            self._process = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._outgoing is not None:
            self._outgoing.put(None)
        if self._io is not None:
            self._io.join(self.config.terminate_grace_s)
            if self._io.is_alive():
                self._emit("process_transport_stuck", reason=reason)
                return False
            self._io = None
        return True

    def _record_fault(self, kind: str, message: str, request: _Request | None, epoch: int) -> None:
        self._emit("process_worker_fault", reason=kind, message=message,
                   request_id=request.request_id if request else None)
        with self._condition:
            if epoch != self._epoch or self._closed:
                return
            if request is not None and self._abort_reason is None:
                self._results.append(PredictionFailure(
                    request.request_id, request.requested_at, request.observation.episode_id, kind, message,
                ))
            self._failures += 1
            if self._failures > self.config.max_restarts:
                self._exhausted = True
                self._emit("process_restart_exhausted", consecutive_failures=self._failures)
                if self._queued is not None:
                    waiting = self._queued
                    self._queued = None
                    self._results.append(PredictionFailure(
                        waiting.request_id, waiting.requested_at, waiting.observation.episode_id,
                        "ProcessUnavailable", "restart budget exhausted; explicit reset required",
                    ))

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    if self._closed:
                        return
                    epoch = self._epoch
                    stale = self._process is not None and self._process_epoch != epoch
                    if not stale and self._queued is None:
                        self._condition.wait(0.01)
                        continue
                if stale:
                    if not self._stop("episode_reset"):
                        return
                    continue
                request = None
                try:
                    with self._condition:
                        if self._exhausted and self._queued is not None:
                            waiting = self._queued
                            self._queued = None
                            self._results.append(PredictionFailure(
                                waiting.request_id, waiting.requested_at, waiting.observation.episode_id,
                                "ProcessUnavailable", "restart budget exhausted; explicit reset required",
                            ))
                            continue
                    if self._process is None:
                        self._start(epoch)
                    with self._condition:
                        self._check_cancelled(epoch, False)
                        if self._queued is None:
                            continue
                        request = self._queued
                        self._queued = None
                        if request.expires_at is not None and time.perf_counter() >= request.expires_at:
                            self._emit("process_request_expired_before_dispatch", request_id=request.request_id)
                            self._results.append(PredictionFailure(
                                request.request_id, request.requested_at, request.observation.episode_id,
                                "ProcessQueueTimeout", "request expired before process dispatch",
                            ))
                            continue
                        self._active = request
                        self._abort_reason = None
                    self._emit("process_request_started", request_id=request.request_id, pid=self._process.pid)
                    self._outgoing.put(request)
                    timeout_s = self.config.execution_timeout_s
                    if request.expires_at is not None:
                        timeout_s = min(timeout_s, request.expires_at - time.perf_counter())
                    kind, payload = self._receive(timeout_s, epoch, active=True)
                    if kind != "result" or payload[0] != request.request_id:
                        raise _Fault("ProcessProtocolError", "response did not match the active request")
                    _, prediction, reset_s = payload
                    now = time.perf_counter()
                    ready_at = request.requested_at + max(0.0, now - request.enqueued_at)
                    result = replace(
                        prediction, request_id=request.request_id, requested_at=request.requested_at,
                        ready_at=ready_at, chunk=replace(prediction.chunk, request_id=request.request_id, ready_at=ready_at),
                        timings_s=(*prediction.timings_s, ("reset", reset_s),
                                   ("end_to_end", ready_at - request.requested_at)),
                    )
                    with self._condition:
                        self._check_cancelled(epoch, True)
                        self._results.append(result)
                        self._failures = 0
                except _Cancelled as error:
                    reason = str(error)
                    if reason == "request_timeout":
                        self._record_fault("ProcessRequestTimeout", reason, request, epoch)
                    if not self._stop(reason):
                        return
                except Exception as error:
                    # A startup failure also retires the currently waiting request.
                    if request is None:
                        with self._condition:
                            if epoch == self._epoch:
                                request = self._queued
                                self._queued = None
                                self._abort_reason = None
                    kind = error.kind if isinstance(error, _Fault) else type(error).__name__
                    self._record_fault(kind, str(error), request, epoch)
                    if not self._stop(kind):
                        return
                finally:
                    with self._condition:
                        self._active = None
                        self._abort_reason = None
        except Exception as error:
            self._emit("process_supervisor_failed", error_type=type(error).__name__, message=str(error))
            with self._condition:
                if not self._closed:
                    for pending in (self._active, self._queued):
                        if pending is not None and pending.epoch == self._epoch:
                            self._results.append(PredictionFailure(
                                pending.request_id, pending.requested_at, pending.observation.episode_id,
                                "ProcessSupervisorError", str(error),
                            ))
                self._queued = None
        finally:
            self._stop("closed")
            with self._condition:
                # An unexpected supervisor/termination failure fails closed.
                self._exhausted = True
