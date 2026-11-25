"""Serializable FluxVLA construction and startup recipes for spawned workers."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from robovla_runtime.backends.fluxvla import FluxVLABackend
from robovla_runtime.backends.fluxvla_inputs import FluxVLAModelInputSpec
from robovla_runtime.backends.process import ProcessBackendConfig, ProcessPolicyBackend
from robovla_runtime.core.types import ActionSpec, Observation, ObservationSpec


@dataclass(frozen=True, slots=True)
class FluxVLALiberoFactory:
    """Model-only upstream recipe; no runner, tensor, or environment in parent."""
    config_path: str
    checkpoint_path: str
    action_spec: ActionSpec
    task_description: str
    fluxvla_root: str | None = None
    task_suite_name: str | None = None
    num_inference_steps: int | None = None
    inference_seed: int | None = None
    timing_mode: str = "throughput"
    reset_model_method: str | None = None
    input_spec: FluxVLAModelInputSpec | None = None

    def __post_init__(self) -> None:
        if self.input_spec is not None and not isinstance(self.input_spec, FluxVLAModelInputSpec):
            raise TypeError("input_spec must be a FluxVLAModelInputSpec")
        if not self.task_description:
            raise ValueError("task_description is required")
        if self.timing_mode not in {"diagnostic", "throughput"}:
            raise ValueError("invalid timing_mode")
        if self.num_inference_steps is not None and self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if self.reset_model_method is not None and any(
            not part.isidentifier() for part in self.reset_model_method.split(".")
        ):
            raise ValueError("reset_model_method must be a dotted model method name")
        for field in ("config_path", "checkpoint_path", "fluxvla_root"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, str(Path(value).resolve()))

    def __call__(self) -> FluxVLABackend:
        from .fluxvla_libero import load_libero_runner

        runner = load_libero_runner(
            config_path=self.config_path, checkpoint_path=self.checkpoint_path,
            fluxvla_root=self.fluxvla_root, task_suite_name=self.task_suite_name,
            num_inference_steps=self.num_inference_steps, inference_seed=self.inference_seed,
        )
        try:
            import torch

            reset_model = None
            if self.reset_model_method is not None:
                reset_model = runner.vla
                for part in self.reset_model_method.split("."):
                    reset_model = getattr(reset_model, part)
                if not callable(reset_model):
                    raise TypeError("reset_model_method must resolve to a callable")
            return FluxVLABackend.from_libero_runner(
                runner, self.action_spec, self.task_description,
                synchronize=torch.cuda.synchronize, timing_mode=self.timing_mode,
                reset_model=reset_model,
                input_spec=self.input_spec,
            )
        except Exception:
            cleanup = getattr(runner, "cleanup", None)
            if callable(cleanup):
                cleanup()
            raise


@dataclass(frozen=True, slots=True)
class FluxVLAProcessFactory:
    """Carry recipes, never initialized models, into a fresh process.

    Both callables must be standard-pickle serializable. backend_factory builds
    the model, transforms, decoder and optional reset hook inside the child.
    observation_factory supplies valid model inputs, not fabricated camera shapes.
    """
    backend_factory: Callable[[], FluxVLABackend]
    observation_factory: Callable[[int], Observation] | None = None
    warmup_steps: int = 2
    model_is_stateless: bool = False
    observation_spec: ObservationSpec | None = None

    def __post_init__(self) -> None:
        if not callable(self.backend_factory):
            raise TypeError("backend_factory must be callable")
        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        if self.warmup_steps and not callable(self.observation_factory):
            raise ValueError("warmup requires an observation_factory")
        if not isinstance(self.model_is_stateless, bool):
            raise TypeError("model_is_stateless must be boolean")
        if self.observation_spec is not None and not isinstance(self.observation_spec, ObservationSpec):
            raise TypeError("observation_spec must be an ObservationSpec")

    def __call__(self) -> _StartingFluxVLA:
        backend = self.backend_factory()
        if not isinstance(backend, FluxVLABackend):
            raise TypeError("backend_factory must build a FluxVLABackend")
        if not backend.has_model_reset and not self.model_is_stateless:
            raise ValueError("FluxVLA requires a model reset hook or explicit model_is_stateless=True")
        return _StartingFluxVLA(backend, self.observation_factory, self.warmup_steps, self.observation_spec)

    def create_backend(self, action_spec: ActionSpec, *, config: ProcessBackendConfig | None = None) -> ProcessPolicyBackend:
        return ProcessPolicyBackend(self, action_spec, config=config)


class _StartingFluxVLA:
    def __init__(self, backend, observation_factory, warmup_steps, observation_spec):
        self.backend = backend
        self.action_spec = backend.action_spec
        self.observation_factory = observation_factory
        self.warmup_steps = warmup_steps
        self.observation_spec = observation_spec
        self._ready = False

    def _validate_observation(self, observation):
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        # Legacy SDK callers infer from their first input, never from ActionSpec.
        spec = self.observation_spec or ObservationSpec(len(observation.state))
        spec.validate(observation)
        self.observation_spec = spec

    def startup(self, report: Callable[..., None]) -> None:
        self._ready = False
        started = time.perf_counter()
        report("warmup_started", steps=self.warmup_steps)
        self.backend.reset("__fluxvla_warmup__")
        for index in range(self.warmup_steps):
            # Keep the factory's source sample pristine even if a transform mutates it.
            source = copy.deepcopy(self.observation_factory(index))
            self._validate_observation(source)
            observation = replace(
                source, episode_id="__fluxvla_warmup__", observation_id=index,
                episode_generation=0,
            )
            self.backend.request(observation, observation.captured_at)  # Deliberately discarded.
        self.backend.synchronize()
        report("warmup_completed", steps=self.warmup_steps, elapsed_s=time.perf_counter() - started)
        # Clear both preprocessor episode tracking and model caches. The process
        # worker performs another reset for the actual episode before its first call.
        self.backend.reset("__fluxvla_ready__")
        self.backend.synchronize()
        report("warmup_state_reset")
        self._ready = True

    def reset(self, episode_id: str) -> None:
        self.backend.reset(episode_id)

    def request(self, observation, requested_at):
        if not self._ready:
            raise RuntimeError("FluxVLA startup has not completed")
        self._validate_observation(observation)
        return self.backend.request(observation, requested_at)
