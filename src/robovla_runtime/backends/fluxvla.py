"""Adapter from FluxVLA-style policy calls to runtime action chunks.

The module deliberately does not import FluxVLA, PyTorch, or NumPy at import
time. A caller supplies an initialized VLA, preprocessing callable, and action
decoder. This keeps the runtime core usable without the optional model stack.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, nullcontext
from typing import Any, ContextManager

from robovla_runtime.core.types import ActionChunk, ActionSpec, Observation

from .base import PendingPrediction

Preprocessor = Callable[[Observation], Mapping[str, Any]]
ActionDecoder = Callable[[Any], Sequence[Sequence[float]]]


def _default_inference_context() -> ContextManager[Any]:
    """Use torch.inference_mode when torch is present."""
    try:
        import torch
    except ImportError:
        return nullcontext()
    return torch.inference_mode()


def _to_python(value: Any) -> Any:
    for method_name in ("detach", "float", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def decode_first_batch(
    raw_actions: Any, action_dimension: int | None = None
) -> tuple[tuple[float, ...], ...]:
    """Decode a rank-2 or rank-3 tensor-like action result.

    FluxVLA evaluation commonly returns ``[B, H, D]``. For rank three this
    decoder selects batch zero. Rank two is interpreted as ``[H, D]``.
    """
    data = _to_python(raw_actions)
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        raise TypeError("actions must be a rank-2 or rank-3 sequence")
    if not data:
        raise ValueError("actions must not be empty")
    first = data[0]
    if not isinstance(first, Sequence) or isinstance(first, (str, bytes)):
        raise ValueError("rank-1 action results are not supported")
    if first and isinstance(first[0], Sequence) and not isinstance(
        first[0], (str, bytes)
    ):
        data = first
    actions: list[tuple[float, ...]] = []
    for raw_action in data:
        if not isinstance(raw_action, Sequence) or isinstance(
            raw_action, (str, bytes)
        ):
            raise ValueError("each action must be a sequence")
        action = tuple(float(value) for value in raw_action)
        if action_dimension is not None:
            action = action[:action_dimension]
        actions.append(action)
    if not actions or not actions[0]:
        raise ValueError("decoded actions must not be empty")
    return tuple(actions)


def make_libero_action_decoder(
    denormalize_action: Callable[[Mapping[str, Any]], Any],
    norm_stats_key: str,
    execute_horizon: int | None = None,
    task_suite_name: str | None = None,
) -> ActionDecoder:
    """Build a decoder matching FluxVLA's LIBERO runner postprocessing."""
    if not norm_stats_key:
        raise ValueError("norm_stats_key must not be empty")
    if execute_horizon is not None and execute_horizon <= 0:
        raise ValueError("execute_horizon must be positive")

    def decode(raw_actions: Any) -> tuple[tuple[float, ...], ...]:
        value = raw_actions
        for method_name in ("detach", "float", "cpu"):
            method = getattr(value, method_name, None)
            if callable(method):
                value = method()
        to_numpy = getattr(value, "numpy", None)
        if callable(to_numpy):
            value = to_numpy()
        ndim = getattr(value, "ndim", None)
        if ndim not in (None, 2, 3):
            raise ValueError("FluxVLA action result must have rank two or three")
        value = _to_python(value)
        if not isinstance(value, Sequence) or not value:
            raise ValueError("FluxVLA action result must not be empty")
        first = value[0]
        if not isinstance(first, Sequence) or not first:
            raise ValueError("FluxVLA action result must have rank two or three")
        if isinstance(first[0], Sequence):
            value = first  # [B, H, D] -> [H, D]
        else:
            value = (first,)  # [B, D] -> one action from batch zero
        if execute_horizon is not None:
            value = value[:execute_horizon]
        decoded = []
        for action in value:
            inputs = {"action": action, "norm_stats_key": norm_stats_key}
            if task_suite_name is not None:
                inputs["task_suite_name"] = task_suite_name
            result = denormalize_action(inputs)
            result = _to_python(result)
            decoded.append(tuple(float(component) for component in result))
        return tuple(decoded)

    return decode


class FluxVLALiberoPreprocessor:
    """Turn an Observation payload into a FluxVLA LIBERO model batch."""

    def __init__(
        self,
        dataset: Callable[[Mapping[str, Any]], Any],
        task_description: str,
        unnorm_key: str,
        predict_options: Mapping[str, Any] | None = None,
    ) -> None:
        if not task_description or not unnorm_key:
            raise ValueError("task_description and unnorm_key are required")
        self.dataset = dataset
        self.task_description = task_description
        self.unnorm_key = unnorm_key
        self.predict_options = dict(predict_options or {})
        self._last_episode_id: str | None = None

    def __call__(self, observation: Observation) -> Mapping[str, Any]:
        if not isinstance(observation.payload, Mapping):
            raise TypeError("LIBERO preprocessing requires a mapping payload")
        raw_observation = dict(observation.payload)
        is_new_episode = observation.episode_id != self._last_episode_id
        raw_observation["task_description"] = self.task_description
        raw_observation["is_new_episode"] = is_new_episode
        result = self.dataset(raw_observation)
        batch = result[0] if isinstance(result, tuple) else result
        if not isinstance(batch, Mapping):
            raise TypeError("FluxVLA dataset must return a batch mapping")
        batch = dict(batch)
        batch["unnorm_key"] = self.unnorm_key
        batch.update(self.predict_options)
        self._last_episode_id = observation.episode_id
        return batch


class FluxVLABackend:
    """Call an initialized FluxVLA policy and return a timed ActionChunk.

    ``request`` is a blocking model call. Its measured latency is projected
    onto the runtime clock through ``ready_at`` for deterministic analysis.
    A worker-backed non-blocking implementation is a separate next step.
    """

    def __init__(
        self,
        vla: Any,
        preprocess: Preprocessor,
        decode_actions: ActionDecoder,
        action_spec: ActionSpec,
        *,
        timer: Callable[[], float] = time.perf_counter,
        synchronize: Callable[[], None] | None = None,
        inference_context: Callable[[], ContextManager[Any]] = (
            _default_inference_context
        ),
    ) -> None:
        predict_action = getattr(vla, "predict_action", None)
        if not callable(predict_action):
            raise TypeError("vla must provide a callable predict_action")
        if not callable(preprocess) or not callable(decode_actions):
            raise TypeError("preprocess and decode_actions must be callable")
        self.vla = vla
        self.preprocess = preprocess
        self.decode_actions = decode_actions
        self.action_spec = action_spec
        self.timer = timer
        self.synchronize = synchronize or (lambda: None)
        self.inference_context = inference_context
        self._next_request_id = 0

    @classmethod
    def from_libero_runner(
        cls,
        runner: Any,
        action_spec: ActionSpec,
        task_description: str,
        *,
        timer: Callable[[], float] = time.perf_counter,
        synchronize: Callable[[], None] | None = None,
    ) -> "FluxVLABackend":
        """Create an adapter from an initialized FluxVLA LiberoEvalRunner."""
        predict_options = {}
        if getattr(runner, "num_inference_steps", None) is not None:
            predict_options["num_inference_steps"] = runner.num_inference_steps
        if getattr(runner, "inference_seed", None) is not None:
            predict_options["seed"] = runner.inference_seed
        preprocess = FluxVLALiberoPreprocessor(
            runner.dataset,
            task_description,
            runner.task_suite_name,
            predict_options,
        )
        decoder = make_libero_action_decoder(
            runner.denormalize_action,
            runner.norm_stats_key,
            runner.eval_chunk_size,
            runner.task_suite_name,
        )

        def inference_context() -> ContextManager[Any]:
            import torch

            stack = ExitStack()
            stack.enter_context(
                torch.autocast(
                    "cuda",
                    dtype=runner.mixed_precision_dtype,
                    enabled=runner.enable_mixed_precision_training,
                )
            )
            stack.enter_context(torch.inference_mode())
            return stack

        return cls(
            runner.vla,
            preprocess,
            decoder,
            action_spec,
            timer=timer,
            synchronize=synchronize,
            inference_context=inference_context,
        )

    def request(
        self, observation: Observation, requested_at: float
    ) -> PendingPrediction:
        request_id = self._next_request_id
        self._next_request_id += 1

        self.synchronize()
        started = self.timer()
        batch = dict(self.preprocess(observation))
        self.synchronize()
        preprocessed = self.timer()
        with self.inference_context():
            raw_actions = self.vla.predict_action(**batch)
        self.synchronize()
        inferred = self.timer()
        decoded = self.decode_actions(raw_actions)
        finished = self.timer()

        actions = tuple(tuple(float(value) for value in action) for action in decoded)
        if not actions:
            raise ValueError("FluxVLA decoder returned no actions")
        if any(len(action) != self.action_spec.dimension for action in actions):
            raise ValueError(
                "decoded FluxVLA actions do not match ActionSpec.dimension"
            )
        total_s = finished - started
        stage_values = (
            preprocessed - started,
            inferred - preprocessed,
            finished - inferred,
        )
        if total_s < 0 or any(value < 0 for value in stage_values):
            raise ValueError("timer must be monotonic")
        ready_at = requested_at + total_s
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
        timings = (
            ("preprocess", stage_values[0]),
            ("inference", stage_values[1]),
            ("postprocess", stage_values[2]),
            ("total", total_s),
        )
        return PendingPrediction(
            request_id,
            requested_at,
            ready_at,
            chunk,
            timings,
        )
