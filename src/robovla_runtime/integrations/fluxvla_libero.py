"""Build a profiling session from an upstream FluxVLA LIBERO runner."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any

from robovla_runtime.backends.fluxvla import FluxVLABackend
from robovla_runtime.core.types import ActionSpec, Observation
from robovla_runtime.profiling.profiler import ProfileSession
from robovla_runtime.profiling.resources import CudaEventTimer


def _get_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _delete_value(config: Any, key: str) -> None:
    if isinstance(config, dict):
        config.pop(key, None)
    elif hasattr(config, key):
        delattr(config, key)


def create_libero_profile_session(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    fluxvla_root: str | Path | None = None,
    task_suite_name: str | None = None,
    task_id: int = 0,
    trial_id: int = 0,
    action_period_s: float = 0.1,
    action_dimension: int = 7,
    timing_mode: str = "diagnostic",
    num_inference_steps: int | None = None,
    inference_seed: int | None = None,
) -> ProfileSession:
    """Initialize one valid LIBERO observation without starting full eval."""
    config_path = Path(config_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"FluxVLA config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"FluxVLA checkpoint not found: {checkpoint_path}")
    if task_id < 0 or trial_id < 0:
        raise ValueError("task_id and trial_id must be non-negative")
    if action_period_s <= 0 or action_dimension <= 0:
        raise ValueError("action_period_s and action_dimension must be positive")
    if num_inference_steps is not None and num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if fluxvla_root is not None:
        root = Path(fluxvla_root).resolve()
        if not (root / "fluxvla").is_dir():
            raise FileNotFoundError(f"FluxVLA package not found under: {root}")
        root_text = os.fspath(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

    try:
        import torch
        from mmengine import Config
        from fluxvla.engines import build_runner_from_cfg
        from fluxvla.engines.runners.libero_eval_runner import (
            _get_libero_benchmark,
        )
        from fluxvla.engines.utils.eval_utils import (
            get_libero_dummy_action,
            get_libero_env,
        )
        from fluxvla.engines.utils.torch_utils import (
            configure_inference_attention_defaults,
        )
    except ImportError as error:
        raise RuntimeError(
            "FluxVLA profiling dependencies are unavailable; activate the "
            "FluxVLA/LIBERO environment before running this command"
        ) from error

    configure_inference_attention_defaults()
    cfg = Config.fromfile(os.fspath(config_path))
    eval_root = cfg.eval
    eval_cfg = copy.deepcopy(
        eval_root.runner if hasattr(eval_root, "runner") else eval_root
    )
    if _get_value(eval_cfg, "type") != "LiberoEvalRunner":
        raise ValueError("profile-fluxvla currently requires LiberoEvalRunner")
    for key in (
        "feishu_sheet_url",
        "feishu_app_id",
        "feishu_app_secret",
        "feishu_timeout",
    ):
        _delete_value(eval_cfg, key)

    configured_suites = _get_value(eval_cfg, "task_suite_name")
    if task_suite_name is None:
        if isinstance(configured_suites, (list, tuple)):
            if not configured_suites:
                raise ValueError("FluxVLA config contains no task suite")
            task_suite_name = str(configured_suites[0])
        else:
            task_suite_name = str(configured_suites)
    eval_cfg.task_suite_name = task_suite_name
    if num_inference_steps is not None:
        eval_cfg.num_inference_steps = num_inference_steps
    if inference_seed is not None:
        eval_cfg.inference_seed = inference_seed
    eval_cfg.cfg = cfg
    eval_cfg.ckpt_path = os.fspath(checkpoint_path)
    processor = _get_value(eval_cfg, "processor")
    if processor is not None:
        if isinstance(processor, dict):
            processor.setdefault("model_path", os.fspath(checkpoint_path))
        elif not hasattr(processor, "model_path"):
            processor.model_path = os.fspath(checkpoint_path)

    runner = None
    environment = None
    try:
        runner = build_runner_from_cfg(eval_cfg)
        runner.run_setup()
        benchmark = _get_libero_benchmark()
        suite = benchmark.get_benchmark_dict()[task_suite_name]()
        if task_id >= suite.n_tasks:
            raise ValueError(
                f"task_id {task_id} is outside suite size {suite.n_tasks}"
            )
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        if trial_id >= len(initial_states):
            raise ValueError(
                f"trial_id {trial_id} is outside initial-state count "
                f"{len(initial_states)}"
            )
        environment, task_description = get_libero_env(task, resolution=256)
        environment.reset()
        raw_observation = environment.set_init_state(initial_states[trial_id])
        for _ in range(runner.num_steps_wait):
            raw_observation, _, done, _ = environment.step(
                get_libero_dummy_action()
            )
            if done:
                raise RuntimeError(
                    "LIBERO episode ended while preparing the profile sample"
                )

        spec = ActionSpec(
            dimension=action_dimension,
            period_s=action_period_s,
            unit="libero_action",
        )
        backend = FluxVLABackend.from_libero_runner(
            runner,
            spec,
            task_description,
            synchronize=torch.cuda.synchronize,
            timing_mode=timing_mode,
            gpu_timer=CudaEventTimer(),
        )
        payload = dict(raw_observation)

        def observation_factory(index: int) -> Observation:
            return Observation(
                episode_id="fluxvla-profile",
                observation_id=index,
                captured_at=index * action_period_s,
                state=(0.0,) * action_dimension,
                payload=payload,
            )

        def cleanup() -> None:
            if environment is not None:
                environment.close()
            cleanup_runner = getattr(runner, "cleanup", None)
            if callable(cleanup_runner):
                cleanup_runner()

        return ProfileSession(
            backend=backend,
            observation_factory=observation_factory,
            metadata={
                "integration": "fluxvla_libero",
                "fluxvla_config": os.fspath(config_path),
                "fluxvla_config_source": config_path.read_text(encoding="utf-8"),
                "checkpoint": os.fspath(checkpoint_path),
                "task_suite_name": task_suite_name,
                "task_id": task_id,
                "trial_id": trial_id,
                "task_description": task_description,
                "model_family": runner.model_family,
                "eval_chunk_size": runner.eval_chunk_size,
                "norm_stats_key": runner.norm_stats_key,
                "num_inference_steps": runner.num_inference_steps,
                "inference_seed": runner.inference_seed,
                "timing_mode": timing_mode,
            },
            cleanup=cleanup,
        )
    except Exception:
        if environment is not None:
            environment.close()
        cleanup_runner = getattr(runner, "cleanup", None)
        if callable(cleanup_runner):
            cleanup_runner()
        raise
