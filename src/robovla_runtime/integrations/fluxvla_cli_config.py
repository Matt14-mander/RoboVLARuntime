"""Strict, dependency-light preflight for the offline FluxVLA CLI."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields as dataclass_fields
from pathlib import Path

from robovla_runtime.backends import ProcessBackendConfig
from robovla_runtime.core import ActionSpec
from .fluxvla_process import FluxVLALiberoFactory
from .offline_observations import fields, integer, load_observations, nonempty, number, sha256


MOCK_FAULTS = {"setup_error", "warmup_error", "warmup_hang", "post_warmup_reset_hang", "request_hang"}


@dataclass(frozen=True)
class OfflineInferenceConfig:
    source: str
    action_spec: ActionSpec
    model: dict
    manifest: str
    required_payload_keys: tuple[str, ...]
    process: ProcessBackendConfig
    repeats: int = 1
    max_attempts_per_sample: int = 2
    poll_interval_s: float = 0.005
    close_timeout_s: float = 2.0

    def resolved(self):
        return {
            "version": 1, "source": self.source, "action": asdict(self.action_spec), "model": self.model,
            "observations": {"manifest": self.manifest, "required_payload_keys": self.required_payload_keys},
            "process": asdict(self.process),
            "inference": {name: getattr(self, name) for name in (
                "repeats", "max_attempts_per_sample", "poll_interval_s", "close_timeout_s",
            )},
        }


def load_inference_config(path: str | Path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    fields(raw, {"version", "action", "model", "observations", "process", "inference"},
           {"version", "action", "model", "observations"}, "config")
    if type(raw["version"]) is not int or raw["version"] != 1:
        raise ValueError("config version must be 1")
    action = dict(raw["action"]) if isinstance(raw["action"], dict) else raw["action"]
    fields(action, {f.name for f in dataclass_fields(ActionSpec)},
           {"dimension", "period_s", "representation", "unit", "action_space", "coordinate_frame"}, "action")
    integer(action["dimension"], "action.dimension", 1)
    number(action["period_s"], "action.period_s", positive=True)
    for key, value in action.items():
        if key not in {"dimension", "period_s", "component_units"}:
            nonempty(value, f"action.{key}")
    if "component_units" in action:
        if not isinstance(action["component_units"], list):
            raise ValueError("action.component_units must be an array")
        action["component_units"] = tuple(nonempty(v, "component unit") for v in action["component_units"])
    spec = ActionSpec(**action)
    model = raw["model"]
    fields(model, {"kind", "config_path", "checkpoint_path", "task_description", "fluxvla_root", "task_suite_name",
                   "num_inference_steps", "inference_seed", "timing_mode", "reset_model_method", "warmup_steps",
                   "model_is_stateless", "fault_plan"}, {"kind", "task_description"}, "model")
    nonempty(model["kind"], "model.kind")
    if model["kind"] not in {"mock", "fluxvla"}:
        raise ValueError("model.kind must be mock or fluxvla")
    nonempty(model["task_description"], "model.task_description")
    integer(model.setdefault("warmup_steps", 2), "model.warmup_steps")
    if type(model.setdefault("model_is_stateless", False)) is not bool:
        raise ValueError("model.model_is_stateless must be boolean")
    for key in ("task_suite_name", "reset_model_method"):
        if key in model:
            nonempty(model[key], f"model.{key}")
    if "num_inference_steps" in model:
        integer(model["num_inference_steps"], "model.num_inference_steps", 1)
    if "inference_seed" in model:
        integer(model["inference_seed"], "model.inference_seed")
    nonempty(model.setdefault("timing_mode", "throughput"), "model.timing_mode")
    if model["timing_mode"] not in {"diagnostic", "throughput"}:
        raise ValueError("model.timing_mode must be diagnostic or throughput")
    if "reset_model_method" in model and any(not p.isidentifier() for p in model["reset_model_method"].split(".")):
        raise ValueError("model.reset_model_method must be a dotted method name")
    if model["kind"] == "mock":
        if {"config_path", "checkpoint_path", "fluxvla_root"} & set(model):
            raise ValueError("mock model must not specify real model/checkpoint paths")
        faults = model.setdefault("fault_plan", [])
        if not isinstance(faults, list) or any(not isinstance(f, str) or f not in MOCK_FAULTS for f in faults):
            raise ValueError(f"model.fault_plan accepts only {sorted(MOCK_FAULTS)}")
        if model.get("reset_model_method", "reset") != "reset":
            raise ValueError("mock model reset_model_method must be reset")
    else:
        if "fault_plan" in model:
            raise ValueError("fault_plan is only available for mock models")
        for key in ("config_path", "checkpoint_path"):
            nonempty(model.get(key), f"model.{key}")
        for key in ("config_path", "checkpoint_path", "fluxvla_root"):
            if key in model:
                candidate = (path.parent / nonempty(model[key], f"model.{key}")).resolve()
                if not candidate.exists() or (key == "config_path" and not candidate.is_file()):
                    raise ValueError(f"model.{key} not found: {candidate}")
                if key == "fluxvla_root" and not (candidate / "fluxvla").is_dir():
                    raise ValueError("model.fluxvla_root must contain the fluxvla package")
                model[key] = str(candidate)
        # Construct only a recipe: no imports or model/CUDA setup during preflight.
        FluxVLALiberoFactory(action_spec=spec, **{k: v for k, v in model.items()
                            if k not in {"kind", "warmup_steps", "model_is_stateless"}})
    process = raw.get("process", {})
    fields(process, {f.name for f in dataclass_fields(ProcessBackendConfig)}, set(), "process")
    for key, value in process.items():
        if key == "max_restarts":
            integer(value, f"process.{key}")
        else:
            number(value, f"process.{key}", positive=True)
    process_config = ProcessBackendConfig(**process)
    inference = raw.get("inference", {})
    fields(inference, {"repeats", "max_attempts_per_sample", "poll_interval_s", "close_timeout_s"}, set(), "inference")
    for key, value in inference.items():
        if key in {"repeats", "max_attempts_per_sample"}:
            integer(value, f"inference.{key}", 1)
        else:
            number(value, f"inference.{key}", positive=True)
    observations = raw["observations"]
    fields(observations, {"manifest", "required_payload_keys"}, {"manifest", "required_payload_keys"}, "observations")
    required = observations["required_payload_keys"]
    if not isinstance(required, list) or not required:
        raise ValueError("observations.required_payload_keys must be a non-empty array")
    required = tuple(nonempty(key, "required payload key") for key in required)
    if len(set(required)) != len(required):
        raise ValueError("duplicate required payload key")
    manifest = (path.parent / nonempty(observations["manifest"], "observations.manifest")).resolve()
    loaded = load_observations(manifest, state_dimension=spec.dimension, required_payload_keys=required)
    if model["kind"] == "mock" and any("camera" not in sample.payload for sample in loaded.samples):
        raise ValueError("mock model observations must contain camera payload")
    config = OfflineInferenceConfig(str(path), spec, model, str(manifest), required, process_config, **inference)
    provenance = [{"path": str(path), "sha256": sha256(path)}, *loaded.provenance]
    return config, loaded, provenance
