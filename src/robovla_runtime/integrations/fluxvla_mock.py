"""Dependency-free stand-ins for the upstream stack, installed only in mock children.

The real Libero factory, config loader, preprocessor, FluxVLABackend and decoder
remain in the path. Only torch/mmengine/upstream model dependencies are replaced.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

from robovla_runtime.core import ActionSpec, Observation
from robovla_runtime.integrations import FluxVLALiberoFactory


def hang():
    while True:
        time.sleep(60)


@dataclass(frozen=True)
class AttemptCounter:
    path: Path

    @property
    def value(self):
        return self.path.read_text(encoding="utf-8").count("\n") if self.path.exists() else 0

    def next(self):
        index = self.value
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write("attempt\n")
        return index


class ConfigNode(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key, value):
        self[key] = value

    @classmethod
    def fromfile(cls, path):
        return json.loads(Path(path).read_text(encoding="utf-8"), object_hook=cls)


@dataclass(frozen=True)
class MockObservationFactory:
    phase: str = "warmup"

    def __call__(self, index: int) -> Observation:
        return Observation("sample", index, index * 0.02, (0.0, 0.0), {
            "camera": b"synthetic-image", "phase": self.phase,
        })


def runtime_payload(environment):
    return MockObservationFactory("inference")(0).payload


@dataclass(frozen=True)
class SimulatedLiberoFactory:
    recipe: FluxVLALiberoFactory
    attempts: AttemptCounter
    audit_path: Path
    faults: tuple[str, ...] = ()
    offline_mode: bool = False

    def __call__(self):
        index = self.attempts.next()
        fault = self.faults[index] if index < len(self.faults) else ""
        dimension = self.recipe.action_spec.dimension
        offline_mode = self.offline_mode

        def audit(kind, **data):
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"kind": kind, "pid": os.getpid(), "generation": index + 1, **data}) + "\n")

        class Model:
            def __init__(self):
                self.history = 999
                self.last_phase = None
                audit("model_built")

            def reset(self):
                audit("model_reset", previous_history=self.history, previous_phase=self.last_phase)
                if fault == "post_warmup_reset_hang" and self.last_phase == "warmup":
                    hang()
                self.history = 0
                self.last_phase = None

            def predict_action(self, **batch):
                phase = batch["phase"]
                self.last_phase = phase
                self.history += 1
                audit("predict", phase=phase, model_history=self.history,
                      dataset_history=batch["dataset_history"], seed=batch.get("seed"),
                      observation_state=batch.get("observation_state"),
                      num_inference_steps=batch.get("num_inference_steps"), unnorm_key=batch["unnorm_key"])
                if (fault == "warmup_hang" and phase == "warmup") or (fault == "request_hang" and phase == "inference"):
                    hang()
                if fault == "warmup_error" and phase == "warmup":
                    raise RuntimeError("injected warmup failure")
                values = [float(self.history), float(batch["dataset_history"])]
                return [[[values[i % 2] for i in range(dimension)] for _ in range(40)]]

        class Runner:
            def __init__(self, cfg):
                self.task_suite_name = cfg.task_suite_name
                self.num_inference_steps = cfg.num_inference_steps
                self.inference_seed = cfg.inference_seed
                self.norm_stats_key = "mock_stats"
                self.eval_chunk_size = 30
                self.enable_mixed_precision_training = False
                self.mixed_precision_dtype = "mock_float32"
                self.vla = Model()
                self.history = 777
                audit("runner_config", checkpoint=cfg.ckpt_path, processor_path=cfg.processor.model_path,
                      task_suite_name=cfg.task_suite_name, has_feishu="feishu_app_secret" in cfg)

            def run_setup(self):
                audit("run_setup")
                if fault == "setup_error":
                    raise RuntimeError("injected runner setup failure")

            def dataset(self, raw):
                if not offline_mode and raw["camera"] != b"synthetic-image":
                    raise ValueError("invalid synthetic camera input")
                if offline_mode and "camera" not in raw:
                    raise ValueError("mock offline model requires camera payload")
                if raw["is_new_episode"]:
                    self.history = 0
                self.history += 1
                audit("preprocess", phase=raw["phase"], is_new_episode=raw["is_new_episode"],
                      task_description=raw["task_description"])
                return {"dataset_history": self.history, "phase": raw["phase"]}, None

            def denormalize_action(self, data):
                if data["norm_stats_key"] != self.norm_stats_key or data["task_suite_name"] != self.task_suite_name:
                    raise ValueError("normalization metadata did not reach decoder")
                return [value / 10 for value in data["action"]]

            def cleanup(self):
                audit("cleanup")

        modules = {name: ModuleType(name) for name in (
            "torch", "mmengine", "fluxvla", "fluxvla.engines", "fluxvla.engines.utils",
            "fluxvla.engines.utils.torch_utils",
        )}
        for module in modules.values():
            module.__path__ = []
        modules["torch"].cuda = SimpleNamespace(synchronize=lambda: audit("synchronize"))
        modules["torch"].inference_mode = nullcontext
        modules["torch"].autocast = lambda *args, **kwargs: nullcontext()
        modules["mmengine"].Config = ConfigNode
        modules["fluxvla.engines"].build_runner_from_cfg = Runner
        modules["fluxvla.engines.utils.torch_utils"].configure_inference_attention_defaults = lambda: None
        sys.modules.update(modules)
        backend = self.recipe()
        if offline_mode:
            preprocess = backend.preprocess
            class OfflinePreprocessor:
                def reset(self, episode_id):
                    preprocess.reset(episode_id)

                def __call__(self, observation):
                    payload = dict(observation.payload)
                    payload["phase"] = "warmup" if observation.episode_id == "__fluxvla_warmup__" else "inference"
                    batch = dict(preprocess(replace(observation, payload=payload)))
                    batch["observation_state"] = observation.state
                    return batch
            backend.preprocess = OfflinePreprocessor()
        return backend


def make_simulated_factory(root: Path, faults=()) -> SimulatedLiberoFactory:
    root.mkdir(parents=True, exist_ok=True)
    config = root / "model.json"
    config.write_text(json.dumps({"eval": {"runner": {
        "type": "LiberoEvalRunner", "task_suite_name": ["mock_suite"],
        "num_inference_steps": 1, "inference_seed": 0, "processor": {},
        "feishu_app_secret": "synthetic-unused-value",
    }}}), encoding="utf-8")
    checkpoint = root / "checkpoint.mock"
    checkpoint.write_text("simulated weights", encoding="utf-8")
    recipe = FluxVLALiberoFactory(
        str(config), str(checkpoint), ActionSpec(2, 0.02), "move the simulated joints",
        num_inference_steps=4, inference_seed=11, reset_model_method="reset",
    )
    return SimulatedLiberoFactory(
        recipe, AttemptCounter(root.resolve() / "attempts.txt"), root.resolve() / "audit.jsonl", tuple(faults),
    )
