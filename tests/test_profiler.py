from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robovla_runtime.backends.base import PendingPrediction
from robovla_runtime.cli import build_parser
from robovla_runtime.core.types import ActionChunk, ActionSpec, Observation
from robovla_runtime.profiling import ProfileConfig, profile_backend, write_profile


class FakeBackend:
    def __init__(self) -> None:
        self.action_spec = ActionSpec(2, 0.1)
        self.calls = 0

    def request(self, observation, requested_at):
        request_id = self.calls
        self.calls += 1
        latency = 0.01 * (request_id + 1)
        chunk = ActionChunk(
            episode_id=observation.episode_id,
            request_id=request_id,
            source_observation_id=observation.observation_id,
            source_observation_time=observation.captured_at,
            ready_at=requested_at + latency,
            start_at=observation.captured_at,
            spec=self.action_spec,
            actions=((0.0, 0.0), (1.0, 1.0)),
        )
        return PendingPrediction(
            request_id,
            requested_at,
            requested_at + latency,
            chunk,
            (("inference", latency), ("total", latency)),
        )


class FakeMemorySampler:
    def __init__(self) -> None:
        self.samples = 0

    def begin_sample(self) -> None:
        self.samples += 1

    def snapshot(self):
        return {"cuda_peak_memory_allocated_bytes": self.samples * 100}


class IncrementingTimer:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        value = self.value
        self.value += 0.001
        return value


class ProfilerTests(unittest.TestCase):
    def test_cli_parses_fluxvla_profile_command(self) -> None:
        args = build_parser().parse_args(
            [
                "profile-fluxvla",
                "--config",
                "model.py",
                "--checkpoint",
                "checkpoint.pt",
                "--output",
                "profile-output",
                "--mode",
                "throughput",
            ]
        )
        self.assertEqual(args.command, "profile-fluxvla")
        self.assertEqual(args.mode, "throughput")
        self.assertEqual(args.samples, 100)

    def test_warmup_is_recorded_but_excluded_from_summary(self) -> None:
        backend = FakeBackend()
        memory = FakeMemorySampler()
        result = profile_backend(
            backend,
            lambda index: Observation("profile", index, index * 0.1, (0.0,)),
            ProfileConfig(warmup=2, samples=3),
            memory_sampler=memory,
            timer=IncrementingTimer(),
            environment={"gpu": "fake"},
            metadata={"checkpoint": "fake.ckpt"},
        )

        self.assertEqual(len(result.records), 5)
        self.assertEqual([record.warmup for record in result.records], [True, True, False, False, False])
        self.assertEqual(result.summary["measured_requests"], 3)
        self.assertEqual(result.summary["backend_timings_s"]["total"]["p50"], 0.04)
        self.assertEqual(
            result.summary["resources"]["cuda_peak_memory_allocated_bytes"]["max"],
            500.0,
        )
        self.assertEqual(result.environment, {"gpu": "fake"})

    def test_writes_reproducible_artifact_set(self) -> None:
        result = profile_backend(
            FakeBackend(),
            lambda index: Observation("profile", index, index * 0.1, (0.0,)),
            ProfileConfig(warmup=1, samples=2, mode="throughput"),
            timer=IncrementingTimer(),
            environment={"platform": "test"},
            metadata={"config": "flux.py"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = write_profile(result, Path(temporary) / "profile")
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "config.json",
                    "requests.jsonl",
                    "summary.json",
                    "environment.json",
                    "metadata.json",
                },
            )
            lines = (output / "requests.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(lines), 3)
            self.assertTrue(json.loads(lines[0])["warmup"])
            saved_config = json.loads(
                (output / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_config["profile"]["mode"], "throughput")
            self.assertEqual(saved_config["metadata"]["config"], "flux.py")


if __name__ == "__main__":
    unittest.main()
