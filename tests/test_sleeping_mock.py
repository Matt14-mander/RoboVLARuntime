import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from robovla_runtime.backends import MockBackend, SleepingMockBackend
from robovla_runtime.cli import main
from robovla_runtime.core import ActionSpec, Observation


class SleepingMockTests(unittest.TestCase):
    def test_blocking_mock_uses_elapsed_time_and_preserves_identity(self):
        clock = [0.0]
        delays = []
        def sleep(delay):
            delays.append(delay)
            clock[0] += delay + 0.005
        backend = SleepingMockBackend(
            MockBackend(ActionSpec(1, 0.01), 10, (0.03,), (1.0,)),
            timer=lambda: clock[0], sleeper=sleep,
        )
        backend.reset("ep")
        result = backend.request(Observation("ep", 3, 0.1, (0.0,), episode_generation=2), 0.1)
        self.assertAlmostEqual(delays[0], 0.03)
        self.assertAlmostEqual(result.ready_at, 0.135)
        self.assertEqual(result.chunk.episode_generation, 2)
        self.assertEqual(result.chunk.source_observation_id, 3)
        self.assertAlmostEqual(dict(result.timings_s)["simulated_inference"], 0.035)

    def test_cli_runs_wall_clock_mock_and_saves_worker_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.toml").write_text(
                '[runtime]\nscheduler="worker_async"\nworker_prefetch="adaptive"\n'
                'duration_s=0.06\nperiod_s=0.01\nlatency_s=[0.0]\n',
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([
                    "run-worker-mock", "--config", str(root / "config.toml"),
                    "--output-root", str(root / "runs"),
                ]), 0)
            directory = next((root / "runs").iterdir())
            events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
            self.assertTrue(any(event["type"] == "prediction_submitted" for event in events))
            self.assertTrue(any(event["type"] == "adaptive_prefetch_started" for event in events))
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["ticks"], 6)


if __name__ == "__main__":
    unittest.main()
