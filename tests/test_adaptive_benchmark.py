from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.run_adaptive_matrix import run_comparison, write_comparison
from robovla_runtime.backends import LatencyTrace
from robovla_runtime.runtime import RuntimeConfig, run_simulation


class AdaptiveBenchmarkTests(unittest.TestCase):
    def test_pairs_share_input_and_configuration_except_scheduler(self):
        runs = run_comparison(duration_s=4.0)
        self.assertEqual(len(runs), 12)
        for fixed, adaptive in zip(runs[::2], runs[1::2]):
            self.assertEqual(fixed[0], adaptive[0])
            self.assertIs(fixed[1], adaptive[1])
            self.assertEqual(
                replace(fixed[2].config, scheduler="adaptive_async"), adaptive[2].config
            )
            for run in (fixed, adaptive):
                trace, result = run[1], run[2]
                requests = [event for event in result.events if event.type == "prediction_requested"]
                for event in requests:
                    self.assertAlmostEqual(
                        event.data["ready_at"] - event.timestamp, trace.at(event.timestamp)
                    )

    def test_saved_configuration_trace_and_events_reproduce_report(self):
        runs = run_comparison(duration_s=0.4)[:2]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_comparison(runs, root / "raw", root / "report")
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["source_sha256"]), 64)
            for name, _, result, row in runs:
                directory = root / "raw" / name / result.config.scheduler
                config = json.loads((directory / "config.json").read_text())
                for field in ("latency_s", "initial_position", "target_position"):
                    config[field] = tuple(config[field])
                replay = run_simulation(
                    RuntimeConfig(**config),
                    latency_trace=LatencyTrace.read(directory / "latency_trace.json"),
                )
                self.assertEqual(replay.summary, result.summary)
                saved_events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
                fallback = sum(event["type"] == "fallback_executed" for event in saved_events)
                self.assertEqual(fallback, row["fallback_ticks"])
                self.assertEqual(
                    json.loads(json.dumps([asdict(event) for event in replay.events])), saved_events
                )
            with self.assertRaises(FileExistsError):
                write_comparison(runs, root / "raw", root / "report")


if __name__ == "__main__":
    unittest.main()
