from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from robovla_runtime.backends import LatencyTrace, MockBackend, make_latency_trace
from robovla_runtime.cli import main
from robovla_runtime.core import ActionSpec, Observation


class LatencyFaultTests(unittest.TestCase):
    def test_time_indexed_trace_is_independent_of_request_count(self):
        trace = LatencyTrace("test", 0.5, (0.04, 0.20, 0.26))
        self.assertEqual([trace.at(t) for t in (0.0, 0.49, 0.5, 1.0, 9.0)], [0.04, 0.04, 0.2, 0.26, 0.26])
        self.assertEqual(trace.at(0.5), 0.2)
        # Boundary is stable under accumulated virtual-clock rounding error.
        self.assertEqual(trace.at(0.5 - 1e-16), 0.2)

    def test_seeded_long_tail_is_reproducible_and_has_bounded_spikes(self):
        first = make_latency_trace("long_tail", seed=7)
        self.assertEqual(first, make_latency_trace("long_tail", seed=7))
        self.assertNotEqual(first.latencies_s, make_latency_trace("long_tail", seed=17).latencies_s)
        self.assertIn(0.04, first.latencies_s)
        self.assertTrue(any(value >= 0.12 for value in first.latencies_s))
        self.assertTrue(all(0.04 <= value <= 0.30 for value in first.latencies_s))

    def test_overload_starts_at_configured_time_slot_and_persists(self):
        trace = make_latency_trace("overload")
        self.assertEqual(trace.at(2.99), 0.04)
        self.assertEqual(trace.at(3.0), 0.26)
        self.assertEqual(trace.at(100.0), 0.26)

    def test_invalid_trace_and_profile_values_are_rejected(self):
        for interval, values in ((0.0, (0.1,)), (math.inf, (0.1,)), (1.0, ()), (1.0, (-1.0,)), (1.0, (math.nan,))):
            with self.subTest(interval=interval, values=values), self.assertRaises(ValueError):
                LatencyTrace("bad", interval, values)
        for settings in ({"duration_s": 0}, {"spike_probability": 2}, {"spike_max_s": 0.01}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                make_latency_trace("long_tail", **settings)
        with self.assertRaises(ValueError):
            make_latency_trace("unknown")

    def test_mock_injection_overrides_sequence_and_reset_preserves_trace(self):
        trace = make_latency_trace("overload")
        backend = MockBackend(ActionSpec(1, 0.02), 10, (99.0,), (1.0,), latency_trace=trace)
        obs = Observation("ep", 0, 0.0, (0.0,))
        first = backend.request(obs, 0.0)
        busy = backend.request(obs, 3.0)
        backend.reset("ep")
        again = backend.request(obs, 0.0)
        self.assertEqual(first.ready_at, 0.04)
        self.assertEqual(busy.ready_at, 3.26)
        self.assertEqual(again.ready_at, first.ready_at)
        self.assertGreater(again.request_id, first.request_id)
        self.assertEqual(dict(busy.timings_s), {"synthetic_latency": 0.26})

    def test_trace_roundtrip_and_cli_preserve_injected_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = LatencyTrace("zero", 1.0, (0.0,), 7)
            trace.write(root / "trace.json")
            self.assertEqual(trace, LatencyTrace.read(root / "trace.json"))
            (root / "config.toml").write_text(
                '[runtime]\nscheduler="adaptive_async"\nduration_s=0.1\nlatency_s=[99.0]\n',
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([
                    "run", "--config", str(root / "config.toml"),
                    "--latency-trace", str(root / "trace.json"),
                    "--output-root", str(root / "runs"),
                ]), 0)
            run = next((root / "runs").iterdir())
            self.assertEqual(LatencyTrace.read(run / "latency_trace.json"), trace)
            summary = json.loads((run / "summary.json").read_text())
            self.assertEqual(summary["buffer_underrun_ticks"], 0)


if __name__ == "__main__":
    unittest.main()
