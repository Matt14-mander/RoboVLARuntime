from __future__ import annotations

import math
import unittest
from dataclasses import replace

from robovla_runtime.backends import MockBackend, LatencyTrace
from robovla_runtime.core import ActionSpec, EventLog, Observation
from robovla_runtime.execution.action_buffer import ActionBuffer
from robovla_runtime.runtime import RuntimeConfig, run_simulation
from robovla_runtime.scheduling import AdaptiveAsyncScheduler, AdaptivePrefetchConfig


class Harness:
    def __init__(self, latencies, **options):
        self.log = EventLog()
        self.buffer = ActionBuffer("ep")
        self.backend = MockBackend(ActionSpec(1, 0.02), 10, latencies, (1.0,))
        self.scheduler = AdaptiveAsyncScheduler(
            self.backend, self.log, 0.08, 0.2, AdaptivePrefetchConfig(**options)
        )
        self.observation_id = 0

    def capture(self, now):
        obs = Observation("ep", self.observation_id, now, (0.0,))
        self.observation_id += 1
        return obs

    def tick(self, now):
        self.scheduler.tick(now, self.buffer, self.capture)

    def events(self, kind):
        return [event for event in self.log.events if event.type == kind]


class AdaptiveSchedulerTests(unittest.TestCase):
    def test_pending_future_latency_is_not_used_before_completion(self):
        harness = Harness((1.0,))
        harness.tick(0.0)
        harness.tick(0.9)
        self.assertEqual(len(harness.scheduler.samples), 0)
        self.assertEqual(harness.scheduler.threshold_s, 0.08)
        self.assertEqual(len(harness.events("prediction_requested")), 1)
        harness.tick(1.0)
        self.assertEqual(list(harness.scheduler.samples), [1.0])
        self.assertEqual(harness.scheduler.threshold_s, 0.2)
        self.assertEqual(harness.events("chunk_rejected")[0].data["reason"], "expired")

    def test_measured_latency_includes_collection_delay(self):
        harness = Harness((0.025,))
        harness.tick(0.0)
        harness.tick(0.04)
        estimate = harness.events("prefetch_estimate_updated")[0].data
        self.assertAlmostEqual(estimate["observed_latency_s"], 0.04)
        self.assertAlmostEqual(estimate["threshold_s"], 0.06)

    def test_hysteresis_and_rolling_window_work_with_single_sample_window(self):
        harness = Harness((0.05, 0.055, 0.09), window_size=1)
        for now in (0.0, 0.05, 0.13, 0.185):
            harness.tick(now)
        self.assertAlmostEqual(harness.scheduler.threshold_s, 0.07)
        for now in (0.26, 0.35):
            harness.tick(now)
        self.assertEqual(len(harness.scheduler.samples), 1)
        self.assertAlmostEqual(harness.scheduler.threshold_s, 0.11)

    def test_quantile_interpolates_completed_samples(self):
        harness = Harness((0.04, 0.08), window_size=2, quantile=0.5)
        for now in (0.0, 0.04, 0.14, 0.22):
            harness.tick(now)
        estimate = harness.events("prefetch_estimate_updated")[-1].data
        self.assertEqual(estimate["sample_count"], 2)
        self.assertAlmostEqual(estimate["estimated_latency_s"], 0.06)
        self.assertAlmostEqual(estimate["threshold_s"], 0.08)

    def test_estimator_recovers_after_spike_leaves_bounded_window(self):
        harness = Harness((0.26,) + (0.02,) * 100, window_size=2, quantile=1.0)
        for tick in range(100):
            harness.tick(tick * 0.02)
        self.assertEqual(
            [event.data["limited"] for event in harness.events("supply_state_changed")],
            [True, False],
        )
        self.assertEqual(len(harness.scheduler.samples), 2)
        self.assertAlmostEqual(harness.scheduler.threshold_s, 0.04)
        self.assertTrue(harness.events("prefetch_estimate_updated")[0].data["threshold_capped"])

    def test_supply_shortfall_can_occur_before_entire_chunk_expires(self):
        harness = Harness((0.12,))
        harness.tick(0.0)
        harness.tick(0.12)
        self.assertEqual(len(harness.events("chunk_accepted")), 1)
        event = harness.events("supply_state_changed")[0]
        self.assertTrue(event.data["limited"])
        self.assertAlmostEqual(event.data["supply_shortfall_s"], 0.04)

    def test_minimum_request_interval_bounds_zero_latency_requests(self):
        harness = Harness((0.0,), min_request_interval_s=0.5)
        for tick in range(50):
            harness.tick(tick * 0.02)
        requests = harness.events("prediction_requested")
        self.assertEqual([event.timestamp for event in requests], [0.0, 0.5])

    def test_invalid_adaptive_parameters_are_rejected_at_configuration_time(self):
        for settings in (
            {"adaptive_window_size": 0}, {"adaptive_window_size": 1.5},
            {"adaptive_quantile": 0.0}, {"adaptive_quantile": 1.1},
            {"adaptive_margin_s": -0.1}, {"adaptive_hysteresis_s": math.nan},
            {"adaptive_min_request_interval_s": 0}, {"prefetch_threshold_s": math.inf},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                RuntimeConfig(scheduler="adaptive_async", **settings)

    def test_fixed_latency_comparison_and_original_zero_latency_behavior(self):
        config = RuntimeConfig(duration_s=4.0, scheduler="fixed_async", latency_s=(0.1,))
        fixed = run_simulation(config)
        adaptive = run_simulation(replace(config, scheduler="adaptive_async"))
        self.assertLess(adaptive.summary["buffer_underrun_ticks"], fixed.summary["buffer_underrun_ticks"])
        self.assertEqual(adaptive.summary["buffer_underrun_ticks"], 5)
        zero = run_simulation(replace(config, scheduler="adaptive_async", latency_s=(0.0,)))
        self.assertEqual(zero.summary["buffer_underrun_ticks"], 0)

    def test_sustained_overload_remains_fallback_with_bounded_threshold(self):
        config = RuntimeConfig(duration_s=2.0, scheduler="adaptive_async")
        result = run_simulation(config, latency_trace=LatencyTrace("overload", 1.0, (0.26,)))
        self.assertEqual(result.summary["buffer_underrun_ratio"], 1.0)
        self.assertIsNone(result.summary["observation_age_s"]["p95"])
        self.assertEqual(result.summary["adaptive_prefetch"]["max_threshold_s"], 0.2)
        self.assertGreater(result.summary["chunk_rejection_counts"]["expired"], 0)

    def test_same_trace_is_reproducible_and_actions_are_never_reused(self):
        config = RuntimeConfig(scheduler="adaptive_async", latency_s=(0.04, 0.14, 0.26))
        first, second = run_simulation(config), run_simulation(config)
        self.assertEqual(first.events, second.events)
        self.assertEqual(first.summary, second.summary)
        keys = [(row.request_id, row.action_index) for row in first.records if row.request_id is not None]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(
            row.observation_age_s < config.execute_horizon * config.period_s + 1e-12
            for row in first.records if row.observation_age_s is not None
        ))


if __name__ == "__main__":
    unittest.main()
