from __future__ import annotations

import unittest

from robovla_runtime.runtime import RuntimeConfig, run_simulation


class RuntimeScenarioTests(unittest.TestCase):
    def config(self, scheduler: str, latency: float, threshold: float = 0.08):
        return RuntimeConfig(
            scheduler=scheduler,
            period_s=0.02,
            duration_s=2.0,
            chunk_size=10,
            execute_horizon=10,
            latency_s=(latency,),
            prefetch_threshold_s=threshold,
            initial_position=(0.0,),
            target_position=(1.0,),
            max_velocity=2.0,
            seed=7,
        )

    def test_zero_latency_has_no_underrun(self) -> None:
        result = run_simulation(self.config("sync", 0.0))
        self.assertEqual(result.summary["buffer_underrun_ticks"], 0)

    def test_fixed_async_reduces_fallback_for_coverable_latency(self) -> None:
        sync = run_simulation(self.config("sync", 0.05))
        async_result = run_simulation(
            self.config("fixed_async", 0.05, threshold=0.08)
        )
        self.assertLess(
            async_result.summary["buffer_underrun_ticks"],
            sync.summary["buffer_underrun_ticks"],
        )

    def test_latency_longer_than_chunk_coverage_causes_underrun(self) -> None:
        result = run_simulation(
            self.config("fixed_async", 0.25, threshold=0.18)
        )
        self.assertGreater(result.summary["buffer_underrun_ticks"], 0)

    def test_same_configuration_is_deterministic(self) -> None:
        config = self.config("fixed_async", 0.05)
        first = run_simulation(config)
        second = run_simulation(config)
        self.assertEqual(first.events, second.events)
        self.assertEqual(first.summary, second.summary)


if __name__ == "__main__":
    unittest.main()
