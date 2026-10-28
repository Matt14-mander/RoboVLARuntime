from __future__ import annotations

from dataclasses import replace
import unittest

from robovla_runtime.backends.base import PendingPrediction
from robovla_runtime.backends.mock import MockBackend
from robovla_runtime.backends.threaded import SubmissionReceipt
from robovla_runtime.core.types import ActionChunk, ActionSpec, Observation
from robovla_runtime.environments import ToyJoint
from robovla_runtime.runtime import RuntimeConfig, run_realtime_runtime, run_runtime


class ExternalEnvironment:
    """Independent adapter: state has two values; commands have one value."""

    def __init__(self, spec=None):
        self.action_spec = spec or ActionSpec(1, 0.02)
        self.reset_calls = []
        self.commands = []
        self.elapsed = 0.0
        self.episode_id = ""

    @property
    def state(self):
        return (self.elapsed, 99.0)

    def reset(self, episode_id):
        self.reset_calls.append(episode_id)
        self.episode_id = episode_id
        self.commands.clear()
        self.elapsed = 0.0

    def observe(self, observation_id, captured_at):
        return Observation(
            self.episode_id, observation_id, captured_at, self.state,
            {"camera": "snapshot", "elapsed": self.elapsed},
        )

    def fallback_command(self, reason):
        return (0.25,)

    def apply_action(self, command):
        self.commands.append(command)

    def advance(self, period_s):
        self.elapsed += period_s
        return self.state

    def metrics(self):
        return {"elapsed_s": self.elapsed}


class RecordingPolicy:
    def __init__(self, spec=None, latency=0.0):
        self.action_spec = spec or ActionSpec(1, 0.02)
        self.latency = latency
        self.observations = []
        self.reset_calls = []
        self.next_id = 0

    def reset(self, episode_id):
        self.reset_calls.append(episode_id)

    def request(self, observation, requested_at):
        self.observations.append(observation)
        request_id = self.next_id
        self.next_id += 1
        ready_at = requested_at + self.latency
        return PendingPrediction(request_id, requested_at, ready_at, ActionChunk(
            observation.episode_id, request_id, observation.observation_id,
            observation.captured_at, ready_at, observation.captured_at,
            self.action_spec, ((0.75,),) * 10,
            episode_generation=observation.episode_generation,
        ))


class NeverReadyPolicy:
    def __init__(self):
        self.action_spec = ActionSpec(1, 0.02)
        self.observations = []
        self.resets = []
        self.cancelled = []
        self.closed = False

    def reset(self, episode_id):
        self.resets.append(episode_id)

    def submit(self, observation, requested_at):
        self.observations.append(observation)
        return SubmissionReceipt(len(self.observations))

    def poll(self):
        return ()

    def cancel_episode(self, episode_id):
        self.cancelled.append(episode_id)
        return ()

    def close(self, timeout_s=None):
        self.closed = True
        return True


class EnvironmentAdapterTests(unittest.TestCase):
    def test_external_environment_supplies_observation_execution_and_metrics(self):
        env = ExternalEnvironment()
        policy = RecordingPolicy()
        result = run_runtime(RuntimeConfig(duration_s=0.08), policy, environment=env)
        self.assertEqual(len(env.commands), 4)
        self.assertEqual(env.commands, [(0.75,)] * 4)
        self.assertEqual(policy.observations[0].state, (0.0, 99.0))
        self.assertEqual(policy.observations[0].payload["camera"], "snapshot")
        self.assertEqual(result.summary["environment_metrics"], {"elapsed_s": 0.08})
        self.assertIsNone(result.summary["final_tracking_error"])

    def test_fallback_ticks_advance_environment_while_prediction_is_pending(self):
        for scheduler in ("sync", "fixed_async"):
            with self.subTest(scheduler=scheduler):
                env = ExternalEnvironment()
                result = run_runtime(
                    RuntimeConfig(duration_s=0.08, scheduler=scheduler),
                    RecordingPolicy(latency=1.0), environment=env,
                )
                self.assertEqual(env.commands, [(0.25,)] * 4)
                self.assertEqual(env.elapsed, 0.08)
                self.assertEqual(result.summary["buffer_underrun_ticks"], 4)

    def test_wall_clock_loop_uses_external_environment_during_pending_inference(self):
        env = ExternalEnvironment()
        backend = NeverReadyPolicy()
        clock = [0.0]

        def sleep(duration):
            clock[0] += duration

        result = run_realtime_runtime(
            RuntimeConfig(scheduler="worker_async", duration_s=0.08), backend,
            environment=env, timer=lambda: clock[0], sleeper=sleep,
        )
        self.assertEqual(env.commands, [(0.25,)] * 4)
        self.assertEqual(env.elapsed, 0.08)
        self.assertEqual(result.summary["deadline_miss_count"], 0)
        self.assertEqual(backend.resets, backend.cancelled)
        self.assertTrue(backend.closed)

    def test_same_episode_run_resets_environment_and_backend(self):
        env = ExternalEnvironment()
        policy = RecordingPolicy()
        config = RuntimeConfig(duration_s=0.08)
        first = run_runtime(config, policy, environment=env)
        second = run_runtime(config, policy, environment=env)
        self.assertEqual(first.summary, second.summary)
        self.assertEqual(env.reset_calls, ["episode-seed-7"] * 2)
        self.assertEqual(policy.reset_calls, env.reset_calls)
        self.assertEqual([obs.state for obs in policy.observations], [(0.0, 99.0)] * 2)

    def test_mismatched_action_contract_is_rejected_before_reset_or_send(self):
        base = ActionSpec(1, 0.02)
        mismatches = [
            replace(base, dimension=2), replace(base, period_s=0.1),
            replace(base, unit="radian"), replace(base, normalization="unnormalized"),
            replace(base, component_units=("radian",)),
            replace(base, coordinate_frame="base"),
            replace(base, rotation_representation="axis_angle"),
            replace(base, gripper_semantics="binary"),
        ]
        for spec in mismatches:
            with self.subTest(spec=spec):
                env = ExternalEnvironment()
                policy = RecordingPolicy(spec)
                with self.assertRaisesRegex(ValueError, "ActionSpec"):
                    run_runtime(RuntimeConfig(), policy, environment=env)
                self.assertEqual(env.commands, [])
                self.assertEqual(env.reset_calls, [])
                self.assertEqual(policy.reset_calls, [])

    def test_matching_delta_contract_is_still_not_executable(self):
        spec = ActionSpec(1, 0.02, representation="delta_position")
        env = ExternalEnvironment(spec)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            run_runtime(RuntimeConfig(), RecordingPolicy(spec), environment=env)
        self.assertEqual(env.reset_calls, [])

    def test_invalid_fallback_is_rejected_before_action_submission(self):
        env = ExternalEnvironment()
        env.fallback_command = lambda reason: (float("nan"),)
        with self.assertRaisesRegex(ValueError, "finite"):
            run_runtime(RuntimeConfig(), RecordingPolicy(latency=1.0), environment=env)
        self.assertEqual(env.commands, [])

    def test_toy_reset_clears_position_and_pending_command(self):
        toy = ToyJoint((0.0,), (1.0,), 2.0)
        toy.apply_action((1.0,))
        toy.advance(0.1)
        toy.reset("same")
        self.assertEqual(toy.state, (0.0,))
        self.assertEqual(toy.advance(0.1), (0.0,))
        self.assertEqual(toy.observe(0, 0.0).episode_id, "same")

    def test_mock_reset_restarts_latency_trace_but_preserves_request_identity(self):
        backend = MockBackend(ActionSpec(1, 0.02), 10, (0.0, 0.1), (1.0,))
        observation = Observation("same", 0, 0.0, (0.0,))
        first = backend.request(observation, 0.0)
        backend.reset("same")
        second = backend.request(observation, 0.0)
        self.assertEqual(first.ready_at, second.ready_at)
        self.assertNotEqual(first.request_id, second.request_id)


if __name__ == "__main__":
    unittest.main()
