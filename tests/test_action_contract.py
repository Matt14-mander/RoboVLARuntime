from dataclasses import replace
import unittest

from robovla_runtime.backends.base import PendingPrediction
from robovla_runtime.core.types import ActionSpec, Observation
from robovla_runtime.integrations.offline_inference import _validate_result
from robovla_runtime.execution.action_buffer import ActionBuffer
from tests.test_action_buffer import make_chunk


class ActionContractTests(unittest.TestCase):
    def test_observation_timestamp_must_be_finite_and_nonnegative(self):
        observation = Observation("ep", 0, 0.0, (0.0,))
        for value in (float("nan"), float("inf"), -float("inf"), -0.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(observation, captured_at=value)
        self.assertEqual(replace(observation, captured_at=0.5).captured_at, 0.5)

    def test_chunk_timestamps_must_be_finite_and_nonnegative(self):
        for field in ("source_observation_time", "ready_at", "start_at"):
            for value in (float("nan"), float("inf"), -float("inf"), -0.01):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(make_chunk(), **{field: value})

    def test_chunk_rejects_overflowing_end_timestamp(self):
        chunk = make_chunk(count=2)
        with self.assertRaises(ValueError):
            replace(chunk, spec=replace(chunk.spec, period_s=1e308))

    def test_prediction_timestamps_must_be_finite_and_nonnegative(self):
        prediction = PendingPrediction(1, 0.0, 0.0, make_chunk())
        for field in ("requested_at", "ready_at"):
            for value in (float("nan"), float("inf"), -float("inf"), -0.01):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(prediction, **{field: value})

    def test_offline_result_validation_rechecks_chunk_timestamps(self):
        chunk = make_chunk()
        observation = Observation(chunk.episode_id, chunk.source_observation_id, 0.0, (0.0,))
        prediction = PendingPrediction(chunk.request_id, 0.0, 0.0, chunk)
        # Simulate a deserialized backend result that bypasses __post_init__.
        object.__setattr__(chunk, "start_at", float("nan"))
        with self.assertRaises(ValueError):
            _validate_result(prediction, chunk.request_id, observation, 0.0, 0.1, chunk.spec)

    def test_cartesian_delta_can_be_described_but_not_time_aligned(self):
        spec = ActionSpec(
            7, 0.1, representation="delta_position", unit="controller_action",
            action_space="cartesian", coordinate_frame="base",
            rotation_representation="axis_angle", normalization="unnormalized",
            component_units=("m", "m", "m", "rad", "rad", "rad", "binary"),
            gripper_semantics="minus_one_open_plus_one_closed",
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            spec.require_time_aligned_execution()

    def test_unknown_contract_fields_and_nonfinite_commands_are_rejected(self):
        spec = ActionSpec(1, 0.1)
        for kwargs in (
            {"representation": "unknown"}, {"action_space": "unknown"},
            {"normalization": "unknown"}, {"coordinate_frame": ""},
            {"rotation_representation": "unknown"}, {"component_units": ("m", "s")},
            {"period_s": float("nan")}, {"period_s": float("inf")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(spec, **kwargs)
        for command in ((float("nan"),), (float("inf"),), (1.0, 2.0)):
            with self.subTest(command=command), self.assertRaises(ValueError):
                spec.validate_command(command)

    def test_reset_rejects_old_generation_even_with_same_episode_id(self):
        buffer = ActionBuffer("episode-a")
        old = make_chunk()
        buffer.accept(old, 0.0)
        buffer.pop(0.0)
        buffer.reset("episode-a")
        self.assertIsNone(buffer.pop(0.0))
        self.assertEqual(buffer.accept(old, 0.0).reason, "wrong_generation")
        new = replace(old, episode_generation=buffer.episode_generation)
        self.assertTrue(buffer.accept(new, 0.0).accepted)
        self.assertIsNotNone(buffer.pop(0.0))

    def test_buffer_rejects_unsupported_or_mismatched_chunk_without_replacing_tail(self):
        chunk = make_chunk()
        buffer = ActionBuffer("episode-a", action_spec=chunk.spec)
        self.assertTrue(buffer.accept(chunk, 0.0).accepted)
        delta = replace(chunk, request_id=2, spec=replace(chunk.spec, representation="delta_position"))
        self.assertEqual(buffer.accept(delta, 0.1).reason, "unsupported_action_spec")
        wrong_unit = replace(chunk, request_id=3, spec=replace(chunk.spec, unit="radian"))
        self.assertEqual(buffer.accept(wrong_unit, 0.1).reason, "action_spec_mismatch")
        self.assertEqual(buffer.pop(0.1).request_id, chunk.request_id)


if __name__ == "__main__":
    unittest.main()
