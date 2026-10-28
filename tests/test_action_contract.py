from dataclasses import replace
import unittest

from robovla_runtime.core.types import ActionSpec
from robovla_runtime.execution.action_buffer import ActionBuffer
from tests.test_action_buffer import make_chunk


class ActionContractTests(unittest.TestCase):
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
