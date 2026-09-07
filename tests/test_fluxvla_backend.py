from __future__ import annotations

import unittest
from contextlib import nullcontext

from robovla_runtime.backends.fluxvla import (
    FluxVLABackend,
    FluxVLALiberoPreprocessor,
    decode_first_batch,
    make_libero_action_decoder,
)
from robovla_runtime.core.types import ActionSpec, Observation
from robovla_runtime.runtime import RuntimeConfig, run_runtime


class FakeVLA:
    def __init__(self, actions=None) -> None:
        self.actions = actions or [[[0.25], [0.5], [0.75], [1.0]]]
        self.calls = []

    def predict_action(self, **batch):
        self.calls.append(batch)
        return self.actions


class SequenceTimer:
    def __init__(self, values) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class FluxVLABackendTests(unittest.TestCase):
    def test_request_builds_timed_chunk_and_forwards_batch(self) -> None:
        vla = FakeVLA()
        synchronizations = []
        backend = FluxVLABackend(
            vla,
            preprocess=lambda obs: {"states": obs.state, "token": 9},
            decode_actions=lambda raw: decode_first_batch(raw, 1),
            action_spec=ActionSpec(1, 0.02),
            timer=SequenceTimer((10.0, 10.1, 10.4, 10.5)),
            synchronize=lambda: synchronizations.append(True),
            inference_context=nullcontext,
        )
        observation = Observation("episode", 3, 1.0, (0.1,))

        pending = backend.request(observation, requested_at=2.0)

        self.assertEqual(vla.calls, [{"states": (0.1,), "token": 9}])
        self.assertEqual(len(synchronizations), 3)
        self.assertAlmostEqual(pending.ready_at, 2.5)
        self.assertEqual(pending.chunk.source_observation_id, 3)
        self.assertEqual(pending.chunk.actions[2], (0.75,))
        self.assertEqual(
            dict(pending.timings_s),
            {
                "preprocess": 0.09999999999999964,
                "inference": 0.3000000000000007,
                "postprocess": 0.09999999999999964,
                "total": 0.5,
            },
        )

    def test_libero_preprocessor_tracks_episode_reset(self) -> None:
        inputs = []

        def dataset(raw):
            inputs.append(raw)
            return {"images": "batch"}, "replay"

        preprocess = FluxVLALiberoPreprocessor(
            dataset,
            "pick up the block",
            "libero_spatial",
            {"num_inference_steps": 5, "seed": 42},
        )
        first = Observation("a", 0, 0.0, (0.0,), {"camera": "frame-0"})
        second = Observation("a", 1, 0.1, (0.0,), {"camera": "frame-1"})
        new_episode = Observation(
            "b", 0, 0.2, (0.0,), {"camera": "frame-2"}
        )

        self.assertEqual(preprocess(first)["unnorm_key"], "libero_spatial")
        preprocess(second)
        preprocess(new_episode)

        self.assertTrue(inputs[0]["is_new_episode"])
        self.assertFalse(inputs[1]["is_new_episode"])
        self.assertTrue(inputs[2]["is_new_episode"])
        self.assertEqual(inputs[0]["task_description"], "pick up the block")
        self.assertEqual(preprocess(first)["num_inference_steps"], 5)
        self.assertEqual(preprocess(first)["seed"], 42)

    def test_libero_decoder_matches_per_action_postprocessing(self) -> None:
        received = []

        def denormalize(data):
            received.append((data["norm_stats_key"], data["task_suite_name"]))
            return [data["action"][0] * 2, data["action"][1] * 2]

        decode = make_libero_action_decoder(
            denormalize,
            "libero_spatial_no_noops",
            execute_horizon=2,
            task_suite_name="libero_spatial",
        )
        result = decode([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

        self.assertEqual(result, ((2.0, 4.0), (6.0, 8.0)))
        self.assertEqual(
            received,
            [("libero_spatial_no_noops", "libero_spatial")] * 2,
        )

    def test_libero_decoder_treats_rank_two_as_batched_single_action(self) -> None:
        decode = make_libero_action_decoder(
            lambda data: data["action"], "stats", execute_horizon=10
        )

        result = decode([[1.0, 2.0], [99.0, 100.0]])

        self.assertEqual(result, ((1.0, 2.0),))

    def test_runtime_accepts_fluxvla_backend_instead_of_mock(self) -> None:
        config = RuntimeConfig(
            scheduler="sync",
            period_s=0.02,
            duration_s=0.2,
            chunk_size=4,
            execute_horizon=4,
            latency_s=(0.0,),
            action_unit="libero_action",
        )
        backend = FluxVLABackend(
            FakeVLA(),
            preprocess=lambda obs: {"states": obs.state},
            decode_actions=lambda raw: decode_first_batch(raw, 1),
            action_spec=ActionSpec(1, 0.02, unit="libero_action"),
            timer=lambda: 0.0,
            inference_context=nullcontext,
        )

        result = run_runtime(config, backend)

        self.assertEqual(result.summary["buffer_underrun_ticks"], 0)
        self.assertGreater(len(backend.vla.calls), 0)
        stages = result.summary["prediction_stage_latency_s"]
        self.assertIn("inference", stages)

    def test_runtime_rejects_backend_action_semantics_mismatch(self) -> None:
        config = RuntimeConfig(duration_s=0.02, action_unit="libero_action")
        backend = FluxVLABackend(
            FakeVLA(),
            preprocess=lambda obs: {},
            decode_actions=lambda raw: decode_first_batch(raw, 1),
            action_spec=ActionSpec(1, 0.02, unit="normalized"),
            timer=lambda: 0.0,
            inference_context=nullcontext,
        )

        with self.assertRaisesRegex(ValueError, "ActionSpec"):
            run_runtime(config, backend)

    def test_rejects_decoder_dimension_mismatch(self) -> None:
        backend = FluxVLABackend(
            FakeVLA(actions=[[[1.0, 2.0]]]),
            preprocess=lambda obs: {},
            decode_actions=decode_first_batch,
            action_spec=ActionSpec(1, 0.02),
            timer=lambda: 0.0,
            inference_context=nullcontext,
        )
        with self.assertRaisesRegex(ValueError, "ActionSpec.dimension"):
            backend.request(Observation("episode", 0, 0.0, (0.0,)), 0.0)


if __name__ == "__main__":
    unittest.main()
