from __future__ import annotations

import pickle
import unittest
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import Mock, patch

from robovla_runtime.backends.fluxvla import FluxVLABackend, FluxVLALiberoPreprocessor, decode_first_batch
from robovla_runtime.core import ActionSpec, Observation, ObservationSpec
from robovla_runtime.integrations import (
    FluxVLAModelInputSpec, FluxVLAProcessFactory, PayloadArraySpec, StatePayloadMapping,
)

try:
    import numpy as np
except ImportError:
    np = None


class InputSchemaTests(unittest.TestCase):
    def test_serializable_and_strict_schema(self):
        arrays = (PayloadArraySpec("camera", "camera", (2, 3, 3), "uint8"),
                  PayloadArraySpec("proprio", "state", (2,), "float32"))
        mapping = StatePayloadMapping("proprio", (2, 0))
        spec = FluxVLAModelInputSpec(arrays, (mapping,))
        self.assertEqual(pickle.loads(pickle.dumps(spec)), spec)
        cases = [
            lambda: PayloadArraySpec("camera", "camera", (3, 2), "uint8"),
            lambda: PayloadArraySpec("camera", "camera", (2, 3, 4), "uint8"),
            lambda: PayloadArraySpec("camera", "camera", (True, 2, 3), "uint8"),
            lambda: PayloadArraySpec("proprio", "state", (1, 2), "float32"),
            lambda: PayloadArraySpec("proprio", "state", (2,), "int32"),
            lambda: PayloadArraySpec("camera", "other", (2,), "uint8"),
            lambda: PayloadArraySpec("task_description", "state", (2,), "float32"),
            lambda: StatePayloadMapping("proprio", (True,)),
            lambda: StatePayloadMapping("proprio", (-1,)),
            lambda: StatePayloadMapping("proprio", (0, 0)),
            lambda: FluxVLAModelInputSpec(()),
            lambda: FluxVLAModelInputSpec((arrays[0], arrays[0])),
            lambda: FluxVLAModelInputSpec(arrays, (mapping, mapping)),
            lambda: FluxVLAModelInputSpec(arrays, (StatePayloadMapping("missing", (0,)),)),
            lambda: FluxVLAModelInputSpec(arrays, (StatePayloadMapping("camera", (0,)),)),
            lambda: FluxVLAModelInputSpec(arrays, (StatePayloadMapping("proprio", (0,)),)),
            lambda: spec.validate_state_dimension(2),
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                case()

    def test_missing_numpy_is_actionable(self):
        spec = FluxVLAModelInputSpec((PayloadArraySpec("camera", "camera", (1, 1, 3), "uint8"),))
        with patch.dict("sys.modules", {"numpy": None}), self.assertRaisesRegex(ValueError, "requires numpy"):
            spec.prepare(Observation("ep", 0, 0.0, (1.0,), {"camera": []}))


@unittest.skipIf(np is None, "numpy is optional")
class ModelInputTests(unittest.TestCase):
    def setUp(self):
        self.spec = FluxVLAModelInputSpec((
            PayloadArraySpec("camera", "camera", (2, 3, 3), "uint8"),
            PayloadArraySpec("proprio", "state", (2,), "float32"),
            PayloadArraySpec("raw_state", "state", (1,), "float64"),
        ), (StatePayloadMapping("proprio", (2, 0)),))
        self.obs = Observation("ep", 0, 0.0, (1.25, 2.5, 3.75), {
            "camera": np.zeros((2, 3, 3), dtype="uint8"),
            "raw_state": np.array([4.5], dtype="float64"),
            "metadata": {"values": [1]},
        })

    def test_mapping_reaches_dataset_and_does_not_mutate_sources(self):
        seen = []
        def dataset(raw):
            seen.append((raw["proprio"].copy(), raw["is_new_episode"]))
            raw["camera"][:] = 255
            raw["raw_state"][:] = 0
            raw["metadata"]["values"].append(2)
            return {"mapped": raw["proprio"]}
        preprocess = FluxVLALiberoPreprocessor(dataset, "task", "suite", input_spec=self.spec)
        for _ in range(2):
            batch = preprocess(self.obs)
            np.testing.assert_array_equal(batch["mapped"], [3.75, 1.25])
            self.assertEqual(batch["mapped"].dtype, np.dtype("float32"))
        preprocess.reset("ep")
        preprocess(self.obs)
        self.assertEqual([new for _, new in seen], [True, False, True])
        self.assertNotIn("proprio", self.obs.payload)
        self.assertFalse(self.obs.payload["camera"].any())
        self.assertEqual(self.obs.payload["raw_state"].tolist(), [4.5])
        self.assertEqual(self.obs.payload["metadata"], {"values": [1]})

    def test_bad_inputs_never_reach_dataset_or_advance_episode(self):
        changes = [
            {"camera": np.zeros((3, 2, 3), dtype="uint8")},
            {"camera": np.zeros((2, 3, 3), dtype="float32")},
            {"camera": [[[0, 0, 0]]]},
            {"camera": None},
            {"raw_state": np.array([1, 2], dtype="float64")},
            {"raw_state": np.array([1], dtype="float32")},
            {"raw_state": np.array([float("nan")])},
            {"raw_state": np.array([float("inf")])},
            {"raw_state": np.array(["1"])},
            {"raw_state": np.array([1], dtype=object)},
            {"proprio": np.zeros(2, dtype="float32")},
        ]
        dataset = Mock(return_value={})
        preprocess = FluxVLALiberoPreprocessor(dataset, "task", "suite", input_spec=self.spec)
        invalid = [replace(self.obs, payload={**self.obs.payload, **change}) for change in changes]
        invalid += [replace(self.obs, state=(1.0,)), replace(self.obs, state=(1e300, 2.0, 3.0)),
                    replace(self.obs, state=(float("nan"), 2.0, 3.0)), replace(self.obs, payload={})]
        for obs in invalid:
            with self.subTest(obs=obs), self.assertRaises(ValueError):
                preprocess(obs)
        dataset.assert_not_called()
        preprocess(self.obs)
        self.assertTrue(dataset.call_args.args[0]["is_new_episode"])

    def test_floating_camera_finiteness(self):
        spec = FluxVLAModelInputSpec((PayloadArraySpec("camera", "camera", (1, 1, 3), "float32"),))
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                spec.prepare(replace(self.obs, payload={"camera": np.full((1, 1, 3), value, dtype="float32")}))

    def test_warmup_and_live_inputs_fail_before_model_prediction(self):
        for steps in (0, 2):
            model = Mock()
            model.predict_action.return_value = [[[0.0]]]
            dataset = Mock(return_value={})
            def build():
                preprocess = FluxVLALiberoPreprocessor(dataset, "task", "suite", input_spec=self.spec)
                return FluxVLABackend(model, preprocess, decode_first_batch, ActionSpec(1, .02),
                                     inference_context=nullcontext)
            bad = replace(self.obs, payload={**self.obs.payload, "camera": np.zeros((1, 1, 3), dtype="uint8")})
            wrapper = FluxVLAProcessFactory(build, lambda index: bad, steps,
                                            observation_spec=ObservationSpec(3))()
            if steps:
                with self.assertRaises(ValueError):
                    wrapper.startup(lambda *args, **kwargs: None)
            else:
                wrapper.startup(lambda *args, **kwargs: None)
                with self.assertRaises(ValueError):
                    wrapper.request(bad, 0.0)
            model.predict_action.assert_not_called()
            dataset.assert_not_called()
