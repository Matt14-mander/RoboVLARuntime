from dataclasses import replace
import pickle
import unittest

from robovla_runtime.core import Observation, ObservationSpec


class ObservationContractTests(unittest.TestCase):
    def test_contract_is_picklable_and_does_not_require_an_action_spec(self):
        spec = ObservationSpec(3, ["camera"])
        restored = pickle.loads(pickle.dumps(spec))
        self.assertEqual(restored, spec)
        self.assertEqual(spec.required_payload_keys, ("camera",))
        restored.validate(Observation("ep", 0, 0.0, (-1.0, 0.0, 1.0), {"camera": []}))

    def test_invalid_dimensions_keys_and_states_are_rejected(self):
        for dimension in (True, False, 0, -1, 2.5, "3"):
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                ObservationSpec(dimension)
        for keys in ("camera", [""], [" "], [1], ["camera", "camera"]):
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                ObservationSpec(3, keys)
        spec = ObservationSpec(3, ("camera",))
        observation = Observation("ep", 0, 0.0, (0.0, 1.0, 2.0), {"camera": []})
        for state in ((0.0,), (True, 0.0, 1.0), ("0", 0.0, 1.0),
                      (float("nan"), 0.0, 1.0), (float("inf"), 0.0, 1.0)):
            with self.subTest(state=state), self.assertRaises(ValueError):
                spec.validate(replace(observation, state=state))
        for payload in (None, {}, []):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                spec.validate(replace(observation, payload=payload))


if __name__ == "__main__":
    unittest.main()
