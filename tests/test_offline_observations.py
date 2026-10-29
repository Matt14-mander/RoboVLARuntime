from __future__ import annotations

import copy
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path

from robovla_runtime.integrations.offline_observations import load_observations, read_json

try:
    import numpy as np
except ImportError:
    np = None


class OfflineObservationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "samples.json"
        self.raw = {"version": 1, "samples": [{"sample_id": "one", "episode_id": "ep",
            "captured_at": 7.5, "state": [0.0, 1.0], "payload": {"camera": [[1, 2], [3, 4]]}}]}

    def load(self, raw=None):
        self.path.write_text(json.dumps(self.raw if raw is None else raw), encoding="utf-8")
        return load_observations(self.path, state_dimension=2, required_payload_keys=("camera",))

    def test_json_samples_are_isolated_picklable_and_keep_source_metadata(self):
        loaded = self.load()
        observation = loaded.observation(0, observation_id=3, captured_at=0.2, generation=2)
        observation.payload["camera"][0][0] = 99
        self.assertEqual(loaded(0).payload["camera"][0][0], 1)
        self.assertEqual(loaded.samples[0].captured_at, 7.5)
        self.assertEqual(observation.episode_generation, 2)
        self.assertEqual(pickle.loads(pickle.dumps(loaded))(0).state, (0.0, 1.0))
        self.assertEqual(len(loaded.provenance[0]["sha256"]), 64)
        self.path.write_text("changed after preflight", encoding="utf-8")
        self.assertEqual(loaded(0).payload["camera"][0][0], 1)

    def test_inferred_state_contract_rejects_inconsistent_samples(self):
        self.raw["samples"][0]["state"] = [0.0, 1.0, 2.0]
        self.path.write_text(json.dumps(self.raw), encoding="utf-8")
        loaded = load_observations(self.path, required_payload_keys=("camera",))
        self.assertEqual(loaded.observation_spec.state_dimension, 3)
        self.assertEqual(loaded(0).state, (0.0, 1.0, 2.0))
        second = copy.deepcopy(self.raw["samples"][0])
        second.update(sample_id="two", state=[0.0, 1.0])
        self.raw["samples"].append(second)
        self.path.write_text(json.dumps(self.raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, r"samples\[1\].state must have 3"):
            load_observations(self.path)

    def test_invalid_explicit_state_dimensions_fail_even_for_valid_manifest(self):
        self.path.write_text(json.dumps(self.raw), encoding="utf-8")
        for dimension in (True, 0, -1, 1.5, "2", 3):
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                load_observations(self.path, state_dimension=dimension)

    def test_rejects_bad_schema_duplicates_dimensions_nonfinite_and_missing_payload(self):
        changes = [
            lambda r: r.update(version=2), lambda r: r.update(samples=[]),
            lambda r: r["samples"].append(copy.deepcopy(r["samples"][0])),
            lambda r: r["samples"][0].update(state=[0.0]),
            lambda r: r["samples"][0].update(state=[True, 0.0]),
            lambda r: r["samples"][0].update(captured_at=float("nan")),
            lambda r: r["samples"][0].update(payload={}),
            lambda r: r["samples"][0].update(typo=1),
            lambda r: r["samples"][0]["payload"].update(camera=float("inf")),
        ]
        for change in changes:
            raw = copy.deepcopy(self.raw)
            change(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.load(raw)
        self.path.write_text('{"version":1,"version":1}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_json(self.path)

    def test_array_format_and_paths_are_checked_without_numpy(self):
        for descriptor in ({"path": "image.pkl", "shape": [2], "dtype": "float32"},
                           {"path": "missing.npy", "shape": [2], "dtype": "float32"},
                           {"path": "image.npz", "shape": [2], "dtype": "float32"}):
            raw = copy.deepcopy(self.raw)
            raw["samples"][0]["arrays"] = {"other": descriptor}
            with self.subTest(descriptor=descriptor), self.assertRaises(ValueError):
                self.load(raw)

    @unittest.skipIf(np is None, "optional numpy is not installed")
    def test_npy_npz_relative_paths_dtype_shape_and_ownership(self):
        camera = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        np.save(self.root / "camera.npy", camera)
        np.savez(self.root / "state.npz", joints=np.array([0.1, 0.2], dtype=np.float32))
        self.raw["samples"][0]["payload"] = {}
        self.raw["samples"][0]["arrays"] = {
            "camera": {"path": "camera.npy", "shape": [2, 2, 3], "dtype": "uint8"},
            "joints": {"path": "state.npz", "key": "joints", "shape": [2], "dtype": "float32"},
        }
        loaded = self.load()
        self.assertEqual(len(loaded.provenance), 3)
        np.testing.assert_array_equal(loaded(0).payload["camera"], camera)
        changed = loaded(0)
        changed.payload["camera"][0, 0, 0] = 99
        self.assertEqual(loaded(0).payload["camera"][0, 0, 0], 0)

    @unittest.skipIf(np is None, "optional numpy is not installed")
    def test_empty_and_truncated_npy_raise_contextual_validation_error(self):
        stream = io.BytesIO()
        np.save(stream, np.zeros((2, 2, 3), dtype=np.uint8))
        complete = stream.getvalue()
        self.raw["samples"][0]["payload"] = {}
        self.raw["samples"][0]["arrays"] = {
            "camera": {"path": "camera.npy", "shape": [2, 2, 3], "dtype": "uint8"},
        }
        for name, data in (("empty", b""), ("truncated_header", complete[:12]),
                           ("truncated_data", complete[:-1])):
            with self.subTest(case=name):
                (self.root / "camera.npy").write_bytes(data)
                with self.assertRaisesRegex(ValueError, "invalid array camera in") as caught:
                    self.load()
                self.assertIn(str(self.root / "camera.npy"), str(caught.exception))
                self.assertIsNotNone(caught.exception.__cause__)

    @unittest.skipIf(np is None, "optional numpy is not installed")
    def test_arrays_reject_objects_nonfinite_wrong_shape_dtype_and_missing_archive_key(self):
        for value, shape, dtype in (
            (np.array([{"unsafe": "object"}], dtype=object), [1], "object"),
            (np.array([float("inf")]), [1], "float64"),
            (np.zeros((2, 2), dtype=np.uint8), [4], "uint8"),
            (np.zeros(2, dtype=np.uint8), [2], "float32"),
        ):
            np.save(self.root / "camera.npy", value)
            raw = copy.deepcopy(self.raw)
            raw["samples"][0]["payload"] = {}
            raw["samples"][0]["arrays"] = {"camera": {"path": "camera.npy", "shape": shape, "dtype": dtype}}
            with self.subTest(dtype=dtype, shape=shape), self.assertRaises(ValueError):
                self.load(raw)
        np.savez(self.root / "camera.npz", camera=np.zeros(2))
        raw["samples"][0]["arrays"] = {"camera": {"path": "camera.npz", "key": "missing", "shape": [2], "dtype": "float64"}}
        with self.assertRaises(ValueError):
            self.load(raw)
        (self.root / "disguised.npy").write_bytes((self.root / "camera.npz").read_bytes())
        raw["samples"][0]["arrays"] = {"camera": {"path": "disguised.npy", "shape": [2], "dtype": "float64"}}
        with self.assertRaisesRegex(ValueError, "archive"):
            self.load(raw)


if __name__ == "__main__":
    unittest.main()
