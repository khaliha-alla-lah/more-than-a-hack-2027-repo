"""Release integrity and input-contract regressions; no target labels or assets required."""

import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from possession.external import MATCHES, paths
from possession.features import extract
from possession.fetch_dfl import download, verify
from possession.learn import collective_matrix, matrix
from possession.release import (fingerprint, read_manifest, run_receipt, save_release,
                                write_json, load_models)
from possession.schema import COLLECTIVE_NAMES, FEATURE_NAMES
from possession.tracking import Tracking


class ModelFile:
    """Lightweight artifact double. Real LightGBM round trips are tested separately."""

    def __init__(self, width):
        self.width = width

    def num_feature(self):
        return self.width

    def save_model(self, path):
        Path(path).write_text(f"synthetic test model; features={self.width}\n", encoding="utf-8")


def example_tracking():
    rng = np.random.default_rng(12)
    players = rng.normal(size=(100, 4, 2)).astype(np.float32)
    ball = rng.normal(size=(100, 2)).astype(np.float32)
    ball[20:30] = np.nan
    return Tracking(players, np.array([0, 0, 1, 1]), ball, (105.3, 68.0))


class FeatureContractTests(unittest.TestCase):
    def test_named_contract_preserves_historical_column_order(self):
        features = extract(example_tracking())
        self.assertEqual(tuple(features), FEATURE_NAMES)
        historical = np.column_stack(list(features.values())).astype(np.float32)
        np.testing.assert_array_equal(matrix(dict(reversed(list(features.items())))), historical)
        np.testing.assert_array_equal(collective_matrix(features),
                                      np.column_stack([features[name] for name in COLLECTIVE_NAMES]).astype(np.float32))

    def test_missing_and_unexpected_features_fail(self):
        features = extract(example_tracking())
        features["unexpected"] = features.pop("ball_seen")
        with self.assertRaisesRegex(ValueError, "Feature schema mismatch"):
            matrix(features)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.destination = self.root / "release"
        self.models = [ModelFile(6), ModelFile(27)]
        self.report = {"collective_status": True, "penalties": [5., 7.], "selected": "test-only"}
        save_release(self.destination, self.models, self.report, {"purpose": "synthetic unit test"})

    def mutate_manifest(self, change):
        path = self.destination / "manifest.json"
        manifest = json.loads(path.read_text())
        change(manifest)
        write_json(path, manifest)

    def test_round_trip_and_refuse_overwrite(self):
        manifest = read_manifest(self.destination)
        self.assertEqual(manifest["feature_schema"]["all"], list(FEATURE_NAMES))
        self.assertEqual(manifest["runtime"]["packages"]["numpy"], np.__version__)
        before = fingerprint(self.destination / "manifest.json")
        with self.assertRaises(FileExistsError):
            save_release(self.destination, self.models, self.report, {})
        self.assertEqual(before, fingerprint(self.destination / "manifest.json"))

    def test_model_tampering_fails(self):
        (self.destination / "model-0.txt").write_text("damaged")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            read_manifest(self.destination)

    def test_benchmark_tampering_fails(self):
        (self.destination / "benchmark.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Benchmark checksum"):
            read_manifest(self.destination)

    def test_reordered_schema_fails(self):
        self.mutate_manifest(lambda manifest: manifest["feature_schema"]["all"].reverse())
        with self.assertRaisesRegex(ValueError, "feature schema"):
            read_manifest(self.destination)

    def test_changed_inference_source_fails(self):
        self.mutate_manifest(lambda manifest: manifest["source"]["files"].update({"possession/features.py": "wrong"}))
        with self.assertRaisesRegex(ValueError, "Inference source differs"):
            read_manifest(self.destination)

    def test_invalid_penalty_and_filename_fail(self):
        self.mutate_manifest(lambda manifest: manifest.update(penalties=[-1, 7]))
        with self.assertRaisesRegex(ValueError, "penalties"):
            read_manifest(self.destination)
        self.mutate_manifest(lambda manifest: manifest.update(penalties=None))
        self.mutate_manifest(lambda manifest: manifest["models"][0].update(file="../outside.txt"))
        with self.assertRaisesRegex(ValueError, "filenames"):
            read_manifest(self.destination)

    def test_failed_save_does_not_publish_partial_release(self):
        destination = self.root / "broken"
        with self.assertRaisesRegex(ValueError, "features"):
            save_release(destination, [ModelFile(5), ModelFile(27)], self.report, {})
        self.assertFalse(destination.exists())

    def test_inference_needs_no_external_training_data(self):
        # Freeze the artifact before disabling the training provider completely.
        fake_lightgbm = SimpleNamespace(Booster=lambda model_file: ModelFile(
            6 if Path(model_file).name == "model-0.txt" else 27))
        with patch("possession.dfl.load", side_effect=AssertionError("training data was accessed")), \
                patch.dict("sys.modules", {"lightgbm": fake_lightgbm}):
            models, penalties, collective = load_models(self.destination)
        self.assertEqual(len(models), 2)
        self.assertEqual(penalties, [5, 7])
        self.assertTrue(collective)

    def test_receipt_binds_output_and_inputs(self):
        output = self.root / "sample.csv"
        output.write_bytes(b"frame,label\n0,H\n")
        receipt = run_receipt(output, {"metadata": output}, mode="rules", penalties=(5, 7), rows=1)
        self.assertEqual(receipt["output"]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertNotIn("model_manifest", receipt)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "sample.xml"
        self.body = b"<root>test input</root>\n"
        self.record = {"size": len(self.body), "lfs": {"oid": hashlib.sha256(self.body).hexdigest()}}

    def test_verified_download_and_existing_file(self):
        download(self.path, self.record, opener=lambda *args, **kwargs: io.BytesIO(self.body))
        verify(self.path, self.record)
        download(self.path, self.record, opener=lambda *args, **kwargs: self.fail("redownloaded verified input"))
        blob = hashlib.sha1(f"blob {len(self.body)}\0".encode() + self.body).hexdigest()
        verify(self.path, {"size": len(self.body), "oid": blob})

    def test_bad_download_is_not_published(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            download(self.path, self.record, opener=lambda *args, **kwargs: io.BytesIO(b"short"))
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_invalid_existing_file_is_preserved(self):
        self.path.write_bytes(b"invalid")
        with self.assertRaises(ValueError):
            download(self.path, self.record)
        self.assertEqual(self.path.read_bytes(), b"invalid")

    def test_expected_input_set(self):
        names = [path.name for match in MATCHES for path in paths("unused", match)]
        self.assertEqual(len(set(names)), 14)


class LightGBMIntegrationTests(unittest.TestCase):
    def test_real_models_round_trip(self):
        try:
            import lightgbm as lgb
        except (ImportError, OSError):
            self.skipTest("LightGBM unavailable; real model loading is not verified in this environment")
        from possession.learn import predict
        from possession.release import load_models

        rng = np.random.default_rng(41)
        x = rng.normal(size=(100, len(FEATURE_NAMES))).astype(np.float32)
        status = x[:, :len(COLLECTIVE_NAMES)]
        params = dict(objective="binary", verbosity=-1, num_threads=1, min_data_in_leaf=3)
        models = [lgb.train(params, lgb.Dataset(values, label=values[:, 0] > 0), num_boost_round=3)
                  for values in (status, x)]
        expected = predict(models, x, status)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "release"
            save_release(path, models, {"collective_status": True, "penalties": [5, 7], "selected": "test"}, {})
            restored, _, _ = load_models(path)
            np.testing.assert_array_equal(predict(restored, x, status), expected)


class SubmissionTests(unittest.TestCase):
    def test_half_masks_lf_encoding_and_repeatability(self):
        from possession.__main__ import generate

        with TemporaryDirectory() as temporary, \
                patch("possession.__main__.half_video_blocks", return_value=[(1, 0, 49), (2, 60, 99)]), \
                patch("possession.__main__.KICKOFF_VIDEO_FRAMES", {"2nd Half": 70}):
            first, second = Path(temporary) / "first.csv", Path(temporary) / "second.csv"
            for output in (first, second):
                generate(example_tracking(), output)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertNotIn(b"\r", first.read_bytes())
            rows = first.read_text().splitlines()
            self.assertEqual(rows[0], "frame,label")
            self.assertEqual(len(rows), 101)
            self.assertTrue(all(rows[frame + 1] == f"{frame},X" for frame in range(50, 60)))
            self.assertTrue(all(rows[frame + 1] == f"{frame},D" for frame in range(60, 70)))

    def test_validation_failure_preserves_existing_submission(self):
        from possession.__main__ import generate

        with TemporaryDirectory() as temporary, \
                patch("possession.__main__.half_video_blocks", return_value=[(1, 0, 99)]), \
                patch("possession.__main__.pm.validate_submission", return_value="invalid fixture"):
            output = Path(temporary) / "existing.csv"
            output.write_bytes(b"preserved")
            with self.assertRaisesRegex(ValueError, "invalid fixture"):
                generate(example_tracking(), output)
            self.assertEqual(output.read_bytes(), b"preserved")
            self.assertEqual(list(Path(temporary).iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
