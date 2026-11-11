"""Compatibility between the upstream experiment tools and Phase 1 releases."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from possession.external import MATCHES, paths
from possession.release import load_models, read_manifest, write_json
from possession.schema import COLLECTIVE_NAMES, FEATURE_NAMES
from possession.tracking import Tracking


class ArtifactModel:
    def __init__(self, width):
        self.width = width

    def num_feature(self):
        return self.width

    def save_model(self, filename):
        write_json(filename, {"test_feature_count": self.width})


FAKE_LIGHTGBM = SimpleNamespace(Booster=lambda model_file: ArtifactModel(
    json.loads(Path(model_file).read_text())["test_feature_count"]))


@unittest.skipUnless(importlib.util.find_spec("scipy"), "Upstream experiment tests require SciPy")
class RebaseIntegrationTests(unittest.TestCase):
    def test_experimental_export_loads_with_the_strict_inference_loader(self):
        from possession.experiments import fit_candidate

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dfl_dir, data = root / "dfl", root / "data"
            matches = MATCHES[:2]
            for match in matches:
                for path in paths(dfl_dir, match):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"synthetic fingerprint fixture")
            for name in ("BAR-ATH_tactical_atd.xml", "BAR-ATH_tactical_atd.txt"):
                path = data / "tactical" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic fingerprint fixture")
            rows = (np.zeros((4, len(FEATURE_NAMES))), np.array([0, 1, 2, 0]),
                    np.zeros((4, len(COLLECTIVE_NAMES))))
            training = {match: {"clean": rows, "correlated": rows} for match in matches}
            for collective in (False, True):
                with self.subTest(collective=collective):
                    destination = root / f"release-{collective}"
                    width = len(COLLECTIVE_NAMES if collective else FEATURE_NAMES)
                    with patch("possession.experiments.gap_lengths", return_value=np.array([5])), \
                            patch("possession.experiments.training_rows", return_value=training), \
                            patch("possession.experiments.fit", return_value=(ArtifactModel(width), ArtifactModel(27))) as fit:
                        fit_candidate(dfl_dir, data, destination, matches, "correlated", collective)
                        self.assertEqual(fit.call_count, 1)
                        with self.assertRaises(FileExistsError):
                            fit_candidate(dfl_dir, data, destination, matches, "correlated", collective)
                        self.assertEqual(fit.call_count, 1)
                    manifest = read_manifest(destination)
                    self.assertEqual(len(manifest["training"]["dfl_inputs"]), 4)
                    with patch.dict("sys.modules", {"lightgbm": FAKE_LIGHTGBM}):
                        models, penalties, saved_collective = load_models(destination)
                    self.assertEqual([model.num_feature() for model in models], [width, 27])
                    self.assertEqual(penalties, [12, 16])
                    self.assertEqual(saved_collective, collective)

    def test_historical_audit_requires_explicit_legacy_mode(self):
        from possession.short_audit import load_audit_models

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ArtifactModel(6).save_model(directory / "live.txt")
            ArtifactModel(27).save_model(directory / "team.txt")
            write_json(directory / "benchmark.json", {"models": ["live.txt", "team.txt"],
                        "collective_status": True, "penalties": None})
            with self.assertRaises(FileNotFoundError):
                load_audit_models(directory)
            with patch.dict("sys.modules", {"lightgbm": FAKE_LIGHTGBM}), \
                    self.assertWarnsRegex(RuntimeWarning, "legacy bundle"):
                models, penalties, collective = load_audit_models(directory, legacy=True)
            self.assertEqual(len(models), 2)
            self.assertIsNone(penalties)
            self.assertTrue(collective)
            # A broken modern release cannot be bypassed with the legacy option.
            write_json(directory / "manifest.json", {"format_version": -1})
            with self.assertRaisesRegex(ValueError, "Unsupported model release"):
                load_audit_models(directory, legacy=True)

    def test_short_audit_accepts_a_release_with_no_decoder(self):
        from possession.short_audit import audit_half

        truth = np.r_[np.zeros(20, dtype=np.int8), np.ones(30, dtype=np.int8)]
        tracking = Tracking(np.zeros((50, 2, 2), dtype=np.float32), np.array([0, 1]),
                            np.zeros((50, 2), dtype=np.float32), (105.3, 68.0))
        probabilities = np.eye(3)[truth]
        with patch("possession.short_audit.predict", return_value=probabilities):
            records = audit_half(tracking, truth, None, None, False)
        self.assertEqual(len(records), 2)
        for row in records:
            self.assertEqual(row["raw_matched"], row["decoded_matched"])
            self.assertEqual(row["raw_best_iou"], row["decoded_best_iou"])


if __name__ == "__main__":
    unittest.main()
