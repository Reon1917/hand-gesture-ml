from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asl_app.modeling import normalize_probabilities, predict_with_bundle


class BrokenProbabilityModel:
    classes_ = np.asarray(["A", "B", "C"])

    def predict_proba(self, _vector: np.ndarray) -> np.ndarray:
        return np.asarray([[np.nan, np.nan, np.nan]], dtype=np.float64)

    def decision_function(self, _vector: np.ndarray) -> np.ndarray:
        return np.asarray([[0.1, 3.2, -1.7]], dtype=np.float64)


class ModelingTests(unittest.TestCase):
    def test_normalize_probabilities_rejects_nan_rows(self) -> None:
        self.assertIsNone(normalize_probabilities(np.asarray([np.nan, np.nan], dtype=np.float64)))

    def test_predict_with_bundle_falls_back_to_decision_scores(self) -> None:
        bundle = {"model": BrokenProbabilityModel()}
        image = Image.new("RGB", (64, 64), color="black")
        prediction = predict_with_bundle(bundle, image)

        self.assertEqual(prediction["label"], "B")
        self.assertGreater(prediction["confidence"], 0.5)
        self.assertTrue(all(np.isfinite(item["confidence"]) for item in prediction["top_predictions"]))


if __name__ == "__main__":
    unittest.main()
