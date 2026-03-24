from __future__ import annotations

import unittest

import numpy as np

from gesture_controller.features import MODEL_FEATURE_DIM, extract_model_features


def make_landmarks() -> np.ndarray:
    rng = np.random.default_rng(7)
    coordinates = rng.normal(size=(21, 3)).astype(np.float32)
    coordinates[0] = np.array([0.25, -0.1, 0.05], dtype=np.float32)
    return coordinates


class FeatureExtractionTests(unittest.TestCase):
    def test_feature_vector_dimension_is_exact(self) -> None:
        features = extract_model_features(make_landmarks())
        self.assertEqual(features.shape, (MODEL_FEATURE_DIM,))

    def test_feature_extraction_is_deterministic(self) -> None:
        landmarks = make_landmarks()
        first = extract_model_features(landmarks)
        second = extract_model_features(landmarks)
        np.testing.assert_allclose(first, second)

    def test_feature_extraction_is_translation_and_scale_invariant(self) -> None:
        landmarks = make_landmarks()
        transformed = landmarks * 2.5 + np.array([3.0, -7.0, 1.5], dtype=np.float32)
        np.testing.assert_allclose(
            extract_model_features(landmarks),
            extract_model_features(transformed),
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
