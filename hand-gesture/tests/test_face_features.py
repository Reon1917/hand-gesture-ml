from __future__ import annotations

import unittest

import numpy as np

from gesture_controller.face_features import (
    FACE_MODEL_FEATURE_DIM,
    analyze_face_landmarks,
    extract_face_features,
)


def make_face_landmarks() -> np.ndarray:
    points = np.zeros((478, 3), dtype=np.float32)
    points[10] = [0.0, -0.45, 0.0]
    points[1] = [0.0, -0.02, 0.0]
    points[152] = [0.0, 0.42, 0.0]
    points[234] = [-0.42, 0.03, 0.0]
    points[454] = [0.42, 0.03, 0.0]
    points[33] = [-0.18, -0.10, 0.0]
    points[133] = [-0.08, -0.10, 0.0]
    points[159] = [-0.12, -0.13, 0.0]
    points[145] = [-0.12, -0.07, 0.0]
    points[362] = [0.08, -0.10, 0.0]
    points[263] = [0.18, -0.10, 0.0]
    points[386] = [0.12, -0.13, 0.0]
    points[374] = [0.12, -0.07, 0.0]
    points[105] = [-0.13, -0.20, 0.0]
    points[334] = [0.13, -0.20, 0.0]
    points[61] = [-0.11, 0.14, 0.0]
    points[291] = [0.11, 0.14, 0.0]
    points[13] = [0.0, 0.11, 0.0]
    points[14] = [0.0, 0.17, 0.0]
    return points


class FaceFeatureTests(unittest.TestCase):
    def test_feature_vector_dimension_is_exact(self) -> None:
        features = extract_face_features(
            make_face_landmarks(),
            {"mouthSmileLeft": 0.82, "mouthSmileRight": 0.84, "jawOpen": 0.1},
            np.eye(4, dtype=np.float32),
        )
        self.assertEqual(features.shape, (FACE_MODEL_FEATURE_DIM,))

    def test_happy_expression_is_inferred(self) -> None:
        analytics = analyze_face_landmarks(
            make_face_landmarks(),
            {"mouthSmileLeft": 0.9, "mouthSmileRight": 0.88, "eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.04},
            np.eye(4, dtype=np.float32),
        )
        self.assertIsNotNone(analytics)
        assert analytics is not None
        self.assertEqual(analytics.expression, "happy")
        self.assertEqual(analytics.attention, "focused")
        self.assertGreater(analytics.expression_confidence, 0.8)


if __name__ == "__main__":
    unittest.main()
