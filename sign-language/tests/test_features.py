from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asl_app.features import extract_feature_vector, feature_dimension


class FeatureTests(unittest.TestCase):
    def test_feature_dimension_is_stable(self) -> None:
        self.assertGreater(feature_dimension(), 1000)

    def test_different_shapes_create_different_features(self) -> None:
        circle = Image.new("RGB", (64, 64), color="black")
        draw_circle = ImageDraw.Draw(circle)
        draw_circle.ellipse((14, 14, 50, 50), fill="white")

        square = Image.new("RGB", (64, 64), color="black")
        draw_square = ImageDraw.Draw(square)
        draw_square.rectangle((14, 14, 50, 50), fill="white")

        circle_features = extract_feature_vector(circle)
        square_features = extract_feature_vector(square)
        distance = float(np.linalg.norm(circle_features - square_features))
        self.assertGreater(distance, 0.5)


if __name__ == "__main__":
    unittest.main()
