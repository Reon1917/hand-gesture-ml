from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import cv2
import numpy as np

from gesture_controller.external_datasets import (
    extract_face_dataset_from_fer2013,
    import_hand_image_dataset,
    iter_labeled_image_paths,
    load_label_map,
)
from gesture_controller.face_features import FACE_MODEL_FEATURE_DIM
from gesture_controller.features import RAW_FEATURE_DIM


class FakeHandDetector:
    def detect(self, frame):
        return SimpleNamespace(features=np.ones(RAW_FEATURE_DIM, dtype=np.float32))


class FakeFaceDetector:
    def detect(self, frame):
        return SimpleNamespace(model_features=np.ones(FACE_MODEL_FEATURE_DIM, dtype=np.float32))


class ExternalDatasetTests(unittest.TestCase):
    def test_load_label_map_accepts_inline_json(self) -> None:
        mapping = load_label_map('{"like":"thumbs_up"}')
        self.assertEqual(mapping["like"], "thumbs_up")

    def test_iter_labeled_image_paths_uses_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            label_dir = root / "peace"
            label_dir.mkdir(parents=True)
            image_path = label_dir / "sample.jpg"
            cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))

            items = list(iter_labeled_image_paths(root))
            self.assertEqual(items, [("peace", image_path)])

    def test_import_hand_image_dataset_appends_landmarks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "like"
            image_dir.mkdir(parents=True)
            image_path = image_dir / "sample.jpg"
            cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))

            dataset_path = root / "gestures.csv"
            stats = import_hand_image_dataset(
                root,
                dataset_path,
                FakeHandDetector(),
                label_map={"like": "thumbs_up"},
            )

            self.assertTrue(dataset_path.exists())
            self.assertEqual(stats["imported"]["thumbs_up"], 1)

    def test_extract_face_dataset_from_fer2013(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "fer2013.csv"
            pixels = " ".join(["0"] * (48 * 48))
            csv_path.write_text(
                "emotion,pixels,Usage\n"
                f"3,{pixels},Training\n"
                f"6,{pixels},Training\n",
                encoding="utf-8",
            )

            X, y, stats = extract_face_dataset_from_fer2013(
                csv_path,
                FakeFaceDetector(),
                usage="training",
            )

            self.assertEqual(X.shape, (2, FACE_MODEL_FEATURE_DIM))
            self.assertEqual(set(y.tolist()), {"happy", "neutral"})
            self.assertEqual(sum(stats["imported"].values()), 2)


if __name__ == "__main__":
    unittest.main()
