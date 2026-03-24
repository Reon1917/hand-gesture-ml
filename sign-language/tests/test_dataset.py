from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asl_app.dataset import build_dataset_index, normalize_label


class DatasetTests(unittest.TestCase):
    def test_normalize_label(self) -> None:
        self.assertEqual(normalize_label("a"), "A")
        self.assertEqual(normalize_label("space"), "space")
        self.assertIsNone(normalize_label("asl_alphabet_train"))

    def test_build_dataset_index_filters_motion_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for label in ["A", "B", "J"]:
                label_dir = root / label
                label_dir.mkdir()
                for index in range(3):
                    Image.new("RGB", (48, 48), color=(index * 30, 20, 20)).save(label_dir / f"{index}.jpg")

            records = build_dataset_index(root)
            labels = sorted({record.label for record in records})
            self.assertEqual(labels, ["A", "B"])


if __name__ == "__main__":
    unittest.main()
