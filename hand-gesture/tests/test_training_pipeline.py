from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gesture_controller.actions import load_bindings
from gesture_controller.modeling import (
    load_bundle,
    save_artifacts,
    train_classifier,
    validate_bundle,
)


def make_template(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    template = rng.normal(size=(21, 3)).astype(np.float32)
    template[0] = np.zeros(3, dtype=np.float32)
    return template


def make_dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    labels = ["peace", "thumbs_up", "fist"]
    templates = {label: make_template(index + 1) for index, label in enumerate(labels)}

    samples: list[np.ndarray] = []
    targets: list[str] = []
    for label in labels:
        for _ in range(12):
            noise = rng.normal(scale=0.03, size=(21, 3)).astype(np.float32)
            scale = rng.uniform(0.8, 1.2)
            translation = rng.normal(scale=0.25, size=(3,)).astype(np.float32)
            sample = templates[label] * scale + translation + noise
            samples.append(sample.reshape(-1))
            targets.append(label)

    return np.asarray(samples, dtype=np.float32), np.asarray(targets, dtype=str)


class TrainingPipelineTests(unittest.TestCase):
    def test_training_writes_all_artifacts(self) -> None:
        X, y = make_dataset()
        artifacts = train_classifier(X, y, n_estimators=50, random_state=123)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metrics = save_artifacts(
                artifacts.bundle,
                artifacts.metrics,
                root / "gesture_model.joblib",
                root / "gesture_model_metrics.json",
                root / "confusion_matrix.png",
                root / "training_report.md",
            )

            self.assertTrue((root / "gesture_model.joblib").exists())
            self.assertTrue((root / "gesture_model_metrics.json").exists())
            self.assertTrue((root / "confusion_matrix.png").exists())
            self.assertTrue((root / "training_report.md").exists())
            self.assertTrue(metrics["trained_with_holdout"])

            loaded_metrics = json.loads((root / "gesture_model_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("artifacts", loaded_metrics)
            self.assertIn("top_confusions", loaded_metrics["holdout"])

    def test_runtime_bundle_validation_and_binding_compatibility(self) -> None:
        X, y = make_dataset()
        artifacts = train_classifier(X, y, n_estimators=25, random_state=7)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "gesture_model.joblib"
            save_artifacts(
                artifacts.bundle,
                artifacts.metrics,
                model_path,
                root / "gesture_model_metrics.json",
                root / "confusion_matrix.png",
                root / "training_report.md",
            )

            loaded_bundle = load_bundle(model_path)
            validate_bundle(loaded_bundle)

            bindings_path = root / "gesture_bindings.json"
            bindings_path.write_text(json.dumps({"peace": "key:space"}), encoding="utf-8")
            self.assertEqual(load_bindings(bindings_path)["peace"], "key:space")

            broken_bundle = dict(loaded_bundle)
            broken_bundle["feature_spec_version"] = "old_version"
            with self.assertRaisesRegex(ValueError, "feature_spec_version"):
                validate_bundle(broken_bundle)


if __name__ == "__main__":
    unittest.main()
