from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


@dataclass(slots=True)
class TrainingArtifacts:
    bundle: dict[str, Any]
    metrics: dict[str, Any]


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 300,
    random_state: int = 42,
    test_size: float = 0.2,
) -> TrainingArtifacts:
    labels, counts = np.unique(y, return_counts=True)
    if labels.size < 2:
        raise ValueError("Need at least two gesture classes to train a classifier.")

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced_subsample",
    )

    metrics: dict[str, Any] = {
        "samples": int(len(y)),
        "labels": {str(label): int(count) for label, count in zip(labels, counts, strict=False)},
    }

    can_stratify = counts.min() >= 2
    holdout_metrics: dict[str, Any] | None = None

    if len(y) >= 10 and can_stratify:
        minimum_test_rows = labels.size
        requested_test_rows = max(int(round(len(y) * test_size)), minimum_test_rows)
        if requested_test_rows >= len(y):
            requested_test_rows = len(y) - 1

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=requested_test_rows,
            random_state=random_state,
            stratify=y,
        )
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracy = float(np.mean(predictions == y_test))
        holdout_metrics = {
            "rows": int(len(y_test)),
            "accuracy": accuracy,
            "classification_report": classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
        }
    else:
        classifier.fit(X, y)

    bundle = {
        "model": classifier,
        "labels": classifier.classes_.tolist(),
        "feature_dim": int(X.shape[1]),
    }

    metrics["holdout"] = holdout_metrics
    metrics["trained_with_holdout"] = holdout_metrics is not None

    return TrainingArtifacts(bundle=bundle, metrics=metrics)


def save_artifacts(bundle: dict[str, Any], metrics: dict[str, Any], model_path: Path, metrics_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_bundle(model_path: Path) -> dict[str, Any]:
    return joblib.load(model_path)


def predict_gesture(bundle: dict[str, Any], features: np.ndarray) -> tuple[str, float, dict[str, float]]:
    model = bundle["model"]
    feature_vector = np.asarray(features, dtype=np.float32).reshape(1, -1)
    probabilities = model.predict_proba(feature_vector)[0]
    best_index = int(np.argmax(probabilities))
    label = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])
    distribution = {
        str(class_name): float(probability)
        for class_name, probability in zip(model.classes_, probabilities, strict=False)
    }
    return label, confidence, distribution
