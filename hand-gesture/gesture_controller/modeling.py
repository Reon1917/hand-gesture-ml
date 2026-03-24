from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import joblib

if "MPLCONFIGDIR" not in os.environ:
    matplotlib_cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .features import FEATURE_SPEC_VERSION, MODEL_FEATURE_DIM, RAW_FEATURE_DIM, extract_model_features


@dataclass(slots=True)
class TrainingArtifacts:
    bundle: dict[str, Any]
    metrics: dict[str, Any]


def transform_landmark_dataset(X_raw: np.ndarray) -> np.ndarray:
    rows = np.asarray(X_raw, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != RAW_FEATURE_DIM:
        raise ValueError(f"Expected raw landmark dataset with shape (n, {RAW_FEATURE_DIM}), got {rows.shape}.")
    return np.asarray([extract_model_features(row) for row in rows], dtype=np.float32)


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 300,
    random_state: int = 42,
    test_size: float = 0.2,
) -> TrainingArtifacts:
    X_model = transform_landmark_dataset(X)
    labels, counts = np.unique(y, return_counts=True)
    if labels.size < 2:
        raise ValueError("Need at least two gesture classes to train a classifier.")

    classifier_template = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced_subsample",
    )

    metrics: dict[str, Any] = {
        "samples": int(len(y)),
        "labels": {str(label): int(count) for label, count in zip(labels, counts, strict=False)},
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "raw_feature_dim": RAW_FEATURE_DIM,
        "model_feature_dim": MODEL_FEATURE_DIM,
        "warnings": {
            "class_imbalance": class_imbalance_warnings(labels, counts),
            "evaluation": [],
        },
    }

    can_stratify = counts.min() >= 2
    holdout_metrics: dict[str, Any] | None = None

    if len(y) >= 10 and can_stratify:
        minimum_test_rows = labels.size
        requested_test_rows = max(int(round(len(y) * test_size)), minimum_test_rows)
        if requested_test_rows >= len(y):
            requested_test_rows = len(y) - 1

        X_train, X_test, y_train, y_test = train_test_split(
            X_model,
            y,
            test_size=requested_test_rows,
            random_state=random_state,
            stratify=y,
        )
        evaluation_model = clone(classifier_template)
        evaluation_model.fit(X_train, y_train)
        predictions = evaluation_model.predict(X_test)
        holdout_metrics = build_holdout_metrics(y_test, predictions, labels)
    else:
        metrics["warnings"]["evaluation"].append(
            "Insufficient evaluation data for stratified holdout. Collect at least 10 total samples and 2 samples per class."
        )

    final_model = clone(classifier_template)
    final_model.fit(X_model, y)

    bundle = {
        "model": final_model,
        "labels": final_model.classes_.tolist(),
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "raw_feature_dim": RAW_FEATURE_DIM,
        "model_feature_dim": MODEL_FEATURE_DIM,
    }

    metrics["holdout"] = holdout_metrics
    metrics["trained_with_holdout"] = holdout_metrics is not None

    return TrainingArtifacts(bundle=bundle, metrics=metrics)


def class_imbalance_warnings(labels: np.ndarray, counts: np.ndarray) -> list[str]:
    warnings: list[str] = []
    if counts.size == 0:
        return warnings

    if counts.min() < 10:
        warnings.append("Some gesture classes have fewer than 10 samples; holdout metrics may be unstable.")
    if counts.min() > 0 and (counts.max() / counts.min()) > 1.5:
        warnings.append("Class imbalance detected; the largest class has more than 1.5x the samples of the smallest class.")

    for label, count in zip(labels, counts, strict=False):
        if count < 30:
            warnings.append(f"Gesture '{label}' has {int(count)} samples; aim for 30+ per class per session.")
    return warnings


def build_holdout_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    label_names = [str(label) for label in labels.tolist()]
    matrix = confusion_matrix(y_true, y_pred, labels=label_names)
    return {
        "rows": int(len(y_true)),
        "accuracy": float(np.mean(y_pred == y_true)),
        "labels": label_names,
        "confusion_matrix": matrix.tolist(),
        "top_confusions": top_confusion_pairs(matrix, label_names),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=label_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def top_confusion_pairs(matrix: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    confusions: list[dict[str, Any]] = []
    for row_index, true_label in enumerate(labels):
        for column_index, predicted_label in enumerate(labels):
            if row_index == column_index:
                continue
            count = int(matrix[row_index, column_index])
            if count <= 0:
                continue
            confusions.append(
                {
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": count,
                }
            )
    return sorted(confusions, key=lambda item: item["count"], reverse=True)[:5]


def save_artifacts(
    bundle: dict[str, Any],
    metrics: dict[str, Any],
    model_path: Path,
    metrics_path: Path,
    confusion_matrix_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_to_save = json.loads(json.dumps(metrics))
    metrics_to_save["artifacts"] = {
        "model": str(model_path),
        "metrics": str(metrics_path),
        "confusion_matrix": str(confusion_matrix_path),
        "report": str(report_path),
    }

    joblib.dump(bundle, model_path)
    write_confusion_matrix_image(confusion_matrix_path, metrics_to_save)
    report_path.write_text(build_training_report(metrics_to_save), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_to_save, indent=2), encoding="utf-8")
    return metrics_to_save


def write_confusion_matrix_image(output_path: Path, metrics: dict[str, Any]) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    holdout = metrics.get("holdout")
    if holdout is None:
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "Insufficient evaluation data for a holdout confusion matrix.",
            ha="center",
            va="center",
            wrap=True,
            fontsize=12,
        )
        axis.set_title("Holdout Confusion Matrix")
    else:
        labels = holdout["labels"]
        matrix = np.asarray(holdout["confusion_matrix"], dtype=np.int32)
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title("Holdout Confusion Matrix")
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    str(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row_index, column_index] > matrix.max() / 2 else "black",
                )

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_training_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Training Report",
        "",
        "## Summary",
        f"- Samples: {metrics['samples']}",
        f"- Feature spec: `{metrics['feature_spec_version']}`",
        f"- Raw feature dimension: {metrics['raw_feature_dim']}",
        f"- Model feature dimension: {metrics['model_feature_dim']}",
        f"- Trained with holdout: {'yes' if metrics['trained_with_holdout'] else 'no'}",
        "",
        "## Sample Counts",
        "",
        "| Gesture | Samples |",
        "| --- | ---: |",
    ]
    for label, count in metrics["labels"].items():
        lines.append(f"| {label} | {count} |")

    warning_groups = metrics.get("warnings", {})
    lines.extend(["", "## Warnings"])
    emitted_warning = False
    for group_name in ("class_imbalance", "evaluation"):
        for warning in warning_groups.get(group_name, []):
            lines.append(f"- {warning}")
            emitted_warning = True
    if not emitted_warning:
        lines.append("- None")

    holdout = metrics.get("holdout")
    lines.extend(["", "## Holdout Evaluation"])
    if holdout is None:
        lines.append("- Insufficient evaluation data for a stratified holdout split.")
    else:
        lines.append(f"- Holdout rows: {holdout['rows']}")
        lines.append(f"- Accuracy: {holdout['accuracy']:.4f}")
        lines.extend(["", "| Gesture | Precision | Recall | F1 | Support |", "| --- | ---: | ---: | ---: | ---: |"])
        for label in holdout["labels"]:
            row = holdout["classification_report"][label]
            lines.append(
                f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1-score']:.3f} | {int(row['support'])} |"
            )
        lines.extend(["", "### Top Confusions"])
        if holdout["top_confusions"]:
            for item in holdout["top_confusions"]:
                lines.append(
                    f"- `{item['true_label']}` predicted as `{item['predicted_label']}` {item['count']} time(s)"
                )
        else:
            lines.append("- None")

    artifacts = metrics.get("artifacts", {})
    lines.extend(
        [
            "",
            "## Artifacts",
            f"- Model: `{artifacts.get('model', '')}`",
            f"- Metrics JSON: `{artifacts.get('metrics', '')}`",
            f"- Confusion matrix: `{artifacts.get('confusion_matrix', '')}`",
            f"- Report: `{artifacts.get('report', '')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def load_bundle(model_path: Path) -> dict[str, Any]:
    return joblib.load(model_path)


def validate_bundle(bundle: dict[str, Any]) -> None:
    expected = {
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "raw_feature_dim": RAW_FEATURE_DIM,
        "model_feature_dim": MODEL_FEATURE_DIM,
    }
    for key, expected_value in expected.items():
        actual_value = bundle.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Model bundle mismatch for {key}: expected {expected_value!r}, found {actual_value!r}. Retrain the model."
            )


def predict_gesture(bundle: dict[str, Any], raw_features: np.ndarray) -> tuple[str, float, dict[str, float]]:
    validate_bundle(bundle)
    raw_feature_vector = np.asarray(raw_features, dtype=np.float32).reshape(-1)
    if raw_feature_vector.size != bundle["raw_feature_dim"]:
        raise ValueError(
            f"Expected {bundle['raw_feature_dim']} raw features, got {raw_feature_vector.size}."
        )

    feature_vector = extract_model_features(raw_feature_vector).reshape(1, -1)
    if feature_vector.shape[1] != bundle["model_feature_dim"]:
        raise ValueError(
            f"Expected {bundle['model_feature_dim']} engineered features, got {feature_vector.shape[1]}."
        )

    model = bundle["model"]
    probabilities = model.predict_proba(feature_vector)[0]
    best_index = int(np.argmax(probabilities))
    label = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])
    distribution = {
        str(class_name): float(probability)
        for class_name, probability in zip(model.classes_, probabilities, strict=False)
    }
    return label, confidence, distribution
