from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .face_features import FACE_FEATURE_SPEC_VERSION, FACE_MODEL_FEATURE_DIM


@dataclass(slots=True)
class FaceTrainingArtifacts:
    bundle: dict[str, Any]
    metrics: dict[str, Any]


def train_face_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 300,
    random_state: int = 42,
    test_size: float = 0.2,
) -> FaceTrainingArtifacts:
    features = np.asarray(X, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != FACE_MODEL_FEATURE_DIM:
        raise ValueError(
            f"Expected face feature dataset with shape (n, {FACE_MODEL_FEATURE_DIM}), got {features.shape}."
        )

    labels, counts = np.unique(y, return_counts=True)
    if labels.size < 2:
        raise ValueError("Need at least two expression classes to train a face model.")

    classifier_template = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced_subsample",
    )
    metrics: dict[str, Any] = {
        "samples": int(len(y)),
        "labels": {str(label): int(count) for label, count in zip(labels, counts, strict=False)},
        "feature_spec_version": FACE_FEATURE_SPEC_VERSION,
        "model_feature_dim": FACE_MODEL_FEATURE_DIM,
        "warnings": {
            "class_imbalance": _face_data_warnings(labels, counts),
            "evaluation": [],
        },
    }

    can_stratify = counts.min() >= 2
    holdout_metrics: dict[str, Any] | None = None
    if len(y) >= 12 and can_stratify:
        minimum_test_rows = labels.size
        requested_test_rows = max(int(round(len(y) * test_size)), minimum_test_rows)
        if requested_test_rows >= len(y):
            requested_test_rows = len(y) - 1

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            y,
            test_size=requested_test_rows,
            random_state=random_state,
            stratify=y,
        )
        evaluation_model = clone(classifier_template)
        evaluation_model.fit(X_train, y_train)
        predictions = evaluation_model.predict(X_test)
        holdout_metrics = _build_holdout_metrics(y_test, predictions, labels)
    else:
        metrics["warnings"]["evaluation"].append(
            "Insufficient evaluation data for a stratified face-expression holdout. Collect more detected faces per class."
        )

    final_model = clone(classifier_template)
    final_model.fit(features, y)

    bundle = {
        "model": final_model,
        "labels": final_model.classes_.tolist(),
        "feature_spec_version": FACE_FEATURE_SPEC_VERSION,
        "model_feature_dim": FACE_MODEL_FEATURE_DIM,
    }
    metrics["holdout"] = holdout_metrics
    metrics["trained_with_holdout"] = holdout_metrics is not None

    return FaceTrainingArtifacts(bundle=bundle, metrics=metrics)


def save_face_artifacts(
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
    _write_confusion_matrix_image(confusion_matrix_path, metrics_to_save, title="Face Expression Confusion Matrix")
    report_path.write_text(_build_face_report(metrics_to_save), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_to_save, indent=2), encoding="utf-8")
    return metrics_to_save


def load_face_bundle(model_path: Path) -> dict[str, Any]:
    return joblib.load(model_path)


def validate_face_bundle(bundle: dict[str, Any]) -> None:
    expected = {
        "feature_spec_version": FACE_FEATURE_SPEC_VERSION,
        "model_feature_dim": FACE_MODEL_FEATURE_DIM,
    }
    for key, expected_value in expected.items():
        actual_value = bundle.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Face model bundle mismatch for {key}: expected {expected_value!r}, found {actual_value!r}. Retrain the face model."
            )


def predict_expression(bundle: dict[str, Any], face_features: np.ndarray) -> tuple[str, float, dict[str, float]]:
    validate_face_bundle(bundle)
    features = np.asarray(face_features, dtype=np.float32).reshape(1, -1)
    if features.shape[1] != bundle["model_feature_dim"]:
        raise ValueError(
            f"Expected {bundle['model_feature_dim']} face features, got {features.shape[1]}."
        )

    model = bundle["model"]
    probabilities = model.predict_proba(features)[0]
    best_index = int(np.argmax(probabilities))
    label = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])
    distribution = {
        str(class_name): float(probability)
        for class_name, probability in zip(model.classes_, probabilities, strict=False)
    }
    return label, confidence, distribution


def _face_data_warnings(labels: np.ndarray, counts: np.ndarray) -> list[str]:
    warnings: list[str] = []
    if counts.min() < 10:
        warnings.append("Some expression classes have fewer than 10 detected faces; metrics may be unstable.")
    if counts.min() > 0 and (counts.max() / counts.min()) > 1.5:
        warnings.append("Expression class imbalance detected; largest class is more than 1.5x the smallest.")
    for label, count in zip(labels, counts, strict=False):
        if count < 40:
            warnings.append(f"Expression '{label}' has {int(count)} samples; aim for 40+ detected faces.")
    return warnings


def _build_holdout_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    label_names = [str(label) for label in labels.tolist()]
    matrix = confusion_matrix(y_true, y_pred, labels=label_names)
    return {
        "rows": int(len(y_true)),
        "accuracy": float(np.mean(y_pred == y_true)),
        "labels": label_names,
        "confusion_matrix": matrix.tolist(),
        "top_confusions": _top_confusion_pairs(matrix, label_names),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=label_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def _top_confusion_pairs(matrix: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
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


def _write_confusion_matrix_image(output_path: Path, metrics: dict[str, Any], *, title: str) -> None:
    import matplotlib.pyplot as plt

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
        axis.set_title(title)
    else:
        labels = holdout["labels"]
        matrix = np.asarray(holdout["confusion_matrix"], dtype=np.int32)
        image = axis.imshow(matrix, cmap="Oranges")
        axis.set_title(title)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        threshold = matrix.max() / 2 if matrix.size else 0
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    str(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row_index, column_index] > threshold else "black",
                )

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _build_face_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Face Expression Training Report",
        "",
        "## Summary",
        f"- Samples: {metrics['samples']}",
        f"- Feature spec: `{metrics['feature_spec_version']}`",
        f"- Model feature dimension: {metrics['model_feature_dim']}",
        f"- Trained with holdout: {'yes' if metrics['trained_with_holdout'] else 'no'}",
        "",
        "## Sample Counts",
        "",
        "| Expression | Samples |",
        "| --- | ---: |",
    ]
    for label, count in metrics["labels"].items():
        lines.append(f"| {label} | {count} |")

    lines.extend(["", "## Warnings"])
    emitted_warning = False
    for group_name in ("class_imbalance", "evaluation"):
        for warning in metrics.get("warnings", {}).get(group_name, []):
            lines.append(f"- {warning}")
            emitted_warning = True
    if not emitted_warning:
        lines.append("- None")

    lines.extend(["", "## Holdout Evaluation"])
    holdout = metrics.get("holdout")
    if holdout is None:
        lines.append("- Insufficient evaluation data for a stratified holdout split.")
    else:
        lines.append(f"- Holdout rows: {holdout['rows']}")
        lines.append(f"- Accuracy: {holdout['accuracy']:.4f}")
        lines.extend(["", "| Expression | Precision | Recall | F1 | Support |", "| --- | ---: | ---: | ---: | ---: |"])
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
