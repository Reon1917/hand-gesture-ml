from __future__ import annotations

import json
import threading
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from joblib import parallel_backend
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

from .asl_reference import MANUAL_TESTING_NOTES, reference_hint, sort_labels
from .dataset import ImageRecord, build_dataset_index
from .features import extract_feature_vector

MODEL_FILE_NAME = "asl_model.joblib"
METRICS_FILE_NAME = "metrics.json"
REPORT_FILE_NAME = "training_report.md"
CONFUSION_MATRIX_FILE_NAME = "confusion_matrix.png"
EXAMPLES_DIR_NAME = "examples"

_RESAMPLING = getattr(Image, "Resampling", Image)
_LANCZOS = _RESAMPLING.LANCZOS


@dataclass(frozen=True)
class TrainingResult:
    model_path: Path
    metrics_path: Path
    report_path: Path
    confusion_matrix_path: Path
    examples_dir: Path
    metrics: dict[str, object]


def build_feature_matrix(
    records: list[ImageRecord],
    *,
    desc: str,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[str] = []

    iterator = tqdm(records, desc=desc, unit="img", disable=not show_progress, dynamic_ncols=True)
    for record in iterator:
        with Image.open(record.path) as image:
            features.append(extract_feature_vector(image))
        labels.append(record.label)

    return np.vstack(features).astype(np.float32), np.asarray(labels)


def build_candidate_models(random_seed: int, *, full_search: bool) -> dict[str, object]:
    calibrated_svm = CalibratedClassifierCV(
        estimator=make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=1.2,
                class_weight="balanced",
                dual="auto",
                max_iter=8000,
                random_state=random_seed,
            ),
        ),
        cv=3,
        method="sigmoid",
        n_jobs=-1,
    )
    sgd_log_loss = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=4000,
            tol=1e-3,
            class_weight="balanced",
            random_state=random_seed,
        ),
    )
    models: dict[str, object] = {
        "sgd-log-loss": sgd_log_loss,
    }
    if full_search:
        models["linear-svm-calibrated"] = calibrated_svm
    return models


def fit_with_heartbeat(
    model: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model_name: str,
    show_progress: bool,
    interval_seconds: float = 10.0,
) -> tuple[object, float]:
    if not show_progress:
        started = time.perf_counter()
        fit_context = nullcontext()
        if hasattr(model, "get_params") and "n_jobs" in model.get_params(deep=False):
            fit_context = parallel_backend("threading", n_jobs=-1)
        with fit_context:
            return model.fit(X_train, y_train), time.perf_counter() - started

    started = time.perf_counter()
    finished = threading.Event()

    def heartbeat() -> None:
        while not finished.wait(interval_seconds):
            elapsed = time.perf_counter() - started
            print(
                f"{model_name} still training... {elapsed:.0f}s elapsed",
                flush=True,
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        fit_context = nullcontext()
        if hasattr(model, "get_params") and "n_jobs" in model.get_params(deep=False):
            fit_context = parallel_backend("threading", n_jobs=-1)
        with fit_context:
            fitted = model.fit(X_train, y_train)
    finally:
        finished.set()
        thread.join(timeout=0.2)

    return fitted, time.perf_counter() - started


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.nan_to_num(scores.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
    shifted = scores - float(np.max(scores))
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(scores.shape, 1.0 / max(scores.shape[0], 1), dtype=np.float64)
    return exp / total


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray | None:
    cleaned = np.nan_to_num(probabilities.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(cleaned))
    if not np.isfinite(total) or total <= 0.0:
        return None
    normalized = cleaned / total
    if not np.all(np.isfinite(normalized)):
        return None
    return normalized


def decision_probabilities(model: object, vector: np.ndarray) -> np.ndarray | None:
    if not hasattr(model, "decision_function"):
        return None

    scores = np.asarray(model.decision_function(vector), dtype=np.float64)
    scores = np.ravel(scores)
    if scores.size == 0:
        return None
    if scores.size == 1 and hasattr(model, "classes_") and len(model.classes_) == 2:
        score = float(scores[0])
        scores = np.asarray([-score, score], dtype=np.float64)
    return softmax(scores)


def predict_with_bundle(bundle: dict[str, object], image: Image.Image, *, top_k: int = 5) -> dict[str, object]:
    model = bundle["model"]
    vector = extract_feature_vector(image).reshape(1, -1)
    probabilities: np.ndarray | None = None

    if hasattr(model, "predict_proba"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw_probabilities = np.asarray(model.predict_proba(vector)[0], dtype=np.float64)
        probabilities = normalize_probabilities(raw_probabilities)

    if probabilities is None:
        probabilities = decision_probabilities(model, vector)

    if probabilities is None:
        prediction = model.predict(vector)[0]
        return {
            "label": str(prediction),
            "confidence": 1.0,
            "top_predictions": [{"label": str(prediction), "confidence": 1.0}],
        }

    classes = [str(label) for label in model.classes_]
    ranking = np.argsort(probabilities)[::-1][:top_k]
    top_predictions = [
        {
            "label": classes[index],
            "confidence": float(probabilities[index]),
        }
        for index in ranking
    ]
    return {
        "label": top_predictions[0]["label"],
        "confidence": float(top_predictions[0]["confidence"]),
        "top_predictions": top_predictions,
    }


def export_reference_examples(records: list[ImageRecord], output_dir: Path) -> list[dict[str, str | None]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples: list[dict[str, str | None]] = []
    seen_labels: set[str] = set()

    for record in records:
        if record.label in seen_labels:
            continue
        seen_labels.add(record.label)
        target_name = f"{record.label.lower()}.jpg"
        target_path = output_dir / target_name
        with Image.open(record.path) as image:
            preview = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (240, 240), method=_LANCZOS)
            preview.save(target_path, format="JPEG", quality=88)
        examples.append(
            {
                "label": record.label,
                "hint": reference_hint(record.label),
                "image_url": f"/artifacts/{EXAMPLES_DIR_NAME}/{target_name}",
            }
        )

    return sorted(examples, key=lambda item: item["label"])


def save_confusion_matrix_png(matrix: np.ndarray, labels: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    font = ImageFont.load_default()
    cell_size = 28
    top_margin = 110
    left_margin = 110
    width = left_margin + cell_size * len(labels) + 30
    height = top_margin + cell_size * len(labels) + 30
    image = Image.new("RGB", (width, height), color=(248, 245, 238))
    draw = ImageDraw.Draw(image)

    max_value = max(int(matrix.max()), 1)

    for row_index, row_label in enumerate(labels):
        y0 = top_margin + row_index * cell_size
        y1 = y0 + cell_size
        draw.text((26, y0 + 7), row_label, fill=(48, 40, 31), font=font)
        for col_index, col_label in enumerate(labels):
            if row_index == 0:
                draw.text((left_margin + col_index * cell_size + 8, 72), col_label, fill=(48, 40, 31), font=font)
            x0 = left_margin + col_index * cell_size
            x1 = x0 + cell_size
            intensity = matrix[row_index, col_index] / max_value
            color = (
                int(236 - 124 * intensity),
                int(226 - 155 * intensity),
                int(208 - 170 * intensity),
            )
            draw.rectangle((x0, y0, x1, y1), fill=color, outline=(214, 206, 194))
            value = str(int(matrix[row_index, col_index]))
            text_width = draw.textlength(value, font=font)
            draw.text(
                (x0 + (cell_size - text_width) / 2, y0 + 8),
                value,
                fill=(23, 19, 15),
                font=font,
            )

    draw.text((left_margin, 20), "Confusion Matrix", fill=(23, 19, 15), font=font)
    draw.text((left_margin, 38), "rows = actual, columns = predicted", fill=(93, 82, 68), font=font)
    image.save(output_path)


def build_report(metrics: dict[str, object]) -> str:
    leaderboard_lines = "\n".join(
        f"| {entry['model']} | {entry['accuracy']:.4f} | {entry['macro_f1']:.4f} |"
        for entry in metrics["leaderboard"]
    )

    labels = ", ".join(metrics["labels"])
    excluded_labels = ", ".join(metrics["excluded_labels"]) or "none"
    notes = "\n".join(f"- {note}" for note in MANUAL_TESTING_NOTES)

    return f"""# ASL Training Report

## Summary

- Selected model: `{metrics['selected_model']}`
- Validation accuracy: `{metrics['accuracy']:.4f}`
- Validation macro F1: `{metrics['macro_f1']:.4f}`
- Total images used: `{metrics['dataset_size']}`
- Labels: `{labels}`
- Excluded labels: `{excluded_labels}`

## Model Leaderboard

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
{leaderboard_lines}

## Classification Report

```text
{metrics['classification_report']}
```

## Notes

{notes}
"""


def train_classifier(
    dataset_root: Path,
    *,
    artifacts_dir: Path,
    include_motion: bool = False,
    include_special: bool = False,
    max_images_per_class: int | None = 1200,
    validation_split: float = 0.2,
    random_seed: int = 42,
    show_progress: bool = False,
    full_search: bool = False,
) -> TrainingResult:
    if show_progress:
        print(f"Indexing dataset in {dataset_root}...", flush=True)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = artifacts_dir / EXAMPLES_DIR_NAME

    records = build_dataset_index(
        dataset_root,
        include_motion=include_motion,
        include_special=include_special,
        max_images_per_class=max_images_per_class,
        random_seed=random_seed,
    )
    labels = [record.label for record in records]
    unique_labels = sort_labels(set(labels))

    if show_progress:
        label_preview = ", ".join(unique_labels)
        print(f"Found {len(unique_labels)} labels across {len(records):,} images.", flush=True)
        print(f"Labels: {label_preview}", flush=True)
        if full_search:
            print("Model search: fast SGD baseline plus slower calibrated SVM benchmark.", flush=True)
        else:
            print("Model search: fast SGD baseline only. Use --full-search to benchmark the slower calibrated SVM.", flush=True)

    minimum_per_label = min(labels.count(label) for label in unique_labels)
    if minimum_per_label < 2:
        raise RuntimeError("Each class needs at least 2 images to create a validation split.")

    train_records, validation_records = train_test_split(
        records,
        test_size=validation_split,
        random_state=random_seed,
        stratify=labels,
    )

    if show_progress:
        print(
            f"Split dataset into {len(train_records):,} training and {len(validation_records):,} validation images.",
            flush=True,
        )

    X_train, y_train = build_feature_matrix(
        train_records,
        desc="Extracting train features",
        show_progress=show_progress,
    )
    X_validation, y_validation = build_feature_matrix(
        validation_records,
        desc="Extracting validation features",
        show_progress=show_progress,
    )

    leaderboard: list[dict[str, float | str]] = []
    best_model = None
    best_name = None
    best_predictions = None
    best_accuracy = -1.0
    best_macro_f1 = -1.0

    for model_name, model in build_candidate_models(random_seed, full_search=full_search).items():
        if show_progress:
            print(
                f"Training {model_name}... this phase can take a few minutes on the full dataset.",
                flush=True,
            )
        fitted, fit_seconds = fit_with_heartbeat(
            model,
            X_train,
            y_train,
            model_name=model_name,
            show_progress=show_progress,
        )
        predictions = fitted.predict(X_validation)
        accuracy = float(accuracy_score(y_validation, predictions))
        macro_f1 = float(f1_score(y_validation, predictions, average="macro"))
        leaderboard.append({"model": model_name, "accuracy": accuracy, "macro_f1": macro_f1})

        if show_progress:
            print(
                f"Finished {model_name} in {fit_seconds:.1f}s: accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}",
                flush=True,
            )

        if accuracy > best_accuracy or (accuracy == best_accuracy and macro_f1 > best_macro_f1):
            best_model = fitted
            best_name = model_name
            best_predictions = predictions
            best_accuracy = accuracy
            best_macro_f1 = macro_f1

    if best_model is None or best_predictions is None or best_name is None:
        raise RuntimeError("Training did not produce a model.")

    confusion = confusion_matrix(y_validation, best_predictions, labels=unique_labels)
    metrics = {
        "selected_model": best_name,
        "accuracy": best_accuracy,
        "macro_f1": best_macro_f1,
        "dataset_root": str(dataset_root),
        "dataset_size": len(records),
        "train_size": len(train_records),
        "validation_size": len(validation_records),
        "labels": unique_labels,
        "excluded_labels": [
            label
            for label in ["J", "Z", "del", "nothing", "space"]
            if (label in {"J", "Z"} and not include_motion) or (label in {"del", "nothing", "space"} and not include_special)
        ],
        "leaderboard": sorted(leaderboard, key=lambda item: (-item["accuracy"], -item["macro_f1"], item["model"])),
        "classification_report": classification_report(y_validation, best_predictions, labels=unique_labels, zero_division=0),
        "feature_dimension": int(X_train.shape[1]),
    }

    if show_progress:
        print("Exporting artifacts...", flush=True)

    example_manifest = export_reference_examples(
        sorted(records, key=lambda record: (record.label, str(record.path).lower())),
        examples_dir,
    )

    bundle = {
        "model": best_model,
        "examples": example_manifest,
        "metadata": {
            "model_name": best_name,
            "labels": unique_labels,
            "dataset_size": len(records),
            "validation_accuracy": best_accuracy,
            "validation_macro_f1": best_macro_f1,
            "feature_dimension": int(X_train.shape[1]),
            "dataset_root": str(dataset_root),
            "excluded_labels": metrics["excluded_labels"],
            "testing_notes": MANUAL_TESTING_NOTES,
        },
    }

    model_path = artifacts_dir / MODEL_FILE_NAME
    metrics_path = artifacts_dir / METRICS_FILE_NAME
    report_path = artifacts_dir / REPORT_FILE_NAME
    confusion_path = artifacts_dir / CONFUSION_MATRIX_FILE_NAME

    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(build_report(metrics), encoding="utf-8")
    save_confusion_matrix_png(confusion, unique_labels, confusion_path)

    if show_progress:
        print("Training complete.", flush=True)

    return TrainingResult(
        model_path=model_path,
        metrics_path=metrics_path,
        report_path=report_path,
        confusion_matrix_path=confusion_path,
        examples_dir=examples_dir,
        metrics=metrics,
    )


def load_bundle(model_path: Path) -> dict[str, object]:
    return joblib.load(model_path)
