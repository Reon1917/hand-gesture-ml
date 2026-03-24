from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .external_datasets import (
    extract_face_dataset_from_fer2013,
    extract_face_dataset_from_image_root,
    iter_labeled_image_paths,
    load_label_map,
)
from .face_landmarks import FaceLandmarkDetector
from .face_modeling import save_face_artifacts, train_face_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a face-expression classifier from an external dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--dataset-format",
        choices=("auto", "image-folder", "fer2013"),
        default="auto",
        help="Dataset source format.",
    )
    parser.add_argument(
        "--usage",
        default="all",
        help="FER2013 usage split to import: all, training, publictest, or privatetest.",
    )
    parser.add_argument("--label-map", help="Inline JSON or a path to a JSON label mapping file.")
    parser.add_argument("--max-images-per-label", type=int)
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/face_expression_model.joblib"))
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("artifacts/face_expression_metrics.json"),
    )
    parser.add_argument(
        "--confusion-matrix-out",
        type=Path,
        default=Path("artifacts/face_expression_confusion_matrix.png"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("artifacts/face_expression_report.md"),
    )
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _resolve_dataset_format(dataset: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if dataset.suffix.lower() == ".csv":
        return "fer2013"
    return "image-folder"


def main() -> int:
    args = parse_args()
    dataset_format = _resolve_dataset_format(args.dataset, args.dataset_format)
    label_map = load_label_map(args.label_map)
    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        return 2
    if dataset_format == "image-folder" and not any(True for _ in iter_labeled_image_paths(args.dataset)):
        print(f"No image files were found under {args.dataset}.", file=sys.stderr)
        return 2

    try:
        detector = FaceLandmarkDetector(static_image_mode=True)
    except Exception as exc:
        print(f"Failed to start the face detector: {exc}", file=sys.stderr)
        return 2

    try:
        if dataset_format == "fer2013":
            X, y, import_stats = extract_face_dataset_from_fer2013(
                args.dataset,
                detector,
                usage=args.usage,
                label_map=label_map,
                max_images_per_label=args.max_images_per_label,
            )
        else:
            X, y, import_stats = extract_face_dataset_from_image_root(
                args.dataset,
                detector,
                label_map=label_map,
                max_images_per_label=args.max_images_per_label,
            )
    except Exception as exc:
        print(f"Failed to load the external face dataset: {exc}", file=sys.stderr)
        detector.close()
        return 2

    detector.close()
    artifacts = train_face_classifier(
        X,
        y,
        n_estimators=args.trees,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    metrics = save_face_artifacts(
        artifacts.bundle,
        artifacts.metrics,
        args.model_out,
        args.metrics_out,
        args.confusion_matrix_out,
        args.report_out,
    )

    print(f"Imported face dataset from {args.dataset}")
    print(json.dumps(import_stats, indent=2))
    print(f"Saved face model to {args.model_out}")
    print(f"Saved face metrics to {args.metrics_out}")
    print(f"Saved face confusion matrix to {args.confusion_matrix_out}")
    print(f"Saved face report to {args.report_out}")
    print(json.dumps(metrics, indent=2))
    return 0
