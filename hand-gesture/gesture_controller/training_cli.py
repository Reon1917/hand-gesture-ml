from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .dataset import load_dataset
from .modeling import save_artifacts, train_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hand gesture classifier from a CSV dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("data/gestures.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/gesture_model.joblib"))
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("artifacts/gesture_model_metrics.json"),
    )
    parser.add_argument(
        "--confusion-matrix-out",
        type=Path,
        default=Path("artifacts/confusion_matrix.png"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("artifacts/training_report.md"),
    )
    parser.add_argument("--trees", type=int, default=300, help="Random forest tree count.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction when enough data exists.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        X, y = load_dataset(args.dataset)
    except FileNotFoundError:
        print(
            f"Dataset not found at {args.dataset}. Run `python collect_data.py` first.",
            file=sys.stderr,
        )
        return 2
    except ValueError as exc:
        print(
            f"{exc} Collect samples with `python collect_data.py` before training.",
            file=sys.stderr,
        )
        return 2

    artifacts = train_classifier(
        X,
        y,
        n_estimators=args.trees,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    metrics = save_artifacts(
        artifacts.bundle,
        artifacts.metrics,
        args.model_out,
        args.metrics_out,
        args.confusion_matrix_out,
        args.report_out,
    )

    print(f"Saved model to {args.model_out}")
    print(f"Saved metrics to {args.metrics_out}")
    print(f"Saved confusion matrix to {args.confusion_matrix_out}")
    print(f"Saved report to {args.report_out}")
    print(json.dumps(metrics, indent=2))
    return 0
