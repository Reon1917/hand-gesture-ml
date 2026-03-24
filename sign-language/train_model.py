from __future__ import annotations

import argparse
from pathlib import Path

from asl_app.runtime_bootstrap import bootstrap_local_venv

bootstrap_local_venv(__file__, ("joblib", "PIL", "sklearn", "tqdm"))

from asl_app.dataset import DEFAULT_DATASET_SLUG, download_kaggle_dataset
from asl_app.modeling import train_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ASL alphabet classifier.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/asl-alphabet"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET_SLUG, help="Kaggle dataset slug.")
    parser.add_argument("--download", action="store_true", help="Download the dataset if it is missing.")
    parser.add_argument("--include-motion", action="store_true", help="Include J and Z.")
    parser.add_argument("--include-special", action="store_true", help="Include del, nothing, and space.")
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=1200,
        help="Cap images per class for faster CPU training. Use 0 for no cap.",
    )
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--full-search",
        action="store_true",
        help="Also benchmark the slower calibrated SVM in addition to the fast SGD model.",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide progress bars and phase logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root

    if args.download and not dataset_root.exists():
        dataset_root = download_kaggle_dataset(dataset_root, dataset_slug=args.dataset)

    max_images = None if args.max_images_per_class == 0 else args.max_images_per_class
    result = train_classifier(
        dataset_root,
        artifacts_dir=args.artifacts_dir,
        include_motion=args.include_motion,
        include_special=args.include_special,
        max_images_per_class=max_images,
        validation_split=args.validation_split,
        random_seed=args.random_seed,
        show_progress=not args.quiet,
        full_search=args.full_search,
    )

    print(f"Selected model: {result.metrics['selected_model']}")
    print(f"Validation accuracy: {result.metrics['accuracy']:.4f}")
    print(f"Validation macro F1: {result.metrics['macro_f1']:.4f}")
    print(f"Artifacts written to: {result.model_path.parent}")


if __name__ == "__main__":
    main()
