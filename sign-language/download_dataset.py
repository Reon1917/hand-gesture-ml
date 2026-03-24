from __future__ import annotations

import argparse
from pathlib import Path

from asl_app.runtime_bootstrap import bootstrap_local_venv

bootstrap_local_venv(__file__, ("kagglehub",))

from asl_app.dataset import DEFAULT_DATASET_SLUG, download_kaggle_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Kaggle ASL dataset for local training.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_SLUG, help="Kaggle dataset slug.")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/asl-alphabet"),
        help="Where to create the local dataset link.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing dataset link.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    linked_path = download_kaggle_dataset(args.destination, dataset_slug=args.dataset, force=args.force)
    print(f"Dataset ready at {linked_path.resolve() if linked_path.exists() else linked_path}")


if __name__ == "__main__":
    main()
