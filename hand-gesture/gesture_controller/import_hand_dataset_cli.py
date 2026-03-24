from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .external_datasets import import_hand_image_dataset, iter_labeled_image_paths, load_label_map
from .landmarks import HandLandmarkDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a labeled hand-image dataset into the landmark CSV.")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--dataset-out", type=Path, default=Path("data/gestures.csv"))
    parser.add_argument("--label-map", help="Inline JSON or a path to a JSON label mapping file.")
    parser.add_argument("--max-images-per-label", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    label_map = load_label_map(args.label_map)
    if not args.image_root.exists():
        print(f"Hand dataset root not found: {args.image_root}", file=sys.stderr)
        return 2
    if not any(True for _ in iter_labeled_image_paths(args.image_root)):
        print(f"No image files were found under {args.image_root}.", file=sys.stderr)
        return 2

    try:
        detector = HandLandmarkDetector(static_image_mode=True)
    except Exception as exc:
        print(f"Failed to start the hand detector: {exc}", file=sys.stderr)
        return 2

    try:
        stats = import_hand_image_dataset(
            args.image_root,
            args.dataset_out,
            detector,
            label_map=label_map,
            max_images_per_label=args.max_images_per_label,
        )
    except Exception as exc:
        print(f"Failed to import the external hand dataset: {exc}", file=sys.stderr)
        detector.close()
        return 2

    detector.close()
    print(f"Imported hand samples into {args.dataset_out}")
    print(json.dumps(stats, indent=2))
    return 0
