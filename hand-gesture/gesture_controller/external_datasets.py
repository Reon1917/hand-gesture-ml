from __future__ import annotations

import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .dataset import append_sample

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FER2013_LABELS = {
    "0": "angry",
    "1": "disgust",
    "2": "fear",
    "3": "happy",
    "4": "sad",
    "5": "surprised",
    "6": "neutral",
}


def load_label_map(value: str | None) -> dict[str, str]:
    if not value:
        return {}

    candidate_path = Path(value)
    if candidate_path.exists():
        data = json.loads(candidate_path.read_text(encoding="utf-8"))
    else:
        data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("Label map must be a JSON object.")
    return {str(key): str(mapped) for key, mapped in data.items()}


def iter_labeled_image_paths(root: Path) -> Iterable[tuple[str, Path]]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = path.parent.name
        if label.startswith("."):
            continue
        yield label, path


def import_hand_image_dataset(
    image_root: Path,
    dataset_path: Path,
    detector: Any,
    *,
    label_map: dict[str, str] | None = None,
    max_images_per_label: int | None = None,
) -> dict[str, Any]:
    label_map = label_map or {}
    if not image_root.exists():
        raise FileNotFoundError(f"Hand dataset root not found: {image_root}")

    image_items = list(iter_labeled_image_paths(image_root))
    if not image_items:
        raise ValueError(f"No image files were found under {image_root}")

    imported = Counter()
    skipped = Counter()
    seen_per_label = defaultdict(int)

    for raw_label, image_path in image_items:
        if max_images_per_label is not None and seen_per_label[raw_label] >= max_images_per_label:
            continue
        seen_per_label[raw_label] += 1
        label = label_map.get(raw_label, raw_label)
        frame = cv2.imread(str(image_path))
        if frame is None:
            skipped["decode_failed"] += 1
            continue
        detection = detector.detect(frame)
        if detection.features is None:
            skipped["no_hand"] += 1
            continue

        append_sample(dataset_path, label, detection.features)
        imported[label] += 1

    return {
        "discovered_images": len(image_items),
        "imported": dict(imported),
        "skipped": dict(skipped),
        "dataset_path": str(dataset_path),
        "source_root": str(image_root),
    }


def extract_face_dataset_from_image_root(
    image_root: Path,
    detector: Any,
    *,
    label_map: dict[str, str] | None = None,
    max_images_per_label: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    label_map = label_map or {}
    if not image_root.exists():
        raise FileNotFoundError(f"Face dataset root not found: {image_root}")

    image_items = list(iter_labeled_image_paths(image_root))
    if not image_items:
        raise ValueError(f"No image files were found under {image_root}")

    features: list[np.ndarray] = []
    labels: list[str] = []
    imported = Counter()
    skipped = Counter()
    seen_per_label = defaultdict(int)

    for raw_label, image_path in image_items:
        if max_images_per_label is not None and seen_per_label[raw_label] >= max_images_per_label:
            continue
        seen_per_label[raw_label] += 1
        label = label_map.get(raw_label, raw_label)
        frame = cv2.imread(str(image_path))
        if frame is None:
            skipped["decode_failed"] += 1
            continue
        detection = detector.detect(frame)
        if detection.model_features is None:
            skipped["no_face"] += 1
            continue

        features.append(detection.model_features)
        labels.append(label)
        imported[label] += 1

    if not features:
        raise ValueError(f"No usable face detections were extracted from {image_root}.")

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=str),
        {
            "discovered_images": len(image_items),
            "imported": dict(imported),
            "skipped": dict(skipped),
            "source_root": str(image_root),
        },
    )


def extract_face_dataset_from_fer2013(
    csv_path: Path,
    detector: Any,
    *,
    usage: str = "all",
    label_map: dict[str, str] | None = None,
    max_images_per_label: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    label_map = label_map or {}
    if not csv_path.exists():
        raise FileNotFoundError(f"FER2013 CSV not found: {csv_path}")

    features: list[np.ndarray] = []
    labels: list[str] = []
    imported = Counter()
    skipped = Counter()
    seen_per_label = defaultdict(int)
    usage_filter = usage.lower()

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_usage = str(row.get("Usage", "")).strip().lower()
            if usage_filter != "all" and row_usage != usage_filter:
                continue

            raw_label = FER2013_LABELS.get(str(row.get("emotion", "")).strip())
            if raw_label is None:
                skipped["unknown_label"] += 1
                continue
            if max_images_per_label is not None and seen_per_label[raw_label] >= max_images_per_label:
                continue
            seen_per_label[raw_label] += 1

            pixels = str(row.get("pixels", "")).strip().split()
            if not pixels:
                skipped["decode_failed"] += 1
                continue
            grayscale = np.asarray([int(pixel) for pixel in pixels], dtype=np.uint8)
            if grayscale.size != 48 * 48:
                skipped["decode_failed"] += 1
                continue
            frame = cv2.cvtColor(grayscale.reshape(48, 48), cv2.COLOR_GRAY2BGR)
            detection = detector.detect(frame)
            if detection.model_features is None:
                skipped["no_face"] += 1
                continue

            label = label_map.get(raw_label, raw_label)
            features.append(detection.model_features)
            labels.append(label)
            imported[label] += 1

    if not features:
        raise ValueError(f"No usable face detections were extracted from {csv_path}.")

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=str),
        {
            "imported": dict(imported),
            "skipped": dict(skipped),
            "source_csv": str(csv_path),
            "usage": usage_filter,
        },
    )
