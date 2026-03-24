from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from .features import RAW_FEATURE_DIM

FEATURE_DIM = RAW_FEATURE_DIM
CSV_HEADER = ["label", *[f"feature_{index}" for index in range(FEATURE_DIM)]]


def ensure_dataset(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)


def append_sample(csv_path: Path, label: str, features: np.ndarray) -> None:
    values = np.asarray(features, dtype=np.float32).reshape(-1)
    if values.size != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} features, got {values.size}.")

    ensure_dataset(csv_path)
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([label, *values.tolist()])


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if len(rows) <= 1:
        raise ValueError(f"Dataset is empty: {csv_path}")

    labels: list[str] = []
    features: list[list[float]] = []

    for row in rows[1:]:
        if not row:
            continue
        labels.append(row[0])
        features.append([float(value) for value in row[1:]])

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=str)

    if X.ndim != 2 or X.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected dataset with {FEATURE_DIM} features, got shape {X.shape}.")

    return X, y


def count_labels(csv_path: Path) -> Counter[str]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return Counter()

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return Counter(row[0] for row in reader if row)
