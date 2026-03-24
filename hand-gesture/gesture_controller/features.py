from __future__ import annotations

import math

import numpy as np

RAW_FEATURE_DIM = 63
MODEL_FEATURE_DIM = 82
FEATURE_SPEC_VERSION = "landmarks_v2_engineered_82"

WRIST = 0
FINGERTIP_INDICES = (4, 8, 12, 16, 20)
ADJACENT_TIP_PAIRS = ((4, 8), (8, 12), (12, 16), (16, 20))
ANGLE_TRIPLETS = (
    (1, 2, 3),
    (2, 3, 4),
    (5, 6, 7),
    (6, 7, 8),
    (9, 10, 11),
    (10, 11, 12),
    (13, 14, 15),
    (14, 15, 16),
    (17, 18, 19),
    (18, 19, 20),
)


def as_landmark_array(raw_landmarks: np.ndarray | list[float]) -> np.ndarray:
    values = np.asarray(raw_landmarks, dtype=np.float32)
    if values.shape == (21, 3):
        return values.copy()
    if values.ndim == 1 and values.size == RAW_FEATURE_DIM:
        return values.reshape(21, 3).copy()
    raise ValueError(f"Expected 21x3 landmarks or {RAW_FEATURE_DIM} flat values, got shape {values.shape}.")


def normalize_landmark_array(raw_landmarks: np.ndarray | list[float]) -> np.ndarray:
    coordinates = as_landmark_array(raw_landmarks)
    coordinates -= coordinates[WRIST]

    scale = float(np.max(np.linalg.norm(coordinates, axis=1)))
    if not np.isfinite(scale) or scale < 1e-6:
        return np.zeros((21, 3), dtype=np.float32)

    coordinates /= scale
    return coordinates


def compute_joint_angle(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
) -> float:
    vector_ab = point_a - point_b
    vector_cb = point_c - point_b

    norm_ab = float(np.linalg.norm(vector_ab))
    norm_cb = float(np.linalg.norm(vector_cb))
    if norm_ab < 1e-6 or norm_cb < 1e-6:
        return 0.0

    cosine = float(np.dot(vector_ab, vector_cb) / (norm_ab * norm_cb))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(math.acos(cosine))


def extract_model_features(raw_landmarks: np.ndarray | list[float]) -> np.ndarray:
    coordinates = normalize_landmark_array(raw_landmarks)
    raw_coordinates = coordinates.reshape(-1)

    tip_to_wrist_distances = np.asarray(
        [np.linalg.norm(coordinates[index] - coordinates[WRIST]) for index in FINGERTIP_INDICES],
        dtype=np.float32,
    )
    adjacent_tip_distances = np.asarray(
        [np.linalg.norm(coordinates[start] - coordinates[end]) for start, end in ADJACENT_TIP_PAIRS],
        dtype=np.float32,
    )
    bend_angles = np.asarray(
        [
            compute_joint_angle(coordinates[start], coordinates[middle], coordinates[end])
            for start, middle, end in ANGLE_TRIPLETS
        ],
        dtype=np.float32,
    )

    features = np.concatenate(
        [raw_coordinates, tip_to_wrist_distances, adjacent_tip_distances, bend_angles]
    ).astype(np.float32, copy=False)
    if features.size != MODEL_FEATURE_DIM:
        raise ValueError(f"Expected {MODEL_FEATURE_DIM} model features, got {features.size}.")
    return features
