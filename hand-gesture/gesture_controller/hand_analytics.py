from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .features import compute_joint_angle, normalize_landmark_array

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_ANGLE_GROUPS = {
    "thumb": ((1, 2, 3), (2, 3, 4)),
    "index": ((5, 6, 7), (6, 7, 8)),
    "middle": ((9, 10, 11), (10, 11, 12)),
    "ring": ((13, 14, 15), (14, 15, 16)),
    "pinky": ((17, 18, 19), (18, 19, 20)),
}
THUMB_TIP_INDEX = 4
THUMB_PINCH_TARGETS = {
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
WRIST_INDEX = 0
INDEX_MCP_INDEX = 5
PINKY_MCP_INDEX = 17
OPEN_ANGLE_THRESHOLD = 2.35
HALF_ANGLE_THRESHOLD = 1.55


@dataclass(slots=True)
class HandAnalytics:
    finger_states: dict[str, str]
    finger_count: int
    pinch_target: str | None
    pinch_strength: float
    thumb_index_distance: float
    openness: float
    palm_rotation_deg: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _classify_finger_state(angle_values: list[float]) -> str:
    mean_angle = float(np.mean(angle_values))
    if mean_angle >= OPEN_ANGLE_THRESHOLD:
        return "open"
    if mean_angle >= HALF_ANGLE_THRESHOLD:
        return "half"
    return "curled"


def _pinch_strength(distance: float) -> float:
    normalized = max(0.0, min(1.0, 1.0 - (distance / 0.45)))
    return float(normalized)


def analyze_hand_landmarks(raw_landmarks: np.ndarray | list[float]) -> HandAnalytics:
    coordinates = normalize_landmark_array(raw_landmarks)
    finger_states: dict[str, str] = {}

    for finger_name, triplets in FINGER_ANGLE_GROUPS.items():
        angles = [
            compute_joint_angle(coordinates[start], coordinates[middle], coordinates[end])
            for start, middle, end in triplets
        ]
        finger_states[finger_name] = _classify_finger_state(angles)

    finger_count = sum(1 for state in finger_states.values() if state == "open")
    thumb_tip = coordinates[THUMB_TIP_INDEX]
    pinch_scores = {
        finger_name: _pinch_strength(float(np.linalg.norm(thumb_tip - coordinates[target_index])))
        for finger_name, target_index in THUMB_PINCH_TARGETS.items()
    }
    pinch_target = max(pinch_scores, key=pinch_scores.get)
    if pinch_scores[pinch_target] < 0.35:
        pinch_target = None

    palm_vector = coordinates[PINKY_MCP_INDEX] - coordinates[INDEX_MCP_INDEX]
    palm_rotation_deg = float(math.degrees(math.atan2(float(palm_vector[1]), float(palm_vector[0]))))
    openness = float(
        np.mean(
            [
                np.linalg.norm(coordinates[index] - coordinates[WRIST_INDEX])
                for index in (4, 8, 12, 16, 20)
            ]
        )
    )
    thumb_index_distance = float(np.linalg.norm(coordinates[4] - coordinates[8]))

    return HandAnalytics(
        finger_states=finger_states,
        finger_count=finger_count,
        pinch_target=pinch_target,
        pinch_strength=pinch_scores.get(pinch_target, max(pinch_scores.values(), default=0.0)),
        thumb_index_distance=thumb_index_distance,
        openness=openness,
        palm_rotation_deg=palm_rotation_deg,
    )
