from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np

LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
LEFT_BROW = 105
RIGHT_BROW = 334
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
UPPER_LIP = 13
LOWER_LIP = 14
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

FACE_BLENDSHAPE_FEATURES = (
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthLeft",
    "mouthPucker",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
)
FACE_GEOMETRY_DIM = 12
FACE_FEATURE_SPEC_VERSION = "face_blendshape_geometry_v1"
FACE_MODEL_FEATURE_DIM = len(FACE_BLENDSHAPE_FEATURES) + FACE_GEOMETRY_DIM


@dataclass(slots=True)
class FaceAnalytics:
    state: str
    attention: str
    expression: str
    expression_confidence: float
    smile: float
    mouth_open: float
    blink: float
    eye_open_left: float
    eye_open_right: float
    brow_raise: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_face_landmark_array(raw_landmarks: object | None) -> np.ndarray | None:
    if raw_landmarks is None:
        return None

    if isinstance(raw_landmarks, np.ndarray):
        values = np.asarray(raw_landmarks, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"Expected face landmarks with shape (n, 3), got {values.shape}.")
        return values

    if hasattr(raw_landmarks, "__len__") and hasattr(raw_landmarks, "__getitem__"):
        points = [
            [float(landmark.x), float(landmark.y), float(landmark.z)]
            for landmark in raw_landmarks
        ]
        return np.asarray(points, dtype=np.float32)

    if hasattr(raw_landmarks, "landmark"):
        return _as_face_landmark_array(raw_landmarks.landmark)

    raise TypeError("Unsupported face landmark format.")


def _to_blendshape_scores(face_blendshapes: object | None) -> dict[str, float]:
    if face_blendshapes is None:
        return {}

    if isinstance(face_blendshapes, dict):
        return {str(key): float(value) for key, value in face_blendshapes.items()}

    if isinstance(face_blendshapes, Iterable):
        scores: dict[str, float] = {}
        for category in face_blendshapes:
            if hasattr(category, "category_name") and hasattr(category, "score"):
                scores[str(category.category_name)] = float(category.score)
            elif isinstance(category, dict):
                name = category.get("category_name")
                score = category.get("score")
                if name is not None and score is not None:
                    scores[str(name)] = float(score)
        return scores

    raise TypeError("Unsupported face blendshape format.")


def _distance(points: np.ndarray, first: int, second: int) -> float:
    return float(np.linalg.norm(points[first] - points[second]))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-6:
        return 0.0
    return float(numerator / denominator)


def _euler_from_matrix(transformation_matrix: object | None) -> tuple[float, float, float]:
    if transformation_matrix is None:
        return 0.0, 0.0, 0.0

    matrix = np.asarray(transformation_matrix, dtype=np.float32)
    if matrix.shape == (4, 4):
        rotation = matrix[:3, :3]
    elif matrix.shape == (3, 3):
        rotation = matrix
    elif matrix.size == 16:
        rotation = matrix.reshape(4, 4)[:3, :3]
    elif matrix.size == 9:
        rotation = matrix.reshape(3, 3)
    else:
        return 0.0, 0.0, 0.0

    sy = math.sqrt(float(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(-rotation[2, 0]), sy)
        roll = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        pitch = math.atan2(float(-rotation[1, 2]), float(rotation[1, 1]))
        yaw = math.atan2(float(-rotation[2, 0]), sy)
        roll = 0.0

    return (
        float(math.degrees(yaw)),
        float(math.degrees(pitch)),
        float(math.degrees(roll)),
    )


def build_face_metrics(
    raw_landmarks: object | None,
    face_blendshapes: object | None = None,
    transformation_matrix: object | None = None,
) -> dict[str, float]:
    points = _as_face_landmark_array(raw_landmarks)
    scores = _to_blendshape_scores(face_blendshapes)
    if points is None or len(points) <= RIGHT_CHEEK:
        return {
            "mouth_open_ratio": 0.0,
            "mouth_width_ratio": 0.0,
            "eye_open_left": 0.0,
            "eye_open_right": 0.0,
            "brow_raise": 0.0,
            "smile": 0.0,
            "blink": 0.0,
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "attention_score": 0.0,
            "jaw_open": 0.0,
        }

    face_width = _distance(points, LEFT_CHEEK, RIGHT_CHEEK)
    eye_span = _distance(points, LEFT_EYE_OUTER, RIGHT_EYE_OUTER)
    reference = max(face_width, eye_span, 1e-6)

    mouth_open = _safe_ratio(_distance(points, UPPER_LIP, LOWER_LIP), reference)
    mouth_width = _safe_ratio(_distance(points, MOUTH_LEFT, MOUTH_RIGHT), reference)
    eye_open_left = _safe_ratio(_distance(points, LEFT_EYE_TOP, LEFT_EYE_BOTTOM), reference)
    eye_open_right = _safe_ratio(_distance(points, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM), reference)
    brow_raise_left = _safe_ratio(_distance(points, LEFT_BROW, LEFT_EYE_TOP), reference)
    brow_raise_right = _safe_ratio(_distance(points, RIGHT_BROW, RIGHT_EYE_TOP), reference)
    brow_raise = float((brow_raise_left + brow_raise_right) / 2.0)

    yaw_deg, pitch_deg, roll_deg = _euler_from_matrix(transformation_matrix)
    if abs(yaw_deg) < 0.1 and abs(pitch_deg) < 0.1 and abs(roll_deg) < 0.1:
        left_span = _distance(points, NOSE_TIP, LEFT_CHEEK)
        right_span = _distance(points, NOSE_TIP, RIGHT_CHEEK)
        yaw_deg = max(-45.0, min(45.0, (right_span - left_span) * 180.0))
        forehead_span = _distance(points, FOREHEAD, NOSE_TIP)
        chin_span = _distance(points, NOSE_TIP, CHIN)
        pitch_deg = max(-45.0, min(45.0, (chin_span - forehead_span) * 180.0))
        roll_deg = math.degrees(
            math.atan2(
                float(points[RIGHT_EYE_OUTER][1] - points[LEFT_EYE_OUTER][1]),
                float(points[RIGHT_EYE_OUTER][0] - points[LEFT_EYE_OUTER][0]),
            )
        )

    smile = float(
        max(
            scores.get("mouthSmileLeft", 0.0),
            scores.get("mouthSmileRight", 0.0),
            min(1.0, mouth_width * 2.6),
        )
    )
    blink = float(
        max(
            scores.get("eyeBlinkLeft", 0.0),
            scores.get("eyeBlinkRight", 0.0),
            max(0.0, 1.0 - ((eye_open_left + eye_open_right) * 24.0)),
        )
    )
    jaw_open = float(max(scores.get("jawOpen", 0.0), min(1.0, mouth_open * 9.0)))
    attention_score = float(
        max(
            0.0,
            1.0 - min(1.0, (abs(yaw_deg) / 30.0) + (abs(pitch_deg) / 30.0)),
        )
    )

    return {
        "mouth_open_ratio": float(mouth_open),
        "mouth_width_ratio": float(mouth_width),
        "eye_open_left": float(eye_open_left),
        "eye_open_right": float(eye_open_right),
        "brow_raise": brow_raise,
        "smile": smile,
        "blink": blink,
        "jaw_open": jaw_open,
        "yaw_deg": float(yaw_deg),
        "pitch_deg": float(pitch_deg),
        "roll_deg": float(roll_deg),
        "attention_score": attention_score,
    }


def infer_expression(face_blendshapes: object | None, metrics: dict[str, float]) -> tuple[str, float]:
    scores = _to_blendshape_scores(face_blendshapes)
    happy = max(scores.get("mouthSmileLeft", 0.0), scores.get("mouthSmileRight", 0.0), metrics["smile"])
    surprised = max(scores.get("jawOpen", 0.0), metrics["jaw_open"]) * 0.7 + max(
        scores.get("eyeWideLeft", 0.0),
        scores.get("eyeWideRight", 0.0),
        metrics["eye_open_left"] * 9.0,
        metrics["eye_open_right"] * 9.0,
    ) * 0.3
    angry = (
        (scores.get("browDownLeft", 0.0) + scores.get("browDownRight", 0.0)) * 0.45
        + (scores.get("mouthPressLeft", 0.0) + scores.get("mouthPressRight", 0.0)) * 0.3
        + (scores.get("eyeSquintLeft", 0.0) + scores.get("eyeSquintRight", 0.0)) * 0.25
    )
    sad = (
        (scores.get("mouthFrownLeft", 0.0) + scores.get("mouthFrownRight", 0.0)) * 0.55
        + scores.get("browInnerUp", 0.0) * 0.45
    )
    neutral = max(
        0.0,
        1.0 - max(happy, surprised, angry, sad),
    )

    candidates = {
        "happy": float(happy),
        "surprised": float(surprised),
        "angry": float(angry),
        "sad": float(sad),
        "neutral": float(neutral),
    }
    label = max(candidates, key=candidates.get)
    confidence = float(max(candidates.values()))
    return label, confidence


def extract_face_features(
    raw_landmarks: object | None,
    face_blendshapes: object | None = None,
    transformation_matrix: object | None = None,
) -> np.ndarray:
    scores = _to_blendshape_scores(face_blendshapes)
    metrics = build_face_metrics(raw_landmarks, face_blendshapes, transformation_matrix)

    feature_values = [float(scores.get(name, 0.0)) for name in FACE_BLENDSHAPE_FEATURES]
    feature_values.extend(
        [
            metrics["mouth_open_ratio"],
            metrics["mouth_width_ratio"],
            metrics["eye_open_left"],
            metrics["eye_open_right"],
            metrics["brow_raise"],
            metrics["smile"],
            metrics["blink"],
            metrics["jaw_open"],
            metrics["yaw_deg"] / 45.0,
            metrics["pitch_deg"] / 45.0,
            metrics["roll_deg"] / 45.0,
            metrics["attention_score"],
        ]
    )
    features = np.asarray(feature_values, dtype=np.float32)
    if features.size != FACE_MODEL_FEATURE_DIM:
        raise ValueError(f"Expected {FACE_MODEL_FEATURE_DIM} face features, got {features.size}.")
    return features


def analyze_face_landmarks(
    raw_landmarks: object | None,
    face_blendshapes: object | None = None,
    transformation_matrix: object | None = None,
    *,
    predicted_expression: str | None = None,
    predicted_confidence: float | None = None,
) -> FaceAnalytics | None:
    if raw_landmarks is None:
        return None

    metrics = build_face_metrics(raw_landmarks, face_blendshapes, transformation_matrix)
    heuristic_expression, heuristic_confidence = infer_expression(face_blendshapes, metrics)
    expression = predicted_expression or heuristic_expression
    expression_confidence = (
        float(predicted_confidence)
        if predicted_confidence is not None
        else float(heuristic_confidence)
    )
    attention = "focused" if metrics["attention_score"] >= 0.58 else "away"

    return FaceAnalytics(
        state="tracking",
        attention=attention,
        expression=expression,
        expression_confidence=expression_confidence,
        smile=metrics["smile"],
        mouth_open=metrics["mouth_open_ratio"],
        blink=metrics["blink"],
        eye_open_left=metrics["eye_open_left"],
        eye_open_right=metrics["eye_open_right"],
        brow_raise=metrics["brow_raise"],
        yaw_deg=metrics["yaw_deg"],
        pitch_deg=metrics["pitch_deg"],
        roll_deg=metrics["roll_deg"],
    )
