from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import mediapipe as mp
import numpy as np

from .face_features import (
    CHIN,
    FOREHEAD,
    LEFT_BROW,
    LEFT_EYE_BOTTOM,
    LEFT_EYE_OUTER,
    LOWER_LIP,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    NOSE_TIP,
    RIGHT_BROW,
    RIGHT_EYE_BOTTOM,
    RIGHT_EYE_OUTER,
    UPPER_LIP,
    analyze_face_landmarks,
    extract_face_features,
)
from .model_assets import ensure_face_landmarker_model

FACE_OVERLAY_INDICES = (
    LEFT_EYE_OUTER,
    LEFT_EYE_BOTTOM,
    RIGHT_EYE_OUTER,
    RIGHT_EYE_BOTTOM,
    LEFT_BROW,
    RIGHT_BROW,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    UPPER_LIP,
    LOWER_LIP,
    NOSE_TIP,
    FOREHEAD,
    CHIN,
)


@dataclass(slots=True)
class FaceDetectionResult:
    model_features: np.ndarray | None
    face_landmarks: object | None
    face_blendshapes: object | None
    transformation_matrix: object | None
    analytics: object | None
    overlay_points: list[dict[str, float]]


def serialize_face_overlay_points(face_landmarks: object | None) -> list[dict[str, float]]:
    if face_landmarks is None:
        return []

    if hasattr(face_landmarks, "landmark"):
        landmarks = face_landmarks.landmark
    else:
        landmarks = face_landmarks

    points: list[dict[str, float]] = []
    for index in FACE_OVERLAY_INDICES:
        if index >= len(landmarks):
            continue
        landmark = landmarks[index]
        points.append(
            {
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
            }
        )
    return points


class FaceLandmarkDetector:
    def __init__(
        self,
        *,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        auto_download_model: bool = True,
    ) -> None:
        if not hasattr(mp, "tasks"):
            raise RuntimeError("This MediaPipe build does not expose the Tasks Face Landmarker API.")

        self._last_timestamp_ms = 0
        self._static_image_mode = static_image_mode
        model_asset_path = ensure_face_landmarker_model(auto_download=auto_download_model)
        vision = mp.tasks.vision
        running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_asset_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=running_mode,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def _next_timestamp_ms(self) -> int:
        now_ms = time.monotonic_ns() // 1_000_000
        self._last_timestamp_ms = max(now_ms, self._last_timestamp_ms + 1)
        return self._last_timestamp_ms

    def detect(self, frame_bgr: np.ndarray) -> FaceDetectionResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if self._static_image_mode:
            results = self._detector.detect(image)
        else:
            results = self._detector.detect_for_video(image, self._next_timestamp_ms())

        if not results.face_landmarks:
            return FaceDetectionResult(
                model_features=None,
                face_landmarks=None,
                face_blendshapes=None,
                transformation_matrix=None,
                analytics=None,
                overlay_points=[],
            )

        face_landmarks = results.face_landmarks[0]
        face_blendshapes = results.face_blendshapes[0] if results.face_blendshapes else None
        transformation_matrix = (
            results.facial_transformation_matrixes[0]
            if results.facial_transformation_matrixes
            else None
        )
        return FaceDetectionResult(
            model_features=extract_face_features(face_landmarks, face_blendshapes, transformation_matrix),
            face_landmarks=face_landmarks,
            face_blendshapes=face_blendshapes,
            transformation_matrix=transformation_matrix,
            analytics=analyze_face_landmarks(face_landmarks, face_blendshapes, transformation_matrix),
            overlay_points=serialize_face_overlay_points(face_landmarks),
        )

    def close(self) -> None:
        self._detector.close()
