from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time

import cv2

if "MPLCONFIGDIR" not in os.environ:
    matplotlib_cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)

import mediapipe as mp
import numpy as np

from .model_assets import ensure_hand_landmarker_model


@dataclass(slots=True)
class DetectionResult:
    features: np.ndarray | None
    hand_landmarks: object | None


def _iterate_landmarks(hand_landmarks: object) -> object:
    if hasattr(hand_landmarks, "landmark"):
        return hand_landmarks.landmark
    return hand_landmarks


def normalize_landmarks(hand_landmarks: object) -> np.ndarray:
    coordinates = np.asarray(
        [[landmark.x, landmark.y, landmark.z] for landmark in _iterate_landmarks(hand_landmarks)],
        dtype=np.float32,
    )
    coordinates -= coordinates[0]

    scale = float(np.max(np.linalg.norm(coordinates, axis=1)))
    if not np.isfinite(scale) or scale < 1e-6:
        return np.zeros(coordinates.size, dtype=np.float32)

    coordinates /= scale
    return coordinates.reshape(-1)


class HandLandmarkDetector:
    def __init__(
        self,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        model_path: Path | None = None,
        auto_download_model: bool = True,
    ) -> None:
        self._backend = "solutions" if hasattr(mp, "solutions") else "tasks"
        self._hands = None
        self._mp_hands = None
        self._drawing_utils = None
        self._drawing_styles = None
        self._hand_connections = None
        self._last_timestamp_ms = 0

        if self._backend == "solutions":
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._drawing_utils = mp.solutions.drawing_utils
            self._drawing_styles = mp.solutions.drawing_styles
            self._hand_connections = self._mp_hands.HAND_CONNECTIONS
            return

        model_asset_path = ensure_hand_landmarker_model(
            model_path,
            auto_download=auto_download_model,
        )
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_asset_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._hands = vision.HandLandmarker.create_from_options(options)
        self._drawing_utils = vision.drawing_utils
        self._drawing_styles = vision.drawing_styles
        self._hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    def _next_timestamp_ms(self) -> int:
        now_ms = time.monotonic_ns() // 1_000_000
        self._last_timestamp_ms = max(now_ms, self._last_timestamp_ms + 1)
        return self._last_timestamp_ms

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._backend == "solutions":
            frame_rgb.flags.writeable = False
            results = self._hands.process(frame_rgb)
            frame_rgb.flags.writeable = True

            if not results.multi_hand_landmarks:
                return DetectionResult(features=None, hand_landmarks=None)

            hand_landmarks = results.multi_hand_landmarks[0]
            return DetectionResult(
                features=normalize_landmarks(hand_landmarks),
                hand_landmarks=hand_landmarks,
            )

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self._hands.detect_for_video(image, self._next_timestamp_ms())
        if not results.hand_landmarks:
            return DetectionResult(features=None, hand_landmarks=None)

        hand_landmarks = results.hand_landmarks[0]
        return DetectionResult(
            features=normalize_landmarks(hand_landmarks),
            hand_landmarks=hand_landmarks,
        )

    def draw(self, frame_bgr: np.ndarray, hand_landmarks: object | None) -> None:
        if hand_landmarks is None:
            return

        self._drawing_utils.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            self._hand_connections,
            self._drawing_styles.get_default_hand_landmarks_style(),
            self._drawing_styles.get_default_hand_connections_style(),
        )

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
