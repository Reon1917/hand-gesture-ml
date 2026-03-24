from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from pathlib import Path
from typing import Any

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory
import numpy as np

from .actions import ActionExecutor, load_bindings
from .dataset import append_sample, count_labels, load_dataset
from .external_datasets import FER2013_LABELS
from .face_features import FaceAnalytics, analyze_face_landmarks
from .face_landmarks import FaceLandmarkDetector
from .face_modeling import load_face_bundle, predict_expression, validate_face_bundle
from .hand_analytics import analyze_hand_landmarks
from .landmarks import HAND_CONNECTIONS, HandLandmarkDetector, serialize_hand_landmarks
from .modeling import load_bundle, predict_gesture, save_artifacts, train_classifier, validate_bundle
from .runtime import GestureStateMachine

DEFAULT_LABELS = ["peace", "thumbs_up", "fist", "open_hand", "point_down"]
DEFAULT_FACE_EXPRESSION_HELP = (
    "Train a face model with `python train_face_model.py --dataset external_datasets/fer2013/fer2013.csv "
    "--dataset-format fer2013` or point it at a labeled image-folder dataset."
)
DEFAULT_HAND_IMPORT_HELP = (
    "Import a labeled hand image dataset with `python import_hand_dataset.py --image-root external_datasets/hands`."
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def decode_data_url_image(data_url: str) -> np.ndarray:
    if "," not in data_url:
        raise ValueError("Invalid image payload.")
    _prefix, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode image payload.")
    return frame


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("Empty image payload.")
    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode image payload.")
    return frame


def _default_hand_analysis(mode: str = "collect") -> dict[str, Any]:
    return {
        "mode": mode,
        "hand_detected": False,
        "prediction": None,
        "confidence": 0.0,
        "state": "idle",
        "active_gesture": None,
        "last_action": "Waiting for a stable gesture",
        "details": {
            "finger_states": {},
            "finger_count": 0,
            "pinch_target": None,
            "pinch_strength": 0.0,
            "thumb_index_distance": 0.0,
            "openness": 0.0,
            "palm_rotation_deg": 0.0,
        },
    }


def _default_face_analysis() -> dict[str, Any]:
    return {
        "state": "no_face",
        "attention": "unknown",
        "expression": "unavailable",
        "expression_confidence": 0.0,
        "smile": 0.0,
        "mouth_open": 0.0,
        "blink": 0.0,
        "eye_open_left": 0.0,
        "eye_open_right": 0.0,
        "brow_raise": 0.0,
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
    }


class WebTestingState:
    def __init__(
        self,
        *,
        dataset_path: Path = Path("data/gestures.csv"),
        bindings_path: Path = Path("config/gesture_bindings.json"),
        model_path: Path = Path("artifacts/gesture_model.joblib"),
        metrics_path: Path = Path("artifacts/gesture_model_metrics.json"),
        confusion_matrix_path: Path = Path("artifacts/confusion_matrix.png"),
        report_path: Path = Path("artifacts/training_report.md"),
        face_model_path: Path = Path("artifacts/face_expression_model.joblib"),
        face_metrics_path: Path = Path("artifacts/face_expression_metrics.json"),
        face_confusion_matrix_path: Path = Path("artifacts/face_expression_confusion_matrix.png"),
        face_report_path: Path = Path("artifacts/face_expression_report.md"),
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.bindings_path = Path(bindings_path)
        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path)
        self.confusion_matrix_path = Path(confusion_matrix_path)
        self.report_path = Path(report_path)
        self.face_model_path = Path(face_model_path)
        self.face_metrics_path = Path(face_metrics_path)
        self.face_confusion_matrix_path = Path(face_confusion_matrix_path)
        self.face_report_path = Path(face_report_path)

        self.hand_detector: HandLandmarkDetector | None = None
        self.face_detector: FaceLandmarkDetector | None = None
        self.hand_detector_error: str | None = None
        self.face_detector_error: str | None = None
        self.lock = threading.RLock()

        self.bindings = load_bindings(self.bindings_path) if self.bindings_path.exists() else {}
        self.labels = self._derive_labels()
        self.counts = count_labels(self.dataset_path)
        self.bundle: dict[str, Any] | None = None
        self.face_bundle: dict[str, Any] | None = None
        self.metrics_summary: dict[str, Any] | None = None
        self.face_metrics_summary: dict[str, Any] | None = None
        self.last_features = None
        self.last_status = "Ready for browser testing"
        self.last_analysis = _default_hand_analysis()
        self.last_face_analysis = _default_face_analysis()
        self._hand_model_mtime: float | None = None
        self._hand_metrics_mtime: float | None = None
        self._face_model_mtime: float | None = None
        self._face_metrics_mtime: float | None = None

        self.state_machine = self._build_state_machine()
        self.executor = ActionExecutor(dry_run=True)
        self._maybe_reload_hand_artifacts()
        self._maybe_reload_face_artifacts()

    def _derive_labels(self) -> list[str]:
        labels = list(self.bindings.keys())
        if self.dataset_path.exists():
            try:
                _X, y = load_dataset(self.dataset_path)
            except Exception:
                pass
            else:
                for label in np.unique(y).tolist():
                    if label not in labels:
                        labels.append(label)
        if not labels:
            labels = list(DEFAULT_LABELS)
        return labels

    def _build_state_machine(self) -> GestureStateMachine:
        return GestureStateMachine(
            window_size=5,
            consensus_threshold=0.6,
            confidence_threshold=0.75,
            release_threshold=0.55,
            release_frames=3,
            cooldown_seconds=1.0,
        )

    def _maybe_reload_hand_artifacts(self) -> None:
        if self.model_path.exists():
            model_mtime = self.model_path.stat().st_mtime
            if self._hand_model_mtime != model_mtime:
                bundle = load_bundle(self.model_path)
                validate_bundle(bundle)
                self.bundle = bundle
                self._hand_model_mtime = model_mtime
        else:
            self.bundle = None
            self._hand_model_mtime = None

        if self.metrics_path.exists():
            metrics_mtime = self.metrics_path.stat().st_mtime
            if self._hand_metrics_mtime != metrics_mtime:
                self.metrics_summary = json.loads(self.metrics_path.read_text(encoding="utf-8"))
                self._hand_metrics_mtime = metrics_mtime
        else:
            self.metrics_summary = None
            self._hand_metrics_mtime = None

    def _maybe_reload_face_artifacts(self) -> None:
        if self.face_model_path.exists():
            model_mtime = self.face_model_path.stat().st_mtime
            if self._face_model_mtime != model_mtime:
                bundle = load_face_bundle(self.face_model_path)
                validate_face_bundle(bundle)
                self.face_bundle = bundle
                self._face_model_mtime = model_mtime
        else:
            self.face_bundle = None
            self._face_model_mtime = None

        if self.face_metrics_path.exists():
            metrics_mtime = self.face_metrics_path.stat().st_mtime
            if self._face_metrics_mtime != metrics_mtime:
                self.face_metrics_summary = json.loads(self.face_metrics_path.read_text(encoding="utf-8"))
                self._face_metrics_mtime = metrics_mtime
        else:
            self.face_metrics_summary = None
            self._face_metrics_mtime = None

    def close(self) -> None:
        if self.hand_detector is not None:
            self.hand_detector.close()
        if self.face_detector is not None:
            self.face_detector.close()

    def reset_runtime(self) -> None:
        self.state_machine = self._build_state_machine()
        self.executor = ActionExecutor(dry_run=True)
        self.last_analysis["last_action"] = self.executor.last_message

    def _ensure_hand_detector(self) -> None:
        if self.hand_detector is not None:
            return
        try:
            self.hand_detector = HandLandmarkDetector()
            self.hand_detector_error = None
        except Exception as exc:
            self.hand_detector_error = str(exc)
            raise

    def _ensure_face_detector(self) -> None:
        if self.face_detector is not None:
            return
        try:
            self.face_detector = FaceLandmarkDetector()
            self.face_detector_error = None
        except Exception as exc:
            self.face_detector_error = str(exc)
            raise

    def ui_status(self) -> dict[str, Any]:
        with self.lock:
            self._maybe_reload_hand_artifacts()
            self._maybe_reload_face_artifacts()
            return {
                "labels": self.labels,
                "bindings": self.bindings,
                "counts": dict(self.counts),
                "model_ready": self.bundle is not None,
                "face_model_ready": self.face_bundle is not None,
                "last_status": self.last_status,
                "analysis": dict(self.last_analysis),
                "face_analysis": dict(self.last_face_analysis),
                "metrics_summary": self._metrics_digest(self.metrics_summary),
                "face_metrics_summary": self._metrics_digest(self.face_metrics_summary),
                "report_preview": self._report_preview(self.report_path),
                "face_report_preview": self._report_preview(self.face_report_path),
                "confusion_matrix_url": self._artifact_url(self.confusion_matrix_path),
                "face_confusion_matrix_url": self._artifact_url(self.face_confusion_matrix_path),
                "hand_detector_ready": self.hand_detector is not None,
                "face_detector_ready": self.face_detector is not None,
                "detector_error": self.hand_detector_error or self.face_detector_error,
                "hand_detector_error": self.hand_detector_error,
                "face_detector_error": self.face_detector_error,
                "hand_connections": HAND_CONNECTIONS,
                "face_dataset_help": DEFAULT_FACE_EXPRESSION_HELP,
                "hand_dataset_help": DEFAULT_HAND_IMPORT_HELP,
                "supported_face_labels": sorted(FER2013_LABELS.values()),
            }

    def analyze_frame(self, frame: np.ndarray, mode: str) -> dict[str, Any]:
        with self.lock:
            if mode not in {"collect", "infer"}:
                raise ValueError("Mode must be 'collect' or 'infer'.")

            started = time.perf_counter()
            hand_detection = None
            face_detection = None

            try:
                self._ensure_hand_detector()
                hand_detection = self.hand_detector.detect(frame)
            except Exception as exc:
                self.hand_detector_error = str(exc)

            try:
                self._ensure_face_detector()
                face_detection = self.face_detector.detect(frame)
            except Exception as exc:
                self.face_detector_error = str(exc)

            hand_points = []
            if hand_detection is not None:
                hand_points = serialize_hand_landmarks(hand_detection.hand_landmarks)

            if mode == "collect":
                self.reset_runtime()
                self._update_collect_hand_analysis(hand_detection)
            else:
                self._update_infer_hand_analysis(hand_detection)

            self._update_face_analysis(face_detection)

            return {
                **self.ui_status(),
                "hand_landmarks": hand_points,
                "face_landmarks": [] if face_detection is None else face_detection.overlay_points,
                "processing_ms": round((time.perf_counter() - started) * 1000, 1),
            }

    def _update_collect_hand_analysis(self, hand_detection: Any | None) -> None:
        if hand_detection is None or hand_detection.features is None:
            self.last_features = None
            analysis = _default_hand_analysis("collect")
            analysis["state"] = "idle"
            analysis["last_action"] = self.last_status
            self.last_analysis = analysis
            return

        self.last_features = hand_detection.features
        details = analyze_hand_landmarks(hand_detection.features).to_dict()
        analysis = _default_hand_analysis("collect")
        analysis["hand_detected"] = True
        analysis["state"] = "tracking"
        analysis["last_action"] = self.last_status
        analysis["details"] = details
        self.last_analysis = analysis

    def _update_infer_hand_analysis(self, hand_detection: Any | None) -> None:
        if self.bundle is None:
            self.last_features = None if hand_detection is None else hand_detection.features
            analysis = _default_hand_analysis("infer")
            analysis["hand_detected"] = hand_detection is not None and hand_detection.features is not None
            analysis["state"] = "model_missing"
            analysis["last_action"] = "Train a hand model before running inference"
            if hand_detection is not None and hand_detection.features is not None:
                analysis["details"] = analyze_hand_landmarks(hand_detection.features).to_dict()
            self.last_analysis = analysis
            return

        if hand_detection is None or hand_detection.features is None:
            self.last_features = None
            update = self.state_machine.update(None, 0.0, hand_present=False, now=time.monotonic())
            analysis = _default_hand_analysis("infer")
            analysis["state"] = update.state
            analysis["active_gesture"] = update.active_label
            analysis["last_action"] = self.executor.last_message
            self.last_analysis = analysis
            return

        self.last_features = hand_detection.features
        label, confidence, _distribution = predict_gesture(self.bundle, hand_detection.features)
        update = self.state_machine.update(label, confidence, hand_present=True, now=time.monotonic())
        if update.should_trigger and update.active_label is not None:
            action_spec = self.bindings.get(update.active_label, "noop")
            self.executor.maybe_execute(update.active_label, action_spec)

        analysis = _default_hand_analysis("infer")
        analysis["hand_detected"] = True
        analysis["prediction"] = label
        analysis["confidence"] = confidence
        analysis["state"] = update.state
        analysis["active_gesture"] = update.active_label
        analysis["last_action"] = self.executor.last_message
        analysis["details"] = analyze_hand_landmarks(hand_detection.features).to_dict()
        self.last_analysis = analysis

    def _update_face_analysis(self, face_detection: Any | None) -> None:
        if face_detection is None or face_detection.analytics is None:
            self.last_face_analysis = _default_face_analysis()
            if self.face_detector_error:
                self.last_face_analysis["state"] = "detector_error"
            return

        analytics: FaceAnalytics | None
        if self.face_bundle is not None and face_detection.model_features is not None:
            label, confidence, _distribution = predict_expression(self.face_bundle, face_detection.model_features)
            analytics = analyze_face_landmarks(
                face_detection.face_landmarks,
                face_detection.face_blendshapes,
                face_detection.transformation_matrix,
                predicted_expression=label,
                predicted_confidence=confidence,
            )
        else:
            analytics = face_detection.analytics

        self.last_face_analysis = analytics.to_dict() if analytics is not None else _default_face_analysis()

    def save_sample(self, label: str) -> dict[str, Any]:
        with self.lock:
            if label not in self.labels:
                raise ValueError(f"Unknown label: {label}")
            if self.last_features is None:
                raise ValueError("No detected hand is available to save right now.")

            append_sample(self.dataset_path, label, self.last_features)
            self.counts[label] += 1
            self.last_status = f"Saved sample for {label} ({self.counts[label]} total)"
            self.last_analysis["last_action"] = self.last_status
            return {
                "ok": True,
                "counts": dict(self.counts),
                "last_status": self.last_status,
            }

    def train_model(self) -> dict[str, Any]:
        with self.lock:
            X, y = load_dataset(self.dataset_path)
            artifacts = train_classifier(X, y)
            metrics = save_artifacts(
                artifacts.bundle,
                artifacts.metrics,
                self.model_path,
                self.metrics_path,
                self.confusion_matrix_path,
                self.report_path,
            )
            self.bundle = artifacts.bundle
            self.metrics_summary = metrics
            self._hand_model_mtime = self.model_path.stat().st_mtime
            self._hand_metrics_mtime = self.metrics_path.stat().st_mtime
            self.last_status = "Hand model retrained from web UI"
            return {
                "ok": True,
                "last_status": self.last_status,
                "metrics_summary": self._metrics_digest(metrics),
                "report_preview": self._report_preview(self.report_path),
                "confusion_matrix_url": self._artifact_url(self.confusion_matrix_path),
            }

    def _metrics_digest(self, summary: dict[str, Any] | None) -> dict[str, Any] | None:
        if summary is None:
            return None
        holdout = summary.get("holdout")
        return {
            "samples": summary.get("samples"),
            "trained_with_holdout": summary.get("trained_with_holdout"),
            "accuracy": None if holdout is None else holdout.get("accuracy"),
            "top_confusions": [] if holdout is None else holdout.get("top_confusions", []),
            "warnings": summary.get("warnings", {}),
        }

    def _artifact_url(self, path: Path) -> str | None:
        if not path.exists():
            return None
        return f"/artifacts/{path.name}?v={int(path.stat().st_mtime)}"

    def _report_preview(self, path: Path) -> str:
        if not path.exists():
            return "No report yet."
        return path.read_text(encoding="utf-8")


def create_app(state: WebTestingState | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent / "web" / "templates"),
        static_folder=str(Path(__file__).resolve().parent / "web" / "static"),
    )
    app.config["STATE"] = state or WebTestingState()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        return jsonify(app.config["STATE"].ui_status())

    @app.post("/api/analyze")
    def analyze():
        try:
            if request.is_json:
                payload = request.get_json(silent=True) or {}
                data_url = payload.get("image")
                mode = payload.get("mode", "collect")
                if not isinstance(data_url, str):
                    return jsonify({"ok": False, "error": "Missing image payload."}), 400
                frame = decode_data_url_image(data_url)
            else:
                mode = request.args.get("mode", "collect")
                frame = decode_image_bytes(request.get_data())
            return jsonify(app.config["STATE"].analyze_frame(frame, mode))
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/save-sample")
    def save_sample():
        payload = request.get_json(silent=True) or {}
        label = payload.get("label")
        if not isinstance(label, str):
            return jsonify({"ok": False, "error": "Missing label."}), 400
        try:
            return jsonify(app.config["STATE"].save_sample(label))
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/train")
    def train():
        try:
            return jsonify(app.config["STATE"].train_model())
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/reset-runtime")
    def reset_runtime():
        app.config["STATE"].reset_runtime()
        return jsonify({"ok": True, "last_status": "Runtime state reset"})

    @app.get("/artifacts/<path:filename>")
    def artifact(filename: str):
        return send_from_directory(PROJECT_ROOT / "artifacts", filename)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local hand gesture testing dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = create_app()
    app.run(host=args.host, port=args.port, debug=False)
    return 0
