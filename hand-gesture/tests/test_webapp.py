from __future__ import annotations

import base64
import unittest

import cv2
import numpy as np

from gesture_controller.webapp import create_app


class FakeState:
    def __init__(self) -> None:
        self.reset_called = False

    def ui_status(self):
        return {
            "labels": ["peace"],
            "bindings": {"peace": "key:space"},
            "counts": {"peace": 1},
            "model_ready": True,
            "last_status": "ok",
            "analysis": {
                "mode": "collect",
                "hand_detected": False,
                "prediction": None,
                "confidence": 0.0,
                "state": "idle",
                "active_gesture": None,
                "last_action": "Waiting",
            },
            "metrics_summary": None,
            "report_preview": "No report",
            "confusion_matrix_url": None,
        }

    def analyze_frame(self, data_url: str, mode: str):
        return {**self.ui_status(), "hand_landmarks": [], "processing_ms": 12.0}

    def save_sample(self, label: str):
        return {"ok": True, "counts": {"peace": 2}, "last_status": f"Saved {label}"}

    def train_model(self):
        return {"ok": True, "last_status": "trained", "metrics_summary": {"samples": 10}}

    def reset_runtime(self):
        self.reset_called = True


class WebAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = FakeState()
        self.app = create_app(self.state)
        self.client = self.app.test_client()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", frame)
        assert ok
        payload = base64.b64encode(encoded.tobytes()).decode("ascii")
        self.image_data_url = f"data:image/jpeg;base64,{payload}"

    def test_status_endpoint(self) -> None:
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["labels"], ["peace"])

    def test_analyze_endpoint(self) -> None:
        response = self.client.post("/api/analyze", json={"image": self.image_data_url, "mode": "collect"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("hand_landmarks", payload)
        self.assertIn("processing_ms", payload)

    def test_save_sample_endpoint(self) -> None:
        response = self.client.post("/api/save-sample", json={"label": "peace"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ok"])

    def test_reset_runtime_endpoint(self) -> None:
        response = self.client.post("/api/reset-runtime")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.state.reset_called)


if __name__ == "__main__":
    unittest.main()
