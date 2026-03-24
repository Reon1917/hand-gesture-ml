from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asl_app.webapp import SignLanguageAppState, create_app


class WebAppTests(unittest.TestCase):
    def test_status_without_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state = SignLanguageAppState(project_root=Path(temp_dir))
            client = TestClient(create_app(state=state))
            response = client.get("/api/status")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertFalse(payload["model_ready"])
            self.assertIn("examples", payload)

    def test_index_route_renders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state = SignLanguageAppState(project_root=Path(temp_dir))
            client = TestClient(create_app(state=state))
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            self.assertIn("ASL Webcam Studio", response.text)


if __name__ == "__main__":
    unittest.main()
