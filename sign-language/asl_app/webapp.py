from __future__ import annotations

import argparse
import base64
import threading
import time
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from PIL import Image, ImageOps

from .asl_reference import MANUAL_TESTING_NOTES, default_reference_cards
from .modeling import MODEL_FILE_NAME, load_bundle, predict_with_bundle

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = PACKAGE_ROOT / "web" / "static"
TEMPLATE_ROOT = PACKAGE_ROOT / "web" / "templates"


class PredictionRequest(BaseModel):
    image: str


def decode_data_url(payload: str) -> Image.Image:
    encoded = payload.split(",", 1)[1] if "," in payload else payload
    try:
        raw_bytes = base64.b64decode(encoded)
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise ValueError("Image payload is not valid base64 data.") from exc

    with Image.open(BytesIO(raw_bytes)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


class SignLanguageAppState:
    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = Path(project_root) if project_root else PACKAGE_ROOT
        self.artifacts_dir = self.project_root / "artifacts"
        self.model_path = self.artifacts_dir / MODEL_FILE_NAME
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.bundle: dict[str, object] | None = None
        self.examples = default_reference_cards()
        self.metadata: dict[str, object] = {}
        self.last_status = "Train the ASL model to enable live webcam inference."
        self.reload_model()

    def reload_model(self) -> None:
        with self.lock:
            if not self.model_path.exists():
                self.bundle = None
                self.examples = default_reference_cards()
                self.metadata = {}
                self.last_status = "Model not found. Run train_model.py before opening the camera UI."
                return

            self.bundle = load_bundle(self.model_path)
            self.examples = self.bundle.get("examples", default_reference_cards())
            self.metadata = self.bundle.get("metadata", {})
            model_name = self.metadata.get("model_name", "model")
            dataset_size = self.metadata.get("dataset_size", 0)
            self.last_status = f"Loaded {model_name} trained on {dataset_size} images."

    def status_payload(self) -> dict[str, object]:
        with self.lock:
            return {
                "model_ready": self.bundle is not None,
                "last_status": self.last_status,
                "examples": self.examples,
                "guidance": self.metadata.get("testing_notes", MANUAL_TESTING_NOTES),
                "model_summary": {
                    "model_name": self.metadata.get("model_name"),
                    "dataset_size": self.metadata.get("dataset_size"),
                    "validation_accuracy": self.metadata.get("validation_accuracy"),
                    "validation_macro_f1": self.metadata.get("validation_macro_f1"),
                    "labels": self.metadata.get("labels", []),
                    "excluded_labels": self.metadata.get("excluded_labels", []),
                },
            }

    def predict(self, image: Image.Image) -> dict[str, object]:
        with self.lock:
            if self.bundle is None:
                raise RuntimeError("Model not trained yet.")
            return predict_with_bundle(self.bundle, image)


def create_app(state: SignLanguageAppState | None = None) -> FastAPI:
    shared_state = state or SignLanguageAppState()
    templates = Jinja2Templates(directory=str(TEMPLATE_ROOT))

    app = FastAPI(title="ASL Webcam Studio", docs_url=None, redoc_url=None)
    app.state.sign_language = shared_state
    app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")
    app.mount("/artifacts", StaticFiles(directory=str(shared_state.artifacts_dir)), name="artifacts")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="index.html", context={})

    @app.get("/api/status")
    async def status() -> dict[str, object]:
        return shared_state.status_payload()

    @app.post("/api/reload-model")
    async def reload_model() -> dict[str, object]:
        shared_state.reload_model()
        return shared_state.status_payload()

    @app.post("/api/predict")
    async def predict(payload: PredictionRequest) -> dict[str, object]:
        started = time.perf_counter()
        try:
            image = decode_data_url(payload.image)
            prediction = shared_state.predict(image)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        prediction["processing_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        prediction["message"] = f"Most likely sign: {prediction['label']}"
        return prediction

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ASL webcam web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()
