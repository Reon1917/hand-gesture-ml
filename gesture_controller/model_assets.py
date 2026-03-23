from __future__ import annotations

import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HAND_LANDMARKER_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
DEFAULT_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_hand_landmarker_model(model_path: Path | None = None, *, auto_download: bool = True) -> Path:
    resolved_path = Path(model_path or DEFAULT_HAND_LANDMARKER_PATH)
    if resolved_path.exists():
        return resolved_path

    if not auto_download:
        raise FileNotFoundError(
            f"Hand landmarker model not found at {resolved_path}. "
            f"Download it from {DEFAULT_HAND_LANDMARKER_URL} and place it there."
        )

    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urllib.request.urlopen(DEFAULT_HAND_LANDMARKER_URL, timeout=30) as response:
            with tempfile.NamedTemporaryFile(delete=False, dir=resolved_path.parent) as handle:
                shutil.copyfileobj(response, handle)
                temp_path = Path(handle.name)
        temp_path.replace(resolved_path)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Unable to download the MediaPipe hand landmarker model automatically. "
            f"Download it from {DEFAULT_HAND_LANDMARKER_URL} and save it to {resolved_path}."
        ) from exc
    except Exception:
        if "temp_path" in locals():
            temp_path.unlink(missing_ok=True)
        raise

    return resolved_path
