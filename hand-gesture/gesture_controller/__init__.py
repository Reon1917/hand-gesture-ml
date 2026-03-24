"""Hand gesture, face analysis, dataset import, and live control utilities."""

from pathlib import Path
import os

if "MPLCONFIGDIR" not in os.environ:
    matplotlib_cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)

from .dataset import FEATURE_DIM
from .features import FEATURE_SPEC_VERSION, MODEL_FEATURE_DIM, RAW_FEATURE_DIM
from .face_features import FACE_FEATURE_SPEC_VERSION, FACE_MODEL_FEATURE_DIM

__all__ = [
    "FEATURE_DIM",
    "RAW_FEATURE_DIM",
    "MODEL_FEATURE_DIM",
    "FEATURE_SPEC_VERSION",
    "FACE_MODEL_FEATURE_DIM",
    "FACE_FEATURE_SPEC_VERSION",
]
