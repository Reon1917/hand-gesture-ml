from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = 64
CELL_SIZE = 8
ORIENTATION_BINS = 9
THUMB_SIZE = 16

_RESAMPLING = getattr(Image, "Resampling", Image)
_BILINEAR = _RESAMPLING.BILINEAR


def image_from_bytes(image_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(image_bytes)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def preprocess_image(image: Image.Image, image_size: int = IMAGE_SIZE) -> np.ndarray:
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    grayscale = ImageOps.grayscale(normalized)
    square = ImageOps.pad(grayscale, (image_size, image_size), method=_BILINEAR, color=0)
    contrasted = ImageOps.autocontrast(square, cutoff=2)
    array = np.asarray(contrasted, dtype=np.float32) / 255.0
    return array


def extract_hog_features(
    gray: np.ndarray,
    *,
    cell_size: int = CELL_SIZE,
    bins: int = ORIENTATION_BINS,
) -> np.ndarray:
    if gray.ndim != 2:
        raise ValueError("Expected a 2D grayscale image.")

    height, width = gray.shape
    if height < cell_size * 2 or width < cell_size * 2:
        raise ValueError("Image is too small for HOG extraction.")

    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gx[:, 0] = gray[:, 1] - gray[:, 0]
    gx[:, -1] = gray[:, -1] - gray[:, -2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    gy[0, :] = gray[1, :] - gray[0, :]
    gy[-1, :] = gray[-1, :] - gray[-2, :]

    magnitude = np.hypot(gx, gy)
    angles = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0

    cells_y = height // cell_size
    cells_x = width // cell_size
    histograms = np.zeros((cells_y, cells_x, bins), dtype=np.float32)
    bin_width = 180.0 / bins

    for cell_y in range(cells_y):
        row_start = cell_y * cell_size
        row_end = row_start + cell_size
        for cell_x in range(cells_x):
            col_start = cell_x * cell_size
            col_end = col_start + cell_size
            cell_magnitude = magnitude[row_start:row_end, col_start:col_end].reshape(-1)
            cell_angles = angles[row_start:row_end, col_start:col_end].reshape(-1)
            indices = np.floor(cell_angles / bin_width).astype(int)
            indices = np.clip(indices, 0, bins - 1)
            histograms[cell_y, cell_x] = np.bincount(indices, weights=cell_magnitude, minlength=bins)

    blocks: list[np.ndarray] = []
    for cell_y in range(cells_y - 1):
        for cell_x in range(cells_x - 1):
            block = histograms[cell_y : cell_y + 2, cell_x : cell_x + 2].reshape(-1)
            block /= np.linalg.norm(block) + 1e-6
            block = np.clip(block, 0.0, 0.2)
            block /= np.linalg.norm(block) + 1e-6
            blocks.append(block.astype(np.float32))

    if not blocks:
        return histograms.reshape(-1)
    return np.concatenate(blocks, dtype=np.float32)


def extract_feature_vector(image: Image.Image) -> np.ndarray:
    gray = preprocess_image(image)
    hog = extract_hog_features(gray)

    thumb_image = Image.fromarray((gray * 255.0).astype(np.uint8)).resize((THUMB_SIZE, THUMB_SIZE), _BILINEAR)
    thumb = np.asarray(thumb_image, dtype=np.float32).reshape(-1) / 255.0

    stats = np.asarray(
        [
            float(gray.mean()),
            float(gray.std()),
            float((gray > 0.55).mean()),
            float((gray < 0.25).mean()),
        ],
        dtype=np.float32,
    )
    return np.concatenate([hog, thumb, stats], dtype=np.float32)


def feature_dimension() -> int:
    blank = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="black")
    return int(extract_feature_vector(blank).shape[0])
