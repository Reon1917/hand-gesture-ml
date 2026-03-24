from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from .asl_reference import MOTION_LABELS, SPECIAL_LABELS, label_sort_key

DEFAULT_DATASET_SLUG = "grassknoted/asl-alphabet"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    label: str
    path: Path


def normalize_label(raw_label: str) -> str | None:
    stripped = raw_label.strip()
    lowered = stripped.lower()
    if lowered in SPECIAL_LABELS:
        return lowered
    if len(stripped) == 1 and stripped.isalpha():
        return stripped.upper()
    return None


def discover_label_directories(dataset_root: Path) -> dict[str, list[Path]]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    buckets: dict[str, list[Path]] = {}
    for directory in sorted((path for path in dataset_root.rglob("*") if path.is_dir()), key=lambda p: str(p).lower()):
        label = normalize_label(directory.name)
        if label is None:
            continue
        image_paths = [
            candidate
            for candidate in sorted(directory.iterdir(), key=lambda p: p.name.lower())
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if image_paths:
            buckets.setdefault(label, []).extend(image_paths)

    if not buckets:
        raise RuntimeError(
            "No ASL label directories were found. Expected subfolders like A/, B/, C/, ... inside the dataset."
        )
    return buckets


def build_dataset_index(
    dataset_root: Path,
    *,
    include_motion: bool = False,
    include_special: bool = False,
    max_images_per_class: int | None = None,
    random_seed: int = 42,
) -> list[ImageRecord]:
    import random

    buckets = discover_label_directories(dataset_root)
    if not include_motion:
        for label in MOTION_LABELS:
            buckets.pop(label, None)
    if not include_special:
        for label in SPECIAL_LABELS:
            buckets.pop(label, None)

    rng = random.Random(random_seed)
    records: list[ImageRecord] = []

    for label in sorted(buckets, key=label_sort_key):
        image_paths = list(buckets[label])
        if max_images_per_class and max_images_per_class > 0 and len(image_paths) > max_images_per_class:
            image_paths = rng.sample(image_paths, max_images_per_class)
            image_paths.sort(key=lambda path: path.name.lower())
        records.extend(ImageRecord(label=label, path=path) for path in image_paths)

    if not records:
        raise RuntimeError("The dataset is empty after applying the current label filters.")
    return records


def ensure_directory_link(target: Path, destination: Path, *, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == target.resolve():
            return destination
        if not force:
            raise RuntimeError(
                f"{destination} already exists. Remove it manually or rerun with force enabled."
            )
        if destination.is_symlink():
            destination.unlink()
        else:
            raise RuntimeError(
                f"{destination} is a real directory. Remove it manually before recreating the dataset link."
            )

    try:
        destination.symlink_to(target, target_is_directory=True)
    except OSError:
        shutil.copytree(target, destination, dirs_exist_ok=True)
    return destination


def _download_with_kagglehub(dataset_slug: str) -> Path:
    import kagglehub

    return Path(kagglehub.dataset_download(dataset_slug)).resolve()


def _download_with_kaggle_cli(dataset_slug: str, destination: Path) -> Path:
    kaggle_binary = shutil.which("kaggle")
    if kaggle_binary is None:
        raise RuntimeError("Kaggle CLI was not found on PATH.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    zip_name = f"{dataset_slug.split('/')[-1]}.zip"
    zip_path = destination.parent / zip_name

    command = [
        kaggle_binary,
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "-p",
        str(destination.parent),
        "--force",
    ]
    subprocess.run(command, check=True)

    if destination.exists():
        if destination.is_symlink():
            destination.unlink()
        else:
            raise RuntimeError(f"Cannot unpack into existing directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        archive.extractall(destination)
    return destination


def download_kaggle_dataset(
    destination: Path,
    *,
    dataset_slug: str = DEFAULT_DATASET_SLUG,
    force: bool = False,
) -> Path:
    try:
        downloaded_root = _download_with_kagglehub(dataset_slug)
    except ModuleNotFoundError:
        return _download_with_kaggle_cli(dataset_slug, destination)
    except Exception as exc:
        if shutil.which("kaggle") is not None:
            try:
                return _download_with_kaggle_cli(dataset_slug, destination)
            except Exception as cli_exc:
                raise RuntimeError(
                    "Kaggle download failed. Check network access and configure "
                    "~/.kaggle/kaggle.json or set KAGGLE_USERNAME and KAGGLE_KEY before retrying."
                ) from cli_exc
        raise RuntimeError(
            "Kaggle download failed. Check network access and configure "
            "~/.kaggle/kaggle.json or set KAGGLE_USERNAME and KAGGLE_KEY before retrying."
        ) from exc

    return ensure_directory_link(downloaded_root, destination, force=force)
