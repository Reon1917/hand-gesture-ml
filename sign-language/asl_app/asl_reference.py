from __future__ import annotations

import string

MOTION_LABELS = ("J", "Z")
SPECIAL_LABELS = ("del", "nothing", "space")
STATIC_LABELS = tuple(letter for letter in string.ascii_uppercase if letter not in MOTION_LABELS)

MANUAL_TESTING_NOTES = [
    "This model is trained for static ASL handshapes, not full sign language translation.",
    "J and Z are excluded by default because they depend on motion across frames.",
    "Use your dominant hand, keep it centered, and match the reference card beside the camera feed.",
    "Plain backgrounds and steady lighting improve results more than larger webcam resolution.",
]


def label_sort_key(label: str) -> tuple[int, str]:
    normalized = label.strip()
    if len(normalized) == 1 and normalized.isalpha():
        return (0, normalized.upper())
    return (1, normalized.lower())


def sort_labels(labels: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    return sorted(labels, key=label_sort_key)


def reference_hint(label: str) -> str:
    if label in MOTION_LABELS:
        return "Motion sign. This build is tuned for single-frame handshapes, so results may be unreliable."
    if label in SPECIAL_LABELS:
        return "Special token from the dataset rather than a letter. Disabled by default in training."
    return "Match the sample handshape and hold it steady for a beat before switching."


def default_reference_cards() -> list[dict[str, str | None]]:
    return [
        {
            "label": label,
            "hint": reference_hint(label),
            "image_url": None,
        }
        for label in STATIC_LABELS
    ]
