from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

from .dataset import append_sample, count_labels
from .landmarks import HandLandmarkDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect hand landmark samples into a CSV dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gestures.csv"),
        help="CSV path for collected landmarks.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="peace,thumbs_up,fist,open_hand,point_down",
        help="Comma-separated gesture labels mapped to number keys.",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    return parser.parse_args()


def draw_overlay(
    frame,
    labels: list[str],
    counts: dict[str, int],
    last_status: str,
    hand_detected: bool,
) -> None:
    instructions = [
        "Tracking landmarks only. Press 1-9 to label and save the current frame",
        "Press q to quit",
        f"Hand detected: {'YES' if hand_detected else 'NO'}",
        "",
        "Labels:",
    ]
    for index, label in enumerate(labels, start=1):
        instructions.append(f"{index}: {label} ({counts.get(label, 0)})")
    instructions.append("")
    instructions.append(last_status)

    for line_index, text in enumerate(instructions):
        y = 30 + (line_index * 24)
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 240, 20),
            2,
            cv2.LINE_AA,
        )


def main() -> int:
    args = parse_args()
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not labels:
        raise ValueError("At least one label is required.")
    if len(labels) > 9:
        raise ValueError("The collector supports up to 9 labels because it maps them to number keys.")

    try:
        detector = HandLandmarkDetector()
    except Exception as exc:
        print(f"Failed to initialize the hand detector: {exc}", file=sys.stderr)
        return 2

    print(f"Collecting samples into {args.output}")
    print("Focus the OpenCV window, hold a gesture, then press its number key:")
    for index, label in enumerate(labels, start=1):
        print(f"  {index} -> {label}")
    print("Press q in the OpenCV window to quit.")

    capture = cv2.VideoCapture(args.camera)

    if not capture.isOpened():
        print(f"Unable to open camera index {args.camera}.", file=sys.stderr)
        detector.close()
        return 2

    counts = count_labels(args.output)
    last_status = "Show a gesture, then press its number key to save a sample"

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Failed to read a frame from the webcam.", file=sys.stderr)
                return 2

            frame = cv2.flip(frame, 1)
            detection = detector.detect(frame)
            detector.draw(frame, detection.hand_landmarks)
            draw_overlay(
                frame,
                labels,
                counts,
                last_status,
                hand_detected=detection.features is not None,
            )
            cv2.imshow("Hand Gesture Collector", frame)

            key_code = cv2.waitKey(1) & 0xFF
            if key_code == ord("q"):
                break

            selected_index = key_code - ord("1")
            if 0 <= selected_index < len(labels):
                label = labels[selected_index]
                if detection.features is None:
                    last_status = f"No hand detected for {label}"
                    print(last_status)
                    continue
                append_sample(args.output, label, detection.features)
                counts[label] += 1
                last_status = f"Saved sample for {label} ({counts[label]} total)"
                print(last_status)
    except KeyboardInterrupt:
        print("\nStopped collection.")
    finally:
        capture.release()
        detector.close()
        cv2.destroyAllWindows()

    return 0
