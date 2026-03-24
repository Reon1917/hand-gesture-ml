from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2

from .actions import ActionExecutor, load_bindings
from .landmarks import HandLandmarkDetector
from .modeling import load_bundle, predict_gesture, validate_bundle
from .runtime import GestureStateMachine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live hand gesture inference and trigger actions.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/gesture_model.joblib"))
    parser.add_argument("--bindings", type=Path, default=Path("config/gesture_bindings.json"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--consensus-threshold", type=float, default=0.6)
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--release-frames", type=int, default=3)
    parser.add_argument("--release-threshold", type=float, default=0.55)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def draw_overlay(
    frame,
    *,
    state: str,
    live_label: str,
    confidence: float,
    active_label: str | None,
    status: str,
) -> None:
    lines = [
        f"State: {state}",
        f"Prediction: {live_label}",
        f"Confidence: {confidence:.2f}",
        f"Active gesture: {active_label or '-'}",
        status,
        "Press q to quit",
    ]
    for line_index, text in enumerate(lines):
        y = 30 + (line_index * 28)
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 240, 20),
            2,
            cv2.LINE_AA,
        )


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        print(
            f"Model not found at {args.model}. Run `python train_model.py` after collecting data.",
            file=sys.stderr,
        )
        return 2

    try:
        bundle = load_bundle(args.model)
        validate_bundle(bundle)
        bindings = load_bindings(args.bindings)
        detector = HandLandmarkDetector()
    except Exception as exc:
        print(f"Failed to start the controller: {exc}", file=sys.stderr)
        return 2

    state_machine = GestureStateMachine(
        window_size=args.window_size,
        consensus_threshold=args.consensus_threshold,
        confidence_threshold=args.confidence_threshold,
        release_threshold=args.release_threshold,
        release_frames=args.release_frames,
        cooldown_seconds=args.cooldown,
    )
    executor = ActionExecutor(dry_run=args.dry_run)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"Unable to open camera index {args.camera}.", file=sys.stderr)
        detector.close()
        return 2

    live_label = "No hand detected"
    live_confidence = 0.0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Failed to read a frame from the webcam.", file=sys.stderr)
                return 2

            frame = cv2.flip(frame, 1)
            detection = detector.detect(frame)
            detector.draw(frame, detection.hand_landmarks)

            if detection.features is None:
                live_label = "No hand detected"
                live_confidence = 0.0
                update = state_machine.update(
                    None,
                    0.0,
                    hand_present=False,
                    now=time.monotonic(),
                )
            else:
                label, confidence, _distribution = predict_gesture(bundle, detection.features)
                live_label = label
                live_confidence = confidence
                update = state_machine.update(
                    label,
                    confidence,
                    hand_present=True,
                    now=time.monotonic(),
                )
                if update.should_trigger and update.active_label is not None:
                    action_spec = bindings.get(update.active_label, "noop")
                    executor.maybe_execute(update.active_label, action_spec)

            draw_overlay(
                frame,
                state=update.state,
                live_label=live_label,
                confidence=live_confidence,
                active_label=update.active_label,
                status=executor.last_message,
            )
            cv2.imshow("Hand Gesture Controller", frame)

            key_code = cv2.waitKey(1) & 0xFF
            if key_code == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nStopped controller.")
    finally:
        capture.release()
        detector.close()
        cv2.destroyAllWindows()

    return 0
