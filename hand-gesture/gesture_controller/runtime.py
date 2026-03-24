from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import time


@dataclass(slots=True)
class StablePrediction:
    label: str
    confidence: float


@dataclass(slots=True)
class RuntimeUpdate:
    state: str
    active_label: str | None
    active_confidence: float
    should_trigger: bool


class GestureStateMachine:
    def __init__(
        self,
        *,
        window_size: int = 5,
        consensus_threshold: float = 0.6,
        confidence_threshold: float = 0.75,
        release_threshold: float = 0.55,
        release_frames: int = 3,
        cooldown_seconds: float = 1.0,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if release_frames < 1:
            raise ValueError("release_frames must be at least 1")

        self.window_size = window_size
        self.consensus_threshold = consensus_threshold
        self.confidence_threshold = confidence_threshold
        self.release_threshold = release_threshold
        self.release_frames = release_frames
        self.cooldown_seconds = cooldown_seconds

        self.state = "no_hand"
        self._window: deque[tuple[str, float]] = deque(maxlen=window_size)
        self._active_label: str | None = None
        self._active_confidence = 0.0
        self._release_counter = 0
        self._cooldown_until = 0.0

    def update(
        self,
        label: str | None,
        confidence: float,
        *,
        hand_present: bool,
        now: float | None = None,
    ) -> RuntimeUpdate:
        current_time = now if now is not None else time.monotonic()

        if self.state == "confirmed":
            self.state = "cooldown"

        if self._active_label is not None and self.state == "cooldown":
            release_frame = (
                not hand_present
                or label != self._active_label
                or confidence < self.release_threshold
            )
            if release_frame:
                self._release_counter += 1
            else:
                self._release_counter = 0

            if self._release_counter >= self.release_frames:
                self._reset_for_rearm()
                return RuntimeUpdate(
                    state=self.state,
                    active_label=self._active_label,
                    active_confidence=self._active_confidence,
                    should_trigger=False,
                )

            return RuntimeUpdate(
                state=self.state,
                active_label=self._active_label,
                active_confidence=self._active_confidence,
                should_trigger=False,
            )

        if current_time < self._cooldown_until:
            self._window.clear()
            self.state = "no_hand"
            return RuntimeUpdate(
                state=self.state,
                active_label=None,
                active_confidence=0.0,
                should_trigger=False,
            )

        if not hand_present or label is None or confidence < self.confidence_threshold:
            self._window.clear()
            self.state = "no_hand"
            return RuntimeUpdate(
                state=self.state,
                active_label=None,
                active_confidence=0.0,
                should_trigger=False,
            )

        self._window.append((label, confidence))
        stable = self._stable_prediction()
        if stable is None:
            self.state = "candidate"
            return RuntimeUpdate(
                state=self.state,
                active_label=None,
                active_confidence=0.0,
                should_trigger=False,
            )

        self._window.clear()
        self._active_label = stable.label
        self._active_confidence = stable.confidence
        self._release_counter = 0
        self._cooldown_until = current_time + self.cooldown_seconds
        self.state = "confirmed"
        return RuntimeUpdate(
            state=self.state,
            active_label=self._active_label,
            active_confidence=self._active_confidence,
            should_trigger=True,
        )

    def _stable_prediction(self) -> StablePrediction | None:
        if len(self._window) < self.window_size:
            return None

        counts = Counter(item[0] for item in self._window)
        best_label, best_count = counts.most_common(1)[0]
        ratio = best_count / len(self._window)
        if ratio < self.consensus_threshold:
            return None

        confidences = [confidence for label, confidence in self._window if label == best_label]
        return StablePrediction(label=best_label, confidence=sum(confidences) / len(confidences))

    def _reset_for_rearm(self) -> None:
        self.state = "no_hand"
        self._window.clear()
        self._active_label = None
        self._active_confidence = 0.0
        self._release_counter = 0
