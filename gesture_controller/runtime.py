from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass


@dataclass(slots=True)
class StablePrediction:
    label: str
    confidence: float


class PredictionSmoother:
    def __init__(self, window_size: int = 5, consensus_threshold: float = 0.6) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.window_size = window_size
        self.consensus_threshold = consensus_threshold
        self._window: deque[tuple[str, float]] = deque(maxlen=window_size)

    def push(self, label: str, confidence: float) -> StablePrediction | None:
        self._window.append((label, confidence))
        if len(self._window) < self.window_size:
            return None

        counts = Counter(item[0] for item in self._window)
        best_label, best_count = counts.most_common(1)[0]
        ratio = best_count / len(self._window)
        if ratio < self.consensus_threshold:
            return None

        confidences = [confidence for label, confidence in self._window if label == best_label]
        return StablePrediction(label=best_label, confidence=sum(confidences) / len(confidences))

    def reset(self) -> None:
        self._window.clear()
