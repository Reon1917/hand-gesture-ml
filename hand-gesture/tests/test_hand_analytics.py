from __future__ import annotations

import unittest

import numpy as np

from gesture_controller.hand_analytics import analyze_hand_landmarks


def make_open_hand() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float32)
    points[0] = [0.0, 0.0, 0.0]
    points[1] = [0.10, 0.03, 0.0]
    points[2] = [0.24, 0.06, 0.0]
    points[3] = [0.38, 0.08, 0.0]
    points[4] = [0.52, 0.09, 0.0]
    points[5] = [0.18, 0.22, 0.0]
    points[6] = [0.18, 0.48, 0.0]
    points[7] = [0.18, 0.74, 0.0]
    points[8] = [0.18, 1.00, 0.0]
    points[9] = [0.02, 0.26, 0.0]
    points[10] = [0.02, 0.56, 0.0]
    points[11] = [0.02, 0.86, 0.0]
    points[12] = [0.02, 1.16, 0.0]
    points[13] = [-0.14, 0.23, 0.0]
    points[14] = [-0.14, 0.48, 0.0]
    points[15] = [-0.14, 0.73, 0.0]
    points[16] = [-0.14, 0.98, 0.0]
    points[17] = [-0.30, 0.18, 0.0]
    points[18] = [-0.30, 0.40, 0.0]
    points[19] = [-0.30, 0.62, 0.0]
    points[20] = [-0.30, 0.84, 0.0]
    return points


class HandAnalyticsTests(unittest.TestCase):
    def test_open_hand_reports_open_fingers(self) -> None:
        analytics = analyze_hand_landmarks(make_open_hand())
        self.assertEqual(analytics.finger_count, 5)
        self.assertEqual(analytics.finger_states["index"], "open")
        self.assertIsNone(analytics.pinch_target)
        self.assertGreater(analytics.openness, 0.4)

    def test_thumb_index_pinch_is_detected(self) -> None:
        hand = make_open_hand()
        hand[4] = hand[8] + np.array([0.01, 0.01, 0.0], dtype=np.float32)
        analytics = analyze_hand_landmarks(hand)
        self.assertEqual(analytics.pinch_target, "index")
        self.assertGreater(analytics.pinch_strength, 0.8)


if __name__ == "__main__":
    unittest.main()
