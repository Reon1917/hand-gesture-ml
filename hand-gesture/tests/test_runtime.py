from __future__ import annotations

import unittest

from gesture_controller.runtime import GestureStateMachine


class GestureStateMachineTests(unittest.TestCase):
    def test_single_trigger_while_holding_and_rearm_after_release(self) -> None:
        machine = GestureStateMachine(
            window_size=3,
            consensus_threshold=2 / 3,
            confidence_threshold=0.8,
            release_threshold=0.55,
            release_frames=2,
            cooldown_seconds=0.5,
        )

        updates = [
            machine.update("peace", 0.95, hand_present=True, now=0.0),
            machine.update("peace", 0.96, hand_present=True, now=0.1),
            machine.update("peace", 0.97, hand_present=True, now=0.2),
            machine.update("peace", 0.98, hand_present=True, now=0.3),
            machine.update("peace", 0.97, hand_present=True, now=0.4),
        ]

        self.assertFalse(updates[0].should_trigger)
        self.assertFalse(updates[1].should_trigger)
        self.assertTrue(updates[2].should_trigger)
        self.assertEqual(updates[2].state, "confirmed")
        self.assertFalse(updates[3].should_trigger)
        self.assertEqual(updates[3].state, "cooldown")
        self.assertFalse(updates[4].should_trigger)

        released = machine.update(None, 0.0, hand_present=False, now=0.5)
        self.assertEqual(released.state, "cooldown")
        rearmed = machine.update(None, 0.0, hand_present=False, now=0.6)
        self.assertEqual(rearmed.state, "no_hand")

        blocked_by_cooldown = machine.update("peace", 0.96, hand_present=True, now=0.65)
        self.assertEqual(blocked_by_cooldown.state, "no_hand")
        self.assertFalse(blocked_by_cooldown.should_trigger)

        machine.update("peace", 0.96, hand_present=True, now=0.75)
        machine.update("peace", 0.96, hand_present=True, now=0.85)
        retrigger = machine.update("peace", 0.96, hand_present=True, now=0.95)
        self.assertTrue(retrigger.should_trigger)
        self.assertEqual(retrigger.state, "confirmed")

    def test_resets_to_no_hand_on_release_signals(self) -> None:
        for label, confidence, hand_present in (
            (None, 0.0, False),
            ("thumbs_up", 0.95, True),
            ("peace", 0.2, True),
        ):
            machine = GestureStateMachine(
                window_size=2,
                consensus_threshold=1.0,
                confidence_threshold=0.8,
                release_threshold=0.55,
                release_frames=1,
                cooldown_seconds=0.0,
            )
            machine.update("peace", 0.9, hand_present=True, now=0.0)
            confirmed = machine.update("peace", 0.9, hand_present=True, now=0.1)
            self.assertTrue(confirmed.should_trigger)

            update = machine.update(label, confidence, hand_present=hand_present, now=0.2)
            self.assertEqual(update.state, "no_hand")
            self.assertIsNone(update.active_label)


if __name__ == "__main__":
    unittest.main()
