"""Checks for camera-based bend exit and safe fallback to model steering."""

import unittest

import numpy as np

from straight_ahead_probe import straight_ahead_probe
from jetson_model_centering_straight_exit import CommandPolicy


def tape_mask(left, right):
    image = np.zeros((240, 320), dtype=np.uint8)
    for y in range(100, 171):
        for x in (left(y), right(y)):
            ix = int(round(x))
            if 2 <= ix < 318:
                image[y, ix - 2:ix + 3] = 255
    return image


class StraightExitTest(unittest.TestCase):
    def policy(self):
        policy = CommandPolicy(auto_speed=0.12)
        policy.good_frames = 3
        policy.first_stamp = 0.0
        policy.last_stamp = 0.3
        return policy

    def command(self, policy, turn, stamp, confidence=0.0):
        return policy.command('AUTO', 'STOP', 0.0, {
            'timestamp': stamp, 'valid': True, 'turn': turn,
            'straight_confidence': confidence,
        }, stamp)

    def test_visible_parallel_boundaries_trigger(self):
        mask = tape_mask(lambda y: 100 - (y - 100),
                         lambda y: 220 + (y - 100))
        confidence, centers = straight_ahead_probe(mask)
        self.assertGreater(confidence, 0.7)
        self.assertEqual(len(centers), 4)

    def test_single_boundary_or_curved_center_does_not_trigger(self):
        single = tape_mask(lambda y: 100 - (y - 100), lambda y: -50)
        self.assertEqual(straight_ahead_probe(single)[0], 0.0)
        curve = tape_mask(lambda y: 100 - (y - 100) + (y - 100) * 0.55,
                          lambda y: 220 + (y - 100) + (y - 100) * 0.55)
        self.assertEqual(straight_ahead_probe(curve)[0], 0.0)

    def test_straight_hint_gradually_eases_persistent_model_turn(self):
        policy = self.policy()
        self.assertEqual(self.command(policy, -0.5, 0.50, 0.8),
                         'DRIVE,0.120,-0.5000')
        values = []
        for i in range(1, 7):
            self.command(policy, -0.5, 0.50 + i * 0.125, 0.8)
            values.append(abs(policy.command_turn))
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
        self.assertLess(values[-1], 0.30)
        self.assertEqual(policy.control_state, 'MODEL DRIVE STRAIGHT')

    def test_uncertain_view_uses_model_and_new_bend_clears_fade(self):
        policy = self.policy()
        self.command(policy, -0.4, 0.5, 0.0)
        self.assertEqual(policy.command_turn, -0.4)
        self.command(policy, -0.4, 0.625, 0.8)
        self.command(policy, -0.4, 0.750, 0.8)
        self.assertGreater(policy.straight_fade, 0)
        self.command(policy, 0.5, 0.875, 0.0)
        self.assertEqual(policy.command_turn, 0.5)
        self.assertEqual(policy.straight_fade, 0.0)

    def test_stale_frame_stops(self):
        policy = self.policy()
        self.command(policy, -0.5, 0.5, 0.8)
        self.assertEqual(policy.command('AUTO', 'STOP', 0.0, None, 1.0), 'STOP')


if __name__ == '__main__':
    unittest.main()
