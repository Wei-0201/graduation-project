"""Geometry/controller regressions for the captured bend-exit failure."""
import unittest
import numpy as np
from visual_exit_guard import VisualExitGuard
from jetson_model_centering_visual_exit import CommandPolicy


def mask_lines(coefs):
    mask = np.zeros((240, 320), np.uint8)
    for slope, intercept in coefs:
        for y in range(94, 220):
            x = int(round(slope*y+intercept))
            if 3 <= x < 317:
                mask[y, x-2:x+3] = 255
    return mask


class VisualExitTests(unittest.TestCase):
    def test_pair_can_be_off_center_and_need_corrective_left_turn(self):
        guard = VisualExitGuard()
        result = guard.process(mask_lines([(-2.6, 380), (-.5, 250)]))
        self.assertEqual(result['visual_source'], 'PAIR')
        self.assertGreater(result['visual_turn'], .3)

    def test_single_known_right_keeps_identity_after_center_leaves_image(self):
        guard = VisualExitGuard()
        guard.process(mask_lines([(-2.6, 380), (-.5, 250)]))
        for intercept in np.linspace(245, 190, 20):
            result = guard.process(mask_lines([(-.5, intercept)]))
            self.assertEqual(result['visual_source'], 'KNOWN RIGHT')
            self.assertGreater(result['visual_turn'], 0)

    def test_single_unknown_line_cannot_choose_a_lane_side(self):
        self.assertIsNone(VisualExitGuard().process(
            mask_lines([(-.5, 190)]))['visual_turn'])

    def test_thin_floor_joint_does_not_complete_a_tape_pair(self):
        mask = mask_lines([(.5, 130)])
        for y in range(94, 220):
            x = int(round(-.8*y+220))
            mask[y, x:x+2] = 255
        self.assertIsNone(VisualExitGuard().process(mask)['visual_turn'])

    def test_lost_line_and_reset_discard_identity(self):
        guard = VisualExitGuard()
        guard.process(mask_lines([(-2.6, 380), (-.5, 250)]))
        for _ in range(9):
            guard.process(np.zeros((240, 320), np.uint8))
        self.assertIsNone(guard.process(mask_lines([(-.5, 190)]))['visual_turn'])

    def policy(self):
        p = CommandPolicy()
        p.good_frames = 3
        p.first_stamp = 0.0
        p.last_stamp = .4
        return p

    def drive(self, p, stamp, raw, visual=None):
        detection = dict(timestamp=stamp, valid=True, turn=raw,
                         visual_turn=visual, visual_weight=.8,
                         visual_source='PAIR')
        return p.command('AUTO', 'STOP', 0.0, detection, stamp)

    def test_straight_image_gradually_reduces_a_persistent_turn(self):
        p = self.policy()
        self.drive(p, .5, -.5)
        values = []
        for i in range(8):
            self.drive(p, .625+i*.125, -.5, 0.0)
            values.append(p.command_turn)
        self.assertGreater(values[-1], values[0])
        self.assertAlmostEqual(values[-1], -.1)
        self.assertLessEqual(max(abs(b-a) for a,b in zip(values,values[1:])), .16001)

    def test_corrective_left_not_suppressed_by_wrong_right_model(self):
        p = self.policy()
        for i in range(6):
            self.drive(p, .5+i*.125, -.2, .7)
        self.assertGreater(p.command_turn, .4)

    def test_duplicate_image_does_not_accumulate_extra_steering(self):
        p = self.policy()
        self.drive(p, .5, -.5, 0.0)
        self.drive(p, .625, -.5, 0.0)
        turn = p.command_turn
        for _ in range(20):
            self.drive(p, .625, -.5, 0.0)
        self.assertEqual(p.command_turn, turn)

    def test_no_visual_evidence_preserves_model_and_stale_frame_stops(self):
        p = self.policy()
        self.drive(p, .5, .4)
        self.drive(p, .625, .2)
        self.assertEqual(p.command_turn, .2)
        self.assertEqual(p.command('AUTO','STOP',0,None,1.2), 'STOP')


if __name__ == '__main__':
    unittest.main()
