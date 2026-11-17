import unittest
import numpy as np

from jetson_model_learned_path_v5 import CommandPolicy, path_decision
import test_review_v4 as legacy


class ExistingControlTests(legacy.ReviewPolicyTest):
    def policy(self, visual_weight=0.):
        p = CommandPolicy(visual_weight=visual_weight, left_gain=1., right_gain=1.)
        p.good_frames, p.first_stamp, p.last_stamp = 3, 0., .4
        return p


class LearnedPathTests(unittest.TestCase):
    def detect(self, near, far, stage, confidence=10.):
        logits = np.zeros(6)
        logits[stage] = confidence
        return path_decision(near, far, logits, [0, 1, 2, 3, 4])

    def test_upcoming_left_cannot_pull_straight_near_target_left(self):
        r = self.detect(160, 0, 1)
        self.assertEqual(r['turn'], 0.)
        self.assertEqual(r['far_weight'], 0.)

    def test_upcoming_right_keeps_near_position_correction(self):
        r = self.detect(144, 300, 3)
        self.assertAlmostEqual(r['turn'], .1)
        self.assertEqual(r['far_weight'], 0.)

    def test_wrong_active_class_cannot_create_far_only_turn(self):
        self.assertEqual(self.detect(160, 0, 2)['turn'], 0.)
        self.assertEqual(self.detect(160, 319, 4)['turn'], 0.)

    def test_active_curve_assistance_is_bounded_and_same_direction(self):
        r = self.detect(128, 0, 2)
        self.assertGreater(r['turn'], .2)
        self.assertLessEqual(r['turn'], .28)
        r = self.detect(128, 300, 2)
        self.assertAlmostEqual(r['turn'], .2)

    def test_low_confidence_uses_near_only(self):
        self.assertAlmostEqual(self.detect(128, 0, 2, confidence=.1)['turn'], .2)

    def test_unsupported_exit_class_never_selected(self):
        logits = [1., 0., 0., 0., 0., 1000.]
        r = path_decision(160, 160, logits, [0, 1, 2, 3, 4])
        self.assertEqual(r['stage'], 'straight')

    def test_invalid_near_stops_invalid_far_does_not_steer(self):
        self.assertFalse(self.detect(float('nan'), 160, 0)['valid'])
        self.assertFalse(self.detect(500, 160, 0)['valid'])
        r = self.detect(128, float('nan'), 2)
        self.assertTrue(r['valid'])
        self.assertAlmostEqual(r['turn'], .2)

    def test_asymmetric_gains_and_fresh_frame_only_filter(self):
        for raw, gain in [(.5, .7), (-.5, .78)]:
            p = CommandPolicy()
            d = dict(timestamp=1., turn=raw)
            self.assertAlmostEqual(p.steering_for_frame(d), raw * gain)
            self.assertAlmostEqual(p.steering_for_frame(d), raw * gain)


if __name__ == '__main__':
    unittest.main()
