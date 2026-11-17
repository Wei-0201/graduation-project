import unittest
import numpy as np
import test_review_v4 as legacy
from jetson_model_learned_path_v51 import CommandPolicy, blend_path_target, path_decision


class ExistingControlTests(legacy.ReviewPolicyTest):
    def policy(self, visual_weight=0.):
        p = CommandPolicy(visual_weight=visual_weight, left_gain=1., right_gain=1.)
        p.good_frames, p.first_stamp, p.last_stamp = 3, 0., .4
        return p


class PreviewTests(unittest.TestCase):
    def test_approach_gets_small_bounded_preview_both_directions(self):
        for far, stage in [(-.5, 'approach_right'), (.5, 'approach_left')]:
            turn, weight, delta = blend_path_target(0., far, stage, .8)
            self.assertGreater(turn * far, 0.)
            self.assertLessEqual(abs(turn), .14)

    def test_sub_threshold_active_confidence_no_longer_disables_preview(self):
        turn, weight, delta = blend_path_target(0., -.5, 'turn_right', .60)
        self.assertLess(turn, -.10)
        self.assertGreater(weight, 0.)

    def test_strong_opposite_near_correction_is_preserved(self):
        self.assertEqual(blend_path_target(.3, -.8, 'turn_right', .9), (.3, 0., 0.))

    def test_preview_cannot_exceed_point_fourteen(self):
        turn, weight, delta = blend_path_target(.1, 1., 'turn_left', 1.)
        self.assertAlmostEqual(delta, .14)
        self.assertLessEqual(abs(turn), 1.)

    def test_same_near_far_points_do_not_invent_heading(self):
        self.assertEqual(blend_path_target(.1, .1, 'turn_left', .9)[0], .1)

    def test_missing_far_uses_near_only(self):
        self.assertEqual(blend_path_target(-.2, None, 'turn_right', .8), (-.2, 0., 0.))
        self.assertEqual(blend_path_target(-.2, float('nan'), 'turn_right', .8), (-.2, 0., 0.))

    def test_invalid_near_stops(self):
        self.assertFalse(path_decision(500, 200, [1, 0, 0, 0, 0, 0], list(range(5)))['valid'])

    def test_untrained_exit_is_never_selected(self):
        r = path_decision(160, 160, [1, 0, 0, 0, 0, 999], list(range(5)))
        self.assertEqual(r['stage'], 'straight')

    def test_frame_duplicate_does_not_keep_adding_preview(self):
        p = CommandPolicy()
        d = path_decision(160, 240, [0, 0, 0, 4, 0, 0], list(range(5)))
        d['timestamp'] = 1.
        first = p.steering_for_frame(d)
        for _ in range(5):
            self.assertEqual(p.steering_for_frame(d), first)


if __name__ == '__main__':
    unittest.main()
