import unittest
import test_review_v4 as legacy
from jetson_model_centering_clean_20261025 import CommandPolicy


class ExistingControlTests(legacy.ReviewPolicyTest):
    def policy(self, visual_weight=0.):
        p = CommandPolicy(visual_weight=visual_weight, left_gain=1., right_gain=1.)
        p.good_frames, p.first_stamp, p.last_stamp = 3, 0., .4
        return p


class UnwindTests(unittest.TestCase):
    def policy(self):
        return CommandPolicy(left_gain=1., right_gain=1.)

    def test_smaller_demand_unwinds_both_directions_without_overshoot(self):
        for direction in (-1, 1):
            p = self.policy()
            old = p.steering_for_frame(dict(timestamp=1., turn=direction * .6))
            for stamp, target in [(2., .4), (3., .2), (4., 0.)]:
                new = p.steering_for_frame(dict(timestamp=stamp, turn=direction * target))
                self.assertTrue(p.unwind_active)
                self.assertLess(abs(new), abs(old))
                self.assertGreaterEqual(new * direction, target)
                old = new

    def test_unwind_follows_reduced_target_faster(self):
        p = self.policy()
        p.steering_for_frame(dict(timestamp=1., turn=.6))
        turn = p.steering_for_frame(dict(timestamp=2., turn=.2))
        self.assertAlmostEqual(turn, .24)
        self.assertLess(turn, .3 * .6 + .7 * .2)

    def test_new_larger_demand_keeps_original_response(self):
        p = self.policy()
        p.steering_for_frame(dict(timestamp=1., turn=.2))
        self.assertAlmostEqual(p.steering_for_frame(dict(timestamp=2., turn=.6)), .48)
        self.assertFalse(p.unwind_active)

    def test_duplicate_frames_cannot_keep_unwinding(self):
        p = self.policy()
        p.steering_for_frame(dict(timestamp=1., turn=.6))
        d = dict(timestamp=2., turn=0.)
        first = p.steering_for_frame(d)
        for _ in range(10):
            self.assertEqual(p.steering_for_frame(d), first)

    def test_unwind_does_not_depend_on_elapsed_bend_time(self):
        outcomes = []
        for times in [(1., 1.1), (1000., 1100.)]:
            p = self.policy()
            p.steering_for_frame(dict(timestamp=times[0], turn=.6))
            outcomes.append(p.steering_for_frame(dict(timestamp=times[1], turn=0.)))
        self.assertEqual(outcomes[0], outcomes[1])


if __name__ == '__main__':
    unittest.main()
