import unittest
import numpy as np
from jetson_model_centering_review_v4 import CommandPolicy, has_tape_evidence, steering_from_x


class ReviewPolicyTest(unittest.TestCase):
    def policy(self,visual_weight=0.0):
        p=CommandPolicy(visual_weight=visual_weight);p.good_frames=3;p.first_stamp=0.;p.last_stamp=.4
        return p

    def drive(self,p,stamp,raw,visual=None,source='NONE',visible=True):
        return p.command('AUTO','STOP',0,dict(timestamp=stamp,valid=True,
            turn=raw,visual_turn=visual,visual_weight=.8,visual_source=source,
            lane_visible=visible),stamp)

    def test_single_line_does_not_amplify_model_turn(self):
        p=self.policy()
        for i in range(4):self.drive(p,.5+i*.125,-.24,-1.,'TRACK LEFT')
        self.assertAlmostEqual(p.command_turn,-.24)

    def test_pair_cannot_increase_a_model_turn(self):
        p=self.policy()
        for i in range(4):self.drive(p,.5+i*.125,-.2,-.8,'CURVE PAIR')
        self.assertAlmostEqual(p.command_turn,-.2)

    def test_confirmed_pair_can_only_ease_by_bounded_amount(self):
        p=self.policy(.2)
        for i in range(4):self.drive(p,.5+i*.125,-.6,-.1,'PAIR')
        self.assertAlmostEqual(p.command_turn,-.52,delta=.003)

    def test_untrusted_geometry_cannot_reverse_model_direction(self):
        p=self.policy(.2)
        for i in range(6):self.drive(p,.5+i*.125,.4,-.6,'CURVE PAIR')
        self.assertAlmostEqual(p.command_turn,.4)
        self.assertFalse(p.review_required)

    def test_missing_lane_stops_after_five_new_images(self):
        p=self.policy()
        for i in range(4):self.assertTrue(self.drive(p,.5+i*.125,.1,visible=False).startswith('DRIVE'))
        self.assertEqual(self.drive(p,1.,.1,visible=False),'STOP')
        self.assertEqual(self.drive(p,1.125,.1,visible=True),'STOP')
        p.command('MANUAL','STOP',0,None,1.25)
        self.assertFalse(p.review_required)

    def test_duplicate_images_do_not_count_as_multiple_losses(self):
        p=self.policy()
        for _ in range(20):self.drive(p,.5,.1,visible=False)
        self.assertEqual(p.unobserved_frames,1)
        self.assertFalse(p.review_required)

    def test_blank_floor_and_thin_joints_do_not_count_as_tape(self):
        mask=np.zeros((240,320),np.uint8)
        self.assertFalse(has_tape_evidence(mask))
        mask[100:220,150:152]=255
        self.assertFalse(has_tape_evidence(mask))

    def test_continuous_bending_tape_is_evidence_without_side_guess(self):
        mask=np.zeros((240,320),np.uint8)
        for y in range(100,220):
            x=int(100+.005*(y-130)**2)
            mask[y,x-5:x+5]=255
        self.assertTrue(has_tape_evidence(mask))

    def test_smoothing_keeps_full_turn_available_and_ignores_duplicate_images(self):
        p=self.policy();self.drive(p,.5,0.)
        self.drive(p,.625,1.);before=p.command_turn
        for _ in range(20):self.drive(p,.625,1.)
        self.assertEqual(p.command_turn,before)
        for i in range(1,10):self.drive(p,.625+i*.125,1.)
        self.assertGreater(p.command_turn,.999)

    def test_stale_or_invalid_prediction_still_stops(self):
        p=self.policy();self.drive(p,.5,.2)
        self.assertEqual(p.command('AUTO','STOP',0,None,1.1),'STOP')
        self.assertEqual(p.command('AUTO','STOP',0,
            dict(timestamp=1.2,valid=False),1.2),'STOP')

    def test_small_edge_overshoot_is_bounded_but_invalid_outputs_reject(self):
        self.assertEqual(steering_from_x(-3),(0.,1.))
        self.assertEqual(steering_from_x(323)[0],319.)
        self.assertIsNone(steering_from_x(350))
        self.assertIsNone(steering_from_x(float('nan')))


if __name__=='__main__':unittest.main()
