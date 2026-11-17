"""Curve support and left/right symmetry regressions for V3."""
import unittest
import numpy as np
from lane_corridor_guard import LaneCorridorGuard
from jetson_model_centering_curve_exit import CommandPolicy


def corridor(right_only=False):
    mask=np.zeros((240,320),np.uint8)
    for y in range(94,200):
        for base in ([180] if right_only else [50,180]):
            x=base+3*(y-110)
            if 5<=x<315:
                mask[y,x-4:x+5]=255
    return mask


class CurveExitTest(unittest.TestCase):
    def test_far_curve_turns_left_even_when_near_center_is_right(self):
        r=LaneCorridorGuard().process(corridor())
        self.assertEqual(r['visual_source'],'CURVE PAIR')
        self.assertGreater(r['visual_turn'],.2)

    def test_mirrored_right_curve_is_symmetric(self):
        left=LaneCorridorGuard().process(corridor())
        right=LaneCorridorGuard().process(np.fliplr(corridor()))
        self.assertAlmostEqual(left['visual_turn'],-right['visual_turn'],delta=.01)

    def test_known_right_border_continues_through_curve(self):
        g=LaneCorridorGuard();g.process(corridor())
        r=g.process(corridor(True))
        self.assertEqual(r['visual_source'],'TRACK RIGHT')
        self.assertGreater(r['visual_turn'],.2)

    def test_unknown_single_border_cannot_guess_its_side(self):
        self.assertIsNone(LaneCorridorGuard().process(corridor(True))['visual_turn'])

    def test_brief_gap_holds_two_images_then_releases(self):
        g=LaneCorridorGuard();g.process(corridor())
        blank=np.zeros((240,320),np.uint8)
        for _ in range(2):
            self.assertEqual(g.process(blank)['visual_source'],'BRIEF GAP')
        self.assertIsNone(g.process(blank)['visual_turn'])

    def test_reset_discards_border_identity(self):
        g=LaneCorridorGuard();g.process(corridor());g.reset()
        self.assertIsNone(g.process(corridor(True))['visual_turn'])

    def policy(self):
        p=CommandPolicy();p.good_frames=3;p.first_stamp=0.;p.last_stamp=.4
        return p

    def apply(self,p,stamp,raw,visual=None):
        r=dict(timestamp=stamp,valid=True,turn=raw,visual_turn=visual,
               visual_weight=.75,visual_source='CURVE PAIR')
        p.command('AUTO','STOP',0,r,stamp)
        return p.command_turn

    def test_neural_direction_jump_is_smoothed_and_geometry_corrects_it(self):
        p=self.policy();before=self.apply(p,.5,-.6)
        after=self.apply(p,.625,.6)
        self.assertLessEqual(abs(after-before),.16001)
        for i in range(8):
            self.apply(p,.75+i*.125,-.5,.6)
        self.assertGreater(p.command_turn,.25)

    def test_duplicate_frame_does_not_repeat_ramp(self):
        p=self.policy();self.apply(p,.5,-.6)
        value=self.apply(p,.625,.6)
        for _ in range(20):self.apply(p,.625,.6)
        self.assertEqual(p.command_turn,value)

    def test_stale_frame_still_stops(self):
        p=self.policy();self.apply(p,.5,.3,.4)
        self.assertEqual(p.command('AUTO','STOP',0,None,1.1),'STOP')


if __name__=='__main__':unittest.main()
