import unittest
from observed_exit_easing import ObservedExitEasing


def frame(i, raw, heading, position=.5, source='PAIR'):
    return dict(timestamp=i, turn=raw, parking_corridor=dict(
        position=position, heading=heading, width=220,
        source=source, right_observed=True))


class ExitTests(unittest.TestCase):
    def test_both_directions_ease_only_after_bend_then_straight(self):
        for sign in (-1, 1):
            p = ObservedExitEasing()
            for i,h in enumerate((.12,.14,.04)):
                self.assertEqual(p.adjust(sign*.3, frame(i, sign*.3,sign*h)), sign*.3)
            out=p.adjust(sign*.28,frame(3,sign*.28,0))
            self.assertLess(abs(out), .28)
            self.assertGreaterEqual(out*sign,0)
            self.assertLessEqual(abs(out-sign*.28),.18+1e-9)

    def test_straight_start_is_not_a_bend_exit(self):
        p=ObservedExitEasing()
        for i in range(10):
            self.assertEqual(p.adjust(.3,frame(i,.3,0)),.3)

    def test_opposite_bend_frames_do_not_arm_exit(self):
        p=ObservedExitEasing()
        p.adjust(-.3,frame(0,-.3,-.15))
        p.adjust(.3,frame(1,.3,.15))
        for i in range(2,6):
            self.assertEqual(p.adjust(.3,frame(i,.3,0)),.3)

    def test_missing_or_inferred_pair_does_not_trigger_exit(self):
        for source in ('RIGHT TRACK','NONE'):
            p=ObservedExitEasing()
            p.adjust(.3,frame(0,.3,.15))
            p.adjust(.3,frame(1,.3,.15))
            for i in range(2,10):
                self.assertEqual(p.adjust(.3,frame(i,.3,0,source=source)),.3)

    def test_increasing_turn_and_duplicate_images_do_not_unwind(self):
        p=ObservedExitEasing()
        p.adjust(.3,frame(0,.3,.15))
        p.adjust(.3,frame(1,.3,.15))
        p.adjust(.25,frame(2,.25,0))
        self.assertEqual(p.adjust(.25,frame(2,.25,0)),.25)
        self.assertEqual(p.adjust(.4,frame(3,.4,0)),.4)

if __name__=='__main__':
    unittest.main()
