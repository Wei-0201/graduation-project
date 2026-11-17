import unittest
from right_entry_guard import RightEntryGuard


def frame(position=.55, heading=-.02, source='PAIR', width=240):
    return dict(parking_corridor=dict(position=position, heading=heading,
                source=source, width=width, right_observed=True))


class RightEntryTests(unittest.TestCase):
    def test_two_observations_then_ease_early_right(self):
        guard = RightEntryGuard()
        self.assertEqual(guard.adjust(-.2, frame()), -.2)
        self.assertAlmostEqual(guard.adjust(-.2, frame()), -.02)

    def test_visible_near_bend_releases_without_timer(self):
        guard = RightEntryGuard()
        guard.adjust(-.4, frame())
        self.assertAlmostEqual(guard.adjust(-.4, frame(heading=-.12)), -.21)
        self.assertEqual(guard.adjust(-.4, frame(heading=-.2)), -.4)

    def test_left_and_small_right_remain_unchanged(self):
        guard = RightEntryGuard()
        guard.adjust(-.2, frame())
        self.assertEqual(guard.adjust(.4, frame()), .4)
        self.assertEqual(guard.adjust(-.01, frame()), -.01)

    def test_left_offset_allows_more_right_recovery(self):
        guard = RightEntryGuard()
        guard.adjust(-.2, frame(position=.3))
        self.assertEqual(guard.adjust(-.2, frame(position=.3)), -.2)

    def test_missing_inferred_or_invalid_pair_cannot_limit(self):
        for bad in ({}, frame(source='RIGHT TRACK'), frame(width=float('nan')),
                    frame(heading=float('nan')), frame(position=1.1)):
            guard = RightEntryGuard()
            self.assertEqual(guard.adjust(-.4, bad), -.4)
            self.assertEqual(guard.adjust(-.4, frame()), -.4)

    def test_segmentation_gap_does_not_restore_large_right_turn(self):
        guard = RightEntryGuard()
        guard.adjust(-.3, frame())
        guard.adjust(-.3, frame())
        self.assertAlmostEqual(guard.adjust(-.4, {}), -.02)
        self.assertAlmostEqual(guard.adjust(-.4, {}), -.02)
        self.assertEqual(guard.adjust(-.4, {}), -.4)

    def test_right_edge_limits_even_when_heading_suggests_bend(self):
        guard = RightEntryGuard()
        guard.adjust(-.3, frame())
        guard.adjust(-.3, frame())
        tracked = frame(position=.85, heading=-.3, source='RIGHT TRACK')
        tracked['parking_corridor']['right_gap_px'] = 36.
        self.assertAlmostEqual(guard.adjust(-.4, tracked), 0.)
        self.assertEqual(guard.adjust(.2, tracked), .2)

    def test_duplicate_frames_do_not_confirm_pair(self):
        guard = RightEntryGuard()
        detection = dict(frame(), timestamp=1.)
        self.assertEqual(guard.adjust(-.3, detection), -.3)
        self.assertEqual(guard.adjust(-.3, detection), -.3)
        detection['timestamp'] = 1.1
        self.assertAlmostEqual(guard.adjust(-.3, detection), -.02)

if __name__ == '__main__':
    unittest.main()
