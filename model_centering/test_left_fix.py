"""Check the left-bend control and stale-frame STOP without connecting motors."""
import sys
import types
import unittest

sys.modules.setdefault('cv2', types.SimpleNamespace())
from jetson_model_centering_left_fix import CommandPolicy


class LeftBendControlTest(unittest.TestCase):
    def setUp(self):
        self.policy = CommandPolicy(auto_speed=0.12, exit_damping=0.6,
                                    left_exit_damping=0.85)
        self.policy.good_frames = 3
        self.policy.first_stamp = 0.0
        self.policy.last_stamp = 0.3

    def drive(self, turn, stamp, now=None):
        return self.policy.command('AUTO', 'STOP', 0, {
            'timestamp': stamp, 'valid': True, 'turn': turn,
        }, stamp if now is None else now)

    def test_left_exit_damps_more_than_right(self):
        self.assertEqual(self.drive(0.6, 0.5), 'DRIVE,0.120,0.6000')
        self.assertEqual(self.drive(0.4, 0.63), 'DRIVE,0.120,0.2300')
        self.assertEqual(self.drive(-0.6, 0.76), 'DRIVE,0.120,-0.6000')
        self.assertEqual(self.drive(-0.4, 0.89), 'DRIVE,0.120,-0.2800')

    def test_single_slow_frame_stays_active_but_old_frame_stops(self):
        self.assertEqual(self.drive(0.5, 0.5), 'DRIVE,0.120,0.5000')
        self.assertEqual(self.drive(0.5, 0.9), 'DRIVE,0.120,0.5000')
        self.assertEqual(self.drive(0.5, 0.9, 1.36), 'STOP')
        self.assertEqual(self.policy.control_state, 'WAIT FRESH MODEL')

    def test_manual_turn_is_unchanged(self):
        self.assertEqual(self.policy.command('MANUAL', 'L', 0.5, None, 0.6),
                         'MANUAL,F,0.300,35')


if __name__ == '__main__':
    unittest.main()
