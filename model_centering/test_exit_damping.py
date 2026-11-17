"""Control-only checks; camera and motors are not connected on the PC."""
import sys
import types
import unittest

sys.modules.setdefault('cv2', types.SimpleNamespace())
from jetson_model_centering_exit_damped import CommandPolicy


class ExitDampingTest(unittest.TestCase):
    def setUp(self):
        self.policy = CommandPolicy(auto_speed=0.12, exit_damping=0.6)
        self.policy.good_frames = 3
        self.policy.first_stamp = 0.0
        self.policy.last_stamp = 0.3

    def drive(self, turn, stamp):
        return self.policy.command('AUTO', 'STOP', 0, {
            'timestamp': stamp, 'valid': True, 'turn': turn,
        }, stamp)

    def test_exit_turn_reduces_without_affecting_entry(self):
        self.assertEqual(self.drive(0.4, 0.5), 'DRIVE,0.120,0.4000')
        self.assertEqual(self.drive(0.6, 0.6), 'DRIVE,0.120,0.6000')
        self.assertEqual(self.drive(0.4, 0.7), 'DRIVE,0.120,0.2800')
        self.assertEqual(self.drive(0.4, 0.7), 'DRIVE,0.120,0.2800')
        self.assertEqual(self.drive(-0.2, 0.8), 'DRIVE,0.120,-0.2000')
        self.assertEqual(self.drive(-0.1, 0.9), 'DRIVE,0.120,-0.0400')

    def test_stale_frame_stops_and_resets_steering_history(self):
        self.drive(0.6, 0.5)
        self.assertEqual(self.policy.command('AUTO', 'STOP', 0, None, 0.9), 'STOP')
        self.assertIsNone(self.policy.previous_raw_turn)

    def test_manual_does_not_apply_exit_damping(self):
        self.assertEqual(self.policy.command('MANUAL', 'L', 0.5, None, 0.6),
                         'MANUAL,F,0.300,35')


if __name__ == '__main__':
    unittest.main()
