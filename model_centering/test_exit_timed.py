"""Control-only regression checks for camera-rate-independent bend exit."""
import sys
import types
import unittest

sys.modules.setdefault('cv2', types.SimpleNamespace())
from jetson_model_centering_exit_timed import CommandPolicy


class TimedExitTest(unittest.TestCase):
    def policy(self):
        p = CommandPolicy(auto_speed=0.12, exit_damping=0.85,
                          left_exit_damping=0.85)
        p.good_frames = 3
        p.first_stamp = 0.0
        p.last_stamp = 0.3
        return p

    def command(self, policy, turn, stamp):
        return policy.command('AUTO', 'STOP', 0, {
            'timestamp': stamp, 'valid': True, 'turn': turn,
        }, stamp)

    def test_right_exit_reduces_turn_and_left_remains_supported(self):
        p = self.policy()
        self.assertEqual(self.command(p, -0.6, 0.5), 'DRIVE,0.120,-0.6000')
        self.assertEqual(self.command(p, -0.4, 0.625), 'DRIVE,0.120,-0.2300')
        self.assertEqual(self.command(p, -0.4, 0.625), 'DRIVE,0.120,-0.2300')
        self.assertEqual(self.command(p, 0.6, 0.75), 'DRIVE,0.120,0.6000')
        self.assertEqual(self.command(p, 0.4, 0.875), 'DRIVE,0.120,0.2300')

    def test_faster_camera_gives_similar_lead(self):
        slow = self.policy()
        self.command(slow, -0.6, 0.5)
        self.command(slow, -0.4, 0.625)
        fast = self.policy()
        self.command(fast, -0.6, 0.5)
        self.command(fast, -0.5, 0.5625)
        self.command(fast, -0.4, 0.625)
        self.assertAlmostEqual(slow.command_turn, fast.command_turn, delta=0.03)

    def test_stale_prediction_stops(self):
        p = self.policy()
        self.command(p, -0.6, 0.5)
        self.assertEqual(p.command('AUTO', 'STOP', 0, None, 1.0), 'STOP')


if __name__ == '__main__':
    unittest.main()
