"""Run without a camera or motor: python3 -m unittest -v test_jetson_recovery."""
import ast
import unittest
from pathlib import Path
import cv2
import numpy as np
import jetson_car_recovery as car


def road(shift=0, left=True, right=True):
    image = np.full((240, 320, 3), 190, np.uint8)
    if left:
        cv2.line(image, (125 + shift, 72), (35 + shift, 220), (25, 25, 25), 7)
    if right:
        cv2.line(image, (195 + shift, 72), (285 + shift, 220), (25, 25, 25), 7)
    return image


def detection(t, quality='dual', offset=-20):
    return dict(valid=quality == 'dual', quality=quality,
                timestamp=t, target_y=139, offset=offset)


class RecoveryTests(unittest.TestCase):
    def test_tracking_rejects_jump_then_expires_at_configured_time(self):
        d = car.LaneDetector()
        self.assertTrue(d.process(road(), 1)[2]['valid'])
        self.assertFalse(d.process(road(35), 1.04)[2]['valid'])
        self.assertTrue(d.process(road(35), 1 + car.TRACK_EXPIRY + .01)[2]['valid'])

    def test_target_path_tracks_dual_center_and_single_prediction(self):
        d = car.LaneDetector()
        for t in (1, 1.05, 1.1):
            r = d.process(road(), t)[2]
        self.assertGreater(len(r['target_path']), 5)
        self.assertTrue(all(abs(x - 160) < 3 for x, y in r['target_path']))
        r = d.process(road(10, left=False), 1.15)[2]
        self.assertEqual(r['quality'], 'single')
        self.assertTrue(all(abs(x - 170) < 3 for x, y in r['target_path']))
        blank = np.full((240, 320, 3), 190, np.uint8)
        r = d.process(blank, 1.2)[2]
        self.assertIn('target_path', r)
        self.assertFalse(r['valid'])
        self.assertIsNone(r['offset'])
        self.assertNotIn('target_path', d.process(blank, 1.71)[2])

    def armed(self):
        p = car.CommandPolicy()
        for t in (1, 1.05, 1.1):
            command = p.command('AUTO', 'STOP', 0, detection(t), t)
        self.assertNotEqual(command, 'CTRL,0.000,999')
        return p

    def test_no_memory_cannot_start_on_single_or_none(self):
        for quality in ('single', 'none'):
            p = car.CommandPolicy()
            self.assertEqual(p.command('AUTO', 'STOP', 0, detection(1, quality), 1), 'CTRL,0.000,999')

    def test_hold_last_direction_and_cap_speed(self):
        p = self.armed()
        p.speed = car.AUTO_MAX_SPEED
        previous = p.steering
        command = p.command('AUTO', 'STOP', 0, detection(1.15, 'none', None), 1.15)
        self.assertEqual(command, 'CTRL,{:.3f},{}'.format(car.RECOVERY_SPEED, previous))
        self.assertEqual(p.control_state, 'MEMORY RECOVERY')

    def test_single_does_not_renew_deadline(self):
        p = self.armed()
        last_real = p.last_dual_time
        for t in (1.15, 1.3, 1.5, 1.69):
            self.assertNotEqual(p.command('AUTO', 'STOP', 0, detection(t, 'single'), t), 'CTRL,0.000,999')
            self.assertEqual(p.last_dual_time, last_real)
        self.assertEqual(p.command('AUTO', 'STOP', 0, detection(1.71, 'single'), 1.71), 'CTRL,0.000,999')

    def test_stale_or_failed_camera_stops_immediately(self):
        for d in (None, detection(1.15, 'single'), detection(1.15, 'none', None)):
            p = self.armed()
            self.assertEqual(p.command('AUTO', 'STOP', 0, d, 1.5), 'CTRL,0.000,999')
            self.assertEqual(p.command('AUTO', 'STOP', 0, detection(1.55, 'single'), 1.55), 'CTRL,0.000,999')

    def test_ambiguity_is_not_recovery_permission(self):
        p = self.armed()
        self.assertEqual(p.command('AUTO', 'STOP', 0, detection(1.15, 'ambiguous'), 1.15), 'CTRL,0.000,999')

    def test_excessive_estimated_offset_stops(self):
        p = self.armed()
        self.assertEqual(p.command('AUTO', 'STOP', 0, detection(1.15, 'single', -120), 1.15), 'CTRL,0.000,999')

    def test_three_distinct_dual_frames_reacquire_then_ramp_from_start(self):
        p = self.armed()
        p.command('AUTO', 'STOP', 0, detection(1.15, 'none', None), 1.15)
        for t in (1.2, 1.2, 1.25):
            p.command('AUTO', 'STOP', 0, detection(t), t)
            self.assertTrue(p.recovering)
        command = p.command('AUTO', 'STOP', 0, detection(1.3), 1.3)
        self.assertFalse(p.recovering)
        self.assertEqual(float(command.split(',')[1]), car.AUTO_START_SPEED)
        self.assertEqual(p.started_at, 1.3)

    def test_intermittent_single_dual_does_not_extend_recovery(self):
        p = self.armed()
        for t, q in ((1.15, 'single'), (1.2, 'dual'), (1.25, 'single'),
                     (1.3, 'dual'), (1.4, 'single'), (1.55, 'dual')):
            p.command('AUTO', 'STOP', 0, detection(t, q), t)
        self.assertEqual(p.last_dual_time, 1.1)
        self.assertEqual(p.command('AUTO', 'STOP', 0, detection(1.71, 'none', None), 1.71), 'CTRL,0.000,999')

    def test_stop_api_clears_recovery(self):
        state = car.Car()
        state.connected = True
        state.mode = 'AUTO'
        state.policy = self.armed()
        client = car.make_app(state).test_client()
        self.assertEqual(client.post('/control', json={'command': 'STOP'}).status_code, 200)
        self.assertIsNone(state.policy.last_dual_time)
        self.assertEqual(state.mode, 'MANUAL')

    def test_detector_infers_each_missing_side_without_refreshing_memory(self):
        for visible in ('left', 'right'):
            d = car.LaneDetector()
            for t in (1, 1.05, 1.1):
                self.assertTrue(d.process(road(), t)[2]['valid'])
            r = d.process(road(10, left=visible == 'left', right=visible == 'right'), 1.15)[2]
            self.assertFalse(r['valid'])
            self.assertEqual(r['quality'], 'single')
            self.assertAlmostEqual(r['offset'], 10, delta=3)
            self.assertEqual(d.last_good, 1.1)
            r = d.process(road(left=False), 1.71)[2]
            self.assertEqual(r['quality'], 'none')

    def test_detector_does_not_infer_with_no_history_or_one_dual_frame(self):
        d = car.LaneDetector()
        self.assertEqual(d.process(road(left=False), 1)[2]['quality'], 'none')
        d.process(road(), 1.05)
        self.assertEqual(d.process(road(left=False), 1.1)[2]['quality'], 'none')

    def test_no_evidence_is_not_misrepresented_as_observed_line(self):
        d = car.LaneDetector()
        for t in (1, 1.05, 1.1):
            d.process(road(), t)
        r = d.process(np.full((240, 320, 3), 190, np.uint8), 1.15)[2]
        self.assertFalse(r['valid'])
        self.assertIsNone(r['offset'])

    def test_python36_syntax(self):
        ast.parse(Path(car.__file__).read_text(encoding='utf-8'), feature_version=(3, 6))


if __name__ == '__main__':
    unittest.main(verbosity=2)
