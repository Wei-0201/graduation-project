"""Offline behavioral tests: no motors, camera, or network server is opened."""
import ast
from pathlib import Path
import time
import unittest
import cv2
import numpy as np
import jetson_car as car


def road(shift=0, left=True, right=True, clutter=False):
    image = np.full((240, 320, 3), 190, np.uint8)
    if left:
        cv2.line(image, (125 + shift, 72), (35 + shift, 220), (25, 25, 25), 7)
    if right:
        cv2.line(image, (195 + shift, 72), (285 + shift, 220), (25, 25, 25), 7)
    if clutter:
        # Furniture above ROI and isolated floor speckles must not become a lane.
        cv2.rectangle(image, (0, 0), (320, 60), (15, 15, 15), -1)
        cv2.line(image, (16, 0), (16, 110), (10, 10, 10), 9)
        for x, y in ((80, 105), (230, 133), (160, 172), (105, 207)):
            cv2.circle(image, (x, y), 2, (20, 20, 20), -1)
    return image


class VisionTests(unittest.TestCase):
    def test_faint_far_left_exits_image_before_default_target(self):
        frame = np.full((240, 320, 3), 150, np.uint8)
        cv2.line(frame, (85, 65), (1, 111), (120, 120, 120), 4)
        cv2.line(frame, (128, 65), (245, 220), (35, 35, 35), 10)
        _, _, result = car.LaneDetector().process(frame, 1)
        self.assertTrue(result['valid'], result)
        self.assertLess(result['target_y'], 112)
        self.assertLess(result['offset'], -40)
        # A single right boundary must not start inventing its absent partner.
        frame = np.full((240, 320, 3), 150, np.uint8)
        cv2.line(frame, (128, 65), (245, 220), (35, 35, 35), 10)
        self.assertFalse(car.LaneDetector().process(frame, 1)[2]['valid'])

    def test_visual_extension_keeps_evidence_separate(self):
        canvas = np.zeros((240, 320, 3), np.uint8)
        model = dict(coef=np.array([0.0, 100.0]), lo=90, hi=180,
                     points=np.array([[100.0, 90.0], [100.0, 180.0]]))
        car.draw_lane_model(canvas, model)
        self.assertGreater(np.count_nonzero(canvas[30:90]), 0)
        self.assertGreater(np.count_nonzero(canvas[181:239]), 0)
        self.assertEqual(np.count_nonzero(canvas[:28]), 0)
        self.assertEqual(len(model['points']), 2)

    def test_clean_and_clutter(self):
        for clutter in (False, True):
            _, _, result = car.LaneDetector().process(road(clutter=clutter), 1)
            self.assertTrue(result['valid'], result)
            self.assertLess(abs(result['offset']), 3)

    def test_both_offset_directions(self):
        for shift in (-20, 20):
            _, _, r = car.LaneDetector().process(road(shift), 1)
            self.assertTrue(r['valid'], r)
            self.assertAlmostEqual(r['offset'], shift, delta=3)

    def test_missing_or_uniform(self):
        for frame in (road(left=False), road(right=False),
                      np.full((240, 320, 3), 20, np.uint8),
                      np.full((240, 320, 3), 190, np.uint8)):
            _, _, r = car.LaneDetector().process(frame, 1)
            self.assertFalse(r['valid'], r)

    def test_short_left_not_extended(self):
        frame = road(left=False)
        cv2.line(frame, (12, 73), (12, 105), (20, 20, 20), 7)
        _, _, r = car.LaneDetector().process(frame, 1)
        self.assertFalse(r['valid'], r)

    def test_tracking_jump_and_expiry(self):
        detector = car.LaneDetector()
        self.assertTrue(detector.process(road(), 1)[2]['valid'])
        self.assertFalse(detector.process(road(35), 1.04)[2]['valid'])
        self.assertTrue(detector.process(road(35), 1.6)[2]['valid'])

    def test_outliers_in_fit(self):
        pts = [(0.4 * y + 20, y) for y in range(80, 220, 5)]
        pts += [(270, y) for y in range(100, 180, 9)]
        models = car.line_models(np.array(pts, np.float32), 139)
        self.assertTrue(any(abs(m['probes'][0] - 75.6) < 2 for m in models))

    def test_ambiguous_third_line(self):
        frame = road()
        cv2.line(frame, (160, 72), (160, 220), (25, 25, 25), 7)
        r = car.LaneDetector().process(frame, 1)[2]
        self.assertFalse(r['valid'], r)


class ControlTests(unittest.TestCase):
    def test_start_hold_accelerate_only_when_stable_and_reset_after_loss(self):
        p = car.CommandPolicy()
        speeds = []
        for i in range(200):
            now = i * 0.05
            d = dict(valid=True, offset=0.0, timestamp=now, target_y=139)
            command = p.command('AUTO', 'STOP', 0, d, now)
            speeds.append(float(command.split(',')[1]))
        self.assertEqual(speeds[2], car.AUTO_START_SPEED)
        self.assertEqual(speeds[15], car.AUTO_START_SPEED)
        self.assertAlmostEqual(speeds[-1], car.AUTO_MAX_SPEED)
        for a, b in zip(speeds[2:], speeds[3:]):
            self.assertLessEqual(b - a, car.AUTO_ACCELERATION * 0.05 + 0.0011)
        d = dict(valid=True, offset=25.0, timestamp=10.05, target_y=139)
        self.assertEqual(float(p.command('AUTO', 'STOP', 0, d, 10.05).split(',')[1]), car.AUTO_START_SPEED)
        self.assertEqual(p.command('AUTO', 'STOP', 0, None, 10.1), 'CTRL,0.000,999')
        for i in range(3):
            now = 10.2 + i * .05
            d = dict(valid=True, offset=0.0, timestamp=now, target_y=139)
            command = p.command('AUTO', 'STOP', 0, d, now)
        self.assertEqual(float(command.split(',')[1]), car.AUTO_START_SPEED)

    def test_large_offset_never_accelerates_even_before_steering_settles(self):
        p = car.CommandPolicy()
        for i in range(120):
            now = i * .05
            d = dict(valid=True, offset=60.0, timestamp=now, target_y=139)
            command = p.command('AUTO', 'STOP', 0, d, now)
            if i >= 2:
                self.assertEqual(float(command.split(',')[1]), car.AUTO_START_SPEED)

    def test_changed_measurement_depth_does_not_create_derivative_kick(self):
        policy = car.CommandPolicy()
        for i in range(3):
            detection = dict(timestamp=1 + i / 30, offset=-30, valid=True, target_y=139)
            policy.command('AUTO', 'STOP', 0, detection, detection['timestamp'])
        detection = dict(timestamp=1.1, offset=-45, valid=True, target_y=98)
        policy.command('AUTO', 'STOP', 0, detection, 1.1)
        self.assertEqual(policy.controller.rate, 0)

    def detection(self, stamp, valid=True):
        return dict(timestamp=stamp, valid=valid, offset=20)

    def test_acquisition_distinct_frames_and_stale_stop(self):
        p = car.CommandPolicy()
        self.assertEqual(p.command('AUTO', 'STOP', 0, None, 1), 'CTRL,0.000,999')
        for now in (1, 1, 1.03):
            self.assertEqual(p.command('AUTO', 'STOP', 0, self.detection(now), now), 'CTRL,0.000,999')
        self.assertNotEqual(p.command('AUTO', 'STOP', 0, self.detection(1.06), 1.06), 'CTRL,0.000,999')
        self.assertEqual(p.command('AUTO', 'STOP', 0, self.detection(1.06), 1.5), 'CTRL,0.000,999')

    def test_lost_stop_and_reacquisition(self):
        p = car.CommandPolicy()
        for i in range(3):
            p.command('AUTO', 'STOP', 0, self.detection(i / 30), i / 30)
        self.assertEqual(p.command('AUTO', 'STOP', 0, self.detection(.1, False), .1), 'CTRL,0.000,999')
        self.assertEqual(p.command('AUTO', 'STOP', 0, self.detection(.13), .13), 'CTRL,0.000,999')

    def test_manual_browser_lease(self):
        p = car.CommandPolicy()
        self.assertEqual(p.command('MANUAL', 'L', 1, None, 1.1), 'MANUAL,F,0.300,35')
        self.assertEqual(p.command('MANUAL', 'L', 1, None, 1.7), 'CTRL,0.000,999')

    def test_sign_and_slew(self):
        for offset, sign in ((30, -1), (-30, 1)):
            p = car.SteeringController()
            previous = 0
            for i in range(30):
                current = p.update(offset, i / 30)
                self.assertLessEqual(abs(current - previous), 5)
                self.assertLessEqual(abs(current), car.MAX_STEERING)
                previous = current
            self.assertGreater(sign * current, 0)

    def test_api_stop_auto_and_late_manual(self):
        state = car.Car()
        state.connected = True
        client = car.make_app(state).test_client()
        epoch = state.generation
        self.assertEqual(client.post('/mode', json={'mode': 'AUTO'}).status_code, 200)
        self.assertEqual(client.post('/control', json={'command': 'STOP'}).status_code, 200)
        self.assertEqual(state.mode, 'MANUAL')
        self.assertEqual(client.post('/control', json={'command': 'F', 'epoch': epoch}).status_code, 409)
        self.assertEqual(client.post('/control', json={'command': 'F', 'epoch': state.generation}).status_code, 200)
        self.assertEqual(client.post('/control', json=[]).status_code, 400)
        self.assertEqual(client.post('/mode', json={'mode': 5}).status_code, 400)

    def test_python36_syntax(self):
        ast.parse(Path(car.__file__).read_text(encoding='utf-8'), feature_version=(3, 6))


if __name__ == '__main__':
    unittest.main(verbosity=2)
