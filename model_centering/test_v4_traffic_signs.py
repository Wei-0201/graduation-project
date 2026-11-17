import time
import threading
import unittest
from unittest.mock import patch
import importlib.util
import subprocess

import cv2
import numpy as np

import test_v4_exit_fix as legacy
from jetson_v4_traffic_signs import CommandPolicy
from jetson_v4_traffic_signs import Car, make_app
from sign_parking_control import RightCorridorTracker, TrafficPolicy
from traffic_sign_worker import SignRecognizer, SignWorker
import user_sign_ocr
import sign_ocr_engine


def signs(now, seq=1, generation=0, stop=False, park=False, stop_present=None, park_present=None):
    return dict(timestamp=now, sequence=seq, generation=generation, error=None,
                stop_confirmed=stop, park_confirmed=park,
                stop_present=stop if stop_present is None else stop_present,
                park_present=park if park_present is None else park_present)


def lane(now, position=.5, heading=0., right=True, gap=None):
    return dict(timestamp=now, valid=True, parking_corridor=dict(
        position=position, heading=heading, right_observed=right, right_gap_px=(200*(1-position) if gap is None else gap)))


class PreservedV4Tests(legacy.ExistingControlTests):
    def policy(self, visual_weight=0.):
        p = CommandPolicy(visual_weight=visual_weight)
        p.good_frames, p.first_stamp, p.last_stamp = 3, 0., .4
        return p


class PreservedUnwindTests(legacy.UnwindTests):
    def policy(self):
        return CommandPolicy()


class TrafficTests(unittest.TestCase):
    def call(self, p, now=1., snapshot=None, detection=None, mode='AUTO', base='DRIVE,0.120,0.0000', generation=0):
        return p.command(mode, base, detection or lane(now), snapshot or signs(now), now, generation)

    def test_follow_keeps_v4_command(self):
        self.assertEqual(self.call(TrafficPolicy(), base='DRIVE,0.120,0.4500'), 'DRIVE,0.120,0.4500')

    def test_speed_arrow_latches_until_down_and_preserves_steering(self):
        policy = TrafficPolicy()
        base = 'DRIVE,0.120,-0.4500'
        up = dict(signs(1), arrow_direction='UP', arrow_confirmed=True)
        self.assertEqual(self.call(policy, snapshot=up, base=base), 'DRIVE,0.300,-0.4500')
        self.assertEqual(self.call(policy, 30, signs(30, seq=2), base=base), 'DRIVE,0.300,-0.4500')
        down = dict(signs(31, seq=3), arrow_direction='DOWN', arrow_confirmed=True)
        self.assertEqual(self.call(policy, 31, down, base=base), base)

    def test_speed_arrow_obeys_stop_parking_and_stale_frame(self):
        policy = TrafficPolicy(stop_seconds=5)
        up = dict(signs(1), arrow_direction='UP', arrow_confirmed=True)
        self.assertEqual(self.call(policy, snapshot=up, base='STOP'), 'STOP')
        self.assertFalse(policy.speed_boosted)
        self.assertEqual(self.call(policy, 3, up), 'STOP')
        up = dict(signs(4, seq=2), arrow_direction='UP', arrow_confirmed=True)
        self.assertEqual(self.call(policy, 4, up), 'DRIVE,0.300,0.0000')
        stop = dict(signs(5, seq=3, stop=True), arrow_direction='UP', arrow_confirmed=True)
        self.assertEqual(self.call(policy, 5, stop), 'STOP')
        self.assertEqual(self.call(policy, 9, signs(9, seq=4)), 'STOP')
        park = dict(signs(11, seq=5, park=True), arrow_direction='UP', arrow_confirmed=True)
        self.assertTrue(self.call(policy, 11, park).startswith('DRIVE,0.120,'))
        self.call(policy, 12, signs(12, seq=6), mode='MANUAL', base='STOP')
        self.assertFalse(policy.speed_boosted)

    def test_stop_holds_until_auto_rearm_and_same_sign_does_not_repeat(self):
        p = TrafficPolicy()
        self.assertEqual(self.call(p, snapshot=signs(1, stop=True)), 'STOP')
        self.assertEqual(self.call(p, 100, signs(100, seq=2, stop=True)), 'STOP')
        p.cancel_motion()  # AUTO button disarms before selecting AUTO again.
        self.assertTrue(self.call(p, 101, signs(101, seq=3, stop=True)).startswith('DRIVE'))

    def test_optional_timed_pause_without_repeated_pause(self):
        p = TrafficPolicy(stop_seconds=3)
        self.assertEqual(self.call(p, snapshot=signs(1, stop=True)), 'STOP')
        self.assertEqual(self.call(p, 3, signs(3, seq=2, stop=True)), 'STOP')
        self.assertTrue(self.call(p, 4.1, signs(4.1, seq=3, stop=True)).startswith('DRIVE'))

    def test_five_second_pause_resumes_with_sign_still_visible(self):
        p = TrafficPolicy(stop_seconds=5)
        self.assertEqual(self.call(p, 1, signs(1, seq=1, stop=True)), 'STOP')
        self.assertEqual(self.call(p, 5.99, signs(5.99, seq=2, stop=True)), 'STOP')
        for seq, now in enumerate((6., 7., 12.), 3):
            self.assertTrue(self.call(p, now, signs(now, seq=seq, stop=True)).startswith('DRIVE'))
        # The timer must not override camera/lane STOP or an expired OCR result.
        self.assertEqual(self.call(p, 13, signs(13, seq=6, stop=True), base='STOP'), 'STOP')
        self.assertEqual(self.call(p, 15, signs(13, seq=6, stop=True)), 'STOP')

    def test_three_absent_ocr_frames_rearm_not_repeated_same_snapshot(self):
        p = TrafficPolicy(stop_seconds=.1)
        self.call(p, snapshot=signs(1, stop=True))
        for _ in range(10):
            self.call(p, 1.2, signs(1.2, seq=2))
        self.assertTrue(p.handled['stop'])
        self.call(p, 1.3, signs(1.3, seq=3))
        self.call(p, 1.4, signs(1.4, seq=4))
        self.assertFalse(p.handled['stop'])
        self.assertEqual(self.call(p, 1.5, signs(1.5, seq=5, stop=True)), 'STOP')

    def test_stop_outranks_parking(self):
        p = TrafficPolicy()
        self.assertEqual(self.call(p, snapshot=signs(1, stop=True, park=True)), 'STOP')
        self.assertIsNone(p.park_started)

    def test_manual_can_cancel_and_preserves_operator_command(self):
        p = TrafficPolicy()
        self.call(p, snapshot=signs(1, park=True))
        result = self.call(p, 1.1, signs(1.1, seq=2), mode='MANUAL', base='MANUAL,F,0.300,35')
        self.assertEqual(result, 'MANUAL,F,0.300,35')
        self.assertIsNone(p.park_started)

    def test_stale_error_or_wrong_epoch_ocr_stops(self):
        for snapshot in [signs(-2), signs(1, generation=1), dict(signs(1), error='timeout')]:
            self.assertEqual(self.call(TrafficPolicy(), snapshot=snapshot), 'STOP')
        self.assertEqual(TrafficPolicy().command('AUTO', 'DRIVE,0.120,0', lane(1), None, 1, 0), 'STOP')

    def test_parking_starts_gently_right(self):
        p = TrafficPolicy()
        result = self.call(p, snapshot=signs(1, park=True))
        self.assertTrue(result.startswith('DRIVE,0.120,'))
        self.assertLess(float(result.split(',')[2]), 0)
        self.assertLessEqual(abs(float(result.split(',')[2])), .30)

    def test_parking_success_needs_three_fresh_position_frames(self):
        p = TrafficPolicy()
        self.call(p, snapshot=signs(1, park=True), detection=lane(1, 1.))
        for _ in range(10):
            self.call(p, snapshot=signs(1, park=True), detection=lane(1, 1.))
        self.assertFalse(p.parked)
        for i in (2, 3):
            self.assertEqual(self.call(p, i/10+1, signs(i/10+1, seq=i, park=True), lane(i/10+1, 1.)), 'STOP')
        self.assertTrue(p.parked)
        self.assertEqual(self.call(p, 2, signs(2, seq=4, park=True)), 'STOP')

    def test_parked_auto_to_leave_ignores_same_p_sign(self):
        p = TrafficPolicy()
        for i in (1, 2, 3):
            self.call(p, i/10, signs(i/10, seq=i, park=True), lane(i/10, 1.))
        self.assertTrue(p.parked)
        p.cancel_motion()
        self.assertTrue(self.call(p, .4, signs(.4, seq=4, park=True)).startswith('DRIVE'))

    def test_near_right_edge_stops(self):
        p = TrafficPolicy()
        self.assertEqual(self.call(p, snapshot=signs(1, park=True), detection=lane(1, 1.14, gap=-28)), 'STOP')
        self.assertIsNotNone(p.aborted)

    def test_heading_must_align_before_parked(self):
        p = TrafficPolicy()
        for i in (1, 2, 3, 4):
            self.call(p, i/10, signs(i/10, seq=i, park=True), lane(i/10, 1., .2))
        self.assertFalse(p.parked)

    def test_missing_right_line_never_drives_and_latches_abort(self):
        p = TrafficPolicy()
        for i in (1, 2, 3):
            self.assertEqual(self.call(p, i/10, signs(i/10, seq=i, park=True), lane(i/10, right=False)), 'STOP')
        self.assertIsNotNone(p.aborted)

    def test_stale_lane_or_base_stop_prevents_parking_motion(self):
        for base, detection in [('STOP', lane(1)), ('DRIVE,0.120,0', lane(0))]:
            p = TrafficPolicy()
            self.assertEqual(self.call(p, snapshot=signs(1, park=True), detection=detection, base=base), 'STOP')

    def test_timeout_is_abort_not_successful_timed_parking(self):
        p = TrafficPolicy(park_timeout=3)
        self.call(p, snapshot=signs(1, park=True))
        self.assertEqual(self.call(p, 5, signs(5, seq=2, park=True)), 'STOP')
        self.assertEqual(p.aborted, 'TIMEOUT')
        self.assertFalse(p.parked)

    def test_auto_from_parked_line_returns_left_then_hands_back_to_v4(self):
        p = TrafficPolicy()
        for i in (1, 2, 3):
            self.call(p, i/10, signs(i/10, seq=i, park=True), lane(i/10, 1.))
        self.assertTrue(p.parked)
        self.assertTrue(p.needs_return)
        p.cancel_motion()
        p.request_auto(.4)
        result = self.call(p, .4, signs(.4, seq=4, park=True), lane(.4, 1.))
        self.assertTrue(p.returning)
        self.assertGreater(float(result.split(',')[2]), 0.)
        for i in (5, 6, 7):
            result = self.call(p, i/10, signs(i/10, seq=i, park=True), lane(i/10, .5))
        self.assertEqual(result, 'DRIVE,0.120,0.0000')
        self.assertFalse(p.needs_return)
        self.assertFalse(p.returning)

    def test_stop_interrupts_return_and_remains_held(self):
        p = TrafficPolicy()
        p.needs_return = True
        p.request_auto(1.)
        self.assertEqual(self.call(p, snapshot=signs(1, stop=True), detection=lane(1, 1.)), 'STOP')
        self.assertTrue(p.needs_return)
        self.assertTrue(p.stop_latched)

    def test_return_can_use_observed_right_line_when_xy_is_outside_training_view(self):
        p = TrafficPolicy()
        p.needs_return = True
        p.request_auto(1.)
        for i in (1, 2, 3):
            d = lane(1+i/10, 1.)
            d.update(valid=False, frame_valid=True)
            result = self.call(p, 1+i/10, signs(1+i/10, seq=i), d, base='STOP')
            if i < 3:
                self.assertEqual(result, 'STOP')
        self.assertTrue(result.startswith('DRIVE'))
        self.assertGreater(float(result.split(',')[2]), 0.)

    def test_lost_lane_guard_cannot_be_overridden_by_geometry_return(self):
        p = TrafficPolicy()
        p.needs_return = True
        p.request_auto(1.)
        for i in (1, 2, 3, 4):
            d = lane(1+i/10, 1.)
            d.update(valid=False, frame_valid=True, lane_guard_latched=True)
            self.assertEqual(self.call(p, 1+i/10, signs(1+i/10, seq=i), d, base='STOP'), 'STOP')


def corridor_mask(position=.5, single=False):
    mask = np.zeros((240, 320), np.uint8)
    for y in range(100, 220):
        width = 200+2*(y-175)
        left = round(160-position*width)
        right = round(left+width)
        if not single:
            cv2.line(mask, (left, y), (left, y), 255, 7)
        cv2.line(mask, (right, y), (right, y), 255, 7)
    return mask


class CorridorTests(unittest.TestCase):
    def test_pair_tracks_center_and_right_parking_position(self):
        for position in (.5, .7, .82):
            r = RightCorridorTracker().process(corridor_mask(position))
            self.assertIsNotNone(r)
            self.assertAlmostEqual(r['position'], position, delta=.01)
            self.assertLess(abs(r['heading']), .01)

    def test_single_ribbon_and_blank_floor_cannot_create_parking_corridor(self):
        for mask in [np.zeros((240, 320), np.uint8), corridor_mask(single=True)]:
            self.assertIsNone(RightCorridorTracker().process(mask))

    def test_old_pair_is_not_reused_when_right_line_disappears(self):
        t = RightCorridorTracker()
        self.assertIsNotNone(t.process(corridor_mask()))
        self.assertIsNone(t.process(np.zeros((240, 320), np.uint8)))

    def test_right_line_remains_observed_when_left_edge_leaves_frame(self):
        tracker = RightCorridorTracker()
        for ratio in (.5, .6, .7, .8, .9, 1.):
            r = tracker.process(corridor_mask(ratio))
            self.assertIsNotNone(r)
            self.assertAlmostEqual(r['position'], ratio, delta=.015)
        self.assertEqual(r['source'], 'RIGHT TRACK')
        self.assertAlmostEqual(r['right_gap_px'], 0., delta=1.)


class OcrTests(unittest.TestCase):
    def test_two_successes_same_object_required(self):
        image = np.zeros((480, 640, 3), np.uint8)
        box = dict(x=100, y=80, w=100, h=100)
        with patch.object(user_sign_ocr, 'find_stop_candidate', return_value=box), \
                patch.object(user_sign_ocr, 'run_stop_ocr', return_value=('STOP', None)):
            r = SignRecognizer()
            self.assertFalse(r.process(image, 1., 0, 1)['stop_confirmed'])
            self.assertTrue(r.process(image, 1.1, 0, 2)['stop_confirmed'])
            self.assertFalse(r.process(image, 1.2, 1, 3)['stop_confirmed'])

    def test_partial_stop_text_is_not_a_confirmed_stop(self):
        box = dict(x=100, y=80, w=100, h=100)
        with patch.object(user_sign_ocr, 'find_stop_candidate', return_value=box), \
                patch.object(user_sign_ocr, 'run_stop_ocr', return_value=('STO', None)):
            r = SignRecognizer()
            for i in range(3):
                self.assertFalse(r.process(np.zeros((480, 640, 3), np.uint8), 1+i/10, 0, i)['stop_confirmed'])

    def test_supplied_p_shape_accepts_p_not_plain_blue_square(self):
        image = np.zeros((161, 161, 3), np.uint8)
        image[:] = (255, 0, 0)
        self.assertFalse(user_sign_ocr.check_white_p_shape(image)[0])
        cv2.putText(image, 'P', (40, 125), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (255, 255, 255), 9)
        self.assertTrue(user_sign_ocr.check_white_p_shape(image)[0])

    def test_worker_keeps_only_latest_pending_frame(self):
        worker = SignWorker(threading.Event())
        a, b = np.zeros((2, 2, 3), np.uint8), np.ones((2, 2, 3), np.uint8)
        worker.submit(a, 1., 0)
        worker.submit(b, 2., 1)
        self.assertIs(worker.pending[0], b)
        self.assertEqual(worker.pending[1:3], (2., 1))

    def test_engine_timeout_is_reported_even_when_supplied_function_catches_it(self):
        image = np.zeros((480, 640, 3), np.uint8)
        box = dict(x=100, y=80, w=100, h=100)
        with patch.object(user_sign_ocr, 'find_stop_candidate', return_value=box), \
                patch.object(sign_ocr_engine, 'executable', return_value='mock_tesseract'), \
                patch.object(sign_ocr_engine.subprocess, 'run', side_effect=subprocess.TimeoutExpired('mock_tesseract', .6)), \
                patch('builtins.print'):
            result = SignRecognizer().process(image, 1., 0, 1)
        self.assertTrue(result['error'])
        self.assertFalse(result['stop_confirmed'])
        self.assertFalse(TrafficPolicy.fresh(result, 1., 0))


@unittest.skipUnless(importlib.util.find_spec('flask'), 'Optional local Flask verification dependencies missing')
class WebTests(unittest.TestCase):
    def setUp(self):
        self.car = Car(port='NO_SERIAL_OPENED')
        self.car.connected = True
        self.client = make_app(self.car).test_client()

    def test_auto_rejected_without_serial_handshake(self):
        self.car.connected = False
        self.assertEqual(self.client.post('/mode', json={'mode': 'AUTO'}).status_code, 409)

    def test_stop_cancels_auto_and_parking_immediately(self):
        self.client.post('/mode', json={'mode': 'AUTO'})
        self.car.traffic.park_started = 1.
        response = self.client.post('/control', json={'command': 'STOP'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.car.mode, 'MANUAL')
        self.assertEqual(self.car.manual, 'STOP')
        self.assertIsNone(self.car.traffic.park_started)

    def test_old_manual_epoch_cannot_drive_after_auto_or_stop(self):
        old = self.car.generation
        self.client.post('/mode', json={'mode': 'AUTO'})
        self.client.post('/control', json={'command': 'STOP'})
        self.assertEqual(self.client.post('/control', json={'command': 'F', 'epoch': old}).status_code, 409)

    def test_rearm_only_while_manual_stop_and_invalidates_old_ocr(self):
        self.client.post('/mode', json={'mode': 'AUTO'})
        self.assertEqual(self.client.post('/signs/rearm', json={}).status_code, 409)
        self.client.post('/control', json={'command': 'STOP'})
        old = self.car.generation
        self.car.traffic.handled = {'stop': True, 'park': True}
        self.assertEqual(self.client.post('/signs/rearm', json={}).status_code, 200)
        self.assertGreater(self.car.generation, old)
        self.assertFalse(any(self.car.traffic.handled.values()))

    def test_status_serializes_ocr_and_parking_info(self):
        response = self.client.get('/status')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['stop_seconds'], 0.)
        self.assertIn('signs', data)
        self.assertIn('parking', data)

    def test_auto_button_starts_return_from_parked(self):
        self.car.traffic.needs_return = True
        self.car.traffic.parked = True
        self.assertEqual(self.client.post('/mode', json={'mode': 'AUTO'}).status_code, 200)
        self.assertTrue(self.car.traffic.returning)

    def test_serial_reset_does_not_release_stop_sign_hold_or_parked_hold(self):
        self.car.mode = 'AUTO'
        self.car.traffic.stop_latched = True
        self.car.pause_connection()
        self.assertTrue(self.car.traffic.stop_latched)
        self.car.traffic.stop_latched = False
        self.car.traffic.parked = True
        self.car.traffic.needs_return = True
        self.car.pause_connection()
        self.assertTrue(self.car.traffic.parked)
        self.assertFalse(self.car.traffic.returning)


if __name__ == '__main__':
    unittest.main()
