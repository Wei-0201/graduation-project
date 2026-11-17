#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Jetson Nano lane follower using train_model.ipynb's ResNet18 target point.
ESP32 protocol: DRIVE,speed,normalized_turn / MANUAL,F|B,speed,steering / STOP.
Positive turn means LEFT. Observed lane geometry assists the model at bend exits.
"""
import argparse
import csv
import logging
import os
import threading
import time

import cv2
import numpy as np
import sign_ocr_engine
from traffic_sign_worker import SignWorker, draw_signs
from sign_parking_control import TrafficPolicy, RightCorridorTracker
from visual_exit_guard import VisualExitGuard
from lane_corridor_guard import LaneCorridorGuard, TARGET_ROW
from right_entry_guard import RightEntryGuard

# Image geometry also supports the model at visually confirmed bend exits.
WIDTH, HEIGHT = 320, 240
ROI_TOP, ROI_BOTTOM = 0.38, 0.92
TARGET_Y_RATIO = 0.58
SLICES = 32
MIN_LINE_WIDTH, MAX_LINE_WIDTH = 2, 30
BLACK_MAX = 125
MIN_LOCAL_CONTRAST = 14
FIT_TOLERANCE = 4.0
MIN_POINTS = 7
MIN_Y_SPAN = 28
EXTENSION_LIMIT = 60

FRAME_TIMEOUT = 0.45
MANUAL_TIMEOUT = 0.60
AUTO_CONFIRM_TIME = 0.30
AUTO_CONFIRM_FRAMES = 3
AUTO_RESTART_DELAY = 0.25
AUTO_SPEED = 0.12
VISUAL_WEIGHT = 0.0
MODEL_SMOOTHING = 0.70  # Preserve V4 response when turning demand increases.
UNWIND_SMOOTHING = 0.90  # Follow a smaller learned turn more promptly.
STEERING_X_MARGIN = 8.0
MANUAL_SPEED = 0.30
CONTROL_PERIOD = 0.02
STREAM_FPS = 12


def steering_from_x(x):
    """Clamp small regression overshoot at image edges; reject distant outputs."""
    if not np.isfinite(x) or not -STEERING_X_MARGIN <= x <= WIDTH-1+STEERING_X_MARGIN:
        return None
    x = float(np.clip(x,0,WIDTH-1))
    return x,(WIDTH/2.0-x)/(WIDTH/2.0)


def has_tape_evidence(mask):
    """Look for a connected dark ribbon; this does not choose a lane side."""
    previous = []
    for y in range(110, 219, 6):
        row = np.count_nonzero(mask[y-1:y+2], axis=0) >= 2
        edges = np.diff(np.r_[False, row, False].astype(np.int8))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        current = []
        for a,b in zip(starts,ends):
            width = b-a
            if not 5 <= width <= 40 or a <= 1 or b >= WIDTH-1:
                continue
            x = (a+b-1)/2.0
            lengths = [length for old,old_width,length in previous
                       if abs(x-old)<=45 and max(width,old_width)<=3*min(width,old_width)]
            length = 1+max(lengths, default=0)
            if length>=6 and y>=170:
                return True
            current.append((x,width,length))
        previous = current
    return False


def fit_line(points):
    """Centered least squares for x=a*y+b; avoid a general SVD for two parameters."""
    x, y = points[:, 0].astype(np.float64), points[:, 1].astype(np.float64)
    dx, dy = x - x.mean(), y - y.mean()
    denominator = float(np.dot(dy, dy))
    slope = float(np.dot(dx, dy) / denominator) if denominator > 0 else 0.0
    return np.array([slope, x.mean() - slope * y.mean()])


class LatestFrame:
    """One-slot mailbox: processing skips old frames instead of building a queue."""
    def __init__(self):
        self.condition = threading.Condition()
        self.frame = None
        self.timestamp = 0.0
        self.sequence = 0

    def publish(self, frame, timestamp):
        with self.condition:
            self.frame, self.timestamp = frame, timestamp
            self.sequence += 1
            self.condition.notify_all()

    def read_after(self, previous, timeout=0.1):
        with self.condition:
            self.condition.wait_for(lambda: self.sequence != previous, timeout=timeout)
            return self.sequence, self.frame, self.timestamp


class CameraReader:
    def __init__(self, camera, stop):
        self.camera, self.stop = camera, stop
        self.mailbox = LatestFrame()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        cap = None
        try:
            while not self.stop.is_set():
                if cap is None:
                    cap = cv2.VideoCapture(self.camera)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                started = time.monotonic()
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.mailbox.publish(None, started)
                    cap.release()
                    cap = None
                    self.stop.wait(0.25)
                else:
                    self.mailbox.publish(frame, started)
        finally:
            if cap is not None:
                cap.release()


def black_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    top, bottom = int(HEIGHT * ROI_TOP), int(HEIGHT * ROI_BOTTOM)
    roi = gray[top:bottom]
    threshold, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshold = float(np.clip(threshold, 25, BLACK_MAX))
    # Horizontal closing estimates nearby floor brightness without removing thin tape.
    background = cv2.morphologyEx(roi, cv2.MORPH_CLOSE,
                                 np.ones((1, 41), np.uint8))
    contrast = cv2.subtract(background, roi)
    # A distant tape can be brighter than near tape. ROI-wide Otsu alone loses it.
    local_dark = contrast.astype(np.float32) >= np.maximum(MIN_LOCAL_CONTRAST,
                                                           background.astype(np.float32) * 0.12)
    selected = ((roi <= threshold) | local_dark) & (contrast >= MIN_LOCAL_CONTRAST)
    mask = np.zeros_like(gray)
    mask[top:bottom] = selected.astype(np.uint8) * 255
    return mask, threshold


def row_candidates(mask):
    points = []
    for y in np.linspace(int(HEIGHT * ROI_TOP), int(HEIGHT * ROI_BOTTOM) - 1,
                         SLICES).astype(int):
        # Majority across three rows reduces isolated floor speckles.
        row = np.count_nonzero(mask[y - 1:y + 2], axis=0) >= 2
        edges = np.diff(np.r_[False, row, False].astype(np.int8))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        for start, end in zip(starts, ends):
            if MIN_LINE_WIDTH <= end - start <= MAX_LINE_WIDTH and start > 1 and end < WIDTH - 1:
                points.append(((start + end - 1) / 2.0, float(y)))
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def line_models(points, target_y):
    """Bounded deterministic RANSAC; score unique scan rows, not raw pixel count."""
    if len(points) < MIN_POINTS or len(points) > 350:
        return []
    x, y = points[:, 0], points[:, 1]
    a, b = np.triu_indices(len(points), 1)
    far = np.abs(y[b] - y[a]) >= MIN_Y_SPAN
    a, b = a[far], b[far]
    if not len(a):
        return []
    if len(a) > 600:
        take = np.random.RandomState(7).choice(len(a), 600, replace=False)
        a, b = a[take], b[take]
    slopes = (x[b] - x[a]) / (y[b] - y[a])
    intercepts = x[a] - slopes * y[a]
    good = np.abs(slopes) < 3.0
    slopes, intercepts = slopes[good], intercepts[good]
    residual = np.abs(x[None, :] - slopes[:, None] * y - intercepts[:, None])
    counts = np.sum(residual <= FIT_TOLERANCE, axis=1)
    models = []
    # Suppress duplicate hypotheses BEFORE refinement. Otherwise the strongest
    # boundary can occupy all 60 slots and hide a slightly shorter opposite line.
    hypotheses = np.column_stack((slopes * target_y + intercepts,
                                  slopes * (target_y + 40) + intercepts))
    remaining = np.argsort(-counts)
    selected = []
    while len(remaining) and len(selected) < 24:
        index = remaining[0]
        if counts[index] < MIN_POINTS:
            break
        selected.append(index)
        distinct = np.max(np.abs(hypotheses[remaining] - hypotheses[index]), axis=1) >= 8
        remaining = remaining[distinct]
    for index in selected:
        inliers = residual[index] <= FIT_TOLERANCE
        # One vote per row; choose the candidate closest to the proposed line.
        ids = np.flatnonzero(inliers)
        ids = ids[np.lexsort((residual[index, ids], y[ids]))]
        _, first = np.unique(y[ids], return_index=True)
        chosen = ids[first]
        if len(chosen) < MIN_POINTS:
            continue
        p = points[chosen]
        # A few furniture points spread over the frame are not a continuous tape.
        pieces = np.split(p, np.flatnonzero(np.diff(p[:, 1]) > 15) + 1)
        p = max(pieces, key=len)
        if len(p) < MIN_POINTS:
            continue
        coef = fit_line(p)
        errors = np.abs(p[:, 0] - np.polyval(coef, p[:, 1]))
        p = p[errors <= FIT_TOLERANCE]
        if len(p) < MIN_POINTS or np.ptp(p[:, 1]) < MIN_Y_SPAN:
            continue
        lo, hi = float(p[:, 1].min()), float(p[:, 1].max())
        expected_rows = (hi - lo) / ((ROI_BOTTOM - ROI_TOP) * HEIGHT / (SLICES - 1)) + 1
        if len(p) / expected_rows < 0.65:
            continue
        # Use a reference row only to deduplicate fitted tape lines.
        coef = fit_line(p)
        probes = np.polyval(coef, [target_y, target_y + 40])
        if any(np.max(np.abs(probes - m['probes'])) < 8 for m in models):
            continue
        rms = float(np.sqrt(np.mean((p[:, 0] - np.polyval(coef, p[:, 1])) ** 2)))
        models.append(dict(coef=coef, points=p, lo=lo, hi=hi, probes=probes,
                           score=len(p) + (hi - lo) / 20.0 - rms, rms=rms))
        if len(models) >= 10:
            break
    return models


def draw_lane_model(frame, model):
    """Solid = observed fit; dashed = automatic extrapolation (not extra evidence)."""
    lo, hi = int(model['lo']), int(model['hi'])
    for start, end, predicted in ((max(0, lo - EXTENSION_LIMIT), lo, True),
                                  (lo, hi, False),
                                  (hi, min(HEIGHT - 1, hi + EXTENSION_LIMIT), True)):
        step = 16 if predicted else max(1, end - start)
        for y in range(start, end, step):
            next_y = min(y + (8 if predicted else step), end)
            x1, x2 = np.polyval(model['coef'], [y, next_y])
            # OpenCV clips image boundaries; bound coordinates before integer conversion.
            x1, x2 = np.clip([x1, x2], -WIDTH * 2, WIDTH * 3).astype(int)
            cv2.line(frame, (int(x1), y), (int(x2), next_y),
                     (255, 80, 20) if predicted else (255, 200, 0), 1 if predicted else 2)
    for x, y in model['points']:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)


class LineExtractor:
    """Extract tape fits for the preview; no center or steering decision."""
    def process(self, frame, now):
        annotated = cv2.resize(frame, (WIDTH, HEIGHT))
        mask, threshold = black_mask(annotated)
        points = row_candidates(mask)
        models = line_models(points, int(HEIGHT * TARGET_Y_RATIO))
        for model in models:
            draw_lane_model(annotated, model)
        cv2.putText(annotated, 'LINES P:{} M:{} TH:{}'.format(
            len(points), len(models), int(threshold)), (5, HEIGHT - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        return annotated, mask, dict(timestamp=now, points=len(points),
                                     models=len(models))


class ModelLaneDetector:
    """Combine the trained target X with reliable observed lane geometry."""
    def __init__(self, checkpoint, show_lines=False, visual_easing=False):
        import torch
        from torchvision import models, transforms
        from PIL import Image

        self.torch = torch
        self.transforms = transforms
        self.Image = Image
        # Tape fitting is preview-only. Its RANSAC work can delay fresh model
        # frames on a Jetson Nano, so keep it optional during AUTO driving.
        self.lines = LineExtractor() if show_lines else None
        self.guard = VisualExitGuard()
        self.corridor = LaneCorridorGuard()
        self.visual_easing = visual_easing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(512, 2)
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        logging.info('Loaded model %s on %s', checkpoint, self.device)

    def reset(self):
        self.guard.reset()
        self.corridor.reset()

    def process(self, frame, now=None):
        now = time.monotonic() if now is None else now
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        if self.lines is None:
            annotated = cv2.resize(frame, (WIDTH, HEIGHT))
            mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            result = dict(timestamp=now)
        else:
            annotated, mask, result = self.lines.process(frame, now)

        # Match XYDataset: PIL RGB resize, tensor conversion, RGB->BGR, then
        # ImageNet mean/std applied in that BGR order. Do not silently swap it.
        image = self.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = self.transforms.functional.resize(image, (224, 224))
        image = self.transforms.functional.to_tensor(image)
        image = self.torch.from_numpy(image.numpy()[::-1].copy())
        image = self.transforms.functional.normalize(
            image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        with self.torch.no_grad():
            xy = self.model(image.unsqueeze(0).to(self.device))[0].cpu().numpy()
        x = float((xy[0] + 1.0) * WIDTH / 2.0)
        y = float((xy[1] + 1.0) * HEIGHT / 2.0)
        steering = steering_from_x(x)
        if steering is None:
            result.update(valid=False, offset=None, turn=None,
                          status='MODEL INVALID: steering X outside image')
            return annotated, mask, result
        x,turn = steering
        # AUTO steers from X only. Keep a bad Y prediction from hiding a valid
        # steering prediction; Y is used solely for the preview dot.
        if not np.isfinite(y):
            y = HEIGHT * TARGET_Y_RATIO
        y = float(np.clip(y, 0, HEIGHT - 1))
        # The model predicts a normalized image x. A point in the valid image
        # already yields a turn between -1 and +1; add no software turn cap.
        tape_mask = mask if self.lines is not None else black_mask(annotated)[0]
        lane_visible = has_tape_evidence(tape_mask)
        mask = tape_mask
        curve_target = False
        geometry = dict(visual_turn=None,visual_source='NONE',visual_weight=0.,visual_center=None)
        if self.visual_easing or self.lines is not None:
            fallback = self.guard.process(tape_mask)
            geometry = self.corridor.process(tape_mask)
            curve_target = geometry['visual_turn'] is not None
            if not curve_target:
                geometry = fallback
        for line in geometry.pop('lines', []):
            lo, hi = int(line['lo']), int(line['hi'])
            xs = np.clip(np.polyval(line['coef'], [lo, hi]), 0, WIDTH-1).astype(int)
            cv2.line(annotated, (int(xs[0]), lo), (int(xs[1]), hi), (0, 255, 0), 2)
        result.update(geometry)
        result.update(valid=True, offset=x - WIDTH / 2.0, turn=turn,
                      model_x=x, model_y=y, lane_visible=lane_visible,
                      status='MODEL X:{:.1f} Y:{:.1f} TURN:{:+.3f}'.format(x, y, turn))
        if geometry['visual_turn'] is not None:
            gx = int(np.clip(geometry['visual_center'], 0, WIDTH-1))
            cv2.circle(annotated, (gx, TARGET_ROW if curve_target else 140),
                       5, (255, 255, 0), 2)
            for px,py in geometry.get('visual_points', []):
                cv2.circle(annotated, (int(px),int(py)),3,(0,255,0),-1)
            visual_label = 'VISUAL {} TURN:{:+.3f}'.format(
                geometry['visual_source'], geometry['visual_turn'])
            cv2.putText(annotated, visual_label, (5, 63), cv2.FONT_HERSHEY_SIMPLEX,
                        .35, (0, 255, 0), 1)
            result['status'] += ' | ' + visual_label
        else:
            cv2.putText(annotated, 'TAPE VISIBLE' if lane_visible else 'NO VISIBLE TAPE',
                        (5,63), cv2.FONT_HERSHEY_SIMPLEX, .35,
                        (0,255,0) if lane_visible else (0,0,255), 1)
        cv2.circle(annotated, (int(x), int(y)), 7, (255, 0, 255), 2)
        cv2.putText(annotated, result['status'], (5, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 0, 255), 1)
        return annotated, mask, result


class CommandPolicy:
    """Fresh model steering assisted by observed lane geometry."""
    def __init__(self, auto_speed=AUTO_SPEED, visual_weight=VISUAL_WEIGHT):
        self.auto_speed = auto_speed
        self.visual_weight = visual_weight
        self.reset()

    def reset(self):
        self.good_frames = 0
        self.first_stamp = None
        self.last_stamp = None
        self.running = False
        self.resume_after = None
        self.control_state = 'STOP'
        self.steering_stamp = None
        self.previous_raw_turn = None
        self.command_turn = 0.0
        self.exit_active = False
        self.unwind_active = False
        self.visual_frames = 0
        self.review_required = False
        self.unobserved_frames = 0
        self.right_entry = RightEntryGuard()

    def steering_for_frame(self, detection):
        """Apply new image evidence once; gradually approach its steering target."""
        stamp = detection['timestamp']
        if stamp == self.steering_stamp:
            return self.command_turn
        raw = float(detection['turn'])
        adjusted = raw
        self.exit_active = False
        visual = detection.get('visual_turn')
        observed_pair = detection.get('visual_source') in ('CURVE PAIR', 'PAIR')
        weight = (min(.20, self.visual_weight, float(detection.get('visual_weight') or 0.0))
                  if observed_pair else 0.0)
        if visual is not None and np.isfinite(visual) and weight > 0:
            self.visual_frames += 1
        else:
            self.visual_frames = 0
        if self.visual_frames >= 2:
            # Assist may ease an already-correct turn, never create a stronger
            # or opposite turn from a guessed border/width.
            if raw*float(visual) >= 0 and abs(float(visual)) < abs(raw):
                target = raw + float(np.clip(weight*(float(visual)-raw),-.08,.08))
                self.exit_active = True
            else:
                target = raw
        else:
            target = raw
        if os.environ.get('CAR_RIGHT_ENTRY_GUARD', '1') != '0':
            target = self.right_entry.adjust(target, detection)
        if self.previous_raw_turn is not None:
            self.unwind_active = abs(target) < abs(self.command_turn)
            smoothing = UNWIND_SMOOTHING if self.unwind_active else MODEL_SMOOTHING
            adjusted = (1.0-smoothing)*self.command_turn + smoothing*target
        else:
            self.unwind_active = False
            adjusted = target
        self.steering_stamp = stamp
        self.previous_raw_turn = raw
        self.command_turn = adjusted
        return adjusted

    def command(self, mode, manual, manual_time, detection, now):
        if mode != 'AUTO':
            self.reset()
            if manual == 'STOP' or now - manual_time > MANUAL_TIMEOUT:
                return 'STOP'
            direction = 'B' if manual == 'B' else 'F'
            steering = {'L': 35, 'R': -35}.get(manual, 0)
            return 'MANUAL,{},{:.3f},{}'.format(direction, MANUAL_SPEED, steering)
        if self.review_required:
            self.control_state = 'STOP: NO VISIBLE TAPE'
            return 'STOP'
        if not detection or now - detection['timestamp'] > FRAME_TIMEOUT or not detection['valid']:
            was_running = self.running
            resume_after = self.resume_after
            self.reset()
            if was_running:
                self.resume_after = now + AUTO_RESTART_DELAY
            else:
                self.resume_after = resume_after
            self.control_state = 'WAIT FRESH MODEL'
            return 'STOP'
        if self.resume_after is not None:
            if now < self.resume_after:
                self.control_state = 'AUTO WAIT'
                return 'STOP'
            self.resume_after = None
        stamp = detection['timestamp']
        if stamp != self.last_stamp:
            observed = detection.get('lane_visible', True)
            self.unobserved_frames = 0 if observed else self.unobserved_frames+1
            if self.unobserved_frames >= 5:
                self.review_required = True
                self.control_state = 'STOP: NO VISIBLE TAPE'
                return 'STOP'
            if self.last_stamp is None or stamp - self.last_stamp > FRAME_TIMEOUT:
                self.good_frames = 0
                self.first_stamp = stamp
            self.good_frames += 1
            self.last_stamp = stamp
        if (self.good_frames < AUTO_CONFIRM_FRAMES or
                stamp - self.first_stamp < AUTO_CONFIRM_TIME):
            self.control_state = 'CONFIRM MODEL'
            return 'STOP'
        self.running = True
        command_turn = self.steering_for_frame(detection)
        self.control_state = ('MODEL RIGHT ENTRY EASE' if self.right_entry.active else
                              'MODEL EASE TURN' if self.unwind_active else
                              'MODEL VISUAL ' + detection.get('visual_source', '')
                              if self.exit_active else 'MODEL DRIVE')
        return 'DRIVE,{:.3f},{:.4f}'.format(self.auto_speed, command_turn)


class Car:
    def __init__(self, port=None, model_path=None, auto_speed=AUTO_SPEED,
                 visual_weight=VISUAL_WEIGHT, show_lines=False, stop_seconds=0., park_ratio=1., park_timeout=15.):
        self.port = port
        self.model_path = model_path
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.mode, self.manual = 'MANUAL', 'STOP'
        self.manual_time = 0.0
        self.detection = None
        self.policy = CommandPolicy(auto_speed, visual_weight)
        self.show_lines = show_lines
        self.connected = False
        self.command = 'STOP'
        self.generation = 0
        self.video_condition = threading.Condition()
        self.jpeg, self.mask_jpeg, self.sequence = None, None, 0
        self.raw_video = LatestFrame()
        self.metrics = {}
        self.esp_status = ''
        self.sign_worker = SignWorker(self.stop)
        self.traffic = TrafficPolicy(stop_seconds, park_ratio, park_timeout)

    def video_worker(self):
        previous = 0
        while not self.stop.is_set():
            cycle = time.monotonic()
            seq, payload, _ = self.raw_video.read_after(previous)
            if seq == previous or payload is None:
                continue
            previous = seq
            annotated, mask = payload
            ok1, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            ok2, debug = cv2.imencode('.jpg', mask)
            if ok1 and ok2:
                with self.video_condition:
                    self.jpeg, self.mask_jpeg = jpeg.tobytes(), debug.tobytes()
                    self.sequence += 1
                    self.video_condition.notify_all()
            self.stop.wait(max(0, 1.0 / STREAM_FPS - (time.monotonic() - cycle)))

    def disarm(self):
        # Caller holds lock. Discard any commands queued before a connection change.
        self.mode, self.manual = 'MANUAL', 'STOP'
        self.generation += 1
        self.detection = None
        self.policy.reset()
        self.traffic.cancel_motion()

    def receive_esp_line(self, text):
        with self.lock:
            if text.startswith('STATUS'):
                self.esp_status = text
            if text == 'READY CENTERING V10':
                self.pause_connection()
                self.connected = True
            elif text.startswith(('READY', 'BOOT', 'EMERGENCY')):
                self.pause_connection()
                self.connected = False

    def pause_connection(self):
        mode = self.mode
        hold, until = self.traffic.stop_latched, self.traffic.stop_until
        parked, aborted = self.traffic.parked, self.traffic.aborted
        self.disarm()
        self.mode = mode
        if mode == 'AUTO':
            self.traffic.stop_latched, self.traffic.stop_until = hold, until
            self.traffic.parked, self.traffic.aborted = parked, aborted
            if self.traffic.needs_return and not (hold or parked or aborted):
                self.traffic.request_auto(time.monotonic())
            self.policy.resume_after = time.monotonic() + AUTO_RESTART_DELAY

    def motor_worker(self):
        import serial
        from serial.tools import list_ports
        connection = None
        next_connect = 0.0
        next_hello = 0.0
        receive = b''
        try:
            while not self.stop.is_set():
                cycle = time.monotonic()
                now = time.monotonic()
                if connection is None and now >= next_connect:
                    next_connect = now + 1.0
                    ports = [self.port] if self.port else [p.device for p in list_ports.comports()
                             if any(s in (p.device + ' ' + (p.description or '')).lower()
                                    for s in ('ttyusb', 'ttyacm', 'cp210', 'ch340', 'ch341'))]
                    if len(ports) != 1:
                        logging.warning('Specify --port: detected %s', ports)
                    else:
                        try:
                            connection = serial.Serial(ports[0], 115200, timeout=0, write_timeout=0.1)
                            self.stop.wait(1.0)  # ESP32 may reset when serial opens
                            connection.reset_input_buffer()
                            with self.lock:
                                self.pause_connection()
                                self.connected = False
                            connection.write(b'CTRL,0.000,999\nSTOP\nHELLO\n')
                            next_hello = time.monotonic() + 0.5
                            logging.info('Serial open; waiting for ESP32 compatible handshake.')
                        except (serial.SerialException, OSError):
                            if connection is not None:
                                connection.close()
                            connection = None
                if connection is not None:
                    try:
                        # Serialize mode changes and motor writes, preventing old AUTO after STOP.
                        with self.lock:
                            base_command = (self.policy.command(self.mode, self.manual, self.manual_time,
                                                                self.detection, time.monotonic())
                                            if self.connected else 'STOP')
                            motion_detection = dict(self.detection) if self.detection else None
                            if motion_detection is not None:
                                motion_detection['lane_guard_latched'] = self.policy.review_required
                            self.command = (self.traffic.command(self.mode, base_command, motion_detection,
                                                               self.sign_worker.snapshot(), time.monotonic(), self.generation)
                                            if self.connected else 'STOP')
                            connection.write((self.command + '\n').encode('ascii'))
                            if not self.connected and time.monotonic() >= next_hello:
                                connection.write(b'HELLO\n')
                                next_hello = time.monotonic() + 0.5
                        receive += connection.read(min(connection.in_waiting, 1024))
                        lines = receive.split(b'\n')
                        receive = lines.pop()[-1024:]
                        for line in lines:
                            text = line.decode('ascii', errors='replace').strip()
                            self.receive_esp_line(text)
                            logging.info('ESP32 %s', text)
                    except (serial.SerialException, OSError):
                        connection.close()
                        connection = None
                        receive = b''
                        with self.lock:
                            self.connected = False
                            self.pause_connection()
                self.stop.wait(max(0.0, CONTROL_PERIOD - (time.monotonic() - cycle)))
        finally:
            if connection is not None:
                try:
                    connection.write(b'STOP\n')
                except (serial.SerialException, OSError):
                    pass
                connection.close()

    def camera_loop(self, camera, display, record=None):
        detector = ModelLaneDetector(self.model_path, self.show_lines,
                                     visual_easing=self.policy.visual_weight>0)
        reader = CameraReader(camera, self.stop)
        self.sign_worker.thread.start()
        corridor_tracker = RightCorridorTracker()
        tracker_generation = None
        reader.thread.start()
        encoder = threading.Thread(target=self.video_worker, daemon=True)
        encoder.start()
        recorder = None
        record_log = None
        record_rows = None
        record_index = 0
        previous_sequence = 0
        last_processed = None
        fps_filtered = 0.0
        last_metric_log = 0.0
        try:
            while not self.stop.is_set():
                with self.lock:
                    generation = self.generation
                seq, frame, started = reader.mailbox.read_after(previous_sequence)
                if seq == previous_sequence:
                    continue
                previous_sequence = seq
                if frame is None:
                    with self.lock:
                        self.detection = None
                    detector.reset()
                    continue
                self.sign_worker.submit(frame, started, generation)
                if generation != tracker_generation:
                    if not self.traffic.needs_return:
                        corridor_tracker.reset()
                    tracker_generation = generation
                if record:
                    if recorder is None:
                        recorder = cv2.VideoWriter(record, cv2.VideoWriter_fourcc(*'MJPG'),
                                                   30.0, (640, 480))
                        if not recorder.isOpened():
                            raise IOError('Cannot open raw recording: ' + record)
                        record_log = open(record + '.csv', 'w', newline='')
                        record_rows = csv.writer(record_log)
                        record_rows.writerow(['frame', 'capture_mono', 'result_mono',
                            'model_turn', 'visual_turn', 'visual_source',
                            'sent_command_at_result', 'state_at_result', 'age_ms', 'lane_visible',
                            'unwind_at_command_snapshot', 'traffic_state', 'stop_text', 'p_text',
                            'stop_confirmed', 'p_confirmed', 'ocr_age_ms', 'ocr_error',
                            'parking_position', 'parking_heading', 'parking_target_ratio'])
                    recorder.write(cv2.resize(frame, (640, 480)))
                processing_start = time.monotonic()
                annotated, mask, result = detector.process(frame, started)
                parking_mask = mask if result.get('valid') else black_mask(cv2.resize(frame, (WIDTH, HEIGHT)))[0]
                result['frame_valid'] = True
                result['parking_corridor'] = corridor_tracker.process(parking_mask)
                finished = time.monotonic()
                processing_ms = (finished - processing_start) * 1000
                age_ms = (finished - started) * 1000
                if last_processed is not None:
                    instantaneous = 1.0 / max(finished - last_processed, 0.001)
                    fps_filtered = instantaneous if fps_filtered == 0 else 0.8 * fps_filtered + 0.2 * instantaneous
                last_processed = finished
                with self.lock:
                    self.metrics = dict(process_fps=round(fps_filtered, 1),
                                        processing_ms=round(processing_ms, 1), age_ms=round(age_ms, 1))
                    if generation == self.generation:
                        self.detection = result
                    else:
                        detector.reset()
                    label = '{} {} {}'.format(self.mode, self.command, self.traffic.state)
                    sign_snapshot = self.sign_worker.snapshot()
                    traffic_snapshot = self.traffic.state
                    command_snapshot = self.command
                    state_snapshot = self.policy.control_state
                    unwind_snapshot = self.policy.unwind_active and command_snapshot.startswith('DRIVE,')
                if record_rows is not None:
                    record_rows.writerow([record_index, started, finished,
                        result.get('turn'), result.get('visual_turn'),
                        result.get('visual_source'), command_snapshot, state_snapshot, age_ms,
                        result.get('lane_visible'), unwind_snapshot, traffic_snapshot,
                        (sign_snapshot or {}).get('stop_text'), (sign_snapshot or {}).get('park_text'),
                        (sign_snapshot or {}).get('stop_confirmed'), (sign_snapshot or {}).get('park_confirmed'),
                        (finished-sign_snapshot['timestamp'])*1000 if sign_snapshot else None,
                        (sign_snapshot or {}).get('error'), (result.get('parking_corridor') or {}).get('position'),
                        (result.get('parking_corridor') or {}).get('heading'),
                        .5 if traffic_snapshot.startswith('RETURN') else self.traffic.target_ratio])
                    record_index += 1
                cv2.putText(annotated, label, (5, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, (0, 255, 255), 1)
                metric_label = 'FPS:{:.1f} PROC:{:.0f}ms AGE:{:.0f}ms'.format(fps_filtered, processing_ms, age_ms)
                cv2.putText(annotated, metric_label, (5, 49), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (255, 255, 255), 1)
                if sign_snapshot and sign_snapshot.get('generation') == generation:
                    draw_signs(annotated, sign_snapshot)
                corridor = result.get('parking_corridor')
                if corridor and traffic_snapshot.startswith(('PARK', 'RETURN')):
                    cv2.line(annotated, (int(corridor['far_right']),145), (int(corridor['right']),175), (0,255,255), 2)
                    active_ratio = .5 if traffic_snapshot.startswith('RETURN') else self.traffic.target_ratio
                    target_x = corridor['left'] + active_ratio*corridor['width']
                    cv2.circle(annotated,(int(np.clip(target_x,0,319)),175),5,(0,255,0),2)
                self.raw_video.publish((annotated, mask), finished)
                if finished - last_metric_log >= 1:
                    logging.info('[VISION] %s lane=%s offset=%s command=%s state=%s',
                                 metric_label, result['status'], result.get('offset'),
                                 self.command, self.policy.control_state)
                    last_metric_log = finished
                if display:
                    cv2.imshow('Lane Detection', annotated)
                    cv2.imshow('Black Mask', mask)
                    if cv2.waitKey(1) & 255 == ord('q'):
                        self.stop.set()
        finally:
            with self.lock:
                self.detection = None
            self.stop.set()
            reader.thread.join(timeout=1)
            encoder.join(timeout=1)
            self.sign_worker.thread.join(timeout=2)
            if recorder is not None:
                recorder.release()
            if record_log is not None:
                record_log.close()
            if display:
                cv2.destroyAllWindows()


HTML = '''<!doctype html><html lang="zh-Hant"><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Jetson Car</title>
<style>body{background:#111;color:white;text-align:center;font-family:Arial}img{width:95%;max-width:640px}
button{font-size:20px;padding:20px;margin:6px;border-radius:12px;touch-action:none;user-select:none}
.stop{background:#ff7766}a{color:#6cf}</style>
<h2>Jetson Car · V4 + STOP / P OCR TEST - Right Entry</h2><img src="/video_feed"><p id="status">Connecting...</p><p id="telemetry"></p>
<button onclick="mode('AUTO')">AUTO</button><button onclick="mode('MANUAL')">MANUAL</button><br>
<button data-cmd="F">前進</button><br><button data-cmd="L">左轉</button>
<button class="stop" onclick="halt()">STOP</button><button data-cmd="R">右轉</button><br>
<button data-cmd="B">後退</button><br>
<button onclick="rearm()">重新辨識路牌（先 STOP）</button>
<p id="stop-note">STOP：依設定暫停後繼續；同一面牌不重複觸發。P：停在右線上，AUTO 先回中央再跟隨。</p>
<script>
let held='STOP', busy=false, epoch=0;
async function post(path,data){return fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});}
async function manual(){if(busy||held==='STOP')return;busy=true;try{await post('/control',{command:held,epoch});}finally{busy=false;}}
async function refresh(){try{let s=await(await fetch('/status')).json();epoch=s.epoch;
document.getElementById('status').textContent=s.mode+' | '+s.state+' | '+s.lane;
document.getElementById('telemetry').textContent=s.command+' | MODEL TURN '+s.turn+' | APPLIED TURN '+s.applied_turn+' | TAPE '+s.lane_visible+' | X '+s.model_x+' Y '+s.model_y+' | FPS '+(s.metrics.process_fps||0)+' | AGE '+(s.metrics.age_ms||0)+'ms | '+(s.connected?s.esp_status:'ESP32 未連線');}catch(e){}}
function halt(){held='STOP';post('/control',{command:'STOP'}).then(refresh);}
async function rearm(){held='STOP';let r=await post('/signs/rearm',{});if(!r.ok)alert('請先按 STOP 再重新辨識');await refresh();}
async function mode(m){held='STOP';await post('/mode',{mode:m});await refresh();}
document.querySelectorAll('[data-cmd]').forEach(b=>{b.onpointerdown=e=>{e.preventDefault();b.setPointerCapture(e.pointerId);held=b.dataset.cmd;manual();};
b.onpointerup=halt;b.onpointercancel=halt;b.onlostpointercapture=()=>{if(held!=='STOP')halt();};});
setInterval(manual,150);setInterval(refresh,250);refresh();
window.addEventListener('blur',()=>{if(held!=='STOP')halt();});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&held!=='STOP')halt();});
</script></html>'''


def make_app(car):
    from flask import Flask, Response, jsonify, request
    app = Flask(__name__)

    @app.route('/')
    def index():
        return HTML

    @app.route('/status')
    def status():
        with car.lock:
            fresh = car.detection and time.monotonic() - car.detection['timestamp'] <= FRAME_TIMEOUT
            return jsonify(mode=car.mode, connected=car.connected, command=car.command,
                           epoch=car.generation,
                            version='v4-right-entry-20261028',
                            state=(car.policy.control_state if car.traffic.state == 'V4 FOLLOW' else car.traffic.state),
                            lane_control_state=car.policy.control_state, esp_status=car.esp_status,
                            signs=car.sign_worker.snapshot(), parking=(car.detection or {}).get('parking_corridor'),
                            stop_seconds=car.traffic.stop_seconds, parking_target=car.traffic.target_ratio,
                            needs_return=car.traffic.needs_return,
                           offset=car.detection.get('offset') if fresh else None,
                            turn=car.detection.get('turn') if fresh else None,
                            visual_turn=car.detection.get('visual_turn') if fresh else None,
                            visual_source=car.detection.get('visual_source') if fresh else None,
                            lane_visible=car.detection.get('lane_visible') if fresh else None,
                            unwind_active=(car.policy.unwind_active
                                           if fresh and car.command.startswith('DRIVE,') else False),
                            applied_turn=(car.policy.command_turn
                                          if fresh and car.command.startswith('DRIVE,') else None),
                            model_x=car.detection.get('model_x') if fresh else None,
                            model_y=car.detection.get('model_y') if fresh else None,
                           metrics=car.metrics,
                           lane=car.detection['status'] if fresh else 'NO FRESH FRAME')

    @app.route('/mode', methods=['POST'])
    def mode():
        data = request.get_json(silent=True)
        value = data.get('mode') if isinstance(data, dict) else None
        if value not in ('AUTO', 'MANUAL'):
            return jsonify(ok=False), 400
        with car.lock:
            if value == 'AUTO' and not car.connected:
                return jsonify(ok=False, error='ESP32 not connected'), 409
            car.disarm()
            car.mode = value
            if value == 'AUTO':
                car.traffic.request_auto(time.monotonic())
        return jsonify(ok=True)

    @app.route('/signs/rearm', methods=['POST'])
    def rearm_signs():
        with car.lock:
            if car.mode != 'MANUAL' or car.command != 'STOP':
                return jsonify(ok=False, error='Press STOP first'), 409
            car.disarm()
            car.traffic.handled = {'stop': False, 'park': False}
            car.traffic.clear_counts = {'stop': 0, 'park': 0}
            car.traffic.last_ocr_sequence = None
        return jsonify(ok=True)

    @app.route('/control', methods=['POST'])
    def control():
        data = request.get_json(silent=True)
        value = data.get('command') if isinstance(data, dict) else None
        if value not in ('F', 'B', 'L', 'R', 'STOP'):
            return jsonify(ok=False), 400
        with car.lock:
            if value == 'STOP':
                car.disarm()  # STOP works in AUTO, even when no manual key was held.
            else:
                if car.mode != 'MANUAL' or data.get('epoch') != car.generation or not car.connected:
                    return jsonify(ok=False), 409
                car.manual, car.manual_time = value, time.monotonic()
        return jsonify(ok=True)

    def frames(mask=False):
        previous = -1
        while not car.stop.is_set():
            with car.video_condition:
                car.video_condition.wait_for(lambda: car.sequence != previous or car.stop.is_set(), timeout=1)
                if car.sequence == previous:
                    continue
                previous = car.sequence
                jpeg = car.mask_jpeg if mask else car.jpeg
            if jpeg is not None:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'

    @app.route('/video_feed')
    def video():
        return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/mask_feed')
    def mask():
        return Response(frames(True), mimetype='multipart/x-mixed-replace; boundary=frame')
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stop-seconds', type=float, default=0., help='STOP pause seconds; default 0 holds until AUTO')
    parser.add_argument('--park-target', type=float, default=1., help='Desired optical-axis position across lane, left=0/right=1')
    parser.add_argument('--park-timeout', type=float, default=15., help='Abort timeout; never timed steering')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--port', help='e.g. /dev/ttyUSB0; required if multiple serial devices exist')
    parser.add_argument('--model', default='best_steering_model_xy_v4_ocr_20261027.pth',
                        help='Weights saved by train_model.ipynb')
    parser.add_argument('--auto-speed', type=float, default=AUTO_SPEED,
                        help='AUTO speed from 0.12 to 0.24 (default: 0.12)')
    parser.add_argument('--visual-weight', type=float, default=VISUAL_WEIGHT,
                        help='Maximum observed-lane easing contribution, 0 (off) to 0.20')
    parser.add_argument('--show-lines', action='store_true',
                        help='Enable CPU-intensive tape fitting for preview only')
    parser.add_argument('--display', action='store_true', help='Enable local OpenCV windows')
    parser.add_argument('--image', help='Offline raw-camera image; does not open serial or web server')
    parser.add_argument('--video', help='Offline raw-camera video; does not open serial or web server')
    parser.add_argument('--record', help='Record unannotated camera frames to an .avi file')
    parser.add_argument('--log', help='Diagnostic log path (default: timestamped log beside script)')
    parser.add_argument('--output', default='lane_debug.jpg')
    args = parser.parse_args()
    if not 0 <= args.stop_seconds <= 30 or not .65 <= args.park_target <= 1. or not 3 <= args.park_timeout <= 60:
        parser.error("Invalid STOP/parking settings")
    if not 0.12 <= args.auto_speed <= 0.24:
        parser.error('--auto-speed must be between 0.12 and 0.24')
    if not 0.0 <= args.visual_weight <= 0.20:
        parser.error('--visual-weight must be between 0 and 0.20')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    cv2.setNumThreads(2)
    model_path = os.path.abspath(args.model if os.path.isabs(args.model) else
                                 os.path.join(os.path.dirname(__file__), args.model))
    if not os.path.isfile(model_path):
        parser.error('Model weights not found: ' + model_path)
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            parser.error('Cannot read image')
        annotated, mask, result = ModelLaneDetector(
            model_path,args.show_lines,args.visual_weight>0).process(frame)
        if not cv2.imwrite(args.output, annotated) or not cv2.imwrite(args.output + '.mask.png', mask):
            raise IOError('Cannot write output')
        print(result)
        return
    if args.video:
        detector = ModelLaneDetector(model_path,args.show_lines,args.visual_weight>0)
        source = cv2.VideoCapture(args.video)
        if not source.isOpened():
            parser.error('Cannot read video')
        fps = source.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and 1 <= fps <= 120 else 30.0
        destination = args.output if args.output.lower().endswith('.avi') else args.output + '.avi'
        writer = cv2.VideoWriter(destination, cv2.VideoWriter_fourcc(*'MJPG'), fps, (WIDTH, HEIGHT))
        if not writer.isOpened():
            source.release()
            raise IOError('Cannot write video')
        count, valid = 0, 0
        try:
            while True:
                ok, frame = source.read()
                if not ok:
                    break
                annotated, _, result = detector.process(frame, count / fps)
                writer.write(annotated)
                count += 1
                valid += int(result['valid'])
        finally:
            source.release()
            writer.release()
        print('Processed {} frames, {} valid; {}'.format(count, valid, destination))
        return
    log_path = args.log or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        'car_model_' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    try:
        handler = logging.FileHandler(log_path, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logging.getLogger().addHandler(handler)
        logging.info('Jetson model diagnostic log: %s', log_path)
    except OSError as exc:
        logging.warning('Cannot create diagnostic log: %s', exc)
    try:
        logging.info('OCR preflight: %s', sign_ocr_engine.check_available())
    except Exception as exc:
        parser.error('OCR unavailable: ' + str(exc))
    car = Car(args.port, model_path, args.auto_speed, args.visual_weight, args.show_lines,
              args.stop_seconds, args.park_target, args.park_timeout)
    app = make_app(car)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    motor = threading.Thread(target=car.motor_worker, daemon=True)
    motor.start()
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True,
                     debug=False, use_reloader=False), daemon=True).start()
    try:
        car.camera_loop(args.camera, args.display, args.record)
    except KeyboardInterrupt:
        pass
    finally:
        car.stop.set()
        with car.video_condition:
            car.video_condition.notify_all()
        motor.join(timeout=2.0)


if __name__ == '__main__':
    main()
