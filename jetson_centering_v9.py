#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Jetson Nano black-tape lane follower. Python 3.6+, OpenCV 4.x.
ESP32 v9 protocol: DRIVE,speed,normalized_turn / MANUAL,F|B,speed,steering / STOP.
Positive turn means LEFT. Pair with ESP32 CENTERING V9 (1% compensation cap).
"""
import argparse
import itertools
import logging
import os
import threading
import time

import cv2
import numpy as np

# All pixel settings below refer to the resized 320 x 240 image.
WIDTH, HEIGHT = 320, 240
ROI_TOP, ROI_BOTTOM = 0.38, 0.92
TARGET_Y_RATIO = 0.58
SLICES = 32
MIN_LINE_WIDTH, MAX_LINE_WIDTH = 2, 30
BLACK_MAX = 125                 # Upper bound for ROI-only Otsu threshold
MIN_LOCAL_CONTRAST = 14         # Reject broad shadows and uniform dark regions
FIT_TOLERANCE = 4.0
MIN_POINTS = 7
MIN_Y_SPAN = 28                 # Short visible line near image edge still needs real support
MIN_SHARED_SPAN = 24
EXTENSION_LIMIT = 60            # User-requested automatic visual extension
MIN_LANE_WIDTH, MAX_LANE_WIDTH = 45, 300
TRACK_MAX_JUMP = 24             # Per-boundary jump at target row
TRACK_EXPIRY = 0.65
RECOVERY_TIMEOUT = 0.60
RECOVERY_SPEED = 0.12
SINGLE_LINE_MAX_SHIFT = 36
MAX_RECOVERY_OFFSET = 90
ACQUIRE_FRAMES = 3
MIN_NEAR_SUPPORT_Y = 0.50
AUTO_START_CONFIRM_TIME = 0.40
ACQUIRE_MAX_GAP = 0.20          # Confirmation requires a continuous stream of frames.
ACQUIRE_MAX_DRIFT = 10.0        # Pixel spread over the entire confirmation window.
ACCEL_FRAME_AGE = 0.18          # Old images may not authorize starting/accelerating.
START_MAX_CENTER_SLOPE = 0.40   # Reject visibly misaligned straight-lane starts.
FRAME_TIMEOUT = 0.30
MANUAL_TIMEOUT = 0.60
AUTO_MAX_SPEED, AUTO_MID_SPEED, AUTO_TURN_SPEED = 0.24, 0.21, 0.18
AUTO_START_SPEED = 0.12
AUTO_START_HOLD = 1.0
STARTUP_STRAIGHT_TIME = 0.70    # Low-speed straight driving before lane-turn correction.
AUTO_STABLE_TIME = 0.8
AUTO_ACCELERATION = 0.03
STABLE_OFFSET = 12
STABLE_OFFSET_RATE = 25
MANUAL_SPEED = 0.30
CENTER_RETURN_BAND = 12         # Stop lane-turn correction inside +/-12 pixels.
CENTER_CORRECTION_TRIGGER = 20  # Start correction only at +/-20 pixels or farther.
DEAD_ZONE = CENTER_RETURN_BAND
FULL_CORRECTION_OFFSET = 45     # Pixels at resized 320x240: maximum compensation here.
TURN_RISE_RATE = 2.0            # Normalized compensation/second when increasing.
OFFSET_KP = 1.0 / (FULL_CORRECTION_OFFSET - CENTER_RETURN_BAND)
OFFSET_KD = 0.025               # Signed damping; may countersteer BEFORE crossing center.
HEADING_KP = 0.60               # Image centerline slope; not a calibrated physical angle.
SETTLED_OFFSET_RATE = 2.0
SETTLED_CENTER_SLOPE = 0.08
PREVIEW_ROWS = 24               # Image-row spacing, not metres; require measured support.
CONTROL_PERIOD = 0.02
STREAM_FPS = 12                # Preview encoding should not compete with perception.


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
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
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
        # Choose the control row only AFTER pairing. A valid line may leave the
        # image before the preferred row and must not be discarded here.
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


def draw_target_path(frame, coef, lo, hi, predicted=False):
    """Green measured centerline; orange dashed remembered/inferred target path."""
    waypoints = []
    start, end = max(0, int(lo)), min(HEIGHT - 1, int(hi))
    for y in range(start, end, 6):
        next_y = min(y + 6, end)
        x1, x2 = np.polyval(coef, [y, next_y])
        if 0 <= x1 < WIDTH and 0 <= x2 < WIDTH:
            waypoints.append([float(x1), float(y)])
            if not predicted or ((y - start) // 6) % 2 == 0:
                cv2.line(frame, (int(x1), y), (int(x2), next_y),
                         (0, 165, 255) if predicted else (50, 230, 50), 2)
    return waypoints


class LaneDetector:
    def __init__(self):
        self.previous = None
        self.last_good = -1e9
        self.center = None
        self.last_target = None
        self.pair_streak = 0
        self.memory_ready = False
        self.path_range = None

    def single_line_estimate(self, models, now):
        if not self.memory_ready or self.previous is None or now - self.last_good >= RECOVERY_TIMEOUT:
            return None
        candidates = []
        for model in models:
            target = int(np.clip(self.last_target, model['lo'] + 4, model['hi'] - 4))
            rows = np.linspace(model['lo'], model['hi'], 5)
            observed = np.polyval(model['coef'], rows)
            distances = [float(np.mean(np.abs(observed - np.polyval(c, rows))))
                         for c in self.previous]
            side = int(np.argmin(distances))
            if distances[side] > SINGLE_LINE_MAX_SHIFT or abs(distances[0] - distances[1]) < 20:
                continue
            lane_width = float(np.polyval(self.previous[1] - self.previous[0], target))
            if not MIN_LANE_WIDTH <= lane_width <= MAX_LANE_WIDTH:
                continue
            observed_x = float(np.polyval(model['coef'], target))
            center = observed_x + lane_width / 2 if side == 0 else observed_x - lane_width / 2
            offset = center - WIDTH / 2
            if abs(offset) > MAX_RECOVERY_OFFSET:
                continue
            candidates.append((distances[side], offset, target, side, model))
        candidates.sort(key=lambda item: item[0])
        if not candidates:
            return None
        best = candidates[0]
        if any(abs(c[1] - best[1]) > 15 and c[0] - best[0] < 8 for c in candidates[1:]):
            return None
        return best

    def process(self, frame, now=None):
        now = time.monotonic() if now is None else now
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        mask, threshold = black_mask(frame)
        points = row_candidates(mask)
        target = int(HEIGHT * TARGET_Y_RATIO)
        models = line_models(points, target)
        if now - self.last_good > TRACK_EXPIRY:
            self.previous = None
            self.center = None
            self.last_target = None
            self.pair_streak = 0
            self.memory_ready = False
            self.path_range = None
        pairs = []
        for first, second in itertools.combinations(models, 2):
            left, right = sorted((first, second), key=lambda m: m['probes'][0])
            # This camera/straight-track configuration must have floor evidence
            # near the vehicle and opposite converging directions.
            if (left['hi'] < HEIGHT * MIN_NEAR_SUPPORT_Y
                    or right['hi'] < HEIGHT * MIN_NEAR_SUPPORT_Y
                    or left['coef'][0] >= -0.12 or right['coef'][0] <= 0.12):
                continue
            lo, hi = max(left['lo'], right['lo']), min(left['hi'], right['hi'])
            if hi - lo < MIN_SHARED_SPAN:
                continue
            pair_target = int(np.clip(target, lo + 4, hi - 4))
            if any(np.min(np.abs(m['points'][:, 1] - pair_target)) > 8
                   for m in (left, right)):
                continue
            rows = np.array([lo, pair_target, hi])
            lx, rx = np.polyval(left['coef'], rows), np.polyval(right['coef'], rows)
            widths = rx - lx
            slope_difference = right['coef'][0] - left['coef'][0]
            if slope_difference <= 0.05:
                continue
            vanishing_y = (left['coef'][1] - right['coef'][1]) / slope_difference
            if not -HEIGHT * 0.10 <= vanishing_y <= lo + 8:
                continue
            # Road boundaries should stay ordered and widen toward the camera.
            if (np.any(lx < -5) or np.any(rx > WIDTH + 5) or np.any(widths < 25)
                    or not MIN_LANE_WIDTH <= widths[1] <= MAX_LANE_WIDTH
                    or widths[2] < widths[0] + 8):
                continue
            boundary = np.array([lx[1], rx[1]])
            penalty = 0.0
            if self.previous is not None:
                old_boundary = np.array([np.polyval(c, pair_target) for c in self.previous])
                change = np.abs(boundary - old_boundary)
                if change.max() > TRACK_MAX_JUMP:
                    continue
                penalty = float(change.sum()) * 0.20
            score = left['score'] + right['score'] - penalty
            pairs.append((score, left, right, boundary, pair_target))
        pairs.sort(key=lambda item: item[0], reverse=True)
        status = 'LOST: no pair P{} M{}'.format(len(points), len(models))
        chosen = None
        if pairs:
            chosen = pairs[0]
            # Competing, substantially different lanes must not silently win by one point.
            for alternative in pairs[1:]:
                other_boundary = np.array([np.polyval(m['coef'], chosen[4])
                                           for m in alternative[1:3]])
                if (np.max(np.abs(chosen[3] - other_boundary)) > 18
                        and chosen[0] - alternative[0] < 4):
                    chosen = None
                    status = 'LOST: ambiguous lines'
                    break
        result = dict(valid=False, offset=None, status=status, threshold=threshold,
                      points=len(points), models=len(models), timestamp=now, target_y=target,
                      quality='ambiguous' if 'ambiguous' in status else 'none')
        # Show rejected candidates as small orange dots for diagnosis, never as a lane.
        for x, y in points:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 140, 255), -1)
        if chosen is not None:
            _, left, right, boundary, target = chosen
            raw_center = float(boundary.mean())
            dt = min(max(now - self.last_good, 0.001), 0.3)
            alpha = 1.0 - np.exp(-dt / 0.055)
            if self.center is not None and self.previous is not None and self.last_target is not None:
                # Move the previous center to this row before temporal filtering.
                self.center += float(np.mean([np.polyval(c, target) - np.polyval(c, self.last_target)
                                              for c in self.previous]))
            self.center = raw_center if self.center is None else self.center + alpha * (raw_center - self.center)
            self.previous = np.array([left['coef'].copy(), right['coef'].copy()])
            self.last_good, self.last_target = now, target
            self.pair_streak += 1
            self.memory_ready = self.memory_ready or self.pair_streak >= ACQUIRE_FRAMES
            self.path_range = (max(left['lo'], right['lo']), min(left['hi'], right['hi']))
            offset = self.center - WIDTH / 2.0
            result.update(valid=True, quality='dual', offset=float(offset), status='TRACK Y{}'.format(target), target_y=target,
                          width=float(boundary[1] - boundary[0]),
                          left_points=len(left['points']), right_points=len(right['points']))
            for model in (left, right):
                draw_lane_model(frame, model)
            path = self.previous.mean(axis=0)
            result['center_slope'] = float(path[0])
            preview_y = max(float(target - PREVIEW_ROWS), self.path_range[0] + 4)
            separation = target - preview_y
            # Separate heading and displacement, inspired by geometric preview control.
            # This pixel-space hint is not a calibrated vehicle or steering angle.
            if separation >= 10:
                near_x = float(np.polyval(path, target))
                preview_x = float(np.polyval(path, preview_y))
                result['heading_hint'] = float(np.arctan2(near_x - preview_x, separation))
                result['preview_y'], result['preview_x'] = preview_y, preview_x
                cv2.circle(frame, (int(preview_x), int(preview_y)), 3, (255, 255, 255), -1)
            else:
                result['heading_hint'] = 0.0
            result['target_path'] = draw_target_path(frame, path, *self.path_range)
            cv2.circle(frame, (int(self.center), target), 5, (0, 255, 0), -1)
        else:
            self.pair_streak = 0
            estimate = self.single_line_estimate(models, now) if result['quality'] != 'ambiguous' else None
            if estimate is not None:
                _, offset, target, side, model = estimate
                side_name = 'LEFT' if side == 0 else 'RIGHT'
                result.update(quality='single', offset=float(offset), target_y=target,
                              status='SINGLE {} OFF:{:+.1f}'.format(side_name, offset))
                draw_lane_model(frame, model)
                remembered_width = self.previous[1] - self.previous[0]
                path = model['coef'] + remembered_width * (0.5 if side == 0 else -0.5)
                result['target_path'] = draw_target_path(frame, path, model['lo'], model['hi'], predicted=True)
                cv2.circle(frame, (int(offset + WIDTH / 2), target), 5, (0, 165, 255), -1)
            elif (result['quality'] == 'none' and self.memory_ready and self.previous is not None
                  and now - self.last_good < RECOVERY_TIMEOUT and self.path_range is not None):
                result['target_path'] = draw_target_path(frame, self.previous.mean(axis=0),
                                                        *self.path_range, predicted=True)
                result['status'] = 'MEMORY PATH {:.2f}s'.format(now - self.last_good)
                status = result['status']
        cv2.rectangle(frame, (1, int(HEIGHT * ROI_TOP)),
                      (WIDTH - 2, int(HEIGHT * ROI_BOTTOM)), (100, 100, 100), 1)
        cv2.line(frame, (WIDTH // 2, 75), (WIDTH // 2, HEIGHT), (255, 255, 255), 1)
        cv2.line(frame, (0, target), (WIDTH - 1, target), (100, 100, 100), 1)
        # Orange outer trigger / green inner return band, centered on camera axis.
        cx = WIDTH // 2
        cv2.rectangle(frame, (cx-CENTER_CORRECTION_TRIGGER, target-6),
                      (cx+CENTER_CORRECTION_TRIGGER, target+6), (0, 165, 255), 1)
        cv2.rectangle(frame, (cx-CENTER_RETURN_BAND, target-4),
                      (cx+CENTER_RETURN_BAND, target+4), (50, 230, 50), 1)
        label = ('OFF:{:+.1f} L:{} R:{} Y:{}'.format(result['offset'], result['left_points'],
                 result['right_points'], target) if result['valid'] else result['status'])
        cv2.putText(frame, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (0, 255, 0) if result['valid'] else (0, 100, 255), 1)
        cv2.putText(frame, 'v9 P:{} M:{} TH:{}'.format(len(points), len(models), int(threshold)),
                    (5, HEIGHT - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        return frame, mask, result


class SteeringController:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.last_time = None
        self.rate = 0.0
        self.steer = 0.0
        self.correcting = False
        self.heading = 0.0
        self.target = 0.0

    def update(self, offset, now, center_slope=0.0):
        dt = 1.0 / 30 if self.last_time is None else np.clip(now - self.last_time, 0.005, 0.2)
        derivative = 0.0 if self.previous is None else np.clip((offset - self.previous) / dt, -250, 250)
        self.rate += (1 - np.exp(-dt / 0.12)) * (derivative - self.rate)
        self.heading += (1 - np.exp(-dt / 0.15)) * (float(np.clip(center_slope, -0.6, 0.6)) - self.heading)
        distance = abs(offset)
        settled = abs(self.rate) <= SETTLED_OFFSET_RATE and abs(self.heading) <= SETTLED_CENTER_SLOPE
        # Neutral requires both small displacement AND small residual lateral motion.
        # Crossing the center does not erase the yaw/lateral velocity to be damped.
        if self.correcting and distance <= CENTER_RETURN_BAND and settled:
            self.correcting = False
        if not self.correcting and (distance >= CENTER_CORRECTION_TRIGGER or not settled):
            self.correcting = True
        position = float(np.sign(offset)) * max(0.0, distance - CENTER_RETURN_BAND)
        # OFFSET positive means lane center is RIGHT: negative turn steers RIGHT.
        # Falling positive OFFSET means approach from the left; -KD*rate brakes LEFT.
        # Positive centerline slope means far path lies to the LEFT of the near path.
        self.target = float(np.clip(-OFFSET_KP * position - OFFSET_KD * self.rate
                                    + HEADING_KP * self.heading, -1.0, 1.0)) if self.correcting else 0.0
        if self.steer * self.target < 0 or abs(self.target) < abs(self.steer):
            # Release the old turn immediately; never slew a wrong-sign turn through zero.
            self.steer = 0.0 if self.steer * self.target < 0 else self.target
        self.steer += float(np.clip(self.target - self.steer, -TURN_RISE_RATE * dt, TURN_RISE_RATE * dt))
        self.previous, self.last_time = offset, now
        return float(self.steer)


class CommandPolicy:
    """Bounded recovery requires fresh camera frames AND a previously acquired lane."""
    def __init__(self):
        self.controller = SteeringController()
        self.reset()

    def reset(self):
        self.controller.reset()
        self.good_frames = 0
        self.last_stamp = None
        self.steering = 0
        self.target_y = None
        self.started_at = None
        self.stable_since = None
        self.speed = AUTO_START_SPEED
        self.speed_time = None
        self.last_dual_time = None
        self.recovering = False
        self.control_state = 'STOP'
        self.acquire_since = None
        self.startup_straight_done = False
        self.acquisition = []
        self.fault = None
        self.has_moved = False

    def stop_fault(self, reason):
        """A moving AUTO fault requires an explicit re-arm, not another kick."""
        moved = self.has_moved
        self.reset()
        if moved:
            self.fault = reason
        self.control_state = reason
        return 'STOP'

    def recovery_command(self, detection, now):
        if (self.started_at is None or self.last_dual_time is None
                or now - self.last_dual_time >= RECOVERY_TIMEOUT):
            return self.stop_fault('LANE LOST - PRESS AUTO')
        if not self.recovering:
            self.recovering = True
            self.good_frames = 0
            self.controller.previous = None
            self.controller.rate = 0.0
        self.stable_since = None
        self.speed = RECOVERY_SPEED
        self.speed_time = now
        offset = detection.get('offset') if (detection['valid'] or detection.get('quality') == 'single') else None
        if offset is not None:
            if abs(offset) > MAX_RECOVERY_OFFSET:
                return self.stop_fault('OFFSET LIMIT - PRESS AUTO')
            if detection['timestamp'] != self.last_stamp:
                target = detection.get('target_y')
                if target is not None and target != self.target_y:
                    self.controller.previous = None
                    self.controller.rate = 0.0
                self.target_y = target
                self.steering = self.controller.update(offset, detection['timestamp'],
                                                       detection.get('heading_hint', detection.get('center_slope', 0.0)))
                self.last_stamp = detection['timestamp']
            self.control_state = 'SINGLE RECOVERY' if not detection['valid'] else 'REACQUIRE'
        else:
            # No new direction evidence: hold the last steering, never sweep blindly.
            self.control_state = 'MEMORY RECOVERY'
        return 'DRIVE,{:.3f},{:.4f}'.format(RECOVERY_SPEED, self.steering)

    def command(self, mode, manual, manual_time, detection, now):
        if mode != 'AUTO':
            self.reset()
            if manual == 'STOP' or now - manual_time > MANUAL_TIMEOUT:
                return 'STOP'
            direction = 'B' if manual == 'B' else 'F'
            steering = {'L': 35, 'R': -35}.get(manual, 0)
            return 'MANUAL,{},{:.3f},{}'.format(direction, MANUAL_SPEED, steering)
        if self.fault:
            self.control_state = self.fault
            return 'STOP'
        if (self.recovering and self.last_dual_time is not None
                and now - self.last_dual_time >= RECOVERY_TIMEOUT):
            return self.stop_fault('LANE LOST - PRESS AUTO')
        if (not detection or now - detection['timestamp'] > FRAME_TIMEOUT
                or detection.get('quality') == 'ambiguous'):
            reason = 'AMBIGUOUS' if detection and detection.get('quality') == 'ambiguous' else 'STALE CAMERA'
            return self.stop_fault(reason + ' - PRESS AUTO')
        starting_straight = (self.started_at is not None and not self.startup_straight_done
                             and now - self.started_at < STARTUP_STRAIGHT_TIME)
        if starting_straight and (not detection['valid']
                                  or abs(detection.get('center_slope', 0.0)) > START_MAX_CENTER_SLOPE):
            return self.stop_fault('START ABORT - PRESS AUTO')
        if not detection['valid']:
            self.good_frames = 0
            return self.recovery_command(detection, now)
        if self.started_at is None:
            if now - detection['timestamp'] > ACCEL_FRAME_AGE:
                self.reset()
                self.control_state = 'WAIT FRESH CAMERA'
                return 'STOP'
            if abs(detection.get('center_slope', 0.0)) > START_MAX_CENTER_SLOPE:
                self.reset()
                self.control_state = 'ALIGN HEADING'
                return 'STOP'
            if not self.acquisition or detection['timestamp'] != self.acquisition[-1][0]:
                sample = (detection['timestamp'], detection['offset'],
                          detection.get('width'), detection.get('target_y', 139))
                if self.acquisition and sample[0] - self.acquisition[-1][0] > ACQUIRE_MAX_GAP:
                    self.acquisition = []
                trial = self.acquisition + [sample]
                unstable = max(s[1] for s in trial) - min(s[1] for s in trial) > ACQUIRE_MAX_DRIFT
                widths = [s[2] for s in trial if s[2] is not None]
                unstable |= bool(widths) and max(widths) - min(widths) > ACQUIRE_MAX_DRIFT * 2
                unstable |= max(s[3] for s in trial) - min(s[3] for s in trial) > 8
                self.acquisition = [sample] if unstable else trial[-60:]
                self.good_frames = len(self.acquisition) - 1
                self.acquire_since = self.acquisition[0][0]
        if self.recovering:
            if detection['timestamp'] != self.last_stamp:
                self.good_frames += 1
            if self.good_frames < ACQUIRE_FRAMES:
                return self.recovery_command(detection, now)
            self.recovering = False
            self.started_at = now
            self.stable_since = None
            self.speed = AUTO_START_SPEED
            self.controller.previous = None
            self.controller.rate = 0.0
        if detection['timestamp'] != self.last_stamp:
            depth_changed = False
            self.good_frames += 1
            self.last_stamp = detection['timestamp']
            target_y = detection.get('target_y')
            if target_y is not None and self.target_y is not None and abs(target_y - self.target_y) > 3:
                # A change of measurement depth is not lateral vehicle velocity.
                self.controller.previous = None
                self.controller.rate = 0.0
                depth_changed = True
            self.target_y = target_y
            self.steering = self.controller.update(detection['offset'], self.last_stamp,
                                                   detection.get('heading_hint', detection.get('center_slope', 0.0)))
            stable = (not depth_changed and abs(detection['offset']) <= STABLE_OFFSET
                      and abs(self.steering) <= 0.10
                      and abs(self.controller.rate) <= STABLE_OFFSET_RATE)
            if not stable:
                self.stable_since = None
            elif self.stable_since is None:
                self.stable_since = now
        if self.good_frames < ACQUIRE_FRAMES:
            self.control_state = 'ACQUIRE'
            return 'STOP'
        if self.started_at is None:
            if self.acquire_since is None:
                self.acquire_since = detection['timestamp']
            if detection['timestamp'] - self.acquire_since < AUTO_START_CONFIRM_TIME:
                self.control_state = 'CONFIRM LANE'
                return 'STOP'
        self.last_dual_time = detection['timestamp']
        self.control_state = 'CORRECT' if self.controller.correcting else 'IN BAND'
        if self.started_at is None:
            self.started_at = now
            self.speed_time = now
            self.stable_since = None
            # Do not reuse correction accumulated while parked during confirmation.
            self.controller.reset()
            self.steering = self.controller.update(detection['offset'], detection['timestamp'],
                                                   detection.get('heading_hint', detection.get('center_slope', 0.0)))
        self.has_moved = True
        if not self.startup_straight_done:
            # An off-center vehicle needs correction immediately, not a blind straight phase.
            if abs(detection['offset']) >= CENTER_CORRECTION_TRIGGER or self.controller.correcting:
                self.startup_straight_done = True
                self.control_state = 'CORRECT'
        if not self.startup_straight_done:
            if now - self.started_at < STARTUP_STRAIGHT_TIME:
                self.steering = 0.0
                self.controller.reset()
                self.speed = AUTO_START_SPEED
                self.speed_time = now
                self.stable_since = None
                self.control_state = 'START STRAIGHT'
                return 'DRIVE,{:.3f},0.0000'.format(AUTO_START_SPEED)
            self.startup_straight_done = True
        dt = min(max(now - self.speed_time, 0.0), 0.10)
        self.speed_time = now
        ready = (now - self.started_at >= AUTO_START_HOLD and self.stable_since is not None
                 and now - self.stable_since >= AUTO_STABLE_TIME
                 and now - detection['timestamp'] <= ACCEL_FRAME_AGE)
        if ready:
            self.speed = min(AUTO_MAX_SPEED, self.speed + AUTO_ACCELERATION * dt)
        else:
            self.speed = AUTO_START_SPEED
        return 'DRIVE,{:.3f},{:.4f}'.format(self.speed, self.steering)


class Car:
    def __init__(self, port=None):
        self.port = port
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.mode, self.manual = 'MANUAL', 'STOP'
        self.manual_time = 0.0
        self.detection = None
        self.policy = CommandPolicy()
        self.connected = False
        self.command = 'STOP'
        self.generation = 0
        self.video_condition = threading.Condition()
        self.jpeg, self.mask_jpeg, self.sequence = None, None, 0
        self.raw_video = LatestFrame()
        self.metrics = {}
        self.esp_status = ''

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

    def receive_esp_line(self, text):
        with self.lock:
            if text.startswith('STATUS'):
                self.esp_status = text
            if text == 'READY CENTERING V9':
                self.disarm()
                self.connected = True
            elif text.startswith(('READY', 'BOOT', 'EMERGENCY')):
                self.disarm()
                if text.startswith(('READY', 'BOOT')):
                    self.connected = False

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
                                self.disarm()
                                self.connected = False
                            connection.write(b'CTRL,0.000,999\nSTOP\nHELLO\n')
                            next_hello = time.monotonic() + 0.5
                            logging.info('Serial open; waiting for ESP32 CENTERING V9 handshake.')
                        except (serial.SerialException, OSError):
                            if connection is not None:
                                connection.close()
                            connection = None
                if connection is not None:
                    try:
                        # Serialize mode changes and motor writes, preventing old AUTO after STOP.
                        with self.lock:
                            self.command = (self.policy.command(self.mode, self.manual, self.manual_time,
                                                                self.detection, time.monotonic())
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
                            self.disarm()
                self.stop.wait(max(0.0, CONTROL_PERIOD - (time.monotonic() - cycle)))
        finally:
            if connection is not None:
                try:
                    connection.write(b'STOP\n')
                except (serial.SerialException, OSError):
                    pass
                connection.close()

    def camera_loop(self, camera, display, record=None):
        detector = LaneDetector()
        reader = CameraReader(camera, self.stop)
        reader.thread.start()
        encoder = threading.Thread(target=self.video_worker, daemon=True)
        encoder.start()
        recorder = None
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
                    detector = LaneDetector()
                    continue
                if record:
                    if recorder is None:
                        recorder = cv2.VideoWriter(record, cv2.VideoWriter_fourcc(*'MJPG'),
                                                   30.0, (WIDTH, HEIGHT))
                        if not recorder.isOpened():
                            raise IOError('Cannot open raw recording: ' + record)
                    recorder.write(cv2.resize(frame, (WIDTH, HEIGHT)))
                processing_start = time.monotonic()
                annotated, mask, result = detector.process(frame, started)
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
                        detector = LaneDetector()
                    label = '{} {} {}'.format(self.mode, self.command, self.policy.control_state)
                cv2.putText(annotated, label, (5, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, (0, 255, 255), 1)
                metric_label = 'FPS:{:.1f} PROC:{:.0f}ms AGE:{:.0f}ms'.format(fps_filtered, processing_ms, age_ms)
                cv2.putText(annotated, metric_label, (5, 49), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (255, 255, 255), 1)
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
            if recorder is not None:
                recorder.release()
            if display:
                cv2.destroyAllWindows()


HTML = '''<!doctype html><html lang="zh-Hant"><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Jetson Car</title>
<style>body{background:#111;color:white;text-align:center;font-family:Arial}img{width:95%;max-width:640px}
button{font-size:20px;padding:20px;margin:6px;border-radius:12px;touch-action:none;user-select:none}
.stop{background:#ff7766}a{color:#6cf}</style>
<h2>Jetson Car v9 · 1%</h2><img src="/video_feed"><p id="status">Connecting...</p><p id="telemetry"></p>
<button onclick="mode('AUTO')">AUTO</button><button onclick="mode('MANUAL')">MANUAL</button><br>
<button data-cmd="F">前進</button><br><button data-cmd="L">左轉</button>
<button class="stop" onclick="halt()">STOP</button><button data-cmd="R">右轉</button><br>
<button data-cmd="B">後退</button><p><a href="/mask_feed" target="_blank">黑線遮罩</a></p>
<script>
let held='STOP', busy=false, epoch=0;
async function post(path,data){return fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});}
async function manual(){if(busy||held==='STOP')return;busy=true;try{await post('/control',{command:held,epoch});}finally{busy=false;}}
async function refresh(){try{let s=await(await fetch('/status')).json();epoch=s.epoch;
document.getElementById('status').textContent=s.mode+' | '+s.state+' | '+s.lane;
document.getElementById('telemetry').textContent=s.command+' | RATE '+s.offset_rate+' | HEAD '+s.heading_hint+' | FPS '+(s.metrics.process_fps||0)+' | AGE '+(s.metrics.age_ms||0)+'ms | '+(s.connected?s.esp_status:'ESP32 未連線');}catch(e){}}
function halt(){held='STOP';post('/control',{command:'STOP'}).then(refresh);}
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
                           version='v9', state=car.policy.control_state, esp_status=car.esp_status,
                           offset=car.detection.get('offset') if fresh else None,
                           offset_rate=round(car.policy.controller.rate, 2),
                           heading_hint=round(car.policy.controller.heading, 3),
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
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--port', help='e.g. /dev/ttyUSB0; required if multiple serial devices exist')
    parser.add_argument('--display', action='store_true', help='Enable local OpenCV windows')
    parser.add_argument('--image', help='Offline raw-camera image; does not open serial or web server')
    parser.add_argument('--video', help='Offline raw-camera video; does not open serial or web server')
    parser.add_argument('--record', help='Record unannotated camera frames to an .avi file')
    parser.add_argument('--log', help='Diagnostic log path (default: timestamped log beside script)')
    parser.add_argument('--output', default='lane_debug.jpg')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    cv2.setNumThreads(2)
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            parser.error('Cannot read image')
        annotated, mask, result = LaneDetector().process(frame)
        if not cv2.imwrite(args.output, annotated) or not cv2.imwrite(args.output + '.mask.png', mask):
            raise IOError('Cannot write output')
        print(result)
        return
    if args.video:
        detector = LaneDetector()
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
                                       'car_v9_' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    try:
        handler = logging.FileHandler(log_path, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logging.getLogger().addHandler(handler)
        logging.info('Jetson v9 diagnostic log: %s', log_path)
    except OSError as exc:
        logging.warning('Cannot create diagnostic log: %s', exc)
    car = Car(args.port)
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
