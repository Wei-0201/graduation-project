"""STOP pause and observed-corridor right-side parking; no timed steering route."""
import math
import numpy as np

SIGN_MAX_AGE = 1.2


def tape_points(mask, y):
    row = np.count_nonzero(mask[y-1:y+2], axis=0) >= 2
    edges = np.diff(np.r_[False, row, False].astype(np.int8))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [((a+b-1)/2., float(b-a)) for a, b in zip(starts, ends)
            if 5 <= b-a <= 45 and a > 1 and b < 319]


def fit(points, y):
    ys = np.array([p[0] for p in points], dtype=float)
    xs = np.array([p[1] for p in points], dtype=float)
    dy = ys-ys.mean()
    slope = np.dot(dy, xs-xs.mean())/max(1., np.dot(dy, dy))
    intercept = xs.mean()-slope*ys.mean()
    residual = float(np.max(np.abs(xs-(slope*ys+intercept))))
    return float(slope*y+intercept), residual, float(slope)


class RightCorridorTracker:
    """Require paired tape evidence and an actually visible right border near y=175."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.misses = 0
        self.width_near = None
        self.width_far = None
        self.last_right_far = None

    def observed_right_track(self, points):
        if self.previous is None or self.width_near is None or self.last_right_far is None:
            return None
        near_right = self.previous[1]
        slope = (near_right-self.last_right_far)/30.
        observed = []
        for y in (145, 155, 165, 175, 185):
            expected = near_right+slope*(y-175)
            options = [(abs(x-expected), x) for x, width in points[y] if abs(x-expected) <= 45]
            if options:
                observed.append((y, min(options)[1]))
        if len(observed) < 4:
            return None
        right, error, _ = fit(observed, 175)
        far_right, _, _ = fit(observed, 145)
        if error > 5 or not any(abs(x-right) <= 8 for x, width in points[175]):
            return None
        self.previous = (right-self.width_near, right)
        self.last_right_far, self.misses = far_right, 0
        return self.result(right-self.width_near, right, far_right-self.width_far, far_right, 'RIGHT TRACK')

    @staticmethod
    def result(left, right, far_left, far_right, source):
        width = right-left
        position = (160-left)/width
        far_position = (160-far_left)/(far_right-far_left)
        return dict(left=left, right=right, width=width, position=position,
                    far_left=far_left, far_right=far_right,
                    heading=far_position-position, right_gap_px=right-160,
                    right_observed=True, source=source)

    def process(self, mask):
        ys = (145, 155, 165, 175, 185)
        points = {y: tape_points(mask, y) for y in ys}
        pairs = {}
        for y in ys:
            pp = points[y]
            pairs[y] = [(l[0], r[0]) for l in pp for r in pp
                        if len(pp) <= 8 and 55 <= r[0]-l[0] <= 310
                        and max(l[1], r[1]) <= 3*min(l[1], r[1])]
        proposals = []
        for left, right in pairs[165]:
            if not left < 160 < right:
                continue
            path = [(165, left, right)]
            for y in (155, 145, 175, 185):
                options = [(abs(l-left)+abs(r-right), l, r) for l, r in pairs[y]
                           if max(abs(l-left), abs(r-right)) <= 65
                           and .5 <= (r-l)/(right-left) <= 1.6]
                if options:
                    _, l, r = min(options)
                    path.append((y, l, r))
            if len(path) < 3 or max(p[0] for p in path)-min(p[0] for p in path) < 20:
                continue
            nl, le, _ = fit([(y, l) for y, l, r in path], 175)
            nr, re, _ = fit([(y, r) for y, l, r in path], 175)
            fl, _, _ = fit([(y, l) for y, l, r in path], 145)
            fr, _, _ = fit([(y, r) for y, l, r in path], 145)
            if max(le, re) > 5 or not 55 <= nr-nl <= 360 or not 40 <= fr-fl <= 330:
                continue
            # No blind approach when only an extrapolated/missing right border exists.
            if not any(abs(x-nr) <= 8 for x, width in points[175]):
                continue
            if self.previous and self.misses < 3 and max(abs(nl-self.previous[0]), abs(nr-self.previous[1])) > 45:
                continue
            score = 20*len(path)-2*(le+re)-.1*abs((nl+nr)/2-160)
            if self.previous and self.misses < 3:
                score -= .2*(abs(nl-self.previous[0])+abs(nr-self.previous[1]))
            proposals.append((score, nl, nr, fl, fr))
        if not proposals:
            tracked = self.observed_right_track(points)
            if tracked is not None:
                return tracked
            self.misses += 1
            return None
        _, left, right, far_left, far_right = max(proposals)
        self.previous, self.misses = (left, right), 0
        self.width_near, self.width_far = right-left, far_right-far_left
        self.last_right_far = far_right
        return self.result(left, right, far_left, far_right, 'PAIR')


class TrafficPolicy:
    def __init__(self, stop_seconds=0., target_ratio=1., park_timeout=15., max_turn=.30):
        self.stop_seconds, self.target_ratio = stop_seconds, target_ratio
        self.park_timeout, self.max_turn = park_timeout, max_turn
        self.handled = {'stop': False, 'park': False}
        self.clear_counts = {'stop': 0, 'park': 0}
        self.last_ocr_sequence = None
        self.state = 'MANUAL'
        self.needs_return = False
        self.cancel_motion()

    def cancel_motion(self):
        # Preserve sign latches: AUTO after parking can depart while the same P is visible.
        if getattr(self, 'park_pending', False) and getattr(self, 'park_started', None) is None:
            self.handled['park'] = False
        self.stop_until = None
        self.stop_latched = False
        self.park_pending = False
        self.park_started = None
        self.parked = False
        self.aborted = None
        self.park_stamp = None
        self.park_stable = 0
        self.park_lost = 0
        self.park_turn = 0.
        self.park_command = 'STOP'
        self.returning = False
        self.geometry_confirm = 0

    def request_auto(self, now):
        if self.needs_return:
            self.returning = True
            self.park_started = now
            self.state = 'RETURN TO LANE CENTER'

    @staticmethod
    def fresh(signs, now, generation):
        return (signs is not None and signs.get('generation') == generation
                and -.05 <= now-signs.get('timestamp', -1e9) <= SIGN_MAX_AGE
                and not signs.get('error'))

    def consume(self, signs, mode, now):
        seq = signs.get('sequence')
        if seq == self.last_ocr_sequence:
            return
        self.last_ocr_sequence = seq
        for kind in ('stop', 'park'):
            if signs.get(kind+'_present'):
                self.clear_counts[kind] = 0
            else:
                self.clear_counts[kind] += 1
                if self.clear_counts[kind] >= 3:
                    self.handled[kind] = False
            if mode == 'AUTO' and signs.get(kind+'_confirmed') and not self.handled[kind]:
                self.handled[kind] = True
                if kind == 'stop':
                    self.stop_latched = self.stop_seconds == 0
                    self.stop_until = now + self.stop_seconds
                    if self.park_started is not None:
                        self.park_started += self.stop_seconds
                elif not self.parked and not self.aborted and not self.returning:
                    self.park_pending = True

    def command(self, mode, base_command, detection, signs, now, generation):
        fresh = self.fresh(signs, now, generation)
        if fresh:
            self.consume(signs, mode, now)
        if mode != 'AUTO':
            self.cancel_motion()
            self.state = 'MANUAL'
            return base_command
        if self.stop_latched or (self.stop_until is not None and now < self.stop_until):
            self.state = 'STOP OCR HOLD' if self.stop_latched else 'STOP OCR PAUSE {:.1f}s'.format(self.stop_until-now)
            return 'STOP'
        if self.parked or self.aborted:
            self.state = 'PARKED ON RIGHT LINE - AUTO TO RETURN' if self.parked else 'MANEUVER ABORT: '+self.aborted
            return 'STOP'
        if not fresh:
            self.state = 'WAIT OCR' if not signs or not signs.get('error') else 'OCR ERROR'
            return 'STOP'
        if self.park_pending:
            self.park_pending = False
            self.park_started = now
            self.needs_return = True
        if self.park_started is None:
            self.state = 'V4 FOLLOW'
            return base_command
        if now-self.park_started > self.park_timeout:
            self.aborted, self.state = 'TIMEOUT', 'MANEUVER ABORT: TIMEOUT'
            return 'STOP'
        if (not detection or not (detection.get('valid') or detection.get('frame_valid')) or
                not -.05 <= now-detection.get('timestamp', -1e9) <= .45 or
                detection.get('lane_guard_latched') or
                (not base_command.startswith('DRIVE,') and detection.get('valid'))):
            self.state = 'PARK WAIT FRESH LANE'
            return 'STOP'
        stamp = detection['timestamp']
        if stamp == self.park_stamp:
            return self.park_command
        self.park_stamp = stamp
        corridor = detection.get('parking_corridor')
        if not corridor or not corridor.get('right_observed'):
            self.park_lost += 1
            self.park_stable = 0
            self.geometry_confirm = 0
            if self.park_lost >= 3:
                self.aborted = 'RIGHT CORRIDOR LOST'
            self.state = 'PARK ABORT: RIGHT CORRIDOR LOST' if self.aborted else 'PARK WAIT RIGHT CORRIDOR'
            self.park_command = 'STOP'
            return 'STOP'
        self.park_lost = 0
        self.geometry_confirm += 1
        position, heading = corridor.get('position'), corridor.get('heading')
        if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (position, heading))
                or not -.05 <= position <= 1.10 or corridor.get('right_gap_px', 0) < -20):
            self.aborted, self.state = 'RIGHT EDGE GUARD', 'PARK ABORT: RIGHT EDGE GUARD'
            self.park_command = 'STOP'
            return 'STOP'
        if not base_command.startswith('DRIVE,') and self.geometry_confirm < 3:
            self.state = 'MANEUVER VERIFY OBSERVED CORRIDOR'
            self.park_command = 'STOP'
            return 'STOP'
        target_ratio = .5 if self.returning else self.target_ratio
        at_target = abs(position-target_ratio) <= (.06 if self.returning else .04)
        if not self.returning and self.target_ratio >= .99:
            at_target = at_target and abs(corridor.get('right_gap_px', 999)) <= 6
        if at_target and abs(heading) <= .10:
            self.park_stable += 1
            if self.returning:
                if self.park_stable >= 3:
                    self.needs_return, self.returning = False, False
                    self.park_started = None
                    self.state = 'V4 FOLLOW'
                    return base_command
            else:
                self.parked = self.park_stable >= 3
                self.state = 'PARKED ON RIGHT LINE - AUTO TO RETURN' if self.parked else 'PARK VERIFY POSITION'
                self.park_command = 'STOP'
                return 'STOP'
        else:
            self.park_stable = 0
        target = float(np.clip(-(target_ratio-position)+1.8*heading, -self.max_turn, self.max_turn))
        self.park_turn = .4*self.park_turn + .6*target
        speed = min(.12, float(base_command.split(',')[1])) if base_command.startswith('DRIVE,') else .12
        self.park_command = 'DRIVE,{:.3f},{:.4f}'.format(speed, self.park_turn)
        self.state = '{} pos:{:.2f} target:{:.2f}'.format(
            'RETURN TO CENTER' if self.returning else 'PARK ON RIGHT LINE', position, target_ratio)
        return self.park_command
