"""Earlier bend unwind from fresh paired lane evidence, never a route timer."""
import math


class ObservedExitEasing:
    def __init__(self):
        self.direction = 0
        self.bend_frames = 0
        self.candidate_direction = 0
        self.straight_frames = 0
        self.misses = 0
        self.active = False
        self.previous_raw = None
        self.stamp = None

    def adjust(self, target, detection):
        self.active = False
        stamp = detection.get('timestamp')
        if stamp is not None and stamp == self.stamp:
            return target
        self.stamp = stamp
        raw = float(detection['turn'])
        geometry = detection.get('parking_corridor') or {}
        try:
            position = float(geometry['position'])
            heading = float(geometry['heading'])
            width = float(geometry['width'])
            valid = (geometry.get('source') == 'PAIR' and geometry.get('right_observed')
                     and all(math.isfinite(v) for v in (position, heading, width))
                     and 55 <= width <= 360 and .25 <= position <= .75)
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            self.misses += 1
            self.straight_frames = 0
            self.bend_frames = 0
            self.candidate_direction = 0
            if self.misses >= 5:
                self.direction = 0
            self.previous_raw = raw
            return target
        self.misses = 0
        direction = 1 if raw > 0 else -1 if raw < 0 else 0
        if abs(raw) >= .15 and heading*direction >= .10:
            self.straight_frames = 0
            self.bend_frames = self.bend_frames+1 if self.candidate_direction == direction else 1
            self.candidate_direction = direction
            if self.bend_frames >= 2:
                self.direction = direction
        else:
            self.bend_frames = 0
            self.candidate_direction = 0
        if self.direction and direction != self.direction:
            self.direction = 0
            self.straight_frames = 0
        shrinking = (self.previous_raw is not None and abs(raw) <= abs(self.previous_raw)+.005)
        self.previous_raw = raw
        near_straight = abs(heading) <= .055 and abs(position-.5) <= .16
        self.straight_frames = self.straight_frames+1 if near_straight else 0
        if not self.direction or self.straight_frames < 2 or not shrinking:
            return target
        # Position gives a recentering request; heading gives near-road direction.
        # Only unwind the current turn, never add or reverse steering here.
        observed = 1.2*(position-.5)+heading
        signed = max(0., observed*self.direction)
        if signed >= abs(target) or target*self.direction <= 0:
            return target
        reduction = min(.18, .65*(abs(target)-signed))
        self.active = reduction > 1e-6
        return target-self.direction*reduction
