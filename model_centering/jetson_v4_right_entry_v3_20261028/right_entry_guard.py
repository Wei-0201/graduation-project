"""Ease premature right turn using a freshly observed near-field lane pair."""
import math


class RightEntryGuard:
    def __init__(self):
        self.pairs = 0
        self.active = False
        self.limit = None
        self.bootstrapped = False
        self.last_stamp = None
        self.missing = 0
        self.previous_cap = None

    def adjust(self, target, detection):
        self.active, self.limit = False, None
        geometry = detection.get('parking_corridor') or {}
        stamp = detection.get('timestamp')
        new_frame = stamp is None or stamp != self.last_stamp
        self.last_stamp = stamp
        try:
            position = float(geometry['position'])
            heading = float(geometry['heading'])
            width = float(geometry['width'])
            paired = geometry.get('source') == 'PAIR'
            gap = float(geometry.get('right_gap_px', float('nan')))
            tracked = (geometry.get('source') == 'RIGHT TRACK' and self.bootstrapped
                       and math.isfinite(gap) and gap >= 0
                       and abs(gap/width-(1-position)) < .05)
            valid = ((paired or tracked) and geometry.get('right_observed')
                     and all(math.isfinite(v) for v in (position, heading, width))
                     and .15 <= position <= 1.0 and 55 <= width <= 360)
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            valid = False
        if new_frame:
            self.pairs = min(2, self.pairs+1) if valid else 0
            self.missing = 0 if valid else self.missing+1
        if not valid:
            # A one/two-frame segmentation gap must not suddenly restore a
            # stronger right command. This only retains attenuation, not geometry.
            if self.previous_cap is not None and self.missing <= 2 and target < 0:
                result = max(target, self.previous_cap)
                self.active = result > target+1e-6
                return result
            self.previous_cap = None
            return target
        if self.pairs >= 2 and paired:
            self.bootstrapped = True
        if self.pairs < 2 or target >= 0:
            if target >= 0:
                self.previous_cap = None
            return target
        # Heading is the far-minus-near normalized lane position. A substantially
        # negative value means the right bend has reached these near scan rows.
        # Release continuously with image geometry, never with elapsed driving time.
        strength = max(0., min(1., (heading+.18)/.12))
        # Still permit rightward recentering when the car is left of the lane.
        allowance = min(.35, .02 + max(0., .5-position)*1.5)
        eased = max(target, -allowance)
        result = target + strength*(eased-target)
        # A yawed camera can make the distant road look strongly curved while
        # the car is already cutting inside. Fresh right-edge proximity must
        # therefore constrain release independently of heading.
        if position > .60:
            edge_allowance = max(0., .06-(position-.60)*.4)
            result = max(result, -edge_allowance)
        self.active = result > target+1e-6
        self.limit = -allowance if self.active else None
        self.previous_cap = result if self.active else None
        return result
