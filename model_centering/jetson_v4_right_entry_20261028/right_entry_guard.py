"""Ease premature right turn using a freshly observed near-field lane pair."""
import math


class RightEntryGuard:
    def __init__(self):
        self.pairs = 0
        self.active = False
        self.limit = None

    def adjust(self, target, detection):
        self.active, self.limit = False, None
        geometry = detection.get('parking_corridor') or {}
        try:
            position = float(geometry['position'])
            heading = float(geometry['heading'])
            width = float(geometry['width'])
            valid = (geometry.get('source') == 'PAIR' and geometry.get('right_observed')
                     and all(math.isfinite(v) for v in (position, heading, width))
                     and .15 <= position <= .85 and 55 <= width <= 360)
        except (KeyError, ValueError, TypeError):
            valid = False
        self.pairs = min(2, self.pairs+1) if valid else 0
        if self.pairs < 2 or target >= 0:
            return target
        # Heading is the far-minus-near normalized lane position. A substantially
        # negative value means the right bend has reached these near scan rows.
        # Release continuously with image geometry, never with elapsed driving time.
        strength = max(0., min(1., (heading+.09)/.06))
        if strength == 0:
            return target
        # Still permit rightward recentering when the car is left of the lane.
        allowance = min(.35, .06 + max(0., .5-position)*1.5)
        eased = max(target, -allowance)
        result = target + strength*(eased-target)
        self.active = result > target+1e-6
        self.limit = -allowance if self.active else None
        return result
