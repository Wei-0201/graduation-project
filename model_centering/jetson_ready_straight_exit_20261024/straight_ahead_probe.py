"""Cheap, conservative two-boundary check for the far half of a 320x240 frame."""

import numpy as np


def straight_ahead_probe(mask, width=320):
    """Return (confidence, centers) only when both tape borders look straight ahead.

    The four scan rows are image coordinates, never elapsed time or track
    position. The caller should fall back to the model when confidence is 0.
    """
    rows = (112, 127, 142, 157)
    pairs = []
    mid = width / 2.0
    for y in rows:
        row = np.count_nonzero(mask[y - 1:y + 2], axis=0) >= 2
        edges = np.diff(np.r_[False, row, False].astype(np.int8))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        centers = [(a + b - 1) / 2.0 for a, b in zip(starts, ends)
                   if 2 <= b - a <= 30 and a > 1 and b < width - 1]
        choices = [(left, right) for left in centers for right in centers
                   if left < mid < right and 45 <= right - left <= 260
                   and abs((left + right) / 2.0 - mid) <= 28]
        if not choices:
            return 0.0, []
        # A centered pair is preferred over two close dark edges from clutter.
        pairs.append(min(choices, key=lambda p: abs((p[0] + p[1]) / 2.0 - mid)))
    lefts = np.asarray([p[0] for p in pairs])
    rights = np.asarray([p[1] for p in pairs])
    centers = (lefts + rights) / 2.0
    widths = rights - lefts
    if (np.ptp(centers) > 18 or
            np.any(np.diff(widths) < -3) or
            np.any(np.abs(np.diff(lefts)) > 45) or
            np.any(np.abs(np.diff(rights)) > 45)):
        return 0.0, centers.tolist()
    # Lower confidence when the apparent lane axis drifts or is off-center.
    drift_weight = max(0.0, 1.0 - float(np.ptp(centers)) / 30.0)
    center_weight = max(0.0, 1.0 - abs(float(np.mean(centers)) - mid) / 40.0)
    return drift_weight * center_weight, centers.tolist()
