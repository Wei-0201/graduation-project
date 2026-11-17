"""Bounded tape geometry for model bend-exit correction (320 x 240)."""
import numpy as np

TARGET_ROW = 140.0


def tape_lines(mask):
    points = []
    for y in range(94, 220, 4):
        row = np.count_nonzero(mask[y-1:y+2], axis=0) >= 2
        edges = np.diff(np.r_[False, row, False].astype(np.int8))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        points.extend(((a+b-1)/2.0, float(y), float(b-a)) for a, b in zip(starts, ends)
                      if 2 <= b-a <= 30 and a > 1 and b < 319)
    if not 7 <= len(points) <= 250:
        return []
    points = np.asarray(points)
    x, y = points[:, :2].T
    a, b = np.triu_indices(len(points), 1)
    keep = np.abs(y[b]-y[a]) >= 20
    a, b = a[keep], b[keep]
    if not len(a):
        return []
    if len(a) > 256:
        chosen = np.random.RandomState(19).choice(len(a), 256, replace=False)
        a, b = a[chosen], b[chosen]
    slopes = (x[b]-x[a])/(y[b]-y[a])
    intercepts = x[a]-slopes*y[a]
    residual = np.abs(x[None, :]-slopes[:, None]*y-intercepts[:, None])
    counts = np.sum(residual <= 3.0, axis=1)
    lines = []
    for index in np.argsort(-counts):
        if counts[index] < 7 or len(lines) >= 6:
            break
        probes = slopes[index]*np.array([120., 160.])+intercepts[index]
        if any(np.max(np.abs(probes-m['probes'])) < 10 for m in lines):
            continue
        ids = np.flatnonzero(residual[index] <= 3.0)
        ids = ids[np.lexsort((residual[index, ids], y[ids]))]
        _, first = np.unique(y[ids], return_index=True)
        p = points[ids[first]]
        pieces = np.split(p, np.flatnonzero(np.diff(p[:, 1]) > 12)+1)
        p = max(pieces, key=len)
        if len(p) < 7 or np.ptp(p[:, 1]) < 24:
            continue
        yy = p[:, 1]-p[:, 1].mean()
        slope = np.dot(yy, p[:, 0]-p[:, 0].mean())/np.dot(yy, yy)
        coef = np.array([slope, p[:, 0].mean()-slope*p[:, 1].mean()])
        rms = float(np.sqrt(np.mean((p[:, 0]-np.polyval(coef, p[:, 1]))**2)))
        if rms > 2.6 or abs(slope) > 3.4:
            continue
        tape_width = float(np.median(p[:, 2]))
        if tape_width < 5:
            continue  # Thin floor joints must not act as a tape boundary.
        lines.append(dict(coef=coef, lo=float(p[:, 1].min()),
                          hi=float(p[:, 1].max()), rms=rms, tape_width=tape_width,
                          probes=np.polyval(coef, [120., 160.])))
    return lines


class VisualExitGuard:
    """Observe straight border pairs anywhere in view; retain a known border.

    This state contains observed border identity/width only. No route schedule
    or elapsed driving time is used to decide where a bend ends.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.width = None
        self.previous = None
        self.misses = 0

    def process(self, mask):
        lines = tape_lines(mask)
        pairs = []
        for left in lines:
            for right in lines:
                lc, rc = left['coef'], right['coef']
                if max(left['tape_width'], right['tape_width']) > 2.8 * min(
                        left['tape_width'], right['tape_width']):
                    continue
                width = float(np.polyval(rc-lc, TARGET_ROW))
                divergence = rc[0]-lc[0]
                if not 45 <= width <= 310 or not 0.4 <= divergence <= 4.5:
                    continue
                vanish_y = -(rc[1]-lc[1])/divergence
                if not 20 <= vanish_y <= 115:
                    continue
                if max(left['hi']-left['lo'], right['hi']-right['lo']) < 78:
                    continue
                if any(m['lo'] > 125 or m['hi'] < 122 or
                       not m['lo']-20 <= TARGET_ROW <= m['hi']+20
                       for m in (left, right)):
                    continue
                shared_lo = max(left['lo'], right['lo'])
                shared_hi = min(left['hi'], right['hi'])
                if shared_hi-shared_lo < 16 or min(np.polyval(
                        rc-lc, [shared_lo, shared_hi])) < 35:
                    continue  # Two pieces of the same curve are not two borders.
                center = float(np.polyval((lc+rc)/2, TARGET_ROW))
                if not -60 <= center <= 380:
                    continue
                score = (left['hi']-left['lo']+right['hi']-right['lo']
                         - 12*(left['rms']+right['rms']))
                if self.previous is not None:
                    score -= .5*sum(abs(float(np.polyval(m['coef']-old, TARGET_ROW)))
                                    for m, old in zip((left, right), self.previous))
                pairs.append((score, left, right, width, center))
        if pairs:
            _, left, right, width, center = max(pairs, key=lambda p: p[0])
            self.width = width
            self.previous = (left['coef'], right['coef'])
            self.misses = 0
            return self.result(center, 'PAIR', .75, [left, right])
        self.misses += 1
        if self.previous is not None and self.misses <= 8:
            matches = []
            for side, old in enumerate(self.previous):
                for line in lines:
                    if line['hi']-line['lo'] < 78 or line['lo'] > 125 or line['hi'] < 185:
                        continue
                    jump = abs(float(np.polyval(line['coef']-old, TARGET_ROW)))
                    slope_jump = abs(float(line['coef'][0]-old[0]))
                    if jump <= 42 and slope_jump <= .65:
                        matches.append((jump+20*slope_jump, side, line))
            if matches:
                _, side, line = min(matches, key=lambda p: p[0])
                x = float(np.polyval(line['coef'], TARGET_ROW))
                center = x + (self.width/2 if side == 0 else -self.width/2)
                previous = list(self.previous)
                previous[side] = line['coef']
                self.previous = tuple(previous)
                self.misses = 0  # A newly observed matching border is fresh evidence.
                return self.result(center, 'KNOWN LEFT' if side == 0 else 'KNOWN RIGHT',
                                   .70, [line])
        if self.misses > 8:
            self.reset()
        return dict(visual_turn=None, visual_source='NONE', visual_weight=0.0,
                    visual_center=None, lines=[])

    @staticmethod
    def result(center, source, weight, lines):
        return dict(visual_turn=float(np.clip((160-center)/160, -1.0, 1.0)),
                    visual_source=source, visual_weight=weight,
                    visual_center=center, lines=lines)
