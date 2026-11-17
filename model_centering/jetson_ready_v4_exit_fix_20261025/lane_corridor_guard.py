"""Observed tape corridor at a forward image row, including curved lanes."""
import numpy as np

TARGET_ROW = 110


def candidates(mask, y):
    row = np.count_nonzero(mask[y-1:y+2], axis=0) >= 2
    edges = np.diff(np.r_[False, row, False].astype(np.int8))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [((a+b-1)/2., float(b-a)) for a,b in zip(starts,ends)
            if 5 <= b-a <= 36 and a > 1 and b < 319]


def row_pairs(points):
    if len(points) > 8:
        return []
    return [(l[0],r[0]) for l in points for r in points
            if 55 <= r[0]-l[0] <= 240 and
            max(l[1],r[1]) <= 2.6*min(l[1],r[1])]


class LaneCorridorGuard:
    """Trace the observed left/right corridor; never infer a timed route."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.width = None
        self.misses = 0
        self.last_result = None
        self.width_slope = 2.0

    def process(self, mask):
        ys = (110,120,130)
        points = {y:candidates(mask,y) for y in ys}
        pairs = {y:row_pairs(points[y]) for y in ys}
        proposals = []
        for left,right in pairs[120]:
            width = right-left
            supporting = []
            for y in (110,130):
                matches = []
                for l,r in pairs[y]:
                    if max(abs(l-left),abs(r-right)) > 65:
                        continue
                    if abs((r-l)-width) > max(30,width*.35):
                        continue
                    if (y-120)*((r-l)-width) < -60:
                        continue
                    matches.append((abs((r-l)-width),l,r,y))
                if matches:
                    supporting.append(min(matches))
            if not supporting:
                continue
            score = 20*len(supporting)
            if self.previous is not None:
                jump = max(abs(left-self.previous[0]),abs(right-self.previous[1]))
                if jump > 75:
                    continue
                score -= .30*(abs(left-self.previous[0])+abs(right-self.previous[1]))
                score -= .30*abs(width-self.width)
            else:
                score -= .1*abs((left+right)/2-160)
            proposals.append((score,left,right,supporting))
        if proposals:
            _,left,right,support = max(proposals,key=lambda p:p[0])
            self.previous = (left,right)
            self.width = right-left
            slopes = [((r-l)-self.width)/(y-120) for _,l,r,y in support]
            self.width_slope = float(np.clip(np.median(slopes),0.,4.))
            self.misses = 0
            far = [p for p in support if p[3]==110]
            if far:
                center = (far[0][1]+far[0][2])/2
            else:
                near = support[0]
                center = 2*(left+right)/2-(near[1]+near[2])/2
            result = self.result(center,'CURVE PAIR',.75,
                                 [(left,120),(right,120)])
            self.last_result = result
            return result
        if self.previous is not None and len(points[120]) <= 4:
            matches = []
            for x,w in points[120]:
                for side,old in enumerate(self.previous):
                    jump = abs(x-old)
                    if jump > 55:
                        continue
                    # A border must continue into an adjacent forward scan row.
                    if not any(any(abs(other-x)<=65 for other,_ in points[y])
                               for y in (110,130)):
                        continue
                    matches.append((jump,side,x))
            matches.sort()
            if matches and (len(matches)==1 or matches[1][0]-matches[0][0]>=12):
                _,side,x = matches[0]
                left = x if side==0 else x-self.width
                right = x+self.width if side==0 else x
                self.previous = (left,right)
                self.misses = 0
                near = [other for other,_ in points[130] if abs(other-x)<=65]
                near_x = min(near,key=lambda v:abs(v-x)) if near else x
                projected = 2*x-near_x
                far = [other for other,_ in points[110] if abs(other-projected)<=35]
                far_x = min(far,key=lambda v:abs(v-projected)) if far else projected
                far_width = max(40.,self.width-10*self.width_slope)
                center = far_x + (far_width/2 if side==0 else -far_width/2)
                result = self.result(center,
                                     'TRACK LEFT' if side==0 else 'TRACK RIGHT',.65,
                                     [(x,120)])
                self.last_result = result
                return result
        self.misses += 1
        if self.last_result is not None and self.misses <= 2:
            result = dict(self.last_result)
            result['visual_source'] = 'BRIEF GAP'
            result['visual_points'] = []
            return result
        if self.misses > 3:
            self.previous = None
            self.width = None
            self.last_result = None
        return dict(visual_turn=None,visual_source='NO CORRIDOR',visual_weight=0.,
                    visual_center=None,visual_points=[],visual_lost=True)

    @staticmethod
    def result(center,source,weight,points):
        return dict(visual_turn=float(np.clip((160-center)/160,-1.,1.)),
                    visual_source=source,visual_weight=weight,
                    visual_center=center,visual_points=points,visual_lost=False)
