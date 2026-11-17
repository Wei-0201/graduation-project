"""Match dark or white vertical arrow silhouettes against sign references."""
from pathlib import Path

import cv2
import numpy as np


SIZE = (40, 56)
ARROW_HITS = 3


def silhouette(contour):
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED, offset=(-x, -y))
    return cv2.resize(mask, SIZE, interpolation=cv2.INTER_AREA) >= 128


def white_arrow_contours(image):
    """Find white cutouts enclosed by a blue or green circular sign."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 115), (179, 65, 255))
    kernel = np.ones((3, 3), dtype=np.uint8)
    for low, high in ((33, 89), (90, 135)):
        color = cv2.inRange(hsv, (low, 75, 65), (high, 255, 255))
        color = cv2.morphologyEx(color, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(color, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        if hierarchy is None:
            continue
        for contour, (_, _, _, parent) in zip(contours, hierarchy[0]):
            if parent < 0:
                continue
            badge = contours[parent]
            bx, by, bw, bh = cv2.boundingRect(badge)
            if min(bw, bh) < 45 or not .74 <= bw/bh <= 1.35:
                continue
            area = cv2.contourArea(badge)
            perimeter = cv2.arcLength(badge, True)
            if area < .60*bw*bh or 4*np.pi*area < .70*perimeter**2:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 22 or h < 30 or not .10*area <= cv2.contourArea(contour) <= .60*area:
                continue
            if not (bx < x and by < y and x+w < bx+bw and y+h < by+bh):
                continue
            interior = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(interior, [contour], -1, 255, cv2.FILLED, offset=(-x, -y))
            white_inside = cv2.bitwise_and(white[y:y+h, x:x+w], interior)
            if cv2.countNonZero(white_inside) >= .80*cv2.countNonZero(interior):
                yield contour


class SpeedArrowDetector:
    def __init__(self):
        with np.load(str(Path(__file__).with_name('speed_arrow_templates.npz'))) as data:
            self.templates = data['masks']
            self.directions = tuple(data['directions'])

    def detect(self, image):
        # The enclosed outline of the blue arrow yields the same solid
        # silhouette as the red or black arrows. Ignore the outside sign frame.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 0, 145)
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        height, width = gray.shape
        matches = {}
        for contour in list(contours) + list(white_arrow_contours(image)):
            x, y, w, h = cv2.boundingRect(contour)
            if not (h >= 30 and w >= 22 and .52 <= w/h <= 1.02
                    and x > 1 and y > 1 and x+w < width-1 and y+h < height-1
                    and .32 <= cv2.contourArea(contour)/(w*h) <= .74):
                continue
            candidate = silhouette(contour)
            scores = (np.count_nonzero(candidate & self.templates, axis=(1, 2))
                      / np.count_nonzero(candidate | self.templates, axis=(1, 2)))
            best = int(np.argmax(scores))
            direction = str(self.directions[best])
            opposite = max(score for score, label in zip(scores, self.directions)
                           if label != direction)
            score = float(scores[best])
            if score >= .83 and score-opposite >= .12 and score > matches.get(direction, (0, None))[0]:
                matches[direction] = (score, dict(x=x, y=y, w=w, h=h))
        # A normal-speed sign wins if both signs are visible.
        for direction in ('DOWN', 'UP'):
            if direction in matches:
                score, box = matches[direction]
                return direction, box, score
        return None, None, None
