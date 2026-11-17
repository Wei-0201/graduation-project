"""Latest-frame OCR worker; no camera ownership and no motor writes."""
import logging
import threading
import time

import cv2
import numpy as np

import sign_ocr_engine
import user_sign_ocr as supplied
from speed_arrow import ARROW_HITS, SpeedArrowDetector


def overlap(a, b):
    if a is None or b is None:
        return 0.
    x = max(a['x'], b['x'])
    y = max(a['y'], b['y'])
    right = min(a['x'] + a['w'], b['x'] + b['w'])
    bottom = min(a['y'] + a['h'], b['y'] + b['h'])
    area = max(0, right-x) * max(0, bottom-y)
    return area / max(1., a['w']*a['h'] + b['w']*b['h'] - area)


def plain_box(box):
    return {k: int(box[k]) for k in ('x', 'y', 'w', 'h')} if box is not None else None


class SignRecognizer:
    def __init__(self):
        self.generation = None
        self.previous = {'stop': None, 'park': None}
        self.counts = {'stop': 0, 'park': 0}
        self.last_stamp = None
        self.arrow_detector = SpeedArrowDetector()
        self.arrow_direction = self.arrow_box = None
        self.arrow_count = 0

    def process(self, frame, stamp, generation, sequence):
        if generation != self.generation or (self.last_stamp is not None and stamp-self.last_stamp > 1.5):
            self.previous = {'stop': None, 'park': None}
            self.counts = {'stop': 0, 'park': 0}
            self.arrow_direction = self.arrow_box = None
            self.arrow_count = 0
        if stamp == self.last_stamp and generation == self.generation:
            raise ValueError('Repeated OCR frame timestamp')
        self.generation, self.last_stamp = generation, stamp
        image = cv2.resize(frame, (640, 480))
        direction, arrow_box, arrow_score = self.arrow_detector.detect(image)
        same_arrow = direction == self.arrow_direction and overlap(self.arrow_box, arrow_box) >= .30
        self.arrow_count = (min(ARROW_HITS, self.arrow_count+1) if same_arrow else 1) if direction else 0
        self.arrow_direction, self.arrow_box = direction, arrow_box
        hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
        red = cv2.bitwise_or(cv2.inRange(hsv, supplied.LOWER_RED_1, supplied.UPPER_RED_1),
                            cv2.inRange(hsv, supplied.LOWER_RED_2, supplied.UPPER_RED_2))
        blue = cv2.inRange(hsv, supplied.LOWER_BLUE, supplied.UPPER_BLUE)
        kernel = np.ones((5, 5), np.uint8)
        masks = []
        for mask in (red, blue):
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            masks.append(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel))
        stop_box = supplied.find_stop_candidate(masks[0])
        blue_box = supplied.find_blue_square_candidate(masks[1])
        park_box, p_shape = None, False
        if blue_box is not None:
            x, y, w, h = [blue_box[k] for k in ('x', 'y', 'w', 'h')]
            p_shape, _ = supplied.check_white_p_shape(image[y:y+h, x:x+w])
            if p_shape:
                park_box = blue_box
        sign_ocr_engine.clear_error()
        texts = {'stop': '', 'park': ''}
        for kind, box in [('stop', stop_box), ('park', park_box)]:
            if box is not None:
                x, y, w, h = [box[k] for k in ('x', 'y', 'w', 'h')]
                if kind == 'stop':
                    x1, y1, x2, y2 = supplied.expand_box(x, y, w, h, 640, 480, .04)
                    texts[kind], _ = supplied.run_stop_ocr(image[y1:y2, x1:x2])
                else:
                    texts[kind], _ = supplied.run_p_ocr(image[y:y+h, x:x+w])
            success = texts[kind] == ('STOP' if kind == 'stop' else 'P')
            same = overlap(self.previous[kind], box) >= .30
            self.counts[kind] = min(2, self.counts[kind]+1) if success and same else 1 if success else 0
            self.previous[kind] = plain_box(box)
        return dict(sequence=sequence, timestamp=stamp, result_mono=time.monotonic(),
                    generation=generation, error=sign_ocr_engine.last_error,
                    stop_present=stop_box is not None, park_present=blue_box is not None,
                    stop_confirmed=self.counts['stop'] >= 2,
                    park_confirmed=self.counts['park'] >= 2,
                    stop_count=self.counts['stop'], park_count=self.counts['park'],
                    stop_text=texts['stop'], park_text=texts['park'], p_shape=bool(p_shape),
                    stop_box=plain_box(stop_box), park_box=plain_box(park_box),
                    arrow_direction=direction, arrow_box=plain_box(arrow_box), arrow_score=arrow_score,
                    arrow_confirmed=self.arrow_count >= ARROW_HITS)


class SignWorker:
    def __init__(self, stop, interval=.20):
        self.stop, self.interval = stop, interval
        self.condition = threading.Condition()
        self.pending = None
        self.sequence = 0
        self.result = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def submit(self, frame, stamp, generation):
        with self.condition:
            self.sequence += 1
            self.pending = (frame, stamp, generation, self.sequence)
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return dict(self.result) if self.result is not None else None

    def run(self):
        recognizer = SignRecognizer()
        previous = 0
        while not self.stop.is_set():
            started = time.monotonic()
            with self.condition:
                self.condition.wait_for(lambda: self.stop.is_set() or self.sequence != previous, timeout=.1)
                if self.stop.is_set():
                    break
                if self.pending is None or self.sequence == previous:
                    continue
                frame, stamp, generation, previous = self.pending
            try:
                result = recognizer.process(frame, stamp, generation, previous)
            except Exception as exc:
                logging.exception('OCR worker failed')
                result = dict(sequence=previous, timestamp=stamp, generation=generation,
                              error=str(exc), stop_present=True, park_present=True,
                              stop_confirmed=False, park_confirmed=False)
            with self.condition:
                self.result = result
            self.stop.wait(max(0., self.interval - (time.monotonic()-started)))


def draw_signs(frame, result):
    if not result:
        return
    sx, sy = frame.shape[1]/640., frame.shape[0]/480.
    for kind, color in [('stop', (0, 0, 255)), ('park', (255, 180, 0)), ('arrow', (0, 220, 110))]:
        box = result.get(kind + '_box')
        if box:
            x, y, w, h = [box[k] for k in ('x', 'y', 'w', 'h')]
            cv2.rectangle(frame, (int(x*sx), int(y*sy)), (int((x+w)*sx), int((y+h)*sy)), color, 1)
            if kind == 'arrow':
                cv2.putText(frame, result.get('arrow_direction') or '', (int(x*sx), max(10, int(y*sy)-3)),
                            cv2.FONT_HERSHEY_SIMPLEX, .38, color, 1)
    label = 'OCR STOP:{}({}) P:{}({})'.format(result.get('stop_text', ''), result.get('stop_count', 0),
                                           result.get('park_text', ''), result.get('park_count', 0))
    cv2.putText(frame, label, (4, 78), cv2.FONT_HERSHEY_SIMPLEX, .32, (255, 255, 0), 1)
