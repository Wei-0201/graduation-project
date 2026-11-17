"""Verify the actual local Tesseract engine without opening serial or driving."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import sign_ocr_engine
from traffic_sign_worker import SignRecognizer


def synthetic_signs():
    image = np.full((480, 640, 3), 190, np.uint8)
    center, radius = np.array([160, 180]), 100
    angles = np.arange(8)*np.pi/4+np.pi/8
    poly = np.rint(center+np.c_[np.cos(angles), np.sin(angles)]*radius).astype(np.int32)
    cv2.fillConvexPoly(image, poly, (0, 0, 220))
    cv2.putText(image, 'STOP', (83, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 5)
    cv2.rectangle(image, (400, 100), (560, 260), (220, 0, 0), -1)
    cv2.putText(image, 'P', (440, 225), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (255, 255, 255), 9)
    return image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, help='Use a real sign image instead of the generated check image')
    args = p.parse_args()
    print(sign_ocr_engine.check_available())
    # Allow a little extra cold-start time during this stationary engine check.
    sign_ocr_engine.TIMEOUT = 3.
    if args.image:
        image = cv2.imdecode(np.frombuffer(args.image.read_bytes(), np.uint8), 1)
        if image is None:
            p.error('Cannot read input image')
    else:
        image = synthetic_signs()
    recognizer = SignRecognizer()
    result = None
    for sequence in (1, 2):
        # Static fixture frame stamps: cold OCR startup here is not live driving timing.
        result = recognizer.process(image, 1000.+sequence*.1, 0, sequence)
        print(json.dumps({k: result.get(k) for k in ['stop_text', 'park_text', 'p_shape',
              'stop_confirmed', 'park_confirmed', 'error']}))
    if result.get('error'):
        raise SystemExit('OCR engine error; do not start AUTO until corrected.')
    if not args.image and not (result['stop_confirmed'] and result['park_confirmed']):
        raise SystemExit('Generated sign check was not fully recognized; verify real signs/camera before AUTO.')


if __name__ == '__main__':
    main()
