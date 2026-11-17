#!/usr/bin/env python3
"""Click a target on the live camera image to save an XY training example.

Requires a graphical desktop. No serial port or motor control is used.
"""
import argparse
import os
import time
import uuid

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--output', default='dataset_xy')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        parser.error('Cannot open camera {}'.format(args.camera))

    current = [None]
    count = [0]

    def save_target(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN or current[0] is None:
            return
        raw = current[0].copy()
        height, width = raw.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return
        name = 'xy_{}_{}_{}.jpg'.format(x, y, uuid.uuid4().hex[:12])
        path = os.path.join(args.output, name)
        if cv2.imwrite(path, raw):
            count[0] += 1
            print('{} saved: {}'.format(count[0], path), flush=True)

    cv2.namedWindow('Click road target; Q to quit', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Click road target; Q to quit', save_target)
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print('Camera frame unavailable; retrying...', flush=True)
                time.sleep(0.1)
                continue
            raw = cv2.resize(frame, (args.width, args.height))
            current[0] = raw
            preview = raw.copy()
            cv2.putText(preview, 'Click target | Q quit | saved {}'.format(count[0]),
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            cv2.imshow('Click road target; Q to quit', preview)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print('Saved {} images in {}'.format(count[0], os.path.abspath(args.output)))


if __name__ == '__main__':
    main()
