"""Local synthetic perception timing; not a measurement of Jetson Nano FPS."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
import jetson_centering_v7 as old
import jetson_centering_v8 as new
from test_centering_v8 import road

cv2.setNumThreads(2)
frames = [road(s) for s in (-20, 0, 20)]
crowded = road()
for x in range(10, 320, 19):
    cv2.line(crowded, (x, 30), (x, 105), (15, 15, 15), 3)
frames.append(crowded)
report = {}
for module in (old, new):
    samples = []
    for i in range(240):
        detector = module.LaneDetector()
        start = time.perf_counter()
        detector.process(frames[i % len(frames)], 1)
        elapsed = (time.perf_counter() - start) * 1000
        if i >= 40:
            samples.append(elapsed)
    report[module.__name__] = dict(median_ms=float(np.median(samples)),
                                   p95_ms=float(np.percentile(samples, 95)))
text = json.dumps(report, indent=2)
Path(__file__).with_name('v8_local_benchmark.json').write_text(text, encoding='utf-8')
print(text)
