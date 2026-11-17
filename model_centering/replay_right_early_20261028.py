"""Offline comparison on the user's recorded frames; no serial or web startup."""
import csv
import json
import os
from pathlib import Path
import cv2
from jetson_v4_traffic_signs import CommandPolicy, black_mask
from sign_parking_control import RightCorridorTracker

root = Path('video_diagnostics/right_early_20261028')
root.mkdir(exist_ok=True)
source = Path(r'C:\Users\USER\Downloads\right_early.avi')
rows = list(csv.DictReader(open(str(source)+'.csv')))
cap = cv2.VideoCapture(str(source))
tracker = RightCorridorTracker()
old, new = CommandPolicy(), CommandPolicy()
results = []
for row in rows:
    ok, image = cap.read()
    if not ok:
        break
    corridor = tracker.process(black_mask(cv2.resize(image,(320,240)))[0])
    now = float(row['capture_mono'])
    raw = float(row['model_turn'])
    mode = 'MANUAL' if row['traffic_state'] == 'MANUAL' else 'AUTO'
    detection = dict(timestamp=now, valid=abs(raw)<=1.05, turn=raw,
                     lane_visible=row['lane_visible']=='True', parking_corridor=corridor)
    os.environ['CAR_RIGHT_ENTRY_GUARD'] = '0'
    before = old.command(mode, 'STOP', now, detection, now)
    os.environ['CAR_RIGHT_ENTRY_GUARD'] = '1'
    after = new.command(mode, 'STOP', now, detection, now)
    results.append(dict(frame=row['frame'], raw=raw, before=before, after=after,
                        active=new.right_entry.active, geometry=corridor))
cap.release()
active = [r for r in results if r['active'] and r['after'].startswith('DRIVE')]
summary = dict(frames=len(results), eased_frames=len(active),
               first_eased=[{k:r[k] for k in ('frame','raw','before','after')} for r in active[:8]],
               limitations='Recorded images only: does not predict the vehicle path after different commands. No Jetson hardware test.')
(root/'replay.json').write_text(json.dumps(dict(summary=summary,frames=results),indent=2),encoding='utf8')
print(json.dumps(summary,indent=2))
