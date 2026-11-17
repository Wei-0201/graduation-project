"""Prepare 240 additional, non-repeated frames, favoring human-readable regions."""
import base64
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'labeling_extra240'
sys.path.insert(0, str(ROOT.parent))
from jetson_model_centering_review_v4 import black_mask, has_tape_evidence


def readable_candidate(image):
    # Selection only, never a road-stage label or a steering target.
    mask, _ = black_mask(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blocked = np.mean(gray[160:235] < 65) > .32
    return has_tape_evidence(mask) and not blocked


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / 'images').mkdir(exist_ok=True)
    original = json.loads((ROOT / 'labeling/manifest.json').read_text(encoding='utf-8'))
    old = {r['id']: r for r in original}
    labels = json.loads((ROOT / 'labels_received_20261025.json').read_text(encoding='utf-8'))
    accepted, skipped = defaultdict(list), defaultdict(list)
    for r in labels['items']:
        accepted[old[r['id']]['group']].append(old[r['id']]['frame'])
    for key in labels['skipped']:
        skipped[old[key]['group']].append(old[key]['frame'])
    used_ids = set(old)
    used_pixels = set()
    # Compare decoded pixels as well as IDs; video compression can repeat frames.
    for r in original:
        image = cv2.imdecode(np.frombuffer((ROOT / 'labeling/images' / (r['id'] + '.jpg')).read_bytes(), np.uint8), 1)
        used_pixels.add(hashlib.sha256(image.tobytes()).hexdigest())
    samples, preview = [], []
    report = []
    for version in (2, 3, 4):
        source = Path.home() / 'Downloads' / ('bend_exit_v{}.avi'.format(version))
        rows = list(csv.DictReader(Path(str(source) + '.csv').open()))
        groups = []
        for i, r in enumerate(rows):
            if r['sent_command_at_result'].startswith('DRIVE'):
                if not groups or i - groups[-1][-1] > 12:
                    groups.append([])
                groups[-1].append(i)
        cap = cv2.VideoCapture(str(source))
        # Decode sequentially, not 240 expensive seeks through the whole video.
        wanted = set()
        bounds = []
        for run, ids in enumerate(groups, 1):
            lo, hi = max(0, ids[0] - 10), min(len(rows) - 1, ids[-1])
            bounds.append((run, lo, hi))
            wanted.update(range(lo, hi + 1))
        images = {}
        frame = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if frame in wanted:
                images[frame] = cv2.resize(image, (320, 240))
            frame += 1
        cap.release()
        # Add exact old decoded frame hashes to exclude duplicate camera frames.
        for r in original:
            if r['source'] == source.name and r['frame'] in images:
                used_pixels.add(hashlib.sha256(images[r['frame']].tobytes()).hexdigest())
        for run, lo, hi in bounds:
            group = '{}_run{}'.format(source.stem, run)
            good = sorted(accepted[group])
            bad = skipped[group]
            candidates = [f for f in range(lo, hi + 1)
                          if f in images and '{}_f{}'.format(group, f) not in used_ids
                          and readable_candidate(images[f])]
            picked = []

            def pick(pool, count, readable=True):
                for _ in range(count):
                    scored = []
                    for f in pool:
                        if f in picked:
                            continue
                        pixels = hashlib.sha256(images[f].tobytes()).hexdigest()
                        if pixels in used_pixels:
                            continue
                        separation = min([abs(f - p) for p in picked] or [hi - lo + 1])
                        if readable and good:
                            dist_good = min(abs(f - p) for p in good)
                            dist_bad = min([abs(f - p) for p in bad] or [999])
                            score = min(separation, 18) - .8 * dist_good + .12 * min(dist_bad, 15)
                        else:
                            score = separation
                        scored.append((score, f, pixels))
                    if not scored:
                        break
                    _, f, pixels = max(scored)
                    picked.append(f)
                    used_pixels.add(pixels)

            # Twelve frames near known-readable positions plus four covering later/gap scenes.
            pick(candidates, 12, readable=True)
            later = [f for f in candidates if not good or f > np.median(good)]
            pick(later, 4, readable=False)
            pick(candidates, 16 - len(picked), readable=False)
            if len(picked) != 16:
                raise RuntimeError('Not enough new unique frames in ' + group)
            for slot, f in enumerate(sorted(picked)):
                image = images[f]
                name = '{}_f{}'.format(group, f)
                data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                (OUT / 'images' / (name + '.jpg')).write_bytes(data)
                samples.append(dict(id=name, source=source.name, run=run, frame=f, slot=slot,
                                    group=group, image='data:image/jpeg;base64,' + base64.b64encode(data).decode()))
                if slot in (0, 5, 10, 15):
                    preview.append((name, image))
            report.append(dict(group=group, frames=sorted(picked)))
    assert len(samples) == 240 and len({r['id'] for r in samples}) == 240
    assert not used_ids.intersection(r['id'] for r in samples)
    samples.sort(key=lambda r: (r['slot'], r['source'], r['run']))
    (OUT / 'manifest.json').write_text(json.dumps([
        {k: v for k, v in r.items() if k != 'image'} for r in samples
    ], indent=2), encoding='utf-8')
    template = (ROOT / 'label_template.html').read_text(encoding='utf-8')
    template = template.replace('learned-path-stage-reset-after-rejection-20261025', 'learned-path-stage-extra240-reset-after-rejection-20261025')
    template = template.replace('path_stage_labels_20261025.json', 'path_stage_labels_extra240_20261025.json')
    template = template.replace('<h1>學習進彎、彎中與回到直線</h1>', '<h1>第二批 240 張：進彎、彎中與回到直線</h1>')
    template = template.replace('<details>', '<p class="note">這是另外 240 張，第一批標註已保留。只缺中央點時可不點；路段仍能判斷就選狀態。不能因為車子跑出去了，就當成「即將回直線」。</p><details>', 1)
    (OUT / 'label_path_extra240.html').write_text(template.replace('__SAMPLES__', json.dumps(samples)), encoding='utf-8')
    (OUT / 'selection_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    from PIL import Image, ImageDraw
    sheet = Image.new('RGB', (320 * 4, 263 * 15), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (name, image) in enumerate(preview):
        x, y = i % 4 * 320, i // 4 * 263
        sheet.paste(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), (x, y))
        draw.text((x + 3, y + 241), name, fill='black')
    sheet.save(OUT / 'selection_preview.jpg')
    print('Prepared 240 new frames in 15 runs; no prior frame IDs or identical decoded frames.')
    print(OUT / 'label_path_extra240.html')


if __name__ == '__main__':
    main()
