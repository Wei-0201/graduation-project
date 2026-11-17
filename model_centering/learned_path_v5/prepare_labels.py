"""Build a two-distance target and bend-stage annotation page from raw videos."""
from pathlib import Path
import base64
import csv
import json

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'labeling'


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / 'images').mkdir(exist_ok=True)
    samples = []
    for version in (2, 3, 4):
        source = Path.home() / 'Downloads' / ('bend_exit_v{}.avi'.format(version))
        rows = list(csv.DictReader(Path(str(source) + '.csv').open()))
        groups = []
        for i, row in enumerate(rows):
            if row['sent_command_at_result'].startswith('DRIVE'):
                if not groups or i - groups[-1][-1] > 12:
                    groups.append([])
                groups[-1].append(i)
        cap = cv2.VideoCapture(str(source))
        for run, ids in enumerate(groups, 1):
            # Include exits; the human may reject already-departed/occluded scenes.
            positions = np.linspace(0, max(0, int(len(ids) * .97) - 1), 8).astype(int)
            for slot, pos in enumerate(positions):
                frame = ids[pos]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError('Cannot read {} frame {}'.format(source, frame))
                image = cv2.resize(image, (320, 240))
                name = '{}_run{}_f{}'.format(source.stem, run, frame)
                data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                (OUT / 'images' / (name + '.jpg')).write_bytes(data)
                samples.append(dict(id=name, source=source.name, run=run, frame=frame,
                                    slot=slot, group='{}_run{}'.format(source.stem, run),
                                    image='data:image/jpeg;base64,' + base64.b64encode(data).decode()))
        cap.release()
    samples.sort(key=lambda r: (r['slot'], r['source'], r['run']))
    (OUT / 'manifest.json').write_text(json.dumps([
        {k: v for k, v in r.items() if k != 'image'} for r in samples
    ], indent=2), encoding='utf-8')
    template = (ROOT / 'label_template.html').read_text(encoding='utf-8')
    (OUT / 'label_path_v5.html').write_text(
        template.replace('__SAMPLES__', json.dumps(samples)), encoding='utf-8')
    print('Prepared {} photos; two targets plus a human bend-stage label.'.format(len(samples)))


if __name__ == '__main__':
    main()
