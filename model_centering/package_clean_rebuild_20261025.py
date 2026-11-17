"""Package the completed clean retrain without any rejected annotation artifacts."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import torch

ROOT = Path(__file__).resolve().parent


def main():
    weight = ROOT / 'best_steering_model_xy_clean_20261025.pth'
    report = json.loads(weight.with_suffix('.report.json').read_text(encoding='utf-8'))
    if not report.get('stop_reason'):
        raise RuntimeError('Training has not completed')
    if report['rejected_annotation_inputs'] or 'ImageNet' not in report['initialization']:
        raise RuntimeError('Unexpected training provenance')
    if set(report['training_hashes']).intersection(report['validation_hashes']):
        raise RuntimeError('Image-content leakage across split')
    state = torch.load(str(weight), map_location='cpu')
    assert tuple(state['fc.weight'].shape) == (2, 512)
    assert all(v.device.type == 'cpu' and torch.isfinite(v).all() for v in state.values())
    assert weight.read_bytes()[:2] != b'PK'
    folder = ROOT / 'jetson_ready_clean_retrained_20261025'
    folder.mkdir(exist_ok=True)
    shutil.copy2(weight, folder / weight.name)
    for name in ['jetson_model_centering_clean_20261025.py', 'visual_exit_guard.py', 'lane_corridor_guard.py']:
        source = ROOT / name
        ast.parse(source.read_text(encoding='utf-8'), feature_version=6)
        shutil.copy2(source, folder / name)
    launcher = '''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python3 jetson_model_centering_clean_20261025.py \\
 --model best_steering_model_xy_clean_20261025.pth \\
 --port /dev/ttyACM0 --auto-speed 0.12 --visual-weight 0 \\
 --left-gain 0.70 --right-gain 0.78 "$@"
'''
    (folder / 'run_clean_retrained_20261025.sh').write_bytes(launcher.encode())
    summary = {k: report[k] for k in ['schema', 'initialization', 'allowed_source',
        'rejected_annotation_inputs', 'training_images', 'validation_images',
        'invalid_images', 'y_weight', 'best_epoch', 'best_metrics', 'stop_reason', 'note']}
    summary.update(model_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(), hardware_tested=False)
    (folder / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    readme = '''CLEAN RETRAINED XY - 2026-10-25 - single camera test

This model was rebuilt from official ImageNet ResNet18 initialization and the
original dataset_xy_full_20261021 photo filenames/XY labels only.
No curve-review or path-stage annotations, V4 review model, or V5 model was loaded.
Rejected annotations/models/packages were moved to a quarantine directory locally.
Direct permanent deletion was blocked by automatic tool review.

Stop old car program using web STOP then Ctrl+C. On Jetson:
cd ~/car
unzip -o jetson_ready_clean_retrained_20261025.zip
bash run_clean_retrained_20261025.sh --record bend_clean_20261025.avi
Refresh http://JETSON_IP:5000. Confirm Clean XY 20261025 TEST.
Wait for fresh images/FPS before selecting AUTO for a low-speed trial.

This is a two-output XY model: X steers; Y is preview only (training weight 0.03).
There is no learned road-stage classifier, learned near/far pair, or dual-camera fusion.
When a new model target requests a smaller turn magnitude, the fresh-frame filter
uses 0.90 instead of 0.70 to reduce the remaining old turn more promptly.
The control page shows MODEL EASE TURN while doing so. Duplicate frames cannot
continue unwinding the turn; there is no bend duration or countdown.
This follows a smaller learned target, not a trained exit-stage detector.
If the model keeps predicting a large turn, this feature cannot identify the exit
by itself. It can also respond more strongly to a mistaken small-turn prediction.
Speed 0.12, left gain 0.70, right gain 0.78. ESP32 firmware/pins are unchanged.
Freshness and lost-tape STOP guards remain. No route clock controls bend timing.
The local package does not remove files or stop any program already on Jetson.
Using the new launcher selects the new uniquely named weights.

See training_summary.json for split, selected epoch and validation errors.
Validation selected the model; similar neighboring photos can still correlate.
This is not an independent driving test or a guarantee of improved track behavior.

RECORDING
Recording begins on launch, including MANUAL/STOP images.
Web STOP then Ctrl+C finalizes bend_clean_20261025.avi and its .avi.csv file.
Return both files.

THIRD-PERSON VIDEO
A phone on a fixed higher support can record the full bend, both lane boundaries,
vehicle body and its heading. Record the same trial with onboard raw video and CSV.
Include a shared start cue visible to both cameras if possible.
Keep actual start/stop under the operator's control; assistant review is not a
continuous real-time safety loop. No second camera is needed on Jetson for this view.
Camera pose/FOV changes or dual-camera model inputs require new matching data.
'''
    (folder / 'README.txt').write_text(readme, encoding='utf-8')
    archive = ROOT / (folder.name + '.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for source in sorted(folder.iterdir()):
            z.write(source, source.name)
    with zipfile.ZipFile(archive) as z:
        assert hashlib.sha256(z.read(weight.name)).hexdigest() == summary['model_sha256']
        assert b'\r' not in z.read('run_clean_retrained_20261025.sh')
    print('Packaged ' + archive.name)
    print(json.dumps(summary['best_metrics']))


if __name__ == '__main__':
    main()
