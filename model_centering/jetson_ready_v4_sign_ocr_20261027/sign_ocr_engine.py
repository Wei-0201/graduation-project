"""Bounded local Tesseract CLI adapter for the supplied image_to_string calls."""
import os
import shlex
import shutil
import subprocess
import glob

import cv2
import numpy as np

TIMEOUT = .60
last_error = None
_resolved = None


def clear_error():
    global last_error
    last_error = None


def executable():
    command = os.environ.get('TESSERACT_CMD') or shutil.which('tesseract')
    if not command:
        raise RuntimeError('Tesseract missing. Run: bash install_ocr_deps.sh')
    return command


def child_environment():
    env = os.environ.copy()
    # An inherited prefix may point at a different Tesseract version or be relative.
    # Explicit --tessdata-dir below is authoritative; leave the user's shell unchanged.
    env.pop('TESSDATA_PREFIX', None)
    env['OMP_THREAD_LIMIT'] = '1'
    return env


def data_directories(binary):
    paths = []
    for variable in ('CAR_TESSDATA_DIR', 'TESSDATA_PREFIX'):
        value = os.environ.get(variable)
        if value:
            paths.extend([value, os.path.join(value, 'tessdata')])
    paths.extend(glob.glob('/usr/share/tesseract-ocr/*/tessdata'))
    paths.extend(['/usr/share/tesseract-ocr/tessdata', '/usr/share/tessdata',
                  '/usr/local/share/tessdata', os.path.join(os.path.dirname(binary), 'tessdata'),
                  os.path.join(os.path.dirname(os.path.dirname(binary)), 'share', 'tessdata')])
    valid = []
    for path in paths:
        path = os.path.realpath(os.path.expanduser(path))
        model = os.path.join(path, 'eng.traineddata')
        if path not in valid and os.path.isfile(model) and os.access(model, os.R_OK):
            if os.path.getsize(model) >= 1024:
                valid.append(path)
    return valid


def command_prefix(binary, directory):
    return [binary, 'stdin', 'stdout', '-l', 'eng'] + (['--tessdata-dir', directory] if directory else [])


def engine_error(result):
    text = result.stderr.decode('utf-8', errors='replace').lower()
    return result.returncode != 0 or any(marker in text for marker in (
        'failed loading language', 'could not initialize tesseract',
        "couldn't load any languages", 'error opening data file'))


def check_available(force=False):
    global _resolved
    if _resolved is not None and not force:
        return 'Tesseract ready: {} ; English data: {}'.format(_resolved[0], _resolved[1] or 'engine default')
    binaries = [executable()]
    # A custom /usr/local build may coexist with the Ubuntu package binary.
    if os.path.isfile('/usr/bin/tesseract') and '/usr/bin/tesseract' not in binaries:
        binaries.append('/usr/bin/tesseract')
    sample = np.full((80, 240), 255, np.uint8)
    cv2.putText(sample, 'TEST', (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
    ok, data = cv2.imencode('.png', sample)
    if not ok:
        raise RuntimeError('Cannot create English-data loading check')
    failures = []
    for binary in binaries:
        for directory in data_directories(binary) + [None]:
            try:
                result = subprocess.run(command_prefix(binary, directory) + ['--psm', '7'],
                                        input=data.tobytes(), stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, timeout=8, env=child_environment())
                if not engine_error(result):
                    _resolved = (binary, directory)
                    return 'Tesseract English model loaded: {} ; data: {}'.format(binary, directory or 'engine default')
                failures.append(result.stderr.decode('utf-8', errors='replace')[-200:])
            except (OSError, subprocess.TimeoutExpired) as exc:
                failures.append(str(exc)[-200:])
    _resolved = None
    raise RuntimeError('Cannot load eng.traineddata. Run bash install_ocr_deps.sh. Last error: ' +
                       (failures[-1] if failures else 'no readable English model found'))


def prepare_text(image, config):
    """Normalize supplied binary masks; keep one bounded OCR call per sign."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if 'whitelist=STOP' in config:
        # STOP mask is white lettering on black. Remove bright scenery / sign
        # borders connected to the ROI edge before reversing polarity.
        count, labels, stats, _ = cv2.connectedComponentsWithStats((gray > 127).astype(np.uint8))
        keep = np.zeros_like(gray)
        h, w = gray.shape
        for i in range(1, count):
            x, y, cw, ch, area = stats[i]
            if x > 2 and y > 2 and x+cw < w-2 and y+ch < h-2 and area >= 8:
                keep[labels == i] = 255
        gray = 255-keep
    elif np.median(np.r_[gray[0], gray[-1], gray[:, 0], gray[:, -1]]) < 128:
        gray = 255-gray
    ys, xs = np.where(gray < 128)
    if not len(xs):
        return gray, config
    gray = gray[ys.min():ys.max()+1, xs.min():xs.max()+1]
    gray = cv2.resize(gray, (max(1, int(round(gray.shape[1]*48./gray.shape[0]))), 48),
                      interpolation=cv2.INTER_AREA)
    gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    # Recognize all letters, then require exact STOP/P in SignRecognizer.
    # A single allowed glyph can otherwise force/suppress classifier output.
    config = config.replace('whitelist=STOP', 'whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    config = config.replace('whitelist=P', 'whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return gray, config


def image_to_string(image, config=''):
    global last_error
    try:
        image, config = prepare_text(image, config)
        ok, data = cv2.imencode('.png', image)
        if not ok:
            raise RuntimeError('Cannot encode OCR ROI')
        binary, directory = _resolved if _resolved is not None else (executable(), None)
        result = subprocess.run(command_prefix(binary, directory) + shlex.split(config),
                                input=data.tobytes(), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=TIMEOUT, env=child_environment())
        if engine_error(result):
            raise RuntimeError(result.stderr.decode('utf-8', errors='replace')[-240:])
        return result.stdout.decode('utf-8', errors='replace')
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        last_error = str(exc)
        raise
