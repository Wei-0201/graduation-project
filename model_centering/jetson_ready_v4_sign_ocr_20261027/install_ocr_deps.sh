#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ready=0
if python3 - <<'PY'
import sys
import sign_ocr_engine
try:
    print(sign_ocr_engine.check_available(force=True))
except Exception as exc:
    print('OCR needs repair:', exc)
    sys.exit(1)
PY
then
  ready=1
fi
if [ "$ready" -eq 0 ]; then
  apt_prefix=()
  if [ "$(id -u)" -ne 0 ]; then apt_prefix=(sudo); fi
  # Repair using the existing package index first. A missing GUI icon index
  # must not prevent installation of the OCR engine and its English data.
  if ! "${apt_prefix[@]}" apt-get install --reinstall -y tesseract-ocr tesseract-ocr-eng; then
    "${apt_prefix[@]}" apt-get \
      -o Acquire::IndexTargets::deb::DEP-11::DefaultEnabled=false \
      -o Acquire::IndexTargets::deb::DEP-11-icons-small::DefaultEnabled=false \
      -o Acquire::IndexTargets::deb::DEP-11-icons::DefaultEnabled=false \
      -o Acquire::IndexTargets::deb::DEP-11-icons-hidpi::DefaultEnabled=false \
      -o Acquire::IndexTargets::deb::DEP-11-icons-large::DefaultEnabled=false \
      -o Acquire::IndexTargets::deb::DEP-11-icons-large-hidpi::DefaultEnabled=false update
    "${apt_prefix[@]}" apt-get install --reinstall -y tesseract-ocr tesseract-ocr-eng
  fi
fi
python3 - <<'PY'
import cv2, numpy
from PIL import Image
import serial
from flask import Flask
import sign_ocr_engine
print(sign_ocr_engine.check_available(force=True))
print('OCR ready: English data was loaded in a real OCR invocation.')
PY
