#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f run_v4_sign_ocr.sh ]; then
  printf '%s
' 'Original V4 OCR package missing here. Extract this fix inside ~/car beside run_v4_sign_ocr.sh.'
  exit 1
fi
bash install_ocr_deps.sh
python3 check_traffic_ocr.py
exec bash run_v4_sign_ocr.sh --record traffic_v4_test.avi "$@"
