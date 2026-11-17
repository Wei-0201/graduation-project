#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python3 jetson_v4_traffic_signs.py \
 --model best_steering_model_xy_v4_ocr_20261027.pth \
 --port /dev/ttyACM0 --auto-speed 0.12 --visual-weight 0 \
 --stop-seconds 5 --park-target 1.0 --park-timeout 15 "$@"
