#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python3 jetson_model_centering_speed_test_v2.py \
  --model best_steering_model_xy_centering_y003.pth \
  --port /dev/ttyACM0 \
  --auto-speed 0.12 "$@"
