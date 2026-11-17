#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python3 jetson_model_centering_speed_test_full.py \
  --model best_steering_model_xy_pc_full_20epochs_20261021.pth \
  --port /dev/ttyACM0 \
  --auto-speed 0.12 "$@"
