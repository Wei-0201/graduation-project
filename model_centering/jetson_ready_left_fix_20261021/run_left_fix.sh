#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python3 jetson_model_centering_left_fix.py \
  --model best_steering_model_xy_centering_y003.pth \
  --port /dev/ttyACM0 \
  --auto-speed 0.12 \
  --exit-damping 0.60 \
  --left-exit-damping 0.85 "$@"
