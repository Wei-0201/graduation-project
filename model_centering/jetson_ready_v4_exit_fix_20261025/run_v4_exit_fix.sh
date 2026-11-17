#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python3 jetson_model_centering_v4_exit_fix.py \
 --model best_steering_model_xy_v4_exit_20261025.pth \
 --port /dev/ttyACM0 --auto-speed 0.12 --visual-weight 0 "$@"
