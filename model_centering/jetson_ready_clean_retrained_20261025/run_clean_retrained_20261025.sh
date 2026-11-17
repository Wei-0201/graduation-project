#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python3 jetson_model_centering_clean_20261025.py \
 --model best_steering_model_xy_clean_20261025.pth \
 --port /dev/ttyACM0 --auto-speed 0.12 --visual-weight 0 \
 --left-gain 0.70 --right-gain 0.78 "$@"
