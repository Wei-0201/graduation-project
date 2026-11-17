#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python3 jetson_model_centering_review_v41.py \
 --model best_steering_model_xy_corrected_v4_20261025.pth \
 --port /dev/ttyACM0 --auto-speed 0.12 --visual-weight 0 --steering-gain 0.70 "$@"
