#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python3 jetson_model_learned_path_v51.py \
 --model learned_path_v5_20261025.pth \
 --port /dev/ttyACM0 --auto-speed 0.12 \
 --left-gain 0.70 --right-gain 0.78 --visual-weight 0 "$@"
