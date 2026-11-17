V4 right entry image guard 2026-10-28
Preserves V4 model weights, STOP 5 second pause, P parking/return, and ESP32 wiring. No model training or rejected labels used.
Based on right_early.avi and .csv: two AUTO runs begin with about -0.2 model right turn while near observed paired corridor is nearly straight, position ~0.55 (already right of center).
Requires two fresh paired observations; no estimated single border/remembered lane width is used for this guard. Limits premature right request according to current lane position. Fades as near-heading changes from -0.03 to -0.09; unchanged at/below -0.09. Left turns unchanged. Missing/invalid geometry falls back to existing V4 with original stale-frame, no-tape and OCR protections. No route delay timer.
Status MODEL RIGHT ENTRY EASE identifies activation. CAR_RIGHT_ENTRY_GUARD=0 bash run_v4_sign_ocr.sh disables only this guard for comparison.
Verification: 64 regression tests pass; Python 3.6 syntax checked; 447 captured frames replayed, 22 frames eased. Representative command -0.2087 -> -0.0617. This is offline command replay, not a measured success rate or closed-loop path prediction. Jetson low-speed test required.
Install after stopping old process: cd ~/car; cp -n jetson_v4_traffic_signs.py jetson_v4_traffic_signs.before_right_entry_20261028.py; unzip -o jetson_v4_right_entry_20261028.zip; bash run_v4_sign_ocr.sh --record right_entry_v2.avi
Rollback after stopping: cp jetson_v4_traffic_signs.before_right_entry_20261028.py jetson_v4_traffic_signs.py; bash run_v4_sign_ocr.sh
