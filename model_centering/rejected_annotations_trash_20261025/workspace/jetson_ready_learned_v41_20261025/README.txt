V4.1 STEERING CALIBRATION TRIAL
Requires the corrected V4 model already installed in ~/car:
best_steering_model_xy_corrected_v4_20261025.pth
No weights or ESP32 firmware included or changed.

Stop old program using web STOP then Ctrl+C. On Jetson:
cd ~/car
unzip -o jetson_ready_learned_v41_20261025.zip
bash run_learned_centering_v41.sh --record bend_exit_v41.avi
Refresh control page; confirm Learned Centering V4.1. Select AUTO for a low-speed trial.

Only control change: model turn scale 1.00 -> 0.70, both directions.
Same speed, smoothing, freshness checks and missing-tape STOP behavior as V4.
Uses image predictions, not route timing. Does not delay the start of a turn.
Does not correct biased model target positions; it only reduces commanded curvature.
Lower gain may understeer on tight bends or recover more slowly from an offset.
Do not assume successful driving from offline replay; verify a short left/right trial.

14 tests passed (11 existing behavior tests at gain=1 and 3 gain tests).
Recorded driving-frame replay confirms consistent 0.70 scaling.
This replay cannot show the changed physical trajectory.

Stop with web STOP then Ctrl+C; return bend_exit_v41.avi plus bend_exit_v41.avi.csv.
To restore old V4: bash run_learned_centering_v4.sh
To test the original gain using this runner: append --steering-gain 1.0
