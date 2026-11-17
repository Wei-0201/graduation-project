V5.1 CONTROL TEST - uses existing learned V5 near/far model
Requires learned_path_v5_20261025.pth and its matching .json already in ~/car.
This is a controller update, not a retrained model. No ESP32 firmware changes.

Stop old program with web STOP then Ctrl+C. On Jetson:
cd ~/car
unzip -o jetson_ready_learned_path_v51_20261025.zip
bash run_learned_path_v51.sh --record bend_exit_v51.avi
Refresh the page; confirm Learned Path V5.1 TEST, wait for frames, test at speed 0.12.

Why: supplied V5 run updated about 14 FPS, while preview weight remained zero.
The hard 75% active-stage gate prevented all far-point assistance during the runs.
V5.1 uses a bounded learned heading preview before/within the bend, with confidence
adjusting weight continuously. No route clock or fixed turn duration is used.
Maximum added preview is 0.14 before left/right gain and fresh-frame smoothing.
A strong opposite near correction (abs >=0.25) keeps priority during lane recovery.
Speed and output gains remain unchanged (left 0.70, right 0.78).

This may start steering sooner and can also cause early steering if far prediction
is wrong. Replay verifies the changed calculation only, not the physical trajectory.
20 tests passed, including existing STOP/stale-frame/lost-tape behavior.
The supplied V5 video contains right-bend trials; left-bend timing is not verified.
The exit_straight class remains untrained and excluded.
No weight files are overwritten. Return bend_exit_v51.avi AND bend_exit_v51.avi.csv
using web STOP then Ctrl+C to finalize recording.
Rollback: bash run_learned_path_v5.sh
