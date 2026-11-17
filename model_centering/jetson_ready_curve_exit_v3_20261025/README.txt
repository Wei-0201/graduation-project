CURVE EXIT V3 - 2026-10-25

Uses the existing best_steering_model_xy_centering_y003.pth checkpoint and
ESP32 firmware. Put these update files beside the checkpoint in /home/tsai/car.

The supplied second recording contains four AUTO attempts: two right turns,
then two left turns. In the last left turn, the old straight-only detector
did not assist any of the 95 moving-frame samples. The model also changed
steering direction while the visible road continued left.

V3 adds a forward corridor detector for curved tape borders. It checks tape
pairs across three forward image rows and tracks a known border when its
partner goes out of view. The forward target uses the farther supported row
or a short local projection, instead of the previous near-row center. The
original straight-border detector remains available as a fallback. Fresh
steering changes are smoothed, including model-only intervals. Maximum
normalized steering remains available. There is no route schedule/timer.

Stop the previous program with web STOP, then terminal Ctrl+C. On Jetson:
    cd /home/tsai/car
    unzip -o jetson_ready_curve_exit_v3_20261025.zip
    bash run_curve_exit_v3.sh --record bend_exit_v3.avi

Refresh http://JETSON_IP:5000. The title should say Curve Exit V3.
Magenta = model target; cyan = visual target. CURVE PAIR, TRACK LEFT and
TRACK RIGHT identify the camera evidence. NONE uses the model. BRIEF GAP
bridges at most two missing images. Default AUTO speed is 0.12.

Press STOP if approaching a lane edge. After the test press STOP then Ctrl+C
to close bend_exit_v3.avi and bend_exit_v3.avi.csv. Recording starts at launch.
The CSV has actual timestamps; the AVI is encoded at 30 FPS and its playback
speed may differ from real driving speed.

Validation: 18 geometry/control tests, Python 3.6 syntax check, and offline
control replay of 416 moving/context frames from all four supplied attempts.
The recorded wrong-way model output in the last left bend is corrected in
offline command replay. Changed commands change the vehicle path, so this
does not establish successful physical driving. Test both directions slowly.
