CLEAN RETRAINED XY - 2026-10-25 - single camera test

This model was rebuilt from official ImageNet ResNet18 initialization and the
original dataset_xy_full_20261021 photo filenames/XY labels only.
No curve-review or path-stage annotations, V4 review model, or V5 model was loaded.
Rejected annotations/models/packages were moved to a quarantine directory locally.
Direct permanent deletion was blocked by automatic tool review.

Stop old car program using web STOP then Ctrl+C. On Jetson:
cd ~/car
unzip -o jetson_ready_clean_retrained_20261025.zip
bash run_clean_retrained_20261025.sh --record bend_clean_20261025.avi
Refresh http://JETSON_IP:5000. Confirm Clean XY 20261025 TEST.
Wait for fresh images/FPS before selecting AUTO for a low-speed trial.

This is a two-output XY model: X steers; Y is preview only (training weight 0.03).
There is no learned road-stage classifier, learned near/far pair, or dual-camera fusion.
When a new model target requests a smaller turn magnitude, the fresh-frame filter
uses 0.90 instead of 0.70 to reduce the remaining old turn more promptly.
The control page shows MODEL EASE TURN while doing so. Duplicate frames cannot
continue unwinding the turn; there is no bend duration or countdown.
This follows a smaller learned target, not a trained exit-stage detector.
If the model keeps predicting a large turn, this feature cannot identify the exit
by itself. It can also respond more strongly to a mistaken small-turn prediction.
Speed 0.12, left gain 0.70, right gain 0.78. ESP32 firmware/pins are unchanged.
Freshness and lost-tape STOP guards remain. No route clock controls bend timing.
The local package does not remove files or stop any program already on Jetson.
Using the new launcher selects the new uniquely named weights.

See training_summary.json for split, selected epoch and validation errors.
Validation selected the model; similar neighboring photos can still correlate.
This is not an independent driving test or a guarantee of improved track behavior.

RECORDING
Recording begins on launch, including MANUAL/STOP images.
Web STOP then Ctrl+C finalizes bend_clean_20261025.avi and its .avi.csv file.
Return both files.

THIRD-PERSON VIDEO
A phone on a fixed higher support can record the full bend, both lane boundaries,
vehicle body and its heading. Record the same trial with onboard raw video and CSV.
Include a shared start cue visible to both cameras if possible.
Keep actual start/stop under the operator's control; assistant review is not a
continuous real-time safety loop. No second camera is needed on Jetson for this view.
Camera pose/FOV changes or dual-camera model inputs require new matching data.
