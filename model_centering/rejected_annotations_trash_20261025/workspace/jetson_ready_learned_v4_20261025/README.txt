# Learned Centering V4 - 2026-10-25

## Run on Jetson
Stop the old car application using web STOP then Ctrl+C first.

    cd ~/car
    unzip -o jetson_ready_learned_v4_20261025.zip
    bash run_learned_centering_v4.sh --record bend_exit_v4.avi

Open http://JETSON_IP:5000 and refresh. Header: Jetson Car - Learned Centering V4.
Starts stopped. Use AUTO for a low-speed trial; speed defaults to 0.12.
No ESP32 firmware or pin changes are needed. Existing model files are preserved.

## Changes
The corrected ResNet18 XY model uses X for steering; Y is preview only.
Default visual-weight is 0: fitted lane geometry cannot override steering.
New-frame target smoothing reduces abrupt changes without timed corner scripts.
Five distinct fresh frames without continuous nearby tape evidence latch STOP.
Reposition on the track and click AUTO to clear that latch, or select MANUAL.
Missing/stale images and invalid predictions also stop the vehicle.
Tape evidence is a basic dark-ribbon check; foreign dark objects can resemble tape.

## Training and validation
108 new annotations: 79 training, 29 held out as two entire driving runs.
477 older height-matched labels retained for training; 150 for retention validation.
Selected epoch: 6. New-label validation X MAE: 68.42 -> 25.48 pixels (320px width).
Wrong direction on 25 clearly off-center validation targets: 8 -> 1.
Left-target error remains higher than right-target error.
Old height-matched validation X MAE: 8.68 -> 10.06 pixels.
Validation was used to select the checkpoint; these are not final independent test scores.
Physical driving improvement still needs a low-speed trial.
Weights are CPU tensor state_dict in legacy serialization for Jetson PyTorch 1.10.

## Recording
Recording starts when this command launches, including MANUAL/STOP frames.
Use web STOP then Ctrl+C to finalize bend_exit_v4.avi and bend_exit_v4.avi.csv.
Return BOTH files for diagnosis. Actual timing is in CSV, not fixed-rate AVI playback.
