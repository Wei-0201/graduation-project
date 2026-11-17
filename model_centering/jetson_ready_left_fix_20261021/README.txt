LEFT-BEND CONTROL UPDATE FOR JETSON

The model remains best_steering_model_xy_centering_y003.pth.
ESP32 firmware and motor calibration are unchanged.

Changes:
* CPU-intensive preview tape fitting is off by default. The blue dashed
  lines vanish; they never controlled AUTO. Add --show-lines to enable them.
* A fresh model frame may be up to 0.45 seconds old before AUTO stops.
  AUTO still stops on an invalid or older frame. Recovery after a brief
  dropout is quicker once fresh predictions return.
* Bend-exit damping is 0.85 on left turns and 0.60 on right turns. This only
  reduces the command while the model asks to straighten the wheels.
* The control page reports MODEL TURN and APPLIED TURN separately.

Install on Jetson:
    cd /home/tsai/car
    unzip -o jetson_ready_left_fix_20261021.zip
    bash run_left_fix.sh

Stop any previous car runner with Ctrl+C before starting this one.
Default serial port is /dev/ttyACM0; AUTO speed is 0.12.
To reduce left exit damping: bash run_left_fix.sh --left-exit-damping 0.60
To turn the preview lines back on: bash run_left_fix.sh --show-lines

Test at low speed with someone ready to press STOP. If the model target dot
is outside the real lane during a left bend, gather more labeled photos of
that specific bend; motor tuning cannot correct a wrong target point.
