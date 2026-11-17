LANE CENTERING - BEND EXIT CONTROL

Uses the previously trained best_steering_model_xy_centering_y003.pth model.
Only the Jetson steering command is changed. ESP32 firmware is unchanged.

The model's steering direction and stronger bend-entry turns stay as before.
When a new camera prediction shows the same-direction turn decreasing, the
Jetson sends a smaller turn to compensate for vehicle response lag. There is
no fixed steering cap. The default damping amount is 0.60; 0 disables it.
The control page shows MODEL TURN and APPLIED TURN for comparison.

Copy the ZIP to /home/tsai/car on Jetson. Then:
    cd /home/tsai/car
    unzip -o jetson_ready_centering_exit_20261021.zip
    bash run_centering_exit.sh

AUTO is selected on the existing page at http://JETSON_IP:5000.
The default serial port is /dev/ttyACM0; AUTO speed is 0.12.
To try less exit damping: bash run_centering_exit.sh --exit-damping 0.30
To revert to the old turn behavior: bash run_centering_exit.sh --exit-damping 0
Press Ctrl+C to stop. Existing Jetson model and runner names are untouched.
