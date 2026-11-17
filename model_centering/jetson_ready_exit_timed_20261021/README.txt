CAMERA-RATE-INDEPENDENT BEND EXIT CONTROL

This package uses the existing lane-centering model and ESP32 firmware.
The default right-bend exit damping is increased from 0.60 to 0.85.
Damping now scales with the time between new camera frames, so disabling the
display-only line fitting does not weaken the exit correction at higher FPS.
The left-bend exit damping stays at 0.85.

Stop the previous Jetson program with Ctrl+C, then:
    cd /home/tsai/car
    unzip -o jetson_ready_exit_timed_20261021.zip
    bash run_exit_timed.sh

On the existing page at http://JETSON_IP:5000, choose AUTO for a low-speed
test. Default port is /dev/ttyACM0 and speed is 0.12.
MODEL TURN is the model's raw command; APPLIED TURN is the sent command.
The blue dashed preview lines are off by default to keep frames fresh.

If the car straightens too early at the end of right bends, try:
    bash run_exit_timed.sh --exit-damping 0.60
If it still turns too far right at the exit, try:
    bash run_exit_timed.sh --exit-damping 1.00

The model, ESP32 firmware, and existing Jetson scripts are not overwritten.
