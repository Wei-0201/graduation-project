STRAIGHT-AHEAD BEND EXIT (2026-10-24)

This update uses the existing ResNet18 model and ESP32 firmware. It does not
memorize a timed route. On each fresh camera frame it looks for both tape
boundaries at four forward image rows. If they form a stable straight lane for
two frames, it gradually reduces the model's continuing turn, up to 60%.
Uncertain/one-sided views fall back to the model. A new bend or opposite turn
clears the old reduction. The control page shows STRAIGHT and FADE readings.

Transfer this ZIP to /home/tsai/car, then on the Jetson:
    cd /home/tsai/car
    unzip -o jetson_ready_straight_exit_20261024.zip
    bash run_straight_exit.sh

Before starting, stop the old Jetson program with Ctrl+C. The trained model
best_steering_model_xy_centering_y003.pth must already be in /home/tsai/car.
ESP32 port remains /dev/ttyACM0 and AUTO speed remains 0.12. Use the web page
STOP button immediately if the car approaches a lane edge or obstacle.

To record raw camera video while testing:
    bash run_straight_exit.sh --record bend_exit_raw.avi
Recording begins when the program launches. Press STOP on the page after the
test, then Ctrl+C in the terminal so the AVI file is finalized.

If the car straightens too early, try --straight-max-fade 0.35.
If the feature should be disabled for comparison, use --straight-max-fade 0.
The old Jetson program and model are not overwritten by this ZIP.
