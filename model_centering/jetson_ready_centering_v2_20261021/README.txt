LANE CENTERING V2 - Jetson package

Model: best_steering_model_xy_centering_y003.pth
Architecture: ResNet18 with a 2-number XY output, compatible with Jetson PyTorch 1.10.
Training: 7729 valid saved photos; 223 unreadable/invalid photos skipped.
Y loss weight: 0.03 relative to X. Best held-out X MAE: 8.52 image pixels.

Copy this ZIP to /home/tsai/car on Jetson. Then run:
    cd /home/tsai/car
    unzip -o jetson_ready_centering_v2_20261021.zip
    bash run_centering_v2.sh

The runner serves the existing car control page on port 5000.
AUTO remains off until selected on the page. Use Ctrl+C to stop.
Default serial port: /dev/ttyACM0. Default AUTO speed: 0.12.

This package uses unique model and runner names and leaves existing files intact.
Offline photo validation cannot establish real track performance; first test
with the wheels lifted and an operator ready to stop the car.
