This package contains the best checkpoint from the full local training run.
Training stopped after epoch 19. Best validation loss: 0.00771.
Valid images: 7729. Unreadable/invalid: 223.

On Jetson, extract all files into /home/tsai/car and run:
  cd /home/tsai/car
  bash run_model_full.sh

Open the existing car control page at http://JETSON_IP:5000 and enable AUTO
only after checking manual steering and wheel direction. Stop with Ctrl+C.
Default serial port is /dev/ttyACM0 and AUTO speed is 0.12.
Override either setting by editing run_model_full.sh or pass supported
command-line flags after the script name.
