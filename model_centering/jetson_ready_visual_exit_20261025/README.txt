VISUAL BEND EXIT UPDATE - 2026-10-25

Uses the existing ResNet18 checkpoint and ESP32 firmware. The checkpoint
best_steering_model_xy_centering_y003.pth must already exist in /home/tsai/car.

The camera geometry can now find two lane borders outside the image center.
After both borders are confirmed, a continuously observed single border can
retain its known LEFT/RIGHT identity. This handles the supplied clip where
the model target eventually falls outside the right border. Thin floor
joints and pairs without shared support are rejected.

Geometry and model targets are combined, with gradual steering changes per
new image. Repeated motor writes do not accumulate changes. The old rule that
weakens every decreasing same-direction command has been removed, because it
can also weaken corrective countersteering. There is no timed route schedule.

In the preview, magenta is the model target; cyan is the observed lane target;
green segments are observed tape fits. VISUAL PAIR / KNOWN LEFT / KNOWN RIGHT
shows which lane evidence is available. NONE uses the model prediction.

1. Stop the running program: press STOP on the web page, then Ctrl+C in its terminal.
2. Transfer the ZIP to /home/tsai/car, then on the Jetson:
     cd /home/tsai/car
     unzip -o jetson_ready_visual_exit_20261025.zip
     bash run_visual_exit.sh --record bend_exit_v2.avi
3. Open http://JETSON_IP:5000 and refresh. The title includes Visual Exit.
4. Use AUTO for a low-speed test. Press STOP if the car approaches a lane edge.
5. Press Ctrl+C after STOP to finalize the video and CSV diagnostic file.

Recording begins at program launch. It creates bend_exit_v2.avi and
bend_exit_v2.avi.csv. The CSV preserves actual capture times, model/visual
targets, and the sent command snapshot at inference completion. The AVI's
encoded frame rate should not be used to infer real vehicle speed.

Validation: 9 control/geometry tests and replay of all 314 supplied frames.
The model's wrong-way commands in the recorded late-exit failure are changed
to corrective commands in offline replay. This does not simulate the changed
vehicle path. Jetson frame rate and physical behavior need a low-speed test.
