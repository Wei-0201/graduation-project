V5 TEST - learned near/far path and road stages (2026-10-25)

UPLOAD FROM WINDOWS POWERSHELL
scp "D:\<project>\model_centering\jetson_ready_learned_path_v5_20261025.zip" tsai@172.20.10.2:/home/tsai/car/
Use the actual local project path.

JETSON
Use web STOP, then Ctrl+C in the old car terminal. Then:
cd ~/car
unzip -o jetson_ready_learned_path_v5_20261025.zip
bash run_learned_path_v5.sh --record bend_exit_v5.avi
Refresh http://172.20.10.2:5000 and confirm Learned Path V5 TEST.
Wait for live frames/FPS, then choose AUTO for a short low-speed trial.
Starts MANUAL/STOP. No ESP32 firmware or pin changes.

WHAT CHANGED
New ResNet18 eight-output model, trained with 214 human-annotated frames.
Two learned target X positions: near at image y=180, far at y=120.
Near target determines current corrective steering, including when vehicle is offset.
Upcoming-turn stages do not steer toward the far target alone.
Confident active-turn classification may add a bounded far-target contribution,
only if the near correction already agrees in direction; correction limited to 0.08.
It does not wait a preset number of seconds before turning.
Per-direction output scales: left 0.70, right 0.78. Speed 0.12.
One shared model per frame; no second inference pass.
The old model and old launchers remain available.

LIMITATIONS OF THIS TEST
Five observed stages supported: straight, approach_left, turn_left,
approach_right, turn_right. No exit_straight training examples were supplied.
EXIT CLASS NOT TRAINED is expected and is NOT an error.
Unsupported exit class is excluded from stage selection. Do not assume exit recognition.
Near-target return toward center can reduce steering, but this is not learned exit classification.
Stage softmax percentages are uncalibrated scores, not safety probabilities.
The model can still misclassify and predict incorrect targets outside training coverage.
Physical trajectory has NOT been verified; PC replay cannot show the changed trajectory.
Do not load these eight-output weights into the old two-output XY runner.

VERIFICATION
159 training images, 55 validation images; four entire driving runs held out.
Validation also selected epoch 20, so it is not a separate final test set.
Runtime near X MAE 15.16px, far X MAE 16.46px (320px-wide images).
Stage correct 42/55 (76.4%). Approach incorrectly called active turn: 3/16.
No wrong near correction direction among 35 validation targets over 20px from center.
These sample results do not guarantee reliable track driving.
19 controller/decision tests passed. Offline V4 replay: 1099 frames, no invalid near outputs.
CPU legacy tensor state_dict, fc.weight=(8,512), Python 3.6 syntax checked.

STOP AND RECORD
Recording begins on program launch, including MANUAL/STOP frames.
Web STOP then Ctrl+C finalizes bend_exit_v5.avi and bend_exit_v5.avi.csv.
Five fresh frames without nearby tape latch STOP; reposition and click AUTO to reset.
Missing/stale images or invalid near predictions also stop the car.
To collect missing exit scenes, record with motors stopped and gently hand-push
through the lane center from mid-bend into a straight with the camera fixed.

ROLLBACK
Web STOP then Ctrl+C. To restore V4.1:
bash run_learned_centering_v41.sh
