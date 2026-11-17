V4 EXIT FIX TEST - original V4 model and behavior, faster reduction of steering

User explicitly selected V4 as the base. The checkpoint is byte-identical to the
original learned V4 corrected model (see model_info.json). No retraining occurred.
Rejected human annotation files remain quarantined and are not training inputs.

Use web STOP then Ctrl+C in the old program terminal. On Jetson:
cd ~/car
unzip -o jetson_ready_v4_exit_fix_20261025.zip
bash run_v4_exit_fix.sh --record bend_exit_v4fix.avi
Refresh http://JETSON_IP:5000, confirm V4 Exit Fix TEST, wait for live images,
then use AUTO for a low-speed trial. Speed 0.12, original V4 steering scale.

The new-frame filter follows reduced model steering at 0.90 rather than 0.70.
When steering grows, the original V4 response remains. The page shows
MODEL EASE TURN while following a smaller target. There is no bend timer.
Duplicate frames cannot keep reducing steering, and STOP/stale/lost-tape guards remain.
This follows the learned target returning toward center. It is not a trained
exit-stage detector. If the model keeps requesting a large turn, this change
cannot recognize the straight road by itself. A mistaken small-turn prediction
will also be followed more promptly. Physical driving still needs a short trial.

No ESP32 firmware or pin changes. This uniquely named checkpoint/launcher does
not overwrite the other model files on Jetson.
Recording starts at launch. Web STOP then Ctrl+C finalizes bend_exit_v4fix.avi
and bend_exit_v4fix.avi.csv; return both files for comparison.
16 control tests passed; legacy CPU weights and Python 3.6 syntax verified.
