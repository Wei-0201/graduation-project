STOP pause update 2026-10-28
Launch via bash run_v4_sign_ocr.sh; default STOP pause is now 5 seconds. Stay in AUTO: no second AUTO click needed. Same visible sign remains handled until absent in three independent OCR results. Fresh OCR and lane/motor safety checks still apply. Manual STOP cancels automatic continuation. P parking behavior unchanged. Includes previous OCR text fix. No model weights or ESP32 changes.
59 regression tests passed, including full 5 second boundary, same visible STOP, stale OCR and base STOP. Actual Jetson test required.
Install: cd ~/car; unzip -o jetson_v4_stop5s_20261028.zip; bash run_v4_sign_ocr.sh
