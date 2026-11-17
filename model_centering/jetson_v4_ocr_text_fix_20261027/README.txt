OCR text preprocessing repair 2026-10-27
Normalizes STOP mask polarity, removes edge-connected background, crops text, scales character height and adds white border. Recognizes A-Z; exact STOP/P validation and two-frame confirmation unchanged. Original supplied functions, V4 weights, motor pins and parking policy unchanged.
Verified on Tesseract 5.5.3 with official tessdata_fast English data: original fixture reproduced SP / empty P before correction, STOP / P after correction. Jetson 5.3 actual self-check and real camera signs still required. 58 existing tests passed.
Jetson: cd ~/car; unzip -o jetson_v4_ocr_text_fix_20261027.zip; python3 check_traffic_ocr.py && bash run_v4_sign_ocr.sh --record traffic_v4_test.avi
Reference: https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
