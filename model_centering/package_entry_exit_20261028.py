from pathlib import Path
import shutil
import zipfile
import ast

root = Path(__file__).parent
runtime = root/'jetson_v4_traffic_signs.py'
source = runtime.read_text('utf8').replace('OCR TEST - Right Entry V3</h2>',
    'OCR TEST - Entry + Exit</h2>').replace("version='v4-right-entry-v3-20261028'",
    "version='v4-entry-exit-20261028'")
runtime.write_text(source, encoding='utf8')
folder = root/'jetson_v4_entry_exit_20261028'
folder.mkdir(exist_ok=True)
for name in ('jetson_v4_traffic_signs.py','right_entry_guard.py','observed_exit_easing.py',
             'sign_ocr_engine.py','run_v4_sign_ocr.sh'):
    shutil.copy2(root/name, folder/name)
    if name.endswith('.py'):
        ast.parse((folder/name).read_text('utf8'),feature_version=6)
(folder/'README_entry_exit.txt').write_text('''V4 Entry + Exit 2026-10-28
Includes revised right-entry guard and earlier observed bend exit easing.
No fixed half-second delay/advance. STOP pause 5 seconds, P parking/return and original V4 weights preserved.
No model training or rejected annotations used.
Exit: after two fresh paired observations of an actual bend, require two fresh near-straight paired observations,
near lane center and model magnitude no longer growing. Ease toward observed near-road request by 65 percent
of difference, at most .18 normalized command. Never reverse steering or strengthen a turn.
Missing pairs or stale images cannot create exit easing. Original motor, OCR, frame freshness and lane-loss
checks retained. Applies to left and right exits.
Right entry retains smaller initial right request and later geometry release; fresh tracked right evidence
can continue attenuation only after two paired observations. Three missing observations reset bootstrap.
Validation: 72 tests passed and Python 3.6 syntax checked. Uploaded AVI decodes 4537 frames but CSV has 779 rows.
The telemetry prefix shows early turn, without a qualifying full bend-to-straight exit sequence.
Exit behavior still needs a measured real vehicle test. No half-second performance claim.
Status: MODEL EXIT EARLY EASE identifies exit activation.
CAR_OBSERVED_EXIT_EASING=0 bash run_v4_sign_ocr.sh disables exit easing alone.
Install after old program STOP and Ctrl+C:
cd ~/car
unzip -o jetson_v4_entry_exit_20261028.zip
bash run_v4_sign_ocr.sh --record entry_exit_test.avi
''',encoding='utf8')
with zipfile.ZipFile(folder.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
    for path in folder.iterdir():
        if path.is_file():
            archive.write(path,path.name)
print(folder.with_suffix('.zip'))
