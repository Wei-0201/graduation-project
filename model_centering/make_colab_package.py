"""Build a browser-only Colab training package using Python's standard library."""

import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "car_training_colab.zip"
NOTEBOOK = ROOT / "train_car_on_colab.ipynb"
LOCAL_NOTEBOOK = ROOT / "train_car_on_local_gpu.ipynb"
TRAINER = ROOT / "train_model_terminal.py"
CHECKPOINT = ROOT / "best_steering_model_xy.pth"
IMAGES = sorted((ROOT / "dataset_xy").glob("*.jpg"))

if not TRAINER.is_file() or not CHECKPOINT.is_file() or len(IMAGES) < 20:
    raise SystemExit("Need trainer, checkpoint, and labelled JPG images first")

with ZipFile(ARCHIVE, "w", compression=ZIP_STORED, allowZip64=True) as archive:
    archive.write(TRAINER, TRAINER.name)
    archive.write(CHECKPOINT, CHECKPOINT.name)
    for image in IMAGES:
        archive.write(image, "dataset_xy/" + image.name)


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
    },
    "cells": [
        markdown("""# Jetson 車道模型：在 Colab 接續訓練

1. 在 Google 雲端硬碟的「我的雲端硬碟」根目錄上傳 `car_training_colab.zip`。
2. 在 Colab 選「執行階段 → 變更執行階段類型 → GPU」。
3. 由上往下依序執行每個程式格。第一次連接雲端硬碟時，請用上傳 ZIP 的 Google 帳號授權。
4. 最後一格會把新模型存到雲端硬碟根目錄。下載後先在 Jetson 驗證能載入，再試車。

這份筆記本接續 ZIP 裡的舊模型，訓練 10 輪，並保留原始模型。Colab 的免費 GPU 與執行時間依當時供應情況而定。
"""),
        code("""from google.colab import drive
drive.mount('/content/drive')
"""),
        code("""from pathlib import Path
from zipfile import ZipFile
import torch, torchvision

archive = Path('/content/drive/MyDrive/car_training_colab.zip')
workdir = Path('/content/car_training')
assert archive.is_file(), '找不到 ZIP；請先把 car_training_colab.zip 上傳到我的雲端硬碟根目錄'
assert torch.cuda.is_available(), '目前沒有 GPU；請在「執行階段 → 變更執行階段類型」選 GPU'
workdir.mkdir(exist_ok=True)
with ZipFile(archive) as package:
    package.extractall(workdir)
for name in ('train_model_terminal.py', 'best_steering_model_xy.pth'):
    assert (workdir / name).is_file(), 'ZIP 缺少 ' + name
count = len(list((workdir / 'dataset_xy').glob('*.jpg')))
assert count >= 20, '沒有找到足夠的標註照片'
print('PyTorch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('GPU:', torch.cuda.get_device_name(0), 'JPG:', count)
"""),
        markdown("""## 接續訓練

這格完成時會顯示 `Training complete`。訓練期間請保持 Colab 執行階段連線；每一輪會顯示 `Epoch ... train=... validation=...`。
"""),
        code("""import subprocess, sys

output = workdir / 'best_steering_model_xy_colab.pth'
command = [
    sys.executable, 'train_model_terminal.py',
    '--data', 'dataset_xy',
    '--resume', 'best_steering_model_xy.pth',
    '--output', output.name,
    '--epochs', '10',
    '--batch-size', '8',
]
subprocess.run(command, cwd=workdir, check=True)
assert output.is_file(), '訓練沒有產出模型'
"""),
        markdown("""## 匯出供 Jetson 測試的模型

Jetson 上使用較舊的 PyTorch。這格把權重另存為舊版序列化格式，然後複製到 Google 雲端硬碟。格式轉換有助於相容，仍須在 Jetson 實際載入確認。
"""),
        code("""import shutil

state = torch.load(output, map_location='cpu', weights_only=True)
legacy = workdir / 'best_steering_model_xy_colab_jetson.pth'
torch.save(state, legacy, _use_new_zipfile_serialization=False)
destination = Path('/content/drive/MyDrive') / legacy.name
shutil.copy2(legacy, destination)
print('完成：', destination, '大小 MB：', round(destination.stat().st_size / 1_000_000, 1))
"""),
    ],
}

NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

local_notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
    },
    "cells": [
        markdown("""# 車道模型：用 Windows 電腦的 GPU 訓練

此筆記本只適用於已連接 **Colab 本機執行階段**、且電腦已安裝 CUDA 版 PyTorch 的情況。
它直接使用 `D:\\畢業專題\\model_centering` 裡的照片和舊模型，不掛載 Google 雲端硬碟，也不解壓 ZIP。
請在開始前先確認前一次雲端訓練已完成或已停止。
"""),
        code("""from pathlib import Path
import torch, torchvision

workdir = Path(r'D:\畢業專題\model_centering')
assert workdir.is_dir(), '找不到本機專案資料夾'
assert (workdir / 'train_model_terminal.py').is_file(), '找不到訓練程式'
assert (workdir / 'best_steering_model_xy.pth').is_file(), '找不到舊模型'
count = len(list((workdir / 'dataset_xy').glob('*.jpg')))
assert count >= 20, '找不到足夠的標註照片'
assert torch.cuda.is_available(), '目前無法使用 RTX 顯示卡；請檢查 CUDA 版 PyTorch'
print('PyTorch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('GPU:', torch.cuda.get_device_name(0), 'JPG:', count)
"""),
        code("""import subprocess, sys

output = workdir / 'best_steering_model_xy_pc.pth'
command = [
    sys.executable, 'train_model_terminal.py',
    '--data', 'dataset_xy',
    '--resume', 'best_steering_model_xy.pth',
    '--output', output.name,
    '--epochs', '10',
    '--batch-size', '8',
]
subprocess.run(command, cwd=workdir, check=True)
assert output.is_file(), '訓練沒有產出模型'
print('完成：', output)
"""),
    ],
}
LOCAL_NOTEBOOK.write_text(
    json.dumps(local_notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)
print("Images:", len(IMAGES))
print("Archive:", ARCHIVE, ARCHIVE.stat().st_size)
print("Notebook:", NOTEBOOK)
print("Local notebook:", LOCAL_NOTEBOOK)
