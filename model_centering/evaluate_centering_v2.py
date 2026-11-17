#!/usr/bin/env python3
"""Report X pixel error by target position for a Jetson-compatible model."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models

from train_local_centering_v2 import CenteringDataset, stratified_split
from train_local_round2 import valid_images


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--data', type=Path, default=ROOT / 'dataset_xy_full_20261021')
    args = parser.parse_args()
    paths, _ = valid_images(args.data)
    _, validation = stratified_split(paths)
    loader = DataLoader(CenteringDataset(validation), batch_size=32,
                        shuffle=False, num_workers=0)
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 2)
    model.load_state_dict(torch.load(str(args.model), map_location='cpu'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    target_x, predicted_x, predicted_y = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            prediction = model(images.to(device)).cpu()
            target_x.extend(((labels[:, 0] + 1) * 160).numpy())
            predicted_x.extend(((prediction[:, 0] + 1) * 160).numpy())
            predicted_y.extend(((prediction[:, 1] + 1) * 120).numpy())
    target_x = np.asarray(target_x)
    predicted_x = np.asarray(predicted_x)
    predicted_y = np.asarray(predicted_y)
    error = predicted_x - target_x
    print('model:', args.model)
    print('validation images:', len(validation))
    valid_x = np.isfinite(predicted_x) & (predicted_x >= 0) & (predicted_x < 320)
    print('steering X accepted by new Jetson runner: {}/{}'.format(
        int(valid_x.sum()), len(valid_x)))
    for title, selected in [('all', np.ones(len(error), dtype=bool)),
                            ('left target', target_x < 160),
                            ('right target', target_x >= 160)] + [
            ('x={}..{}'.format(lo, lo + 63), (target_x >= lo) & (target_x < lo + 64))
            for lo in range(0, 320, 64)]:
        local = error[selected]
        print('{}: n={} MAE={:.2f}px bias={:+.2f}px P90={:.2f}px'.format(
            title, len(local), np.mean(np.abs(local)), np.mean(local),
            np.percentile(np.abs(local), 90)))


if __name__ == '__main__':
    main()
