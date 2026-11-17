#!/usr/bin/env python3
"""Train a Jetson-compatible lane-center point model from saved XY photos.

The existing Jetson runner uses only model X to steer. This version gives X
more weight, mirrors left/right examples with matching labels, and applies
label-correct horizontal crops. It starts from ImageNet, not a car checkpoint.
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from train_local_round2 import valid_images


ROOT = Path(__file__).resolve().parent


class CenteringDataset(Dataset):
    def __init__(self, paths, augment=False):
        self.paths = paths
        self.augment = augment
        self.jitter = transforms.ColorJitter(0.25, 0.25, 0.25, 0.08)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        x, y = [int(part) for part in Path(path).stem.split('_')[1:3]]
        with Image.open(path) as source:
            image = source.convert('RGB')
        width, height = image.size
        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                x = width - 1 - x
            # Keep the target in view, then transform its label with the crop.
            left = random.randint(0, min(16, x))
            right = random.randint(0, min(16, width - 1 - x))
            if left or right:
                image = image.crop((left, 0, width - right, height))
                x = (x - left) * width / (width - left - right)
            image = self.jitter(image)
        label = torch.tensor([(x - width / 2) / (width / 2),
                              (y - height / 2) / (height / 2)], dtype=torch.float32)
        image = transforms.functional.resize(image, (224, 224))
        rgb = transforms.functional.to_tensor(image)
        bgr = torch.from_numpy(rgb.numpy()[::-1].copy())
        bgr = transforms.functional.normalize(
            bgr, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return bgr, label


def stratified_split(paths, seed=20261021):
    """Keep 10% of each horizontal target-position band for validation."""
    groups = [[] for _ in range(5)]
    for path in paths:
        x = int(Path(path).stem.split('_')[1])
        groups[min(4, x // 64)].append(path)
    rng = np.random.RandomState(seed)
    train, validation = [], []
    for group in groups:
        order = rng.permutation(len(group))
        n_val = max(1, round(len(group) * 0.1))
        validation.extend(group[i] for i in order[:n_val])
        train.extend(group[i] for i in order[n_val:])
    random.Random(seed).shuffle(train)
    return train, validation


def evaluate(model, loader, device):
    model.eval()
    n = 0
    x_absolute = y_absolute = squared = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            delta = model(images) - labels
            x_absolute += delta[:, 0].abs().sum().item()
            y_absolute += delta[:, 1].abs().sum().item()
            squared += delta.square().sum().item()
            n += len(labels)
    return dict(x_mae_px=x_absolute / n * 160,
                y_mae_px=y_absolute / n * 120,
                xy_mse=squared / (2 * n))


def save_jetson(model, destination):
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, str(destination), _use_new_zipfile_serialization=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'dataset_xy_full_20261021')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'best_steering_model_xy_centering_v2.pth')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.00015)
    parser.add_argument('--resume', type=Path, help='Continue from an existing compatible checkpoint')
    parser.add_argument('--y-weight', type=float, default=0.03)
    parser.add_argument('--stop-x-mae', type=float, default=0.0,
                        help='Optional early target in image pixels; 0 disables this threshold')
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or
            args.lr <= 0 or args.y_weight < 0 or args.stop_x_mae < 0):
        parser.error('invalid training settings')
    if args.output.exists():
        parser.error('output already exists; refusing to overwrite it')
    if args.resume and not args.resume.is_file():
        parser.error('resume checkpoint not found')
    if not torch.cuda.is_available():
        parser.error('CUDA GPU unavailable')

    paths, invalid = valid_images(args.data)
    if len(paths) < 20:
        parser.error('need at least 20 valid labeled images')
    train, validation = stratified_split(paths)
    print('Images: {} train, {} validation, {} unreadable/invalid'.format(
        len(train), len(validation), len(invalid)), flush=True)
    train_loader = DataLoader(CenteringDataset(train, augment=True),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(CenteringDataset(validation),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    torch.manual_seed(20261021)
    random.seed(20261021)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    print('GPU: {}'.format(torch.cuda.get_device_name(0)), flush=True)
    model = models.resnet18(pretrained=not bool(args.resume))
    model.fc = torch.nn.Linear(512, 2)
    if args.resume:
        model.load_state_dict(torch.load(str(args.resume), map_location='cpu'))
        print('Resuming from {}'.format(args.resume), flush=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_x = float('inf')
    if args.resume:
        baseline = evaluate(model, val_loader, device)
        best_x = baseline['x_mae_px']
        save_jetson(model, args.output)
        print('Baseline val_x_MAE={:.2f}px val_XY_MSE={:.5f}; saved {}'.format(
            best_x, baseline['xy_mse'], args.output), flush=True)
    stale = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for batch, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            prediction = model(images)
            loss = F.smooth_l1_loss(prediction[:, 0], labels[:, 0], beta=0.1)
            loss += args.y_weight * F.smooth_l1_loss(
                prediction[:, 1], labels[:, 1], beta=0.1)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            n += len(labels)
            if batch % 100 == 0:
                print('Epoch {}/{} batch {}/{} elapsed {:.0f}s'.format(
                    epoch, args.epochs, batch, len(train_loader),
                    time.monotonic() - started), flush=True)
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        print('Epoch {}/{} train={:.5f} val_x_MAE={:.2f}px val_y_MAE={:.2f}px '
              'val_XY_MSE={:.5f}'.format(epoch, args.epochs, loss_sum / n,
              metrics['x_mae_px'], metrics['y_mae_px'], metrics['xy_mse']), flush=True)
        if metrics['x_mae_px'] < best_x:
            best_x = metrics['x_mae_px']
            stale = 0
            save_jetson(model, args.output)
            print('Saved {} (best x MAE {:.2f}px)'.format(args.output, best_x),
                  flush=True)
        else:
            stale += 1
        if args.stop_x_mae and best_x <= args.stop_x_mae:
            print('Reached requested X MAE target {:.2f}px'.format(args.stop_x_mae),
                  flush=True)
            break
        if stale >= args.patience:
            print('Early stop after {} epochs without better x MAE'.format(stale),
                  flush=True)
            break
    print('Done: {} best x MAE {:.2f}px'.format(args.output, best_x), flush=True)


if __name__ == '__main__':
    main()
