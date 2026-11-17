#!/usr/bin/env python3
"""Train a new XY steering model on the entire received dataset using the PC GPU.

Starts from generic ImageNet ResNet18 weights, never from an earlier car model.
Only unreadable images and images without valid XY labels are skipped.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from train_local_round2 import valid_images
from train_model_terminal import XYDataset


ROOT = Path(__file__).resolve().parent


def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            errors = F.mse_loss(model(images), labels, reduction="none").mean(dim=1)
            total += errors.sum().item()
            count += len(images)
    return total / count


def save_jetson_weights(model, path):
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(state, str(path), _use_new_zipfile_serialization=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset_xy_full_20261021")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "best_steering_model_xy_pc_full_20261021.pth")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0003)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, patience, and lr must be positive")
    if not args.data.is_dir():
        parser.error("dataset directory missing: " + str(args.data))
    if args.output.exists():
        parser.error("output already exists; choose another --output path")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU unavailable")

    paths, invalid = valid_images(args.data)
    print("Dataset: {} valid images, {} unreadable or invalid".format(len(paths), len(invalid)),
          flush=True)
    for name, reason in invalid[:5]:
        print("Skipped {}: {}".format(name, reason), flush=True)
    if len(paths) < 20:
        parser.error("need at least 20 valid labeled photos")

    order = np.random.RandomState(42).permutation(len(paths))
    val_count = max(1, round(len(paths) * 0.10))
    train_paths = [paths[index] for index in order[val_count:]]
    val_paths = [paths[index] for index in order[:val_count]]
    print("Split: {} train, {} validation (all valid photos included)".format(
          len(train_paths), len(val_paths)), flush=True)
    train_loader = DataLoader(XYDataset("", augment=True, paths=train_paths),
                              batch_size=args.batch_size, shuffle=True, num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(XYDataset("", paths=val_paths),
                            batch_size=args.batch_size, shuffle=False, num_workers=0,
                            pin_memory=True)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print("GPU: {}".format(torch.cuda.get_device_name(0)), flush=True)
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 2)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float("inf")
    stale_epochs = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch_index, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = F.mse_loss(model(images), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(images)
            count += len(images)
            if batch_index % 100 == 0:
                print("Epoch {}/{} batch {}/{} elapsed {:.0f}s".format(
                      epoch, args.epochs, batch_index, len(train_loader),
                      time.monotonic() - started), flush=True)
        val_loss = evaluate(model, val_loader, device)
        print("Epoch {}/{} train={:.5f} validation={:.5f}".format(
              epoch, args.epochs, total / count, val_loss), flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0
            save_jetson_weights(model, args.output)
            print("Saved {} (best validation={:.5f})".format(args.output, best_loss),
                  flush=True)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stop: validation did not improve for {} epochs".format(
                      args.patience), flush=True)
                break
    print("Training complete: {} (best validation={:.5f})".format(
          args.output, best_loss), flush=True)


if __name__ == "__main__":
    main()
