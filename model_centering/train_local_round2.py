#!/usr/bin/env python3
"""Fine-tune the XY lane model on old and newly collected photos on a PC GPU.

Invalid images are excluded. New photos are repeated only in the training set;
validation images are held out before repetition. Existing weights are read-only.
"""

import argparse
import random
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from train_model_terminal import XYDataset


ROOT = Path(__file__).resolve().parent
LABEL = re.compile(r"^xy_(\d+)_(\d+)_.+\.jpg$", re.IGNORECASE)


def valid_images(directory):
    paths = []
    invalid = []
    for path in sorted(directory.glob("*.jpg")):
        try:
            match = LABEL.fullmatch(path.name)
            if match is None:
                raise ValueError("filename has no XY label")
            x, y = map(int, match.groups())
            with Image.open(path) as image:
                width, height = image.size
                if not (0 <= x < width and 0 <= y < height):
                    raise ValueError("XY label outside image")
                image.load()
            paths.append(str(path))
        except (OSError, ValueError, IndexError) as exc:
            invalid.append((path.name, str(exc)))
    return paths, invalid


def split_paths(paths, seed):
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(paths))
    val_count = max(1, round(len(paths) * 0.10))
    return [paths[i] for i in order[val_count:]], [paths[i] for i in order[:val_count]]


def make_model(checkpoint, device):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 2)
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
    return model.to(device)


def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            per_image = F.mse_loss(model(images), labels, reduction="none").mean(dim=1)
            loss_sum += per_image.sum().item()
            count += len(images)
    return loss_sum / count


def save_weights(model, output):
    state = {name: weight.detach().cpu() for name, weight in model.state_dict().items()}
    torch.save(state, str(output), _use_new_zipfile_serialization=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-data", type=Path, default=ROOT / "dataset_xy")
    parser.add_argument("--new-data", type=Path, default=ROOT / "dataset_xy_new_20261021")
    parser.add_argument("--checkpoint", type=Path, action="append", default=None,
                        help="Candidate existing model; can be passed more than once")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "best_steering_model_xy_pc_round2.pth")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--new-repeat", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.00003)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.new_repeat < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, new-repeat, and lr must be positive")
    checkpoints = args.checkpoint or [
        ROOT / "best_steering_model_xy.pth",
        Path.home() / "Downloads" / "best_steering_model_xy_colab_jetson.pth",
    ]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            parser.error("checkpoint missing: " + str(checkpoint))
        if checkpoint.resolve() == args.output.resolve():
            parser.error("output must differ from every input checkpoint")
    if args.output.exists():
        parser.error("output already exists; choose a new --output path")

    old_paths, old_bad = valid_images(args.old_data)
    new_paths, new_bad = valid_images(args.new_data)
    print("Photos: old {} valid / {} invalid; new {} valid / {} invalid".format(
        len(old_paths), len(old_bad), len(new_paths), len(new_bad)), flush=True)
    if len(old_paths) < 20 or len(new_paths) < 20:
        parser.error("need at least 20 valid photos in each data directory")
    for label, invalid in (("old", old_bad), ("new", new_bad)):
        for name, reason in invalid[:3]:
            print("Skipped {}: {} ({})".format(label, name, reason), flush=True)

    old_train, old_val = split_paths(old_paths, 42)
    new_train, new_val = split_paths(new_paths, 20261021)
    print("Split: old train {} / val {}; new train {} / val {}; new repeat {}".format(
        len(old_train), len(old_val), len(new_train), len(new_val), args.new_repeat),
        flush=True)
    train_paths = old_train + new_train * args.new_repeat
    random.Random(20261021).shuffle(train_paths)
    train_loader = DataLoader(XYDataset("", augment=True, paths=train_paths),
                              batch_size=args.batch_size, shuffle=True, num_workers=0,
                              pin_memory=True)
    old_loader = DataLoader(XYDataset("", paths=old_val), batch_size=args.batch_size,
                            num_workers=0, pin_memory=True)
    new_loader = DataLoader(XYDataset("", paths=new_val), batch_size=args.batch_size,
                            num_workers=0, pin_memory=True)
    if not torch.cuda.is_available():
        parser.error("CUDA GPU unavailable; refusing to run a long CPU training job")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print("GPU: {} ({:.1f} GB)".format(torch.cuda.get_device_name(0),
          torch.cuda.get_device_properties(0).total_memory / 1e9), flush=True)

    best_model = None
    best_score = float("inf")
    best_checkpoint = None
    for checkpoint in checkpoints:
        model = make_model(checkpoint, device)
        old_loss = evaluate(model, old_loader, device)
        new_loss = evaluate(model, new_loader, device)
        score = 0.3 * old_loss + 0.7 * new_loss
        print("Candidate {}: old={:.5f} new={:.5f} weighted={:.5f}".format(
            checkpoint.name, old_loss, new_loss, score), flush=True)
        if score < best_score:
            best_model, best_score, best_checkpoint = model, score, checkpoint
        else:
            del model
    model = best_model
    print("Starting from {}".format(best_checkpoint), flush=True)
    save_weights(model, args.output)  # Preserve a working baseline even if fine-tuning worsens.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stale = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch_index, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = F.mse_loss(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(images)
            count += len(images)
            if batch_index % 100 == 0:
                print("Epoch {}/{} batch {}/{} elapsed {:.0f}s".format(
                    epoch, args.epochs, batch_index, len(train_loader),
                    time.monotonic() - started), flush=True)
        old_loss = evaluate(model, old_loader, device)
        new_loss = evaluate(model, new_loader, device)
        score = 0.3 * old_loss + 0.7 * new_loss
        print("Epoch {}/{} train={:.5f} old_val={:.5f} new_val={:.5f} weighted={:.5f}".format(
            epoch, args.epochs, loss_sum / count, old_loss, new_loss, score), flush=True)
        if score < best_score:
            save_weights(model, args.output)
            best_score = score
            stale = 0
            print("Saved {} (best weighted={:.5f})".format(args.output, best_score),
                  flush=True)
        else:
            stale += 1
            if stale >= 3:
                print("Early stop: no validation improvement for 3 epochs", flush=True)
                break
        model.train()
    print("Training complete: {} (best weighted={:.5f})".format(args.output, best_score),
          flush=True)


if __name__ == "__main__":
    main()
