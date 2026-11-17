"""Rebuild XY using only the original Jetson photo dataset and ImageNet weights."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models

from train_local_centering_v2 import CenteringDataset, save_jetson
from train_local_round2 import valid_images

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'dataset_xy_full_20261021'


def split_original(paths):
    """Keep byte-identical images together; no recent annotation directories."""
    groups = defaultdict(list)
    for name in paths:
        p = Path(name).resolve()
        if p.parent != DATA.resolve() or 'bend_exit' in p.name or 'review' in p.name:
            raise ValueError('Input outside the original photo whitelist: ' + str(p))
        groups[hashlib.sha256(p.read_bytes()).hexdigest()].append(str(p))
    bins = [[] for _ in range(5)]
    for digest, names in sorted(groups.items()):
        x = int(Path(names[0]).stem.split('_')[1])
        bins[min(4, x // 64)].append((digest, names))
    rng = random.Random(20261025)
    training, validation, train_hash, val_hash = [], [], [], []
    for group in bins:
        rng.shuffle(group)
        n = max(1, round(len(group) * .1))
        for i, (digest, names) in enumerate(group):
            (validation if i < n else training).extend(names)
            (val_hash if i < n else train_hash).append(digest)
    assert not set(train_hash).intersection(val_hash)
    rng.shuffle(training)
    return training, validation, train_hash, val_hash


def worker_init(_):
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def evaluate(model, loader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for image, labels in loader:
            pred = model(image.to(device, non_blocking=True)).cpu().float()
            errors.append((pred - labels).numpy())
    error = np.concatenate(errors)
    return dict(n=len(error), x_mae_px=float(np.abs(error[:, 0]).mean() * 160),
                x_p90_px=float(np.percentile(np.abs(error[:, 0]) * 160, 90)),
                y_mae_px=float(np.abs(error[:, 1]).mean() * 120),
                x_mse=float(np.mean(error[:, 0] ** 2)),
                xy_mse=float(np.mean(error ** 2)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--stop-x-mse', type=float, default=.0075)
    p.add_argument('--output', type=Path, default=ROOT / 'best_steering_model_xy_clean_20261025.pth')
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix('.report.json').exists():
        p.error('Output already exists; choose a new output name')
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.stop_x_mse < 0:
        p.error('Invalid training settings')
    if not torch.cuda.is_available():
        p.error('CUDA unavailable')
    random.seed(20261025)
    np.random.seed(20261025)
    torch.manual_seed(20261025)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    paths, invalid = valid_images(DATA)
    training, validation, train_hash, val_hash = split_original(paths)
    report = dict(schema='clean-xy-rebuild-v1', allowed_source=str(DATA),
                  initialization='ImageNet ResNet18 only; no trained-car checkpoint loaded',
                  rejected_annotation_inputs=[], y_weight=.03,
                  preprocessing='PIL RGB resize224; RGB-to-BGR; ImageNet normalization',
                  training_images=len(training), validation_images=len(validation),
                  invalid_images=len(invalid), invalid_files=invalid,
                  training_files=[Path(x).name for x in training],
                  validation_files=[Path(x).name for x in validation],
                  training_hashes=train_hash, validation_hashes=val_hash,
                  max_epochs=args.epochs, stop_x_mse=args.stop_x_mse, epochs=[],
                  note='Identical image bytes do not cross the split; nearby photographs may still correlate. Validation selects checkpoint, not independent driving test.')
    report_path = args.output.with_suffix('.report.json')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('CLEAN INPUT: {} train / {} validation / {} invalid; zero rejected-label inputs'.format(
        len(training), len(validation), len(invalid)), flush=True)
    print('GPU: ' + torch.cuda.get_device_name(0), flush=True)
    print('Initialization: ImageNet only, no car-model resume', flush=True)
    options = dict(num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init,
                   persistent_workers=args.workers > 0)
    train_loader = DataLoader(CenteringDataset(training, augment=True), batch_size=args.batch_size,
                              shuffle=True, **options)
    val_loader = DataLoader(CenteringDataset(validation), batch_size=64, **options)
    # torchvision reads the official cached ImageNet file, never review/V5 weights.
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 2)
    model.to('cuda')
    opt = torch.optim.AdamW(model.parameters(), lr=.00015, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    best, stale, start = float('inf'), 0, time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0., 0
        for batch, (image, label) in enumerate(train_loader, 1):
            image, label = image.cuda(non_blocking=True), label.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                pred = model(image)
                loss = F.smooth_l1_loss(pred[:, 0], label[:, 0], beta=.1)
                loss = loss + .03 * F.smooth_l1_loss(pred[:, 1], label[:, 1], beta=.1)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.item()) * len(image)
            n += len(image)
            if batch % 100 == 0:
                print('Epoch {}/{} batch {}/{} elapsed {:.0f}s'.format(
                    epoch, args.epochs, batch, len(train_loader), time.monotonic() - start), flush=True)
        scheduler.step()
        metrics = evaluate(model, val_loader, 'cuda')
        improved = metrics['x_mae_px'] < best
        if improved:
            best, stale = metrics['x_mae_px'], 0
            save_jetson(model, args.output)
            report['best_epoch'], report['best_metrics'] = epoch, metrics
        else:
            stale += 1
        report['epochs'].append(dict(epoch=epoch, training_loss=total/n, saved=improved, **metrics))
        print('Epoch {}/{} X_MAE={:.2f}px X_MSE={:.6f} Y_MAE={:.2f}px saved={} elapsed={:.0f}s'.format(
            epoch, args.epochs, metrics['x_mae_px'], metrics['x_mse'], metrics['y_mae_px'],
            improved, time.monotonic() - start), flush=True)
        stop = ('requested_x_mse_target' if improved and args.stop_x_mse and metrics['x_mse'] <= args.stop_x_mse
                else 'plateau' if stale >= 6 else 'max_epochs' if epoch == args.epochs else None)
        report['stop_reason'] = stop
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        if stop:
            break
    print('DONE {} best epoch {} X_MAE {:.2f}px'.format(args.output.name, report['best_epoch'], best), flush=True)


if __name__ == '__main__':
    main()
