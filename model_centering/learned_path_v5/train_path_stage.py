"""Train six road stages and optional near/far centers; never deploy automatically."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
STAGES = ['straight', 'approach_left', 'turn_left', 'approach_right',
          'turn_right', 'exit_straight']
MIRROR = [0, 3, 4, 1, 2, 5]


def load_records(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema') != 'learned-path-stage-v1':
        raise ValueError('Wrong annotation schema')
    if [data.get(k) for k in ('width', 'height', 'near_y', 'far_y')] != [320, 240, 180, 120]:
        raise ValueError('Unexpected image geometry')
    known = {}
    for folder in (ROOT / 'labeling', ROOT / 'labeling_extra240'):
        if (folder / 'manifest.json').is_file():
            manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
            for r in manifest:
                if r['id'] in known:
                    raise ValueError('Duplicate manifest ID: ' + r['id'])
                known[r['id']] = dict(r, path=str(folder / 'images' / (r['id'] + '.jpg')))
    records, seen = [], set()
    skipped = data.get('skipped', [])
    if len(set(skipped)) != len(skipped) or any(k not in known for k in skipped):
        raise ValueError('Invalid skipped IDs')
    for r in data['items']:
        key = r['id']
        if key not in known or key in seen or key in skipped:
            raise ValueError('Unknown, duplicated or skipped ID: ' + key)
        seen.add(key)
        if r.get('stage') not in STAGES + ['unknown']:
            raise ValueError('Invalid stage: ' + key)
        for k in ('near_x', 'far_x'):
            x = r.get(k)
            if x is not None and (isinstance(x, bool) or not isinstance(x, (int, float))
                                  or not np.isfinite(x) or not 0 <= x <= 319):
                raise ValueError('Invalid point: ' + key)
        records.append(dict(known[key], **{k: r.get(k) for k in ('near_x', 'far_x', 'stage')}))
    return records


def mirror_label(points, mask, stage):
    return np.where(mask, -points - 1. / 160., points), MIRROR[stage] if stage >= 0 else -1


class Samples(Dataset):
    def __init__(self, rows, augment=False):
        self.rows, self.augment = rows, augment
        self.jitter = transforms.ColorJitter(.12, .12, .10, .02)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(r['path']) as f:
            image = f.convert('RGB')
        mask = np.array([r.get(k) is not None for k in ('near_x', 'far_x')])
        points = np.array([(r[k] - 160) / 160 if r.get(k) is not None else 0.
                           for k in ('near_x', 'far_x')], dtype=np.float32)
        stage = STAGES.index(r['stage']) if r['stage'] in STAGES else -1
        if self.augment:
            if random.random() < .5:
                image = ImageOps.mirror(image)
                points, stage = mirror_label(points, mask, stage)
            image = self.jitter(image)
        image = transforms.functional.resize(image, (224, 224))
        x = transforms.functional.to_tensor(image)
        x = torch.from_numpy(x.numpy()[::-1].copy())
        x = transforms.functional.normalize(x, [.485, .456, .406], [.229, .224, .225])
        return x, torch.tensor(points), torch.tensor(mask), stage


def build_model(checkpoint=None):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 8)  # near X, far X, six stage logits
    if checkpoint:
        state = torch.load(str(checkpoint), map_location='cpu')
        model.load_state_dict({k: v for k, v in state.items() if not k.startswith('fc.')}, strict=False)
        # Initialize both X heads from the known X regressor, not its Y head.
        # Stages start uniformly; an arbitrary random stage bias has no meaning.
        with torch.no_grad():
            for i in (0, 1):
                model.fc.weight[i].copy_(state['fc.weight'][0])
                model.fc.bias[i].copy_(state['fc.bias'][0])
            model.fc.weight[2:].zero_()
            model.fc.bias[2:].zero_()
    return model


def describe(rows):
    return dict(n=len(rows), stages=dict(Counter(r['stage'] for r in rows)),
                near_points=sum(r.get('near_x') is not None for r in rows),
                far_points=sum(r.get('far_x') is not None for r in rows))


def evaluate(model, loader, device, active=None):
    confusion = np.zeros((6, 6), dtype=int)
    errors = [[], []]
    model.eval()
    with torch.no_grad():
        for images, points, mask, stage in loader:
            out = model(images.to(device)).cpu()
            for i in range(2):
                errors[i].extend((160 * (out[mask[:, i], i] - points[mask[:, i], i])).abs().tolist())
            scores = out[:, 2:].clone()
            if active is not None:
                for i in range(6):
                    if i not in active:
                        scores[:, i] = -float('inf')
            for true, pred in zip(stage.tolist(), scores.argmax(1).tolist()):
                if true >= 0:
                    confusion[true, pred] += 1
    n = confusion.sum()
    approach = confusion[[1, 3]]
    exit_n = int(confusion[5].sum())
    represented = confusion.sum(1) > 0
    recall = np.diag(confusion)[represented] / confusion.sum(1)[represented]
    return dict(stage_accuracy=float(confusion.trace() / n) if n else None,
                balanced_accuracy=float(recall.mean()) if len(recall) else None,
                confusion=confusion.tolist(),
                near_mae_px=float(np.mean(errors[0])) if errors[0] else None,
                far_mae_px=float(np.mean(errors[1])) if errors[1] else None,
                approach_as_active_turn=int(approach[:, [2, 4]].sum()),
                approach_count=int(approach.sum()),
                exit_recall=float(confusion[5, 5] / exit_n) if exit_n else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('labels', type=Path, nargs='+')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--observed-stages-only', action='store_true',
                   help='Offline exploration only: explicitly mask classes without training examples')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--holdout', nargs='+', default=['bend_exit_v4_run2', 'bend_exit_v4_run5',
                                                  'bend_exit_v2_run2', 'bend_exit_v3_run3'])
    p.add_argument('--output', type=Path, default=ROOT / 'path_stage_v5.pth')
    args = p.parse_args()
    rows = [r for path in args.labels for r in load_records(path)]
    if len({r['id'] for r in rows}) != len(rows):
        p.error('Repeated labeled ID across input files')
    train = [r for r in rows if r['group'] not in args.holdout]
    val = [r for r in rows if r['group'] in args.holdout]
    print(json.dumps(dict(training=describe(train), validation=describe(val)), indent=2))
    if args.validate_only:
        return
    if args.output.exists() or args.output.with_suffix('.json').exists():
        p.error('Output already exists')
    if args.epochs <= 0 or not train or not val:
        p.error('Need positive epochs and labeled training/held-out runs')
    active = [i for i, k in enumerate(STAGES) if any(r['stage'] == k for r in train)]
    unsupported = [k for i, k in enumerate(STAGES) if i not in active]
    if unsupported:
        print('UNSUPPORTED STAGES: ' + ', '.join(unsupported), flush=True)
    # Real class coverage is required before training; mirroring is not a substitute.
    for split, records in [('training', train), ('validation', val)]:
        counts = Counter(r['stage'] for r in records)
        required = [STAGES[i] for i in active] if args.observed_stages_only else STAGES
        missing = [k for k in required if counts[k] < (3 if split == 'training' else 1)]
        if missing:
            p.error('{} lacks examples of {}; collect/label them before training'.format(split, missing))
    hashes = {hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() for r in train}
    if any(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() in hashes for r in val):
        p.error('Identical image crosses the group split')
    random.seed(20261025)
    np.random.seed(20261025)
    torch.manual_seed(20261025)
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(ROOT.parent / 'best_steering_model_xy_review_aligned_v4_20261025.pth').to(device)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(('layer4.', 'fc.'))
    opt = torch.optim.AdamW([{'params': model.layer4.parameters(), 'lr': 2e-5},
                            {'params': model.fc.parameters(), 'lr': 4e-4}], weight_decay=1e-4)
    counts = Counter(r['stage'] for r in train)
    # Mirroring exchanges paired classes. Weight the distribution AFTER augmentation.
    expected_counts = [.5 * (counts[STAGES[i]] + counts[STAGES[MIRROR[i]]]) for i in active]
    weights = torch.tensor([1. / np.sqrt(n) for n in expected_counts], dtype=torch.float32, device=device)
    weights /= weights.mean()
    train_loader = DataLoader(Samples(train, True), batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(Samples(val), batch_size=16, num_workers=0)
    best = -float('inf')
    report = dict(schema='path-stage-model-v1', stages=STAGES, width=320, height=240,
                  near_y=180, far_y=120, preprocessing='pil_rgb_resize224_bgr_imagenet',
                  training=describe(train), validation=describe(val),
                  train_ids=[r['id'] for r in train], validation_ids=[r['id'] for r in val],
                  epochs=[], deployment_ready=False,
                  active_stage_indices=active, unsupported_stages=unsupported,
                  class_weights_after_mirroring=weights.detach().cpu().tolist(),
                  note='Validation selects checkpoint; not an independent test or driving verification.')
    for epoch in range(1, args.epochs + 1):
        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        for images, points, mask, stage in train_loader:
            images, points, mask, stage = [t.to(device) for t in (images, points, mask, stage)]
            out = model(images)
            loss = out.sum() * 0
            known = stage >= 0
            if known.any():
                mapping = torch.tensor([active.index(i) if i in active else -1 for i in range(6)], device=device)
                loss = loss + F.cross_entropy(out[known, 2:][:, active], mapping[stage[known]], weight=weights)
            for i in range(2):
                valid = mask[:, i]
                if valid.any():
                    loss = loss + F.smooth_l1_loss(out[valid, i], points[valid, i], beta=.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        metrics = evaluate(model, val_loader, device, active)
        score = metrics['balanced_accuracy'] - .5 * metrics['approach_as_active_turn'] / max(1, metrics['approach_count'])
        score -= .001 * sum(metrics[k] or 0 for k in ('near_mae_px', 'far_mae_px'))
        report['epochs'].append(dict(epoch=epoch, **metrics))
        if score > best:
            best = score
            report['best_epoch'], report['best_metrics'] = epoch, metrics
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                       str(args.output), _use_new_zipfile_serialization=False)
        args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('Epoch {} {}'.format(epoch, json.dumps(metrics)), flush=True)
    print('Saved offline checkpoint only. Review class/point coverage and replay before Jetson integration.')


if __name__ == '__main__':
    main()
