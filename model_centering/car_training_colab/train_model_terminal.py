#!/usr/bin/env python3
"""Train the same ResNet18 XY model as train_model_ready.ipynb from a terminal."""
import argparse
import glob
import os

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directory, augment=False, paths=None):
        if paths is None:
            self.paths = []
            for path in sorted(glob.glob(os.path.join(directory, '*.jpg'))):
                try:
                    parts = os.path.basename(path).split('_')
                    int(parts[1])
                    int(parts[2])
                    with PIL.Image.open(path) as image:
                        image.load()
                except (OSError, ValueError, IndexError) as exc:
                    print('Skipping invalid image {}: {}'.format(path, exc), flush=True)
                    continue
                self.paths.append(path)
        else:
            self.paths = paths
        self.augment = augment
        self.jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = PIL.Image.open(path).convert('RGB')
        width, height = image.size
        parts = os.path.basename(path).split('_')
        x = (int(parts[1]) - width / 2.0) / (width / 2.0)
        y = (int(parts[2]) - height / 2.0) / (height / 2.0)
        if self.augment:
            image = self.jitter(image)
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = torch.from_numpy(image.numpy()[::-1].copy())
        image = transforms.functional.normalize(
            image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, torch.tensor([x, y]).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='dataset_xy')
    parser.add_argument('--output', default='best_steering_model_xy.pth')
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--resume', help='Existing XY model to continue training from')
    parser.add_argument('--lr', type=float, help='Learning rate (default: 0.0001 when resuming, 0.001 otherwise)')
    args = parser.parse_args()
    if args.resume:
        if not os.path.isfile(args.resume):
            parser.error('Resume model not found: ' + args.resume)
        if os.path.abspath(args.resume) == os.path.abspath(args.output):
            parser.error('--output must differ from --resume to preserve the original model')
    learning_rate = args.lr if args.lr is not None else (0.0001 if args.resume else 0.001)
    if learning_rate <= 0:
        parser.error('--lr must be positive')
    dataset = XYDataset(args.data)
    if len(dataset) < 20:
        parser.error('Need at least 20 labelled JPG images in ' + args.data)
    indices = np.random.RandomState(42).permutation(len(dataset))
    num_test = max(1, int(0.1 * len(dataset)))
    train_set = torch.utils.data.Subset(
        XYDataset(args.data, augment=True, paths=dataset.paths), indices[num_test:])
    test_set = torch.utils.data.Subset(dataset, indices[:num_test])
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    if args.resume:
        model = models.resnet18(pretrained=False)
    else:
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print('Continuing from {}'.format(os.path.abspath(args.resume)), flush=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print('Images: {} train, {} validation; device: {}'.format(
        len(train_set), len(test_set), device), flush=True)
    best_loss = float('inf')
    if args.resume:
        model.eval()
        baseline_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                baseline_loss += F.mse_loss(model(images), labels).item()
        best_loss = baseline_loss / len(test_loader)
        torch.save(model.state_dict(), args.output)
        print('Existing model validation={:.5f}; saved baseline to {}'.format(
            best_loss, args.output), flush=True)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                test_loss += F.mse_loss(model(images), labels).item()
        test_loss /= len(test_loader)
        print('Epoch {:02d}/{:02d} train={:.5f} validation={:.5f}'.format(
            epoch + 1, args.epochs, train_loss, test_loss), flush=True)
        if test_loss < best_loss:
            torch.save(model.state_dict(), args.output)
            best_loss = test_loss
            print('Saved {} (best validation={:.5f})'.format(args.output, best_loss),
                  flush=True)
    print('Training complete. Best model: {}'.format(os.path.abspath(args.output)),
          flush=True)


if __name__ == '__main__':
    main()
