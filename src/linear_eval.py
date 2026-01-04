import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import SimCLRModel
from utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Linear evaluation on CIFAR-10")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    train_set = datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    projection_dim = checkpoint.get("args", {}).get("projection_dim", 128)
    simclr = SimCLRModel(projection_dim=projection_dim)
    simclr.load_state_dict(checkpoint["model_state_dict"])
    encoder = simclr.encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    in_dim = simclr.projection_head[0].in_features
    classifier = nn.Linear(in_dim, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9)

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        train_loss = 0.0
        train_acc = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, targets in loop:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.no_grad():
                features = encoder(images)

            logits = classifier(features)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy(logits, targets)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        classifier.eval()
        test_acc = 0.0
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                features = encoder(images)
                logits = classifier(features)
                test_acc += accuracy(logits, targets)

        test_acc /= len(test_loader)
        print(
            f"Epoch {epoch:03d} | train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | test_acc: {test_acc:.4f}"
        )


if __name__ == "__main__":
    main()
