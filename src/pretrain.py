import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from augmentations import SimCLRTransform
from loss import NTXentLoss
from model import SimCLRModel
from utils import get_device, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="SimCLR pretraining on CIFAR-10")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/simclr_latest.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=SimCLRTransform(size=32),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    model = SimCLRModel(projection_dim=args.projection_dim).to(device)
    criterion = NTXentLoss(temperature=args.temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        loop = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for (x1, x2), _ in loop:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch:03d} | loss: {avg_loss:.4f}")

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            },
            args.checkpoint_path,
        )


if __name__ == "__main__":
    main()
