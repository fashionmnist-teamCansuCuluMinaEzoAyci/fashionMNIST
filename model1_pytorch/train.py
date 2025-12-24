from __future__ import annotations

import os
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import FashionCNN


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    val_ratio: float = 0.1
    num_workers: int = 0  # Windows için genelde 0 problemsiz


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(base_dir: Path) -> tuple[Path, Path]:
    ckpt_dir = base_dir / "outputs" / "checkpoints"
    fig_dir = base_dir / "outputs" / "figures"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir, fig_dir


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = loss_sum / total
    acc = correct / total
    return avg_loss, acc


def plot_curves(history: dict, out_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.with_name("training_loss.png"), dpi=220)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.with_name("training_accuracy.png"), dpi=220)
    plt.close()


def main() -> None:
    cfg = TrainConfig()
    base_dir = Path(__file__).resolve().parent
    ckpt_dir, fig_dir = ensure_dirs(base_dir)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Fashion-MNIST normalization (mean/std). İstersen 0.5/0.5 de kullanabilirsin.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    full_train = datasets.FashionMNIST(
        root=str(base_dir / "data"),
        train=True,
        download=True,
        transform=transform
    )

    val_size = int(len(full_train) * cfg.val_ratio)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = FashionCNN(num_classes=10).to(device)

    # LOSS + OPTIMIZER (istediğin gibi)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_acc = -1.0
    best_path = ckpt_dir / "model1_best.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, correct = 0, 0
        loss_sum = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = loss_sum / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{cfg.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "config": asdict(cfg),
                "val_acc": best_val_acc
            }, best_path)

    # Save curves
    plot_curves(history, fig_dir / "training_curves.png")

    # Save metadata
    meta = {
        "best_val_acc": best_val_acc,
        "device": str(device),
        "config": asdict(cfg)
    }
    (ckpt_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved best checkpoint: {best_path}")
    print(f"Saved training curves to: {fig_dir}")


if __name__ == "__main__":
    main()
