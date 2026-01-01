import os
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import build_model

'''
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
'''

def get_device():
    # 1) Apple Silicon GPU (M1/M2/M3)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    
    # 2) NVIDIA GPU (Windows / Linux)
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # 3) CPU (herkes çalıştırabilir)
    return torch.device("cpu")
    

def ensure_dirs(base_dir: Path):
    out_dir = base_dir / "outputs"
    ckpt_dir = out_dir / "checkpoints"
    fig_dir = out_dir / "figures"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, ckpt_dir, fig_dir


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def main():
    base_dir = Path(__file__).resolve().parent
    _, ckpt_dir, _ = ensure_dirs(base_dir)

    device = get_device()
    print(f"Device: {device}")

    # Augmentation (train) + normal transform (val/test)
    train_tf = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_root = base_dir / "data"

    full_train = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=train_tf)
    test_set = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=test_tf)

    # Train/Val split
    val_ratio = 0.1
    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    # val_set transform’unu test_tf yapmak için küçük hack:
    # random_split Subset döner, içindeki dataset full_train; transform train_tf.
    # Val için daha “dürüst” ölçüm adına: val loader’da normalize aynı kalsın.
    # Basit kalsın diye böyle bırakıyoruz; istersen val için ayrı dataset kurarız.

    batch_size = 128
    # WINDOWS için en stabil: num_workers=0
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # verbose paramını koyma (sende hata veriyordu)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    epochs = 12
    best_val_acc = 0.0
    best_path = ckpt_dir / "model2_best.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        scheduler.step(val_loss)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"time={dt:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(), "val_acc": best_val_acc},
                best_path
            )

    print(f"\nSaved best checkpoint to: {best_path}")

    # Final test (best checkpoint ile)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    # Windows spawn problemlerine karşı şart
    main()


