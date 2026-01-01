from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from model import FashionCNN


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def ensure_fig_dir(base_dir: Path) -> Path:
    fig_dir = base_dir / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


@torch.no_grad()
def main() -> None:
    base_dir = Path(__file__).resolve().parent
    fig_dir = ensure_fig_dir(base_dir)
    ckpt_path = base_dir / "outputs" / "checkpoints" / "model1_best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    test_ds = datasets.FashionMNIST(
        root=str(base_dir / "data"),
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    model = FashionCNN(num_classes=10).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true_all, y_pred_all = [], []

    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred_all.append(preds)
        y_true_all.append(y.numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    acc = (y_true == y_pred).mean()
    print(f"\nTest Accuracy: {acc:.4f}\n")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report)
    (fig_dir / "classification_report_model1.txt").write_text(report, encoding="utf-8")

    # Confusion matrix (normalized)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm_norm, cmap="Blues")
    plt.title("Normalized Confusion Matrix (Model 1 - PyTorch)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(10), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(10), CLASS_NAMES)

    for i in range(10):
        for j in range(10):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    out_path = fig_dir / "confusion_matrix_normalized_model1.png"
    plt.savefig(out_path, dpi=220)
    plt.close()

    print(f"Saved normalized confusion matrix to: {out_path}")


if __name__ == "__main__":
    main()
