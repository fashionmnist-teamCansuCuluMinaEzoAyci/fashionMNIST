from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from model import build_model


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main():
    base_dir = Path(__file__).resolve().parent
    ckpt_path = base_dir / "outputs" / "checkpoints" / "model2_best.pt"

    device = get_device()
    print(f"Device: {device}")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_set = datasets.FashionMNIST(root=base_dir / "data", train=False, download=True, transform=tf)
    loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    model = build_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        y_pred.append(pred)
        y_true.append(y.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = (y_true == y_pred).mean()
    print(f"\nTest Accuracy: {acc:.4f}\n")

    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


if __name__ == "__main__":
    main()
