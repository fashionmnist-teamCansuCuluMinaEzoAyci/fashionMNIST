from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import FashionCNN


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def ensure_dirs(base_dir: Path) -> Path:
    fig_dir = base_dir / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def unnormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """
    img_tensor: (1,28,28) normalized
    return: (28,28) in [0,1] for display
    """
    mean, std = 0.2860, 0.3530
    x = img_tensor.clone()
    x = x * std + mean
    x = x.clamp(0, 1)
    return x.squeeze(0).cpu().numpy()


def save_grid(images: np.ndarray, title: str, out_path: Path, cols: int = 4) -> None:
    """
    images: (N, H, W)
    """
    n = images.shape[0]
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(cols * 2.2, rows * 2.2))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        img = images[i]
        # normalize per-map for visibility
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        plt.imshow(img, cmap="gray")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def main() -> None:
    base_dir = Path(__file__).resolve().parent
    fig_dir = ensure_dirs(base_dir)
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

    model = FashionCNN(num_classes=10).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # -------------------------
    # 1) Figure 1: Sample images (dataset overview)
    # -------------------------
    idxs = list(range(10))
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(idxs):
        x, y = test_ds[idx]
        plt.subplot(2, 5, i + 1)
        plt.imshow(unnormalize(x), cmap="gray")
        plt.title(CLASS_NAMES[int(y)], fontsize=9)
        plt.axis("off")

    plt.suptitle("Sample Images from Fashion-MNIST Dataset")
    plt.tight_layout()
    out_path = fig_dir / "sample_images_fashion_mnist_pytorch.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Saved Figure 1 to: {out_path}")

    # -------------------------
    # 2) Filters visualization (conv1 weights)
    # -------------------------
    w = model.conv1.weight.detach().cpu().numpy()  # (outC, inC, k, k)
    w = w[:, 0, :, :]  # grayscale -> (outC, k, k)
    n_filters = min(16, w.shape[0])
    filters = w[:n_filters]

    save_grid(filters, "Learned Filters (conv1) - Model 1", fig_dir / "filters_conv1_model1.png", cols=4)
    print(f"Saved filters to: {fig_dir / 'filters_conv1_model1.png'}")

    # -------------------------
    # 3) Activation maps with forward hooks
    # -------------------------
    activations = {}

    def hook_fn(name):
        def _hook(module, inp, out):
            activations[name] = out.detach()
        return _hook

    h1 = model.conv1.register_forward_hook(hook_fn("conv1"))
    h2 = model.conv2.register_forward_hook(hook_fn("conv2"))

    # Choose a test index to visualize
    idx = 123
    x, y_true = test_ds[idx]
    x_batch = x.unsqueeze(0).to(device)

    logits = model(x_batch)
    y_pred = int(torch.argmax(logits, dim=1).item())

    # remove hooks
    h1.remove()
    h2.remove()

    # Save sample preview
    plt.figure()
    plt.imshow(unnormalize(x), cmap="gray")
    plt.axis("off")
    plt.title(f"idx={idx} | True: {CLASS_NAMES[int(y_true)]} | Pred: {CLASS_NAMES[y_pred]}")
    sample_path = fig_dir / f"sample_{idx}_true_{int(y_true)}_pred_{y_pred}_model1.png"
    plt.tight_layout()
    plt.savefig(sample_path, dpi=220)
    plt.close()
    print(f"Saved sample preview to: {sample_path}")

    # Activation maps: (1, C, H, W) -> take first N channels
    for layer_name in ["conv1", "conv2"]:
        act = activations[layer_name][0].cpu().numpy()  # (C, H, W)
        n_maps = min(16, act.shape[0])
        maps = act[:n_maps]

        out_path = fig_dir / f"activation_{layer_name}_idx{idx}_model1.png"
        save_grid(maps, f"Activation Maps ({layer_name}) - idx {idx} - Model 1", out_path, cols=4)
        print(f"Saved activation maps to: {out_path}")

    print("Done. Check outputs/figures/ folder.")


if __name__ == "__main__":
    main()
