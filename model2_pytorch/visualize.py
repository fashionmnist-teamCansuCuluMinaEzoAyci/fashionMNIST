from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import build_model


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main():
    base_dir = Path(__file__).resolve().parent
    fig_dir = base_dir / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_set = datasets.FashionMNIST(root=base_dir / "data", train=False, download=True, transform=tf)
    loader = DataLoader(test_set, batch_size=16, shuffle=True, num_workers=0)

    ckpt_path = base_dir / "outputs" / "checkpoints" / "model2_best.pt"
    model = build_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    images, labels = next(iter(loader))
    images = images.to(device)
    logits = model(images)
    preds = logits.argmax(dim=1).cpu()

    images_cpu = images.cpu() * 0.5 + 0.5  # denormalize

    plt.figure()
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images_cpu[i, 0], cmap="gray")
        plt.title(f"T:{labels[i].item()} P:{preds[i].item()}")
        plt.axis("off")
    out_path = fig_dir / "sample_images_model2.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved sample figure to: {out_path}")


if __name__ == "__main__":
    main()
