from __future__ import annotations
import torch
import torch.nn as nn


class FashionCNN(nn.Module):
    """
    Baseline CNN for Fashion-MNIST (28x28 grayscale).
    conv1, conv2 layer names are important for activation visualization (hooks).
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # 28x28
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # 28x28
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14
        self.dropout = nn.Dropout(p=0.25)

        # After pool: 64 x 14 x 14
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # logits
        return x
