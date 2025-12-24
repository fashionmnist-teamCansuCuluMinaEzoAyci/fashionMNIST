import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Model2CNN(nn.Module):
    """
    Fashion-MNIST için: overfit azaltmaya odaklı CNN
    - BatchNorm + Dropout
    - Global Average Pooling (daha az parametre)
    """
    def __init__(self, num_classes=10, dropout=0.30):
        super().__init__()

        self.features = nn.Sequential(
            ConvBNReLU(1, 32),
            ConvBNReLU(32, 32),
            nn.MaxPool2d(2),            # 28 -> 14
            nn.Dropout2d(0.10),

            ConvBNReLU(32, 64),
            ConvBNReLU(64, 64),
            nn.MaxPool2d(2),            # 14 -> 7
            nn.Dropout2d(0.15),

            ConvBNReLU(64, 128),
            nn.Dropout2d(0.20),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)  # (B, 128, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


def build_model():
    return Model2CNN(num_classes=10, dropout=0.30)
