import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(in_channels=150, out_channels=3, dim=128, depth=8, kernel_size=9):
    return nn.Sequential(
        nn.Conv2d(in_channels, dim, kernel_size=kernel_size, padding='same', stride=1),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.Conv2d(dim, out_channels, kernel_size=1),
        nn.GELU(),
        nn.BatchNorm2d(out_channels),
        # nn.AdaptiveAvgPool2d((1,1)),
        # nn.Flatten(),
        # nn.Linear(dim, n_classes)
    )