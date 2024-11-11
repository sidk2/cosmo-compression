"""
Implements a ResNet, which will be used to encode CMD data for compression
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    """Basic building block for ResNet"""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super(ResnetBlock, self).__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                ),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                ),
            ]
        )
        self.batch_norms = nn.ModuleList(
            [
                nn.BatchNorm2d(out_channels),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        logits = self.convs[0](x)
        logits = self.batch_norms[0](logits)
        logits = F.relu(logits)
        logits = self.convs[1](x)
        logits = self.batch_norms[1](logits)
        logits += self.downsample(x)
        logits = F.relu(logits)
        return logits


class ResNet(nn.Module):
    """Residual convolutional network (ResNet 18 architecture)"""

    def __init__(self, in_channels: int, latent_dim: int):
        super(ResNet, self).__init__()
        # CAMELS Multifield Dataset is 256x256
        self.in_channels = 64
        self.in_layer = nn.Sequential(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        self.resnet_layers = nn.ModuleList(
            [
                self._make_layer(64, 2, stride=1),
                self._make_layer(64, 2, stride=1),
                self._make_layer(128, 2, stride=1),
                self._make_layer(256, 2, stride=1),
                self._make_layer(512, 2, stride=1),
            ]
        )
        # Downsampled to 64x64
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, latent_dim)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResnetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        x = self.in_layer(x)
        for layer in self.resnet_layers:
            x = layer(x)
        x = self.avgpool(x)
        return self.fc(x)
