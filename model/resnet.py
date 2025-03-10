"""
Implements a ResNet, which will be used to encode CMD data for compression
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T 

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
                    padding=1,
                    padding_mode='circular',
                ),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode='circular',
                ),
            ]
        )
        
        self.batch_norms = nn.ModuleList(
            [
                nn.BatchNorm2d(out_channels),
                nn.BatchNorm2d(out_channels),
            ]
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False, padding_mode='circular',
                ),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        
        logits = self.convs[0](x)
        logits = self.batch_norms[0](logits)
        logits = F.relu(logits)
        logits = self.convs[1](logits)
        logits = self.batch_norms[1](logits)
        logits = F.relu(logits)
        return logits


class ResNet(nn.Module):
    """Residual convolutional network (ResNet 18 architecture)"""

    def __init__(self, in_channels: int, latent_img_channels: int = 32, blur_kernel_size = 1, fc_out_dim: int = 512):
        super(ResNet, self).__init__()
        # CAMELS Multifield Dataset is 256x256
        self.in_layer = nn.Sequential(
            
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False, padding_mode='circular'
                ),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        )
        self.resnet_layers = nn.ModuleList(
            [
                self._make_layer(in_channels=64, out_channels=64, num_blocks=1, stride=1),
                self._make_layer(in_channels=64, out_channels=64, num_blocks=1, stride=1),
                self._make_layer(in_channels=64, out_channels=128, num_blocks=1, stride=2),
                self._make_layer(in_channels=128, out_channels=128, num_blocks=1, stride=2),
                self._make_layer(in_channels=128, out_channels=256, num_blocks=1, stride=1),
                self._make_layer(in_channels=256, out_channels=latent_img_channels, num_blocks=1, stride=2),
            ]
        )
        self.blur = T.GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_kernel_size / 3)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(latent_img_channels, fc_out_dim)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResnetBlock(in_channels, out_channels, stride))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        x = self.in_layer(x)
        for i, layer in enumerate(self.resnet_layers):
            x = layer(x)
        return x, self.fc(self.pool(x).view(x.shape[0], -1))
    
class ResNetEncoder(nn.Module):
    """Residual convolutional network (ResNet 18 architecture)"""

    def __init__(self, in_channels: int, latent_img_channels: int = 32,):
        super(ResNetEncoder, self).__init__()
        self.resnet_list = nn.ModuleList(
            [
                ResNet(in_channels=in_channels, latent_img_channels=latent_img_channels, blur_kernel_size = 1),
                ResNet(in_channels=in_channels, latent_img_channels=latent_img_channels, blur_kernel_size = 1),
                ResNet(in_channels=in_channels, latent_img_channels=latent_img_channels, blur_kernel_size = 5),
                ResNet(in_channels=in_channels, latent_img_channels=latent_img_channels, blur_kernel_size = 7),
                ResNet(in_channels=in_channels, latent_img_channels=latent_img_channels, blur_kernel_size = 11),
            ]
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        latent_list = [layer(x) for layer in self.resnet_list]
        spatial = torch.cat([spatial for spatial, _ in latent_list], dim=1)
        repr = torch.cat([repr for _, repr in latent_list], dim=1)
        return spatial, repr