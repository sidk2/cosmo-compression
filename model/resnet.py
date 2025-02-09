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
        
        self.t_emb_dim = out_channels
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1, padding_mode='zeros'
                ),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1, padding_mode='zeros'
                ),
            ]
        )
        
        self.time_layers = nn.ModuleList(
            [
                nn.Linear(in_features=out_channels, out_features=out_channels),
                nn.Linear(in_features=out_channels, out_features=out_channels),
            ]
        )
        self.batch_norms = nn.ModuleList(
            [
                nn.BatchNorm2d(out_channels),
                nn.BatchNorm2d(out_channels),
            ]
        )
        # self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False, padding_mode='zeros'
                ),
                nn.BatchNorm2d(out_channels),
            )
    def pos_encoding(self, t: int, channels: int) -> torch.Tensor:
        """Generate sinusoidal timestep embedding"""
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if channels == 1:
            t = t.view(-1, 1).repeat(1, max(1,channels // 2))
            inv_freq = 1.0 / (
                10000 ** (torch.arange(0, channels, 2, device=device).float() / channels)
            )
            pos_enc_a = torch.sin(t * inv_freq)
            return pos_enc_a
        
        t = t.view(-1, 1).repeat(1, max(1,channels // 2))
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=device).float() / channels)
        )
        pos_enc_a = torch.sin(t * inv_freq)
        pos_enc_b = torch.cos(t * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x, t = x
        
        t_enc = self.pos_encoding(t, channels=self.t_emb_dim)
        
        t_emb_1 = self.time_layers[0](t_enc)
        t_emb_2 = self.time_layers[1](t_enc)
        
        logits = self.convs[0](x) * t_emb_1.unsqueeze(-1).unsqueeze(-1)
        logits = self.batch_norms[0](logits)
        logits = F.relu(logits)
        logits = self.convs[1](logits) * t_emb_2.unsqueeze(-1).unsqueeze(-1)
        logits = self.batch_norms[1](logits)
        logits = F.relu(logits)
        return logits


class ResNet(nn.Module):
    """Residual convolutional network (ResNet 18 architecture)"""

    def __init__(self, in_channels: int, latent_dim: int, latent_img_channels: int = 32,):
        super(ResNet, self).__init__()
        # CAMELS Multifield Dataset is 256x256
        self.in_channels = 64
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
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(latent_img_channels, latent_dim)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResnetBlock(in_channels, out_channels, stride))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        x = self.in_layer(x)
        # print("latent: ", x)
        for i, layer in enumerate(self.resnet_layers):
            x = layer((x, t))
        return x