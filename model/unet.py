"""Implements a UNet"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Implementation of a self-attention module"""

    def __init__(self, channels: int):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Overloads forward pass of nn.Module"""
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class UNetConv(nn.Module):
    """2 sets of convolution plus batch norm. Basic UNet building block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        int_channels: int | None = None,
        residual: bool = False,
    ):
        super().__init__()
        self.residual = residual
        if not int_channels:
            int_channels = out_channels
        self.unet_conv = nn.Sequential(
            nn.Conv2d(in_channels, int_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, int_channels),
            nn.GELU(),
            nn.Conv2d(int_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        if self.residual:
            return F.gelu(input=(x + self.unet_conv(x)))
        else:
            return self.unet_conv(x)


class DownStep(nn.Module):
    """Downscaling input with max pool and double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        int_channels: int | None = None,
        emb_dim: int = 256,
    ):
        super(
            DownStep,
            self,
        ).__init__()
        int_channels = int_channels if int_channels else in_channels
        self.layers = nn.Sequential(
            [
                nn.MaxPool2d(kernel_size=2),
                UNetConv(
                    in_channels=in_channels,
                    out_channels=int_channels,
                    residual=True,
                ),
                UNetConv(
                    in_channels=int_channels,
                    out_channels=out_channels,
                ),
            ]
        )
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return self.layers(x) + emb


class UpStep(nn.Module):
    """Upsample latent and incorporate residual"""

    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            UNetConv(in_channels, in_channels, residual=True),
            UNetConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(
        self, x: torch.Tensor, res_x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        x = self.up(x)
        x = torch.cat([res_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    """Creates a UNet using the building block modules in this file"""

    def __init__(
        self,
        n_channels: int,
        time_dim: int = 256,
    ):
        super(UNet, self).__init__()
        self.time_dim = time_dim
        self.n_channels = n_channels
        
        self.embedding_layer = nn.Embedding()

        self.inc = UNetConv(in_channels=n_channels, out_channels=64)
        self.down1 = DownStep(in_channels=64, out_channels=128)
        self.sa1 = SelfAttention(channels=128)
        self.down2 = DownStep(in_channels=128, out_channels=256)
        self.sa2 = SelfAttention(channels=256)
        self.down3 = DownStep(in_channels=256, out_channels=512)
        self.sa3 = SelfAttention(channels=512)
        
        self.up1 = UpStep(in_channels=1024, out_channels=256)
        self.sa4 = SelfAttention(channels=256)
        self.up2 = UpStep(in_channels=512, out_channels=128)
        self.sa5 = SelfAttention(channels=128)
        self.up3 = UpStep(in_channels=256, out_channels=64)
        self.sa6 = SelfAttention(channels=64)
        self.outc = nn.Conv2d(in_channels=64, out_channels=n_channels, kernel_size=1)
        
    def pos_encoding(self, t: int, channels: int) -> torch.Tensor:
        '''Generate sinusoidal timestep embedding'''
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x: torch.Tensor, t: torch.Tensor, ) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
