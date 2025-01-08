"""Implements a UNet

"""

import torch
import torch.nn as nn

class AdaGN(nn.Module):
    """
    AdaGN allows model to modulate layer activations with conditioning latent
    """

    def __init__(self, num_channels: int, num_groups: int):
        super().__init__()
        self.gn = nn.GroupNorm(num_channels=num_channels, num_groups=num_groups)

    def forward(
        self,
        x: torch.Tensor,
        z_s: torch.Tensor,
        z_b: torch.Tensor,
        t_s: torch.Tensor | None = None,
        t_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        norm_x = z_s[:,:,None, None] * (t_s[:,:,None, None] * self.gn(x) + t_b[:,:,None, None]) + z_b[:,:,None, None]
        return norm_x


class SelfAttention(nn.Module):
    """Implementation of a self-attention module"""

    def __init__(self, channels: int):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 2, batch_first=True)
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
        x, _ = self.mha(x, x, x)
        # attention_value = attention_value + x
        x = self.ff_self(x) + x
        return x.swapaxes(2, 1).view(-1, self.channels, size, size)


class UNetConv(nn.Module):
    """2 sets of convolution plus batch norm. Basic UNet building block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        time_dim: int,
        int_channels: int | None = None,
        residual: bool = False,
    ):
        super().__init__()
        self.residual = residual
        if not int_channels:
            int_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels, int_channels, kernel_size=3, padding=1, bias=False, padding_mode='circular'
        )
        self.gn_1 = AdaGN(num_channels=int_channels, num_groups=(8 if int_channels % 8 == 0 else (13 if int_channels % 13 == 0 else int_channels)))
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(
            int_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode='circular'
        )
        self.gn_2 = AdaGN(num_channels=out_channels, num_groups=(8 if out_channels % 8 == 0 else (13 if out_channels % 13 == 0 else out_channels)))

        self.z_scale_proj_1 = nn.Linear(latent_dim, int_channels)
        self.z_bias_proj_1 = nn.Linear(latent_dim, int_channels)
        self.t_scale_proj_1 = nn.Linear(time_dim, int_channels)
        self.t_bias_proj_1 = nn.Linear(time_dim, int_channels)

        self.z_scale_proj_2 = nn.Linear(latent_dim, out_channels)
        self.z_bias_proj_2 = nn.Linear(latent_dim, out_channels)
        self.t_scale_proj_2 = nn.Linear(time_dim, out_channels)
        self.t_bias_proj_2 = nn.Linear(time_dim, out_channels)

    def forward(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        # t is shape [batch_size]
        z_s1 = self.z_scale_proj_1(z)
        z_b1 = self.z_bias_proj_1(z)
        t_s1 = self.t_scale_proj_1(t)
        t_b1 = self.t_bias_proj_1(t)

        z_s2 = self.z_scale_proj_2(z)
        z_b2 = self.z_bias_proj_2(z)
        t_s2 = self.t_scale_proj_2(t)
        t_b2 = self.t_bias_proj_2(t)

        x = self.conv1(x)
        x = self.gn_1(x, z_s1, z_b1, t_s1, t_b1)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gn_2(x, z_s2, z_b2, t_s2, t_b2)
        x = x + self.gelu(x)

        return x


class DownStep(nn.Module):
    """Downscaling input with max pool and double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        int_channels: int | None = None,
        latent_dim: int = 256,
        time_dim: int = 256,
    ):
        super(
            DownStep,
            self,
        ).__init__()
        int_channels = int_channels if int_channels else in_channels

        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.conv1 = UNetConv(
            in_channels=in_channels,
            out_channels=int_channels,
            latent_dim=latent_dim,
            time_dim = time_dim,
            residual=True,
        )
        self.conv2 = UNetConv(
            in_channels=int_channels,
            out_channels=out_channels,
            latent_dim=latent_dim,
            time_dim=time_dim,
            residual=True,
        )
    def forward(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        return self.conv2(self.conv1(self.pooling(x), z, t), z, t)

class UpStep(nn.Module):
    """Upsample latent and incorporate residual"""

    def __init__(self, in_channels: int, out_channels: int, latent_dim: int = 256, time_dim: int = 256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = UNetConv(
                in_channels=in_channels,
                out_channels=in_channels,
                latent_dim=latent_dim,
                time_dim=time_dim,
                residual=True,
            )
        
        self.conv2 = UNetConv(
                in_channels=in_channels,
                int_channels=in_channels // 2,
                out_channels=out_channels,
                latent_dim=latent_dim,
                time_dim=time_dim,
                residual=True,
            )
        
    def forward(
        self, x: torch.Tensor, res_x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        x = self.up(x)
        x = torch.cat([res_x, x], dim=1)
        x = self.conv1(x, z, t)
        x = self.conv2(x,z,t)
        return x


class UNet(nn.Module):
    """Creates a UNet using the building block modules in this file"""

    def __init__(
        self,
        n_channels: int,
        latent_dim: int = 256,
        time_dim: int = 256
    ):
        super(UNet, self).__init__()
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        self.n_channels = n_channels

        self.inc = UNetConv(
            in_channels=n_channels,
            out_channels=64,
            latent_dim=latent_dim,
            time_dim = time_dim,
            residual=True,
        )
        self.down1 = DownStep(in_channels=320, out_channels=128, latent_dim=latent_dim, time_dim=time_dim)
        # self.sa1 = SelfAttention(channels=128)
        self.down2 = DownStep(in_channels=128, out_channels=256, latent_dim=latent_dim, time_dim=time_dim)
        self.sa2 = SelfAttention(channels=256)
        self.down3 = DownStep(in_channels=256, out_channels=512, latent_dim=latent_dim, time_dim=time_dim)
        self.sa3 = SelfAttention(channels=512)
        self.down4 = DownStep(in_channels=512, out_channels=512, latent_dim=latent_dim, time_dim=time_dim)

        self.up0 = UpStep(in_channels=1024, out_channels=256, latent_dim=latent_dim, time_dim=time_dim)
        self.up1 = UpStep(in_channels=768, out_channels=256, latent_dim=latent_dim, time_dim=time_dim)
        self.sa4 = SelfAttention(channels=256)
        self.up2 = UpStep(in_channels=384, out_channels=128, latent_dim=latent_dim, time_dim=time_dim)
        self.sa5 = SelfAttention(channels=128)
        self.up3 = UpStep(in_channels=448, out_channels=64, latent_dim=latent_dim, time_dim=time_dim)
        # self.sa6 = SelfAttention(channels=64)
        self.outc = nn.Conv2d(in_channels=64, out_channels=n_channels, kernel_size=1, padding_mode='circular')
        
        self.upsampler = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def pos_encoding(self, t: int, channels: int) -> torch.Tensor:
        """Generate sinusoidal timestep embedding"""
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Overloads forward method of nn.Module
            t is the full timestep embedding, with dimension time_dim
            z is the full latent, which will be split into latent_dim chunks
        """
        
        # There are 9 up/down sampling layers, so z must be latent_dim*9 elements long
        latent_dim = self.latent_dim
        z, latent_img = z
        
        # Latent img is 512*16*16
        latent_img = self.upsampler(latent_img)
        assert z.shape[-1] == self.latent_dim*9

        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        x1 = self.inc(x, z[:, :latent_dim], t)
        x1 = torch.cat([latent_img, x1], dim=1)
        x2 = self.down1(x1, z[:, latent_dim:2*latent_dim], t)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2, z[:, 2*latent_dim:3*latent_dim], t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, z[:, 3*latent_dim:4*latent_dim], t)
        x4 = self.sa3(x4)
        x5 = self.down4(x4, z[:, 4*latent_dim:5*latent_dim], t)

        
        x = self.up0(x5, x4, z[:, 5*latent_dim:6*latent_dim], t)
        x = self.up1(x4, x3, z[:, 6*latent_dim:7*latent_dim], t)
        x = self.sa4(x)
        x = self.up2(x, x2, z[:, 7*latent_dim:8*latent_dim], t)
        # x = self.sa5(x)
        x = self.up3(x, x1, z[:, 8*latent_dim:], t)
        # x = self.sa6(x)
        output = self.outc(x)
        return output
