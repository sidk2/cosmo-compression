"""Implements a UNet"""

# import torch
# import torch.nn as nn

# from cosmo_compression.model import gdn


# def compute_groups(channels: int) -> int:
#     """Compute the number of groups for GroupNorm"""
#     num_groups = 1
#     while channels % 2 == 0:
#         channels //= 2
#         num_groups *= 2

#     return min(num_groups, 8)


# class AdaGN(nn.Module):
#     """
#     AdaGN allows model to modulate layer activations with conditioning latent
#     """

#     def __init__(self, num_channels: int, num_groups: int):
        
#         super().__init__()
#         self.gn = nn.GroupNorm(num_channels=num_channels, num_groups=num_groups)

#     def forward(
#         self,
#         x: torch.Tensor,
#         t_s: torch.Tensor | None = None,
#         t_b: torch.Tensor | None = None,
#         z_s: torch.Tensor | None = None,
#         z_b: torch.Tensor | None = None,
#     ) -> torch.Tensor:
#         """Overloads forward method of nn.Module"""

#         # Channelwise modulation of the latent by the timestep embedding
#         if t_s is None or t_b is None:
#             return self.gn(x)
#         elif z_s is None or z_b is None:
#             return t_s[:, :, None, None] * self.gn(x) + t_b[:, :, None, None]

#         else:
#             return (
#                 z_s[:, :, None, None]
#                 * (t_s[:, :, None, None] * self.gn(x) + t_b[:, :, None, None])
#                 + z_b[:, :, None, None]
#             )


# class SelfAttention(nn.Module):
#     """Implementation of a self-attention module"""

#     def __init__(self, channels: int):
#         super(SelfAttention, self).__init__()
#         self.channels = channels
#         self.mha = nn.MultiheadAttention(channels, 1, batch_first=True)
#         self.ln = nn.LayerNorm([channels])
#         self.ff_self = nn.Sequential(
#             nn.LayerNorm([channels]),
#             nn.Linear(channels, channels),
#             nn.GELU(),
#             nn.Linear(channels, channels),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Overloads forward pass of nn.Module"""
#         size = x.shape[-1]
#         x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
#         x, _ = self.mha(x, x, x)
#         # attention_value = attention_value + x
#         x = self.ff_self(x) + x
#         return x.swapaxes(2, 1).view(-1, self.channels, size, size)

# def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
#     """3x3 sub-pixel convolution for up-sampling."""
#     return nn.Sequential(
#         nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3, padding=1), nn.PixelShuffle(r)
#     )


# class UpsamplingUNetConv(nn.Module):
#     """2 sets of convolution plus batch norm. Basic UNet building block."""

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         int_channels: int | None = None,
#         residual: bool = False,
#         time_dim: int = 256,
#         latent_vec_dim: int = 14,
#     ):
#         super().__init__()
#         self.residual = residual
#         if not int_channels:
#             int_channels = out_channels
#         self.conv1 = nn.Conv2d(
#             in_channels=in_channels, out_channels=int_channels, kernel_size=3, padding=1
#         )
#         self.gn_1 = AdaGN(
#             num_channels=int_channels,
#             num_groups=compute_groups(int_channels),
#         )
#         self.gelu = nn.GELU()
#         self.conv2 = subpel_conv3x3(in_ch=int_channels, out_ch=out_channels, r=2)
#         self.gn_2 = AdaGN(
#             num_channels=out_channels,
#             num_groups=compute_groups(out_channels),
#         )

#         self.t_scale_proj_1 = nn.Linear(time_dim, int_channels)
#         self.t_bias_proj_1 = nn.Linear(time_dim, int_channels)

#         self.t_scale_proj_2 = nn.Linear(time_dim, out_channels)
#         self.t_bias_proj_2 = nn.Linear(time_dim, out_channels)

#         self.z_scale_proj_1 = nn.Linear(latent_vec_dim, int_channels)
#         self.z_bias_proj_1 = nn.Linear(latent_vec_dim, int_channels)

#         self.z_scale_proj_2 = nn.Linear(latent_vec_dim, out_channels)
#         self.z_bias_proj_2 = nn.Linear(latent_vec_dim, out_channels)

#     def forward(self, x: torch.Tensor, t=None, z=None) -> torch.Tensor:
#         """Overloads forward method of nn.Module"""
#         # t is shape [batch_size]

#         t_s1 = self.t_scale_proj_1(t) if t is not None else None
#         t_b1 = self.t_bias_proj_1(t) if t is not None else None

#         t_s2 = self.t_scale_proj_2(t) if t is not None else None
#         t_b2 = self.t_bias_proj_2(t) if t is not None else None

#         z_s1 = self.z_scale_proj_1(z) if z is not None else None
#         z_b1 = self.z_bias_proj_1(z) if z is not None else None

#         z_s2 = self.z_scale_proj_2(z) if z is not None else None
#         z_b2 = self.z_bias_proj_2(z) if z is not None else None

#         x = self.conv1(x)
#         x = self.gn_1(x, t_s1, t_b1, z_s1, z_b1)
#         x = self.gelu(x)
#         x = self.conv2(x)
#         x = self.gn_2(x, t_s2, t_b2, z_s2, z_b2)
#         x = x + self.gelu(x)

#         return x


# class UNetConv(nn.Module):
#     """2 sets of convolution plus batch norm. Basic UNet building block."""

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         time_dim: int,
#         int_channels: int | None = None,
#         residual: bool = False,
#         latent_vec_dim: int = 14 * 9,
#     ):
#         super().__init__()
#         self.residual = residual
#         if not int_channels:
#             int_channels = out_channels
#         self.conv1 = nn.Conv2d(
#             in_channels,
#             int_channels,
#             kernel_size=3,
#             padding=1,
#             bias=False,
#             padding_mode="circular",
#         )
#         self.gn_1 = AdaGN(
#             num_channels=int_channels,
#             num_groups=compute_groups(int_channels),
#         )
#         self.gelu = nn.GELU()
#         self.conv2 = nn.Conv2d(
#             int_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             bias=False,
#             padding_mode="circular",
#         )
#         self.gn_2 = AdaGN(
#             num_channels=out_channels,
#             num_groups=compute_groups(out_channels),
#         )

#         self.t_scale_proj_1 = nn.Linear(time_dim, int_channels)
#         self.t_bias_proj_1 = nn.Linear(time_dim, int_channels)

#         self.t_scale_proj_2 = nn.Linear(time_dim, out_channels)
#         self.t_bias_proj_2 = nn.Linear(time_dim, out_channels)

#         self.z_scale_proj_1 = nn.Linear(latent_vec_dim, int_channels)
#         self.z_bias_proj_1 = nn.Linear(latent_vec_dim, int_channels)

#         self.z_scale_proj_2 = nn.Linear(latent_vec_dim, out_channels)
#         self.z_bias_proj_2 = nn.Linear(latent_vec_dim, out_channels)

#     def forward(
#         self,
#         x: torch.Tensor,
#         t: torch.Tensor | None = None,
#         z: torch.Tensor | None = None,
#     ) -> torch.Tensor:
#         """Overloads forward method of nn.Module"""
#         # t is shape [batch_size]

#         t_s1 = self.t_scale_proj_1(t) if t is not None else None
#         t_b1 = self.t_bias_proj_1(t) if t is not None else None

#         t_s2 = self.t_scale_proj_2(t) if t is not None else None
#         t_b2 = self.t_bias_proj_2(t) if t is not None else None

#         z_s1 = self.z_scale_proj_1(z) if z is not None else None
#         z_b1 = self.z_bias_proj_1(z) if z is not None else None

#         z_s2 = self.z_scale_proj_2(z) if z is not None else None
#         z_b2 = self.z_bias_proj_2(z) if z is not None else None

#         x = self.conv1(x)
#         x = self.gn_1(x, t_s1, t_b1, z_s1, z_b1)
#         x = self.gelu(x)
#         x = self.conv2(x)
#         x = self.gn_2(x, t_s2, t_b2, z_s2, z_b2)
#         x = x + self.gelu(x)

#         return x


# class DownStep(nn.Module):
#     """Downscaling input with max pool and double conv"""

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         int_channels: int | None = None,
#         time_dim: int = 256,
#     ):
#         super(
#             DownStep,
#             self,
#         ).__init__()
#         int_channels = int_channels if int_channels else in_channels

#         self.pooling = nn.MaxPool2d(kernel_size=2)
#         self.conv1 = UNetConv(
#             in_channels=in_channels,
#             out_channels=int_channels,
#             time_dim=time_dim,
#             residual=True,
#         )
#         self.conv2 = UNetConv(
#             in_channels=int_channels,
#             out_channels=out_channels,
#             time_dim=time_dim,
#             residual=True,
#         )
#         self.gdn_layer = gdn.GDN(ch=out_channels, device="cuda")

#     def forward(self, x: torch.Tensor, t, z = None) -> torch.Tensor:
#         """Overloads forward method of nn.Module"""
#         return self.gdn_layer(self.conv2(self.conv1(self.pooling(x), t, z), t, z))


# class UpStep(nn.Module):
#     """Upsample latent and incorporate residual"""

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         res_channels: int,
#         time_dim: int = 256,
#     ):
#         super().__init__()
#         self.conv1 = UpsamplingUNetConv(
#             in_channels=in_channels,
#             int_channels=in_channels,
#             out_channels=in_channels,
#             residual=True,
#             time_dim=time_dim,
#         )
#         self.conv2 = UNetConv(
#             in_channels=(in_channels + res_channels),
#             int_channels=(in_channels + res_channels) // 2,
#             out_channels=out_channels,
#             time_dim=time_dim,
#         )

#         self.gdn_layer = gdn.GDN(ch=out_channels, device="cuda", inverse=True)

#     def forward(self, x: torch.Tensor, res_x: torch.Tensor, t, z = None) -> torch.Tensor:
#         """Overloads forward method of nn.Module"""
#         x = self.conv1(x, t, z)
#         x = torch.cat([res_x, x], dim=1)
#         x = self.conv2(x, t, z)
#         return self.gdn_layer(x)


# class UpStepWoutRes(nn.Module):
#     """Upsample latent and incorporate residual"""

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         time_dim: int = 256,
#     ):
#         super().__init__()

#         self.conv1 = UNetConv(
#             in_channels=in_channels,
#             out_channels=in_channels,
#             time_dim=time_dim,
#         )

#         self.conv2 = UpsamplingUNetConv(
#             in_channels=in_channels,
#             int_channels=in_channels // 2,
#             out_channels=out_channels,
#             residual=True,
#             time_dim=time_dim,
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Overloads forward method of nn.Module"""
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return x


# class UNet(nn.Module):
#     """Creates a UNet using the building block modules in this file"""

#     def __init__(
#         self,
#         n_channels: int,
#         time_dim: int = 256,
#         latent_img_channels: int = 32,
#         latent_vec_dim: int = 14 * 9,
#     ):
#         super(UNet, self).__init__()
#         self.time_dim = time_dim
#         self.n_channels = n_channels
#         self.num_latent_channels = latent_img_channels

#         self.dropout = nn.Dropout2d(p=0.1)
#         self.inc = UNetConv(
#             in_channels=n_channels,
#             out_channels=64,
#             time_dim=time_dim,
#             residual=True,
#         )
#         self.down1 = DownStep(
#             in_channels=64 + self.num_latent_channels,
#             out_channels=128,
#             time_dim=time_dim,
#         )
#         self.down2 = DownStep(
#             in_channels=128 ,
#             out_channels=256,
#             time_dim=time_dim,
#         )
#         self.sa2 = SelfAttention(channels=256)
#         self.down3 = DownStep(
#             in_channels=256 ,
#             out_channels=512,
#             time_dim=time_dim,
#         )
#         self.sa3 = SelfAttention(channels=512)
#         self.down4 = DownStep(
#             in_channels=512 ,
#             out_channels=512,
#             time_dim=time_dim,
#         )

#         self.up0 = UpStep(
#             in_channels=512,
#             res_channels=512 ,
#             out_channels=256,
#             time_dim=time_dim,
#         )
#         self.sa0_inv = SelfAttention(channels=256)
#         self.up1 = UpStep(
#             in_channels=256,
#             res_channels=256 ,
#             out_channels=256,
#             time_dim=time_dim,
#         )
#         self.sa1_inv = SelfAttention(channels=256)
#         self.up2 = UpStep(
#             in_channels=256,
#             res_channels=128 ,
#             out_channels=128,
#             time_dim=time_dim,
#         )
#         self.up3 = UpStep(
#             in_channels=128,
#             res_channels=64 + self.num_latent_channels,
#             out_channels=64,
#             time_dim=time_dim,
#         )

#         self.outc = nn.Conv2d(
#             in_channels=64,
#             out_channels=n_channels,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             bias=True,
#         )
#         self.latent_upsampler_0 = nn.Sequential(
#             UpStepWoutRes(
#                 in_channels=int(self.num_latent_channels),
#                 out_channels=int(self.num_latent_channels ),
#                 time_dim=time_dim,
#             ),
#             UpStepWoutRes(
#                 in_channels=int(self.num_latent_channels ),
#                 out_channels=int(self.num_latent_channels ),
#                 time_dim=time_dim,
#             ),
#             UpStepWoutRes(
#                 in_channels=int(self.num_latent_channels ),
#                 out_channels=int(self.num_latent_channels ),
#                 time_dim=time_dim,
#             ),
#             UpStepWoutRes(
#                 in_channels=int(self.num_latent_channels ),
#                 out_channels=int(self.num_latent_channels ),
#                 time_dim=time_dim,
#             ),
#         )

#         self.latent_vec_dim = latent_vec_dim

#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(self.num_latent_channels, 14 * 9)

#     def pos_encoding(self, t: int, channels: int) -> torch.Tensor:
#         """Generate sinusoidal timestep embedding"""
#         device = (
#             torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#         )
#         inv_freq = 1.0 / (
#             10000 ** (torch.arange(0, channels, 2, device=device).float() / channels)
#         )
#         pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
#         return pos_enc

#     def forward(
#         self,
#         x: torch.Tensor,
#         t: torch.Tensor | None = None,
#         z: torch.Tensor | None = None,
#     ) -> torch.Tensor:
#         """Overloads forward method of nn.Module
#         t is the full timestep embedding, with dimension time_dim
#         z is the full latent, which will be split into latent_dim chunks
#         """

#         spatial = z  # [B, C, H, W]
#         B, C, H, W = spatial.shape

#         t = t.unsqueeze(-1)
#         if t.shape[0] != B:
#             t = t.expand(B, t.shape[0])

#         start = 0
#         end = C
                
#         for b in range(B):
#             t_val = float(t[b])
#             num_mask = int(C * t_val)
#             unmasked = C - num_mask

#             # index of the last channel we want to keep gradients on
#             last_unmasked = unmasked - 1
#             # zero-out everything *after* last_unmasked
#             if num_mask > 0:
#                 spatial[b, last_unmasked+1 :, ...] = 0

#         # --- End of masking ---

#         vec_latent = self.fc(self.pool(spatial).squeeze())
#         if vec_latent.dim() == 1:
#             vec_latent = vec_latent.unsqueeze(0)

#         t = self.pos_encoding(t, self.time_dim)

#         # Downsampling stages
#         x1 = self.inc(x, t, vec_latent)
#         x1 = torch.cat(
#             [
#                 self.latent_upsampler_0(
#                     spatial
#                 ),
#                 x1,
#             ],
#             dim=1,
#         )
#         x2 = self.down1(x1, t)
#         x3 = self.down2(
#             x2, t
#         )
#         x3 = self.sa2(x3)
#         x4 = self.down3(
#             x3, t
#         )
#         x4 = self.sa3(x4)
#         x5 = self.down4(
#             x4, t
#         )

#         # Upsampling stages
#         x = self.up0(
#             x5, x4, t
#         )
#         x = self.up1(
#             x, x3, t
#         )
#         x = self.sa1_inv(x)
#         x = self.up2(
#             x, x2, t
#         )
#         x = self.up3(
#             x, x1, t
#         )

#         return self.outc(x)


import torch
import torch.nn as nn
from typing import Optional

from cosmo_compression.model import gdn


def compute_groups(channels: int) -> int:
    """Compute the number of groups for GroupNorm (max 8)."""
    num_groups = 1
    while channels % 2 == 0 and num_groups < 8:
        channels //= 2
        num_groups *= 2
    return num_groups


class AdaGN(nn.Module):
    """
    Adaptive GroupNorm: applies GroupNorm followed by optional
    timestep and latent conditioning.
    """

    def __init__(self, num_channels: int, num_groups: int):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

    def forward(
        self,
        x: torch.Tensor,
        t_s: Optional[torch.Tensor] = None,
        t_b: Optional[torch.Tensor] = None,
        z_s: Optional[torch.Tensor] = None,
        z_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_norm = self.gn(x)
        if t_s is None or t_b is None:
            return x_norm

        y = t_s[..., None, None] * x_norm + t_b[..., None, None]
        if z_s is not None and z_b is not None:
            y = z_s[..., None, None] * y + z_b[..., None, None]
        return y


class SelfAttention(nn.Module):
    """Simple self-attention block with residual feed-forward."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=1, batch_first=True)
        self.layer_norm = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, N, C]
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).transpose(1, 2)
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        out = self.ff(attn_out) + attn_out
        out = out.transpose(1, 2).view(b, c, h, w)
        return out


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 conv followed by PixelShuffle upsampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r * r, kernel_size=3, padding=1),
        nn.PixelShuffle(r),
    )


class UNetBlock(nn.Module):
    """Generic double-conv block with AdaGN and optional upsampling."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_dim: int,
        lat_dim: int,
        use_subpel: bool = False,
    ):
        super().__init__()
        mid_ch = out_ch if not use_subpel else in_ch

        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=not use_subpel)
        self.norm1 = AdaGN(mid_ch, compute_groups(mid_ch))
        self.act = nn.GELU()

        if use_subpel:
            self.conv2 = subpel_conv3x3(mid_ch, out_ch, r=2)
        else:
            self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = AdaGN(out_ch, compute_groups(out_ch))

        # Projections for conditioning
        self.t_s1 = nn.Linear(time_dim, mid_ch)
        self.t_b1 = nn.Linear(time_dim, mid_ch)
        self.t_s2 = nn.Linear(time_dim, out_ch)
        self.t_b2 = nn.Linear(time_dim, out_ch)
        self.z_s1 = nn.Linear(lat_dim, mid_ch)
        self.z_b1 = nn.Linear(lat_dim, mid_ch)
        self.z_s2 = nn.Linear(lat_dim, out_ch)
        self.z_b2 = nn.Linear(lat_dim, out_ch)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # compute conditioning vectors
        t_s1 = self.t_s1(t) if t is not None else None
        t_b1 = self.t_b1(t) if t is not None else None
        t_s2 = self.t_s2(t) if t is not None else None
        t_b2 = self.t_b2(t) if t is not None else None
        z_s1 = self.z_s1(z) if z is not None else None
        z_b1 = self.z_b1(z) if z is not None else None
        z_s2 = self.z_s2(z) if z is not None else None
        z_b2 = self.z_b2(z) if z is not None else None

        x = self.conv1(x)
        x = self.norm1(x, t_s1, t_b1, z_s1, z_b1)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x, t_s2, t_b2, z_s2, z_b2)
        return x + self.act(x)


class DownStep(nn.Module):
    """Downsample by maxpool + double-conv + GDN."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, lat_dim: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block1 = UNetBlock(in_ch, out_ch, time_dim, lat_dim)
        self.block2 = UNetBlock(out_ch, out_ch, time_dim, lat_dim)
        self.gdn = gdn.GDN(ch=out_ch, device="cuda")

    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.block1(x, t, z)
        x = self.block2(x, t, z)
        return self.gdn(x)


class UpStep(nn.Module):
    """Upsample with subpixel conv, concat residual, double-conv, then inverse GDN."""

    def __init__(
        self,
        in_ch: int,
        res_ch: int,
        out_ch: int,
        time_dim: int,
        lat_dim: int,
    ):
        super().__init__()
        self.up_block = UNetBlock(in_ch, in_ch, time_dim, lat_dim, use_subpel=True)
        self.conv = UNetBlock(in_ch + res_ch, out_ch, time_dim, lat_dim)
        self.igdn = gdn.GDN(ch=out_ch, device="cuda", inverse=True)

    def forward(
        self,
        x: torch.Tensor,
        res: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up_block(x, t, z)
        x = torch.cat([res, x], dim=1)
        x = self.conv(x, t, z)
        return self.igdn(x)


class UNet(nn.Module):
    """Full UNet with attention and latent conditioning."""

    def __init__(
        self,
        n_channels: int,
        time_dim: int = 256,
        lat_img_ch: int = 32,
        lat_vec_dim: int = 14 * 9,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.latent_img_ch = lat_img_ch
        self.latent_vec_dim = lat_vec_dim

        self.inc = UNetBlock(n_channels, 64, time_dim, lat_vec_dim)
        self.latent_ups = nn.Sequential(
            *[UNetBlock(lat_img_ch, lat_img_ch, time_dim, lat_vec_dim, use_subpel=True) for _ in range(4)]
        )

        self.down1 = DownStep(64 + lat_img_ch, 128, time_dim, lat_vec_dim)
        self.down2 = DownStep(128, 256, time_dim, lat_vec_dim)
        self.sa2 = SelfAttention(256)
        self.down3 = DownStep(256, 512, time_dim, lat_vec_dim)
        self.sa3 = SelfAttention(512)
        self.down4 = DownStep(512, 512, time_dim, lat_vec_dim)

        self.up0 = UpStep(512, 512, 256, time_dim, lat_vec_dim)
        self.sa0 = SelfAttention(256)
        self.up1 = UpStep(256, 256, 256, time_dim, lat_vec_dim)
        self.sa1 = SelfAttention(256)
        self.up2 = UpStep(256, 128, 128, time_dim, lat_vec_dim)
        self.up3 = UpStep(128, 64 + lat_img_ch, 64, time_dim, lat_vec_dim)

        self.out_conv = nn.Conv2d(64, n_channels, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(lat_img_ch, lat_vec_dim)

    def pos_encoding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal position encoding for timestep."""
        device = t.device
        channels = self.time_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=device) / channels))
        t_expanded = t.unsqueeze(1)
        sinusoid = torch.cat([
            torch.sin(t_expanded * inv_freq),
            torch.cos(t_expanded * inv_freq),
        ], dim=-1)
        return sinusoid

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        # Mask latent channels based on t
        
        # If t is 0 dimensional, reshape it to a 1-dimensional tensor
        if t.dim() == 0:
            t = t.unsqueeze(0)
        b, c, h, w = z.shape
        mask_count = (c * t.float()).long()
        for i in range(b):
            if mask_count[i] > 0:
                z[i, c - mask_count[i]:] = 0

        # latent vector from spatial latent
        vec = self.fc(self.pool(z).view(b, -1))
        t_emb = self.pos_encoding(t)

        # Encoding path
        x1 = self.inc(x, t_emb, vec)
        x1 = torch.cat([self.latent_ups(z), x1], dim=1)
        x2 = self.down1(x1, t_emb, vec)
        x3 = self.down2(x2, t_emb, vec)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t_emb, vec)
        x4 = self.sa3(x4)
        x5 = self.down4(x4, t_emb, vec)

        # Decoding path
        y = self.up0(x5, x4, t_emb, vec)
        y = self.up1(y, x3, t_emb, vec)
        y = self.sa1(y)
        y = self.up2(y, x2, t_emb, vec)
        y = self.up3(y, x1, t_emb, vec)
        return self.out_conv(y)
