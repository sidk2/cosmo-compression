"""Implements a UNet"""

import torch
import torch.nn as nn

from cosmo_compression.model import gdn


def compute_groups(channels: int) -> int:
    """Compute the number of groups for GroupNorm"""
    num_groups = 1
    while channels % 2 == 0:
        channels //= 2
        num_groups *= 2

    return min(num_groups, 8)


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
        t_s: torch.Tensor | None = None,
        t_b: torch.Tensor | None = None,
        z_s: torch.Tensor | None = None,
        z_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Overloads forward method of nn.Module"""

        # Channelwise modulation of the latent by the timestep embedding
        if t_s is None or t_b is None:
            return self.gn(x)
        elif z_s is None or z_b is None:
            return t_s[:, :, None, None] * self.gn(x) + t_b[:, :, None, None]

        else:
            return (
                z_s[:, :, None, None]
                * (t_s[:, :, None, None] * self.gn(x) + t_b[:, :, None, None])
                + z_b[:, :, None, None]
            )


class SelfAttention(nn.Module):
    """Implementation of a self-attention module"""

    def __init__(self, channels: int):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 1, batch_first=True)
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


# class SelfAttention(nn.Module):
#     """Self-attention module using patching strategy to reduce sequence length"""

#     def __init__(self, channels: int, patch_size: int = 2):
#         """
#         Args:
#             channels (int): Number of input channels.
#             patch_size (int): Size of each patch (patch_size x patch_size).
#         """
#         super(SelfAttention, self).__init__()
#         self.channels = channels
#         self.patch_size = patch_size

#         # Patching: Embed each non-overlapping patch using a convolution.
#         # This reduces the spatial dimensions by a factor of patch_size.
#         self.patch_embed = nn.Conv2d(
#             channels, channels, kernel_size=patch_size, stride=patch_size
#         )

#         # Multi-head attention: expects input shape (batch, tokens, channels)
#         self.mha = nn.MultiheadAttention(channels, num_heads=1, batch_first=True)

#         # Feedforward network with residual connection.
#         self.ff_self = nn.Sequential(
#             nn.LayerNorm(channels),
#             nn.Linear(channels, channels),
#             nn.GELU(),
#             nn.Linear(channels, channels),
#         )

#         # Unpatching: Recover the original spatial resolution using a transposed convolution.
#         self.patch_unembed = nn.ConvTranspose2d(
#             channels, channels, kernel_size=patch_size, stride=patch_size
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x (torch.Tensor): Input tensor of shape (batch, channels, height, width).
#         Returns:
#             torch.Tensor: Output tensor with the same spatial resolution as the input.
#         """
#         # 1. Patchify the input: embed non-overlapping patches.
#         #    The output shape will be (batch, channels, H_p, W_p) where H_p = height/patch_size.
#         patches = self.patch_embed(x)

#         # 2. Flatten spatial dimensions to form tokens.
#         batch, c, H_p, W_p = patches.shape
#         # Reshape to (batch, tokens, channels)
#         tokens = patches.view(batch, c, H_p * W_p).transpose(1, 2)

#         # 3. Apply multi-head self-attention.
#         tokens, _ = self.mha(tokens, tokens, tokens)

#         # 4. Apply the feedforward network with a residual connection.
#         tokens = tokens + self.ff_self(tokens)

#         # 5. Reshape tokens back to patch grid.
#         patches = tokens.transpose(1, 2).view(batch, c, H_p, W_p)

#         # 6. Unpatch: Reconstruct the spatial dimensions.
#         out = self.patch_unembed(patches)
#         return out


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


class UpsamplingUNetConv(nn.Module):
    """2 sets of convolution plus batch norm. Basic UNet building block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        int_channels: int | None = None,
        residual: bool = False,
        time_dim: int = 256,
        latent_vec_dim: int = 14,
    ):
        super().__init__()
        self.residual = residual
        if not int_channels:
            int_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=int_channels, kernel_size=3, padding=1
        )
        self.gn_1 = AdaGN(
            num_channels=int_channels,
            num_groups=compute_groups(int_channels),
        )
        self.gelu = nn.GELU()
        self.conv2 = subpel_conv3x3(in_ch=int_channels, out_ch=out_channels, r=2)
        self.gn_2 = AdaGN(
            num_channels=out_channels,
            num_groups=compute_groups(out_channels),
        )

        self.t_scale_proj_1 = nn.Linear(time_dim, int_channels)
        self.t_bias_proj_1 = nn.Linear(time_dim, int_channels)

        self.t_scale_proj_2 = nn.Linear(time_dim, out_channels)
        self.t_bias_proj_2 = nn.Linear(time_dim, out_channels)

        self.z_scale_proj_1 = nn.Linear(latent_vec_dim, int_channels)
        self.z_bias_proj_1 = nn.Linear(latent_vec_dim, int_channels)

        self.z_scale_proj_2 = nn.Linear(latent_vec_dim, out_channels)
        self.z_bias_proj_2 = nn.Linear(latent_vec_dim, out_channels)

    def forward(self, x: torch.Tensor, t=None, z=None) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        # t is shape [batch_size]

        t_s1 = self.t_scale_proj_1(t) if t is not None else None
        t_b1 = self.t_bias_proj_1(t) if t is not None else None

        t_s2 = self.t_scale_proj_2(t) if t is not None else None
        t_b2 = self.t_bias_proj_2(t) if t is not None else None

        z_s1 = self.z_scale_proj_1(z) if z is not None else None
        z_b1 = self.z_bias_proj_1(z) if z is not None else None

        z_s2 = self.z_scale_proj_2(z) if z is not None else None
        z_b2 = self.z_bias_proj_2(z) if z is not None else None

        x = self.conv1(x)
        x = self.gn_1(x, t_s1, t_b1, z_s1, z_b1)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gn_2(x, t_s2, t_b2, z_s2, z_b2)
        x = x + self.gelu(x)

        return x


class UNetConv(nn.Module):
    """2 sets of convolution plus batch norm. Basic UNet building block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        int_channels: int | None = None,
        residual: bool = False,
        latent_vec_dim: int = 14,
    ):
        super().__init__()
        self.residual = residual
        if not int_channels:
            int_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels,
            int_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            padding_mode="circular",
        )
        self.gn_1 = AdaGN(
            num_channels=int_channels,
            num_groups=compute_groups(int_channels),
        )
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(
            int_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            padding_mode="circular",
        )
        self.gn_2 = AdaGN(
            num_channels=out_channels,
            num_groups=compute_groups(out_channels),
        )

        self.t_scale_proj_1 = nn.Linear(time_dim, int_channels)
        self.t_bias_proj_1 = nn.Linear(time_dim, int_channels)

        self.t_scale_proj_2 = nn.Linear(time_dim, out_channels)
        self.t_bias_proj_2 = nn.Linear(time_dim, out_channels)

        self.z_scale_proj_1 = nn.Linear(latent_vec_dim, int_channels)
        self.z_bias_proj_1 = nn.Linear(latent_vec_dim, int_channels)

        self.z_scale_proj_2 = nn.Linear(latent_vec_dim, out_channels)
        self.z_bias_proj_2 = nn.Linear(latent_vec_dim, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        # t is shape [batch_size]

        t_s1 = self.t_scale_proj_1(t) if t is not None else None
        t_b1 = self.t_bias_proj_1(t) if t is not None else None

        t_s2 = self.t_scale_proj_2(t) if t is not None else None
        t_b2 = self.t_bias_proj_2(t) if t is not None else None

        z_s1 = self.z_scale_proj_1(z) if z is not None else None
        z_b1 = self.z_bias_proj_1(z) if z is not None else None

        z_s2 = self.z_scale_proj_2(z) if z is not None else None
        z_b2 = self.z_bias_proj_2(z) if z is not None else None

        x = self.conv1(x)
        x = self.gn_1(x, t_s1, t_b1, z_s1, z_b1)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gn_2(x, t_s2, t_b2, z_s2, z_b2)
        x = x + self.gelu(x)

        return x


class DownStep(nn.Module):
    """Downscaling input with max pool and double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        int_channels: int | None = None,
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
            time_dim=time_dim,
            residual=True,
        )
        self.conv2 = UNetConv(
            in_channels=int_channels,
            out_channels=out_channels,
            time_dim=time_dim,
            residual=True,
        )
        self.gdn_layer = gdn.GDN(ch=out_channels, device="cuda")

    def forward(self, x: torch.Tensor, t, z) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        return self.gdn_layer(self.conv2(self.conv1(self.pooling(x), t, z), t, z))


class UpStep(nn.Module):
    """Upsample latent and incorporate residual"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res_channels: int,
        time_dim: int = 256,
    ):
        super().__init__()
        self.conv1 = UpsamplingUNetConv(
            in_channels=in_channels,
            int_channels=in_channels,
            out_channels=in_channels,
            residual=True,
            time_dim=time_dim,
        )
        self.conv2 = UNetConv(
            in_channels=(in_channels + res_channels),
            int_channels=(in_channels + res_channels) // 2,
            out_channels=out_channels,
            time_dim=time_dim,
        )

        self.gdn_layer = gdn.GDN(ch=out_channels, device="cuda", inverse=True)

    def forward(self, x: torch.Tensor, res_x: torch.Tensor, t, z) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        x = self.conv1(x, t, z)
        x = torch.cat([res_x, x], dim=1)
        x = self.conv2(x, t, z)
        return self.gdn_layer(x)


class UpStepWoutRes(nn.Module):
    """Upsample latent and incorporate residual"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int = 256,
    ):
        super().__init__()

        self.conv1 = UNetConv(
            in_channels=in_channels,
            out_channels=in_channels,
            time_dim=time_dim,
        )

        self.conv2 = UpsamplingUNetConv(
            in_channels=in_channels,
            int_channels=in_channels // 2,
            out_channels=out_channels,
            residual=True,
            time_dim=time_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Overloads forward method of nn.Module"""
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    """Creates a UNet using the building block modules in this file"""

    def __init__(
        self,
        n_channels: int,
        time_dim: int = 256,
        latent_img_channels: int = 32,
        latent_vec_dim: int = 14,
    ):
        super(UNet, self).__init__()
        self.time_dim = time_dim
        self.n_channels = n_channels
        self.num_latent_channels = latent_img_channels

        self.dropout = nn.Dropout2d(p=0.1)
        self.inc = UNetConv(
            in_channels=n_channels,
            out_channels=64,
            time_dim=time_dim,
            residual=True,
        )
        self.down1 = DownStep(
            in_channels=64 + self.num_latent_channels // 4,
            out_channels=128,
            time_dim=time_dim,
        )
        # self.sa1 = SelfAttention(channels=128)
        self.down2 = DownStep(
            in_channels=128 + self.num_latent_channels // 4,
            out_channels=256,
            time_dim=time_dim,
        )
        self.sa2 = SelfAttention(channels=256)
        self.down3 = DownStep(
            in_channels=256 + self.num_latent_channels // 4,
            out_channels=512,
            time_dim=time_dim,
        )
        self.sa3 = SelfAttention(channels=512)
        self.down4 = DownStep(
            in_channels=512 + self.num_latent_channels // 4,
            out_channels=512,
            time_dim=time_dim,
        )

        self.up0 = UpStep(
            in_channels=512,
            res_channels=512 + self.num_latent_channels // 4,
            out_channels=256,
            time_dim=time_dim,
        )
        self.sa0_inv = SelfAttention(channels=256)
        self.up1 = UpStep(
            in_channels=256,
            res_channels=256 + self.num_latent_channels // 4,
            out_channels=256,
            time_dim=time_dim,
        )
        self.sa1_inv = SelfAttention(channels=256)
        self.up2 = UpStep(
            in_channels=256,
            res_channels=128 + self.num_latent_channels // 4,
            out_channels=128,
            time_dim=time_dim,
        )
        self.up3 = UpStep(
            in_channels=128,
            res_channels=64 + self.num_latent_channels // 4,
            out_channels=64,
            time_dim=time_dim,
        )

        self.outc = nn.Conv2d(
            in_channels=64,
            out_channels=n_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.latent_upsampler_0 = nn.Sequential(
            UpStepWoutRes(
                in_channels=int(self.num_latent_channels // 4),
                out_channels=int(self.num_latent_channels // 4),
                time_dim=time_dim,
            ),
            UpStepWoutRes(
                in_channels=int(self.num_latent_channels // 4),
                out_channels=int(self.num_latent_channels // 4),
                time_dim=time_dim,
            ),
            UpStepWoutRes(
                in_channels=int(self.num_latent_channels // 4),
                out_channels=int(self.num_latent_channels // 4),
                time_dim=time_dim,
            ),
            UpStepWoutRes(
                in_channels=int(self.num_latent_channels // 4),
                out_channels=int(self.num_latent_channels // 4),
                time_dim=time_dim,
            ),
        )

        self.latent_upsampler_1 = nn.Sequential(
            UpStepWoutRes(
                in_channels=int(self.num_latent_channels // 4),
                out_channels=int(self.num_latent_channels // 4),
                time_dim=time_dim,
            ),
            UpStepWoutRes(
                in_channels=int(self.num_latent_channels // 4),
                out_channels=int(self.num_latent_channels // 4),
                time_dim=time_dim,
            ),
            UpStepWoutRes(
                in_channels=int(self.num_latent_channels // 4),
                out_channels=int(self.num_latent_channels // 4),
                time_dim=time_dim,
            ),
        )

        self.latent_upsampler_2 = nn.Sequential(
            UpStepWoutRes(
                in_channels=int(self.num_latent_channels // 4),
                out_channels=int(self.num_latent_channels // 4),
                time_dim=time_dim,
            ),
            UpStepWoutRes(
                in_channels=int(self.num_latent_channels // 4),
                out_channels=int(self.num_latent_channels // 4),
                time_dim=time_dim,
            ),
        )

        self.latent_upsampler_3 = nn.Sequential(
            UpStepWoutRes(
                in_channels=int(self.num_latent_channels // 4),
                out_channels=int(self.num_latent_channels // 4),
                time_dim=time_dim,
            ),
        )

        self.latent_vec_dim = latent_vec_dim
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_latent_channels, 14*9)

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
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Overloads forward method of nn.Module
        t is the full timestep embedding, with dimension time_dim
        z is the full latent, which will be split into latent_dim chunks
        """
        t = t.unsqueeze(-1)
        spatial = z
        
        num_segments_spatial = 4
        seg_size_spatial = self.num_latent_channels // num_segments_spatial
        num_mask_channels = (seg_size_spatial * t).floor().int()
        
        # For each sample in the batch, compute the mask per segment

        for seg in range(num_segments_spatial):
            start = seg * seg_size_spatial
            end = (seg + 1) * seg_size_spatial

            for b in range(t.shape[0]):
                t_val = t[b].item()
                num_mask_channels = int(seg_size_spatial * t_val)
                unmasked = seg_size_spatial - num_mask_channels

                # Copy masked part
                if num_mask_channels > 0:
                    spatial[b, start + unmasked:end, :, :] = 0  # mask
                
                
        # --- End of masking ---
        
        repr = self.fc(self.pool(spatial).squeeze())
        
        t = self.pos_encoding(t, self.time_dim)

        # Downsampling stages
        x1 = self.inc(x, t, repr[:, 0:self.latent_vec_dim])
        x1 = torch.cat([self.latent_upsampler_0(spatial[:, 0:self.num_latent_channels // 4, :, :]), x1], dim=1)
        x2 = self.down1(x1, t, repr[:, self.latent_vec_dim:self.latent_vec_dim*2])
        x2 = torch.cat([self.latent_upsampler_1(spatial[:, (self.num_latent_channels // 4):2*self.num_latent_channels // 4, :, :]), x2], dim=1)
        x3 = self.down2(x2, t, repr[:, self.latent_vec_dim*2:self.latent_vec_dim*3])
        x3 = self.sa2(x3)
        x3 = torch.cat([self.latent_upsampler_2(spatial[:, (2*self.num_latent_channels // 4):3*self.num_latent_channels // 4, :, :]), x3], dim=1)
        x4 = self.down3(x3, t, repr[:, self.latent_vec_dim*3:self.latent_vec_dim*4])
        x4 = self.sa3(x4)
        x4 = torch.cat([self.latent_upsampler_3(spatial[:, (3*self.num_latent_channels // 4):4*self.num_latent_channels // 4, :, :]), x4], dim=1)
        x5 = self.down4(x4, t, repr[:, self.latent_vec_dim*4:self.latent_vec_dim*5])

        # Upsampling stages
        x = self.up0(x5, x4, t, repr[:, self.latent_vec_dim*5:self.latent_vec_dim*6])
        # x = self.sa0_inv(x)
        x = self.up1(x, x3, t, repr[:, self.latent_vec_dim*6:self.latent_vec_dim*7])
        x = self.sa1_inv(x)
        x = self.up2(x, x2, t, repr[:, self.latent_vec_dim*7:self.latent_vec_dim*8])
        # x = self.sa2_inv(x)
        x = self.up3(x, x1, t, repr[:, self.latent_vec_dim*8:self.latent_vec_dim*9])
        # x = self.sa3_inv(x)
        # output = self.outc(x, t, repr[:, self.latent_vec_dim // 2 : self.latent_vec_dim])

        return self.outc(x)
