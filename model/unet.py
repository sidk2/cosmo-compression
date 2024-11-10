'''Implements a UNet'''

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetConv(nn.Module):
    '''2 sets of convolution plus batch norm. Basic UNet building block.'''

    def __init__(
        self, in_channels: int, out_channels: int, int_channels: Optional[int] = None
    ):
        super(
            UNetConv,
            self,
        ).__init__()

        int_channels = int_channels if int_channels else out_channels

        self.layers = nn.Sequential(
            [
                nn.Conv2d(
                    in_channels, int_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(int_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    int_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        )

    def forward(self, x: torch.Tensor):
        '''Overloads the forward method of nn.Module'''
        return self.layers(x)


class DownStep(nn.Module):
    '''Downscaling input with max pool and double conv'''

    def __init__(
        self, in_channels: int, out_channels: int, int_channels: Optional[int] = None
    ):
        super(
            DownStep,
            self,
        ).__init__()
        self.layers = nn.Sequential(
            [
                nn.MaxPool2d(kernel_size=2),
                UNetConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    int_channels=int_channels,
                ),
            ]
        )

    def forward(self, x):
        '''Overloads forward method of nn.Module'''
        return self.layers(x)


class UpStep(nn.Module):
    '''Upsample latent and incorporate residual'''

    def __init__(self, in_channels: int, out_channels: int):
        super(UpStep, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode="", align_corners=True)
        self.conv = UNetConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x, res):
        '''Overloads forward method of nn.Module'''
        x = self.up(x)

        # input is (channels, height, width)
        diff_y = res.size()[2] - x.size()[2]
        diff_x = res.size()[3] - x.size()[3]

        x = F.pad(
            x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([res, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    '''Creates a UNet using the building block modules in this file'''

    def __init__(
        self,
        n_channels: int,
    ):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        self.inc = UNetConv(in_channels=n_channels, out_channels=64)
        self.down1 = DownStep(in_channels=64, out_channels=128)
        self.down2 = DownStep(in_channels=128, out_channels=256)
        self.down3 = DownStep(in_channels=256, out_channels=512)
        self.down4 = DownStep(in_channels=512, out_channels=512)
        self.up1 = UpStep(in_channels=1024, out_channels=256)
        self.up2 = UpStep(in_channels=512, out_channels=128)
        self.up3 = UpStep(in_channels=256, out_channels=64)
        self.up4 = UpStep(in_channels=128, out_channels=64)
        self.outc = nn.Conv2d(in_channels=64, out_channels=n_channels, kernel_size=1)
        
    def forward(self, x):
        '''Overloads forward method of nn.Module'''
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits