'''
Steps for implementation:
    - Implement the encoder from scratch. This should be some kind of ResNet probably?
    - Implement the flow matching decoder, using Neural ODE. Use Carol's code as reference, but DIY to understand it better.
    - Train the compression model
    - Implement a normalizing flow model for parameter estimation. CAMELs has a reference for this sort of thing.
        - Train it on the latents.
'''

import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        

class ResNet(nn.Module):
    ...
    def __init__(self, input_dim, in_channels):
        # CAMELS Multifield Dataset is 256x256
        self.in_layer = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2), 
                                       nn.MaxPool2d(kernel_size=3, stride=2)])
        # Downsampled to 64x64
        
        
class Encoder(nn.Module):
    def __init__(self,):
        super.__init__()