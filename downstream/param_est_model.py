import torch
import torch.nn as nn

# Define the CNN classifier
# class ParamEstimatorImg(nn.Module):
#     def __init__(self, hidden, dr, channels, output_size=2):
#         super(ParamEstimatorImg, self).__init__()
        
#         # input: 1x256x256 ---------------> output: 2*hiddenx128x128
#         self.C01 = nn.Conv2d(channels,  2*hidden, kernel_size=3, stride=1, padding=1, 
#                             padding_mode='circular', bias=True)
#         self.C02 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=3, stride=1, padding=1, 
#                             padding_mode='circular', bias=True)
#         self.C03 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=2, stride=2, padding=0, 
#                             padding_mode='circular', bias=True)
#         self.B01 = nn.BatchNorm2d(2*hidden)
#         self.B02 = nn.BatchNorm2d(2*hidden)
#         self.B03 = nn.BatchNorm2d(2*hidden)
        
#         # input: 2*hiddenx128x128 ----------> output: 4*hiddenx64x64
#         self.C11 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
#                             padding_mode='circular', bias=True)
#         self.C12 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
#                             padding_mode='circular', bias=True)
#         self.C13 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=2, stride=2, padding=0,
#                             padding_mode='circular', bias=True)
#         self.B11 = nn.BatchNorm2d(4*hidden)
#         self.B12 = nn.BatchNorm2d(4*hidden)
#         self.B13 = nn.BatchNorm2d(4*hidden)
        
#         # input: 4*hiddenx64x64 --------> output: 8*hiddenx32x32
#         self.C21 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
#                             padding_mode='circular', bias=True)
#         self.C22 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
#                             padding_mode='circular', bias=True)
#         self.C23 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=2, stride=2, padding=0,
#                             padding_mode='circular', bias=True)
#         self.B21 = nn.BatchNorm2d(8*hidden)
#         self.B22 = nn.BatchNorm2d(8*hidden)
#         self.B23 = nn.BatchNorm2d(8*hidden)
        
#         # input: 8*hiddenx32x32 ----------> output: 16*hiddenx16x16
#         self.C31 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
#                             padding_mode='circular', bias=True)
#         self.C32 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=3, stride=1, padding=1,
#                             padding_mode='circular', bias=True)
#         self.C33 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=2, stride=2, padding=0,
#                             padding_mode='circular', bias=True)
#         self.B31 = nn.BatchNorm2d(16*hidden)
#         self.B32 = nn.BatchNorm2d(16*hidden)
#         self.B33 = nn.BatchNorm2d(16*hidden)
        
#         # input: 16*hiddenx16x16 ----------> output: 32*hiddenx8x8
#         self.C41 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
#                             padding_mode='circular', bias=True)
#         self.C42 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
#                             padding_mode='circular', bias=True)
#         self.C43 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=2, stride=2, padding=1,
#                             padding_mode='circular', bias=True)
#         self.B41 = nn.BatchNorm2d(32*hidden)
#         self.B42 = nn.BatchNorm2d(32*hidden)
#         self.B43 = nn.BatchNorm2d(32*hidden)
        
#         # input: 32*hiddenx8x8 ----------> output:64*hiddenx4x4
#         self.C51 = nn.Conv2d(32*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
#                             padding_mode='circular', bias=True)
#         self.C52 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
#                             padding_mode='circular', bias=True)
#         self.C53 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=2, stride=2, padding=0,
#                             padding_mode='circular', bias=True)
#         self.B51 = nn.BatchNorm2d(64*hidden)
#         self.B52 = nn.BatchNorm2d(64*hidden)
#         self.B53 = nn.BatchNorm2d(64*hidden)

#         # input: 64*hiddenx4x4 ----------> output: 128*hiddenx1x1
#         self.C61 = nn.Conv2d(64*hidden, 128*hidden, kernel_size=4, stride=1, padding=0,
#                             padding_mode='circular', bias=True)
#         self.B61 = nn.BatchNorm2d(128*hidden)

#         self.P0  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

#         self.FC1  = nn.Linear(128*hidden, 64*hidden)  
#         self.FC2  = nn.Linear(64*hidden,  output_size)  

#         self.dropout   = nn.Dropout(p=dr)
#         self.ReLU      = nn.ReLU()
#         self.LeakyReLU = nn.LeakyReLU(0.2)
#         self.tanh      = nn.Tanh()

#         def _init_weights(self):
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
#                 elif isinstance(m, nn.BatchNorm2d):
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)
#                 elif isinstance(m, nn.Linear):
#                     nn.init.xavier_normal_(m.weight)
#                     nn.init.constant_(m.bias, 0)


#     def forward(self, image):
#         x = self.LeakyReLU(self.C01(image))
#         x = self.LeakyReLU(self.B02(self.C02(x)))
#         x = self.LeakyReLU(self.B03(self.C03(x)))

#         x = self.LeakyReLU(self.B11(self.C11(x)))
#         x = self.LeakyReLU(self.B12(self.C12(x)))
#         x = self.LeakyReLU(self.B13(self.C13(x)))

#         x = self.LeakyReLU(self.B21(self.C21(x)))
#         x = self.LeakyReLU(self.B22(self.C22(x)))
#         x = self.LeakyReLU(self.B23(self.C23(x)))

#         x = self.LeakyReLU(self.B31(self.C31(x)))
#         x = self.LeakyReLU(self.B32(self.C32(x)))
#         x = self.LeakyReLU(self.B33(self.C33(x)))

#         x = self.LeakyReLU(self.B41(self.C41(x)))
#         x = self.LeakyReLU(self.B42(self.C42(x)))
#         x = self.LeakyReLU(self.B43(self.C43(x)))

#         x = self.LeakyReLU(self.B51(self.C51(x)))
#         x = self.LeakyReLU(self.B52(self.C52(x)))
#         x = self.LeakyReLU(self.B53(self.C53(x)))

#         x = self.LeakyReLU(self.B61(self.C61(x)))

#         x = x.view(image.shape[0],-1)
#         x = self.dropout(self.LeakyReLU(self.FC1(x)))
#         x = self.FC2(x)

#         return x

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic ResNet block with two 3x3 convolutions"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dr=0.0):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.dropout = nn.Dropout2d(p=dr) if dr > 0 else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ParamEstimatorImg(nn.Module):
    """ResNet-18 architecture for parameter estimation from images"""
    
    def __init__(self, hidden=64, dr=0.1, channels=1, output_size=2):
        super(ParamEstimatorImg, self).__init__()
        
        # Initial convolution layer (like original ResNet)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-18 layers: [2, 2, 2, 2] blocks per layer
        self.layer1 = self._make_layer(hidden, hidden, 2, stride=1, dr=dr*0.5)
        self.layer2 = self._make_layer(hidden, hidden*2, 2, stride=2, dr=dr*0.7)
        self.layer3 = self._make_layer(hidden*2, hidden*4, 2, stride=2, dr=dr*0.8)
        self.layer4 = self._make_layer(hidden*4, hidden*8, 2, stride=2, dr=dr)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dr)
        self.fc = nn.Linear(hidden*8, output_size)
        
        # Initialize weights
        self._init_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, dr=0.0):
        """Create a ResNet layer with specified number of blocks"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample, dr))
        
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, dr=dr))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, image):
        # Initial convolution and pooling
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
    
class SummaryNet(nn.Module):
    def __init__(self, hidden=5, last_layer=10):
        super().__init__()
        # input: 1x100x100 ---------------> output: hiddenx100x100
        self.conv1 = nn.Conv2d(1, hidden, kernel_size = 3, stride=1, padding=1)
        self.B1 = nn.BatchNorm2d(hidden)
        
        # input: hiddenx100x100 ---------------> output: hiddenx100x100
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size = 3, stride=1, padding=1)
        self.B2 = nn.BatchNorm2d(hidden)
        # input: hiddenx100x100 ---------------> output: hiddenx50x50
        # pool
        
        # input: hiddenx50x50 ---------------> output: 2*hiddenx50x50
        self.conv3 = nn.Conv2d(hidden, 2*hidden, kernel_size = 3, stride=1, padding=1)
        self.B3 = nn.BatchNorm2d(2*hidden)
        
        # input: 2*hiddenx50x50 ---------------> output: 2*hiddenx50x50
        self.conv4 = nn.Conv2d(2*hidden, 2*hidden, kernel_size = 3, stride=1, padding=1)
        self.B4 = nn.BatchNorm2d(2*hidden)
        # input: 2*hiddenx50x50 ---------------> output: 2*hiddenx25x25
        # pool
        
        # input: 2*hiddenx25x25 ---------------> output: 4*hiddenx24x24
        self.conv5 = nn.Conv2d(2*hidden, 4*hidden, kernel_size = 2, stride=1, padding=0)
        self.B5 = nn.BatchNorm2d(4*hidden)
        
        # input: 4*hiddenx24x24 ---------------> output: 4*hiddenx24x24
        self.conv6 = nn.Conv2d(4*hidden, 4*hidden, kernel_size = 3, stride=1, padding=1)
        self.B6 = nn.BatchNorm2d(4*hidden)
        # input: 4*hiddenx24x24 ---------------> output: 4*hiddenx12x12
        # pool
        
        # input: 4*hiddenx12x12 ---------------> output: 8*hiddenx10x10
        self.conv7 = nn.Conv2d(4*hidden, 8*hidden, kernel_size = 3, stride=1, padding=0)
        self.B7 = nn.BatchNorm2d(8*hidden)
        
        # input: 8*hiddenx10x10 ---------------> output: 8*hiddenx8x8
        self.conv8 = nn.Conv2d(8*hidden, 8*hidden, kernel_size = 3, stride=1, padding=0)
        self.B8 = nn.BatchNorm2d(8*hidden)
        # input: 8*hiddenx8*8 ---------------> output: 8*hiddenx4x4
        # pool
        
        # input: 8*hiddenx4x4---------------> output: 8*hiddenx2x2
        self.conv9 = nn.Conv2d(8*hidden, 16*hidden, kernel_size = 3, stride=1, padding=0)
        self.B9 = nn.BatchNorm2d(16*hidden)
        # input: 16*hiddenx2x2 ---------------> output: 16*hiddenx1x1
        # pool
        
        self.maxpool = nn.MaxPool2d(2, 2)
        self.pool = nn.AvgPool2d(2, 2) #nn.MaxPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(6, 6)
        # input: hiddenx16x16 ---------------> output: last_layer
        self.fc1 = nn.Linear(16*hidden, 16*hidden)
        self.fc2 = nn.Linear(16*hidden, last_layer)
        
    def forward(self, x):
        x = F.relu(self.B1(self.conv1(x)))
        x = self.pool(F.relu(self.B2(self.conv2(x))))
        
        x = F.relu(self.B3(self.conv3(x)))
        x = self.pool(F.relu(self.B4(self.conv4(x))))
        
        x = F.relu(self.B5(self.conv5(x)))
        x = self.pool(F.relu(self.B6(self.conv6(x))))
        
        x = F.relu(self.B7(self.conv7(x)))
        x = self.pool(F.relu(self.B8(self.conv8(x))))
        x = self.pool(F.relu(self.B9(self.conv9(x))))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class ParamEstVec(nn.Module):
    def __init__(self, hidden_dim, num_hiddens, in_dim, output_size):
        super(ParamEstVec, self).__init__()
        self.in_transform = nn.Linear(in_dim, hidden_dim)
        self.out_transform = nn.Linear(256, output_size)
        
        self.hiddens = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hiddens)]
        )
        self.hiddens.append(nn.Linear(hidden_dim, 256))
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.LeakyReLU(self.in_transform(x))
        for hidden in self.hiddens:
            x = self.LeakyReLU(hidden(x))
        x = self.out_transform(x)
        return x
    