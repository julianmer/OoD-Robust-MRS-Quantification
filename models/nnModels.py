####################################################################################################
#                                           nnModels.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 15/08/22                                                                                #
#                                                                                                  #
# Purpose: Definitions of various neural networks to be used in hybrid data-driven/model-based     #
#          optimization.                                                                           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch
import torch.nn as nn
import torch.nn.functional as F


#**************************************************************************************************#
#                                          Class BaseNet                                           #
#**************************************************************************************************#
#                                                                                                  #
# Base class for all neural networks.                                                              #
#                                                                                                  #
#**************************************************************************************************#
class BaseNet(nn.Module):
    def __init__(self, shapeX, shapeY, dropout=0.0, act='relu', weight_init=None):
        super(BaseNet, self).__init__()
        self.shapeX = shapeX
        self.shapeY = shapeY
        self.dropout = nn.Dropout(dropout)
        self.fl = nn.Flatten()
        self.bn1d = nn.BatchNorm1d(num_features=shapeX[0])

        self.act = self.get_activation(act)
        self.weight_init = weight_init

    def get_activation(self, act):
        if act == 'elu': return F.elu
        elif act == 'relu': return F.relu
        elif act == 'tanh': return F.tanh
        elif act == 'sigmoid': return F.sigmoid
        elif act == 'softplus': return F.softplus
        else: raise ValueError(f'Activation function {act} not recognized.')

    def initialize_weights(self):
        if self.weight_init is None:
            return
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.weight_init == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif self.weight_init == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif self.weight_init == 'normal':
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                elif self.weight_init == 'uniform':
                    nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                elif self.weight_init == 'constant':
                    nn.init.constant_(m.weight, 0.0)
                elif self.weight_init == 'zeros':
                    nn.init.zeros_(m.weight)
                else:
                    raise ValueError(f'Initialization mode {self.weight_init} not recognized.')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def format_output(self, x):
        xc = F.softplus(x[:, :self.shapeY])
        xs = F.softplus(x[:, self.shapeY:self.shapeY + 2]) + 1
        xo = x[:, self.shapeY + 2:-7]
        xp1 = F.tanh(x[:, -7:-6]) * 1e-4
        xr = x[:, -6:]
        return torch.cat((xc, xs, xo, xp1, xr), dim=-1)

    def set_bn_mode(self, mode='train'):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.train() if mode == 'train' else module.eval()


#**************************************************************************************************#
#                                             Class MLP                                            #
#**************************************************************************************************#
#                                                                                                  #
# A simple multi-layer perceptron (MLP).                                                           #
#                                                                                                  #
#**************************************************************************************************#
class MLP(BaseNet):
    def __init__(self, shapeX, shapeY, dropout=0.0, width=256, depth=3, act='relu',
                 weight_init=None, **kwargs):
        super().__init__(shapeX, shapeY, dropout, act, weight_init)
        self.depth = depth

        self.fc0 = nn.Linear(shapeX[0] * shapeX[1], width)
        for i in range(1, depth):
            setattr(self, f'fc{i}', nn.Linear(width // (2 ** (i - 1)), width // (2 ** i)))
        self.out = nn.Linear(width // (2 ** (depth - 1)), shapeY + 11)

        self.initialize_weights()

    def forward(self, x):
        x = self.bn1d(x)
        x = self.fl(x)

        x = self.fc0(x)
        x = self.act(x)
        x = self.dropout(x)

        for i in range(1, self.depth):
            x = getattr(self, f'fc{i}')(x)
            x = self.act(x)
            x = self.dropout(x)

        x = self.out(x)
        return self.format_output(x)


#**************************************************************************************************#
#                                            Class CNN                                             #
#**************************************************************************************************#
#                                                                                                  #
# A simple convolutional neural network (CNN).                                                     #
#                                                                                                  #
#**************************************************************************************************#
class CNN(BaseNet):
    def __init__(self, shapeX, shapeY, dropout=0.0, width=256, depth=3, act='relu',
                 conv_depth=5, kernel_size=5, stride=2, weight_init=None, **kwargs):
        super().__init__(shapeX, shapeY, dropout, act, weight_init)
        self.depth = depth
        self.conv_depth = conv_depth

        in_ch = shapeX[0]
        for i in range(conv_depth):
            out_ch = 2 ** (i + 2)
            setattr(self, f'con{i}', nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride))
            setattr(self, f'bn1d{i}', nn.BatchNorm1d(out_ch))
            in_ch = out_ch

        # compute flattened shape after conv
        x = torch.randn(1, shapeX[0], shapeX[1])
        for i in range(conv_depth):
            x = getattr(self, f'con{i}')(x)
        flattened = x.shape[1] * x.shape[2]

        self.fc0 = nn.Linear(flattened, width)
        for i in range(1, depth):
            setattr(self, f'fc{i}', nn.Linear(width // (2 ** (i - 1)), width // (2 ** i)))
        self.out = nn.Linear(width // (2 ** (depth - 1)), shapeY + 11)

        self.initialize_weights()

    def forward(self, x):
        x = self.bn1d(x)
        for i in range(self.conv_depth):
            x = getattr(self, f'con{i}')(x)
            x = getattr(self, f'bn1d{i}')(x)
            x = self.act(x)
            x = self.dropout(x)

        x = self.fl(x)
        x = self.fc0(x)
        x = self.act(x)
        x = self.dropout(x)

        for i in range(1, self.depth):
            x = getattr(self, f'fc{i}')(x)
            x = self.act(x)
            x = self.dropout(x)

        x = self.out(x)
        return self.format_output(x)


#**************************************************************************************************#
#                                         Class DoubleConv                                         #
#**************************************************************************************************#
#                                                                                                  #
# 2D convolutional layer with batchnorm and relu.                                                  #
#                                                                                                  #
#**************************************************************************************************#
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


#**************************************************************************************************#
#                                             Class Down                                           #
#**************************************************************************************************#
#                                                                                                  #
# Downscaling with maxpooling and double convolutional layer.                                      #
#                                                                                                  #
#**************************************************************************************************#
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


#**************************************************************************************************#
#                                              Class Up                                            #
#**************************************************************************************************#
#                                                                                                  #
# Upscaling then double convolutional layer.                                                       #
#                                                                                                  #
#**************************************************************************************************#
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                         kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


#**************************************************************************************************#
#                                           Class OutConv                                          #
#**************************************************************************************************#
#                                                                                                  #
# Final convolutional layer.                                                                       #
#                                                                                                  #
#**************************************************************************************************#
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


#**************************************************************************************************#
#                                             Class UNet                                           #
#**************************************************************************************************#
#                                                                                                  #
# U-Net model.                                                                                     #
#                                                                                                  #
#**************************************************************************************************#
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, dropout=0.0, **kwargs):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout = dropout

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = nn.functional.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.down1(x1)
        x2 = nn.functional.dropout(x2, p=self.dropout, training=self.training)
        x3 = self.down2(x2)
        x3 = nn.functional.dropout(x3, p=self.dropout, training=self.training)
        x4 = self.down3(x3)
        x4 = nn.functional.dropout(x4, p=self.dropout, training=self.training)
        x5 = self.down4(x4)
        x5 = nn.functional.dropout(x5, p=self.dropout, training=self.training)
        u = self.up1(x5, x4)
        u = nn.functional.dropout(u, p=self.dropout, training=self.training)
        u = self.up2(u, x3)
        u = nn.functional.dropout(u, p=self.dropout, training=self.training)
        u = self.up3(u, x2)
        u = nn.functional.dropout(u, p=self.dropout, training=self.training)
        u = self.up4(u, x1)
        u = nn.functional.dropout(u, p=self.dropout, training=self.training)
        u = self.outc(u)
        return u


#**************************************************************************************************#
#                                    Class DepthwiseSeparableConv                                  #
#**************************************************************************************************#
#                                                                                                  #
# Depthwise separable convolution to reduce number of parameters.                                  #
#                                                                                                  #
#**************************************************************************************************#
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels,
                                   kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


#**************************************************************************************************#
#                                     Class ReducedDoubleConv                                      #
#**************************************************************************************************#
#                                                                                                  #
# Reduced double convolutional layer with depthwise separable conv and batchnorm.                  #
#                                                                                                  #
#**************************************************************************************************#
class ReducedDoubleConv(nn.Module):
    """(depthwise separable conv => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


#**************************************************************************************************#
#                                             Class Down                                           #
#**************************************************************************************************#
#                                                                                                  #
# Downscaling with maxpooling and double convolutional layer.                                      #
#                                                                                                  #
#**************************************************************************************************#
class ReDown(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ReducedDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


#**************************************************************************************************#
#                                              Class Up                                            #
#**************************************************************************************************#
#                                                                                                  #
# Upscaling then double convolutional layer.                                                       #
#                                                                                                  #
#**************************************************************************************************#
class ReUp(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = ReducedDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


#**************************************************************************************************#
#                                             Class UNet                                           #
#**************************************************************************************************#
#                                                                                                  #
# U-Net model with reduced parameters.                                                             #
#                                                                                                  #
#**************************************************************************************************#
class ReUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, dropout=0.0, **kwargs):
        super(ReUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout = dropout

        self.inc = ReducedDoubleConv(n_channels, 32)
        self.down1 = ReDown(32, 64)
        self.down2 = ReDown(64, 128)
        self.down3 = ReDown(128, 256)
        self.down4 = ReDown(256, 256)
        self.up1 = ReUp(512, 128, bilinear)
        self.up2 = ReUp(256, 64, bilinear)
        self.up3 = ReUp(128, 32, bilinear)
        self.up4 = ReUp(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = nn.functional.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.down1(x1)
        x2 = nn.functional.dropout(x2, p=self.dropout, training=self.training)
        x3 = self.down2(x2)
        x3 = nn.functional.dropout(x3, p=self.dropout, training=self.training)
        x4 = self.down3(x3)
        x4 = nn.functional.dropout(x4, p=self.dropout, training=self.training)
        x5 = self.down4(x4)
        x5 = nn.functional.dropout(x5, p=self.dropout, training=self.training)
        u = self.up1(x5, x4)
        u = nn.functional.dropout(u, p=self.dropout, training=self.training)
        u = self.up2(u, x3)
        u = nn.functional.dropout(u, p=self.dropout, training=self.training)
        u = self.up3(u, x2)
        u = nn.functional.dropout(u, p=self.dropout, training=self.training)
        u = self.up4(u, x1)
        u = nn.functional.dropout(u, p=self.dropout, training=self.training)
        u = self.outc(u)
        return u

