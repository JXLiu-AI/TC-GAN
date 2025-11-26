import math

import torch.nn.functional as F
from torch import nn
import torch


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            #nn.Conv2d(2, 64, kernel_size=9, padding=4),
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.PReLU()
        )
        #self.block2 = ResidualBlock(64)
        self.block2 = ConvLayer(16, 32, 5, last = nn.LeakyReLU)
        #self.block3 = ResidualBlock(64)
        self.block3 = ConvLayer(32, 64, 7, last=nn.LeakyReLU)
        #self.block4 = ResidualBlock(64)
        self.block4 = ConvLayer(64, 64, 5, last = nn.LeakyReLU)
        #self.block5 = ResidualBlock(64)
        self.block5 = ConvLayer(64, 32, 5, last=nn.LeakyReLU)
        #self.block6 = ResidualBlock(64)
        self.block6 = ConvLayer(32, 16, 5, last=nn.LeakyReLU)
        self.block7 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.PReLU()
        )

        block8 = []
        block8.append(nn.Conv2d(16, 1, kernel_size=3, padding=1))#amazing
        self.block8 = nn.Sequential(*block8)
        self.se = SELayer(64, )
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block4 = self.se(block4)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
#        block8 = self.block8(block1 + block7)
        block8 = self.block8(block7)
        block_size = list(block8.size())
        block_size[1] = 3
        block8 = block8.expand(block_size)
        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x1 = self.net(x).view(batch_size,1)
        x2 = torch.sigmoid(x1)
        return torch.sigmoid(x1)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()# 负数部分的参数会变
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class ConvLayer(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 16, kernel_size = 5, last = nn.ReLU):
        super().__init__()
        if kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = padding),
            nn.BatchNorm2d(out_channels),
            last()
        )

    def forward(self, x):
        out = self.main(x)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction_ratio=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel, int(channel/reduction_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel/reduction_ratio),channel),
            nn.Sigmoid()
            )
    def forward(self,x):
        b,c,_,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x*y

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
