import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGG19Encoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
        features = vgg.features

        if freeze:
            for param in features.parameters():
                param.requires_grad = False

        self.stage1 = features[:6]   # 64
        self.stage2 = features[6:13] # 128
        self.stage3 = features[13:26]# 256
        self.stage4 = features[26:39]# 512
        self.stage5 = features[39:]  # 512

    def forward(self, x):
        x0 = self.stage1(x)
        x1 = self.stage2(x0)
        x2 = self.stage3(x1)
        x3 = self.stage4(x2)
        x4 = self.stage5(x3)
        return x0, x1, x2, x3, x4

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNetPlusPlusVGG19(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, freeze=True):
        super().__init__()
        self.encoder = VGG19Encoder(freeze=freeze)

        self.conv0_0 = ConvBlock(64, 32)
        self.conv1_0 = ConvBlock(128, 64)
        self.conv2_0 = ConvBlock(256, 128)
        self.conv3_0 = ConvBlock(512, 256)
        self.conv4_0 = ConvBlock(512, 256)

        # Nested Dense Blocks
        self.conv0_1 = ConvBlock(32 + 64, 32)
        self.conv1_1 = ConvBlock(64 + 128, 64)
        self.conv2_1 = ConvBlock(128 + 256, 128)
        self.conv3_1 = ConvBlock(256 + 256, 256)

        self.conv0_2 = ConvBlock(32*2 + 64, 32)
        self.conv1_2 = ConvBlock(64*2 + 128, 64)
        self.conv2_2 = ConvBlock(128*2 + 256, 128)

        self.conv0_3 = ConvBlock(32*3 + 64, 32)
        self.conv1_3 = ConvBlock(64*3 + 128, 64)

        self.conv0_4 = ConvBlock(32*4 + 64, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)

        # 0-th column
        x0_0 = self.conv0_0(x0)
        x1_0 = self.conv1_0(x1)
        x2_0 = self.conv2_0(x2)
        x3_0 = self.conv3_0(x3)
        x4_0 = self.conv4_0(x4)

        # 1st column
        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(x1_0, size=x0_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, size=x1_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, size=x2_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(x4_0, size=x3_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        # 2nd column
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1, size=x0_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1, size=x1_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(x3_1, size=x2_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        # 3rd column
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, size=x0_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2, size=x1_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        # 4th column
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, size=x0_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        output = self.final(x0_4)
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=True)

        return output
