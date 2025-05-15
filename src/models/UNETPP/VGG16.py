import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGG16Encoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)

        if freeze:
            for param in vgg.parameters():
                param.requires_grad = False

        features = vgg.features
        self.enc0 = features[0:6]   # conv1
        self.enc1 = features[6:13]  # conv2
        self.enc2 = features[13:23] # conv3
        self.enc3 = features[23:33] # conv4
        self.enc4 = features[33:43] # conv5

    def forward(self, x):
        x0 = self.enc0(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        return x0, x1, x2, x3, x4


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetPlusPlusVGG16(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, freeze=True):
        super().__init__()
        self.encoder = VGG16Encoder(freeze=freeze)

        # Decoder layers
        self.conv0_1 = ConvBlock(64 + 128, 64)
        self.conv0_2 = ConvBlock(64 * 2 + 128, 64)
        self.conv0_3 = ConvBlock(64 * 3 + 128, 64)
        self.conv0_4 = ConvBlock(64 * 4 + 128, 64)

        self.conv1_0 = ConvBlock(128, 128)
        self.conv1_1 = ConvBlock(128 + 256, 128)
        self.conv1_2 = ConvBlock(128 * 2 + 256, 128)
        self.conv1_3 = ConvBlock(128 * 3 + 256, 128)

        self.conv2_0 = ConvBlock(256, 256)
        self.conv2_1 = ConvBlock(256 + 512, 256)
        self.conv2_2 = ConvBlock(256 * 2 + 512, 256)

        self.conv3_0 = ConvBlock(512, 512)
        self.conv3_1 = ConvBlock(512 + 512, 512)

        self.conv4_0 = ConvBlock(512, 512)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)

        # Level 0
        x0_0 = x0
        x1_0 = self.conv1_0(x1)
        x2_0 = self.conv2_0(x2)
        x3_0 = self.conv3_0(x3)
        x4_0 = self.conv4_0(x4)

        # Level 1
        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(x1_0, x0_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, x1_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, x2_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(x4_0, x3_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        # Level 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1, x0_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1, x1_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(x3_1, x2_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        # Level 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, x0_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2, x1_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        # Level 4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, x0_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        # Final prediction
        out = self.final(x0_4)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out
