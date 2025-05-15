import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet50Encoder(nn.Module):
    def __init__(self, freeze=True):  # Corrected here
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        if freeze:
            for param in resnet.parameters():
                param.requires_grad = False

        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.encoder1 = resnet.layer1  # 256
        self.encoder2 = resnet.layer2  # 512
        self.encoder3 = resnet.layer3  # 1024
        self.encoder4 = resnet.layer4  # 2048

    def forward(self, x):
        x0 = self.initial(x)     # 64 channels
        x1 = self.encoder1(x0)   # 256
        x2 = self.encoder2(x1)   # 512
        x3 = self.encoder3(x2)   # 1024
        x4 = self.encoder4(x3)   # 2048
        return x0, x1, x2, x3, x4


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):  # Corrected here
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


class UNetPlusPlusResNet50(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, freeze=True):  # Corrected here
        super().__init__()
        self.encoder = ResNet50Encoder(freeze=freeze)

        # Decoder blocks
        self.conv4_0 = ConvBlock(2048, 1024)
        self.conv3_0 = ConvBlock(1024, 512)
        self.conv2_0 = ConvBlock(512, 256)
        self.conv1_0 = ConvBlock(256, 128)

        self.conv3_1 = ConvBlock(1024 + 512, 512)
        self.conv2_1 = ConvBlock(512 + 256, 256)
        self.conv1_1 = ConvBlock(256 + 128, 128)

        self.conv2_2 = ConvBlock(512 + 256, 256)
        self.conv1_2 = ConvBlock(256 + 128, 128)

        self.conv1_3 = ConvBlock(256 + 128, 128)

        self.final = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        # Encode
        x0, x1, x2, x3, x4 = self.encoder(x)

        # Decoder
        x4_0 = self.conv4_0(x4)

        x3_0 = self.conv3_0(x3)
        x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(x4_0, size=x3_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        x2_0 = self.conv2_0(x2)
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, size=x2_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_1, F.interpolate(x3_1, size=x2_1.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        x1_0 = self.conv1_0(x1)
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, size=x1_0.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_1, F.interpolate(x2_1, size=x1_1.shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_2, F.interpolate(x2_2, size=x1_2.shape[2:], mode='bilinear', align_corners=True)], dim=1))

        out = self.final(x1_3)

        # Upsample final output to match input size
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)

        return out
