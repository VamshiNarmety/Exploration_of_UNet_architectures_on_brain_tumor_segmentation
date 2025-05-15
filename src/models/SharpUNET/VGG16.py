import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGG16Encoder(nn.Module):
    def __init__(self, freeze=True):
        super(VGG16Encoder, self).__init__()
        vgg16 = models.vgg16_bn(pretrained=True)

        if freeze:
            for param in vgg16.parameters():
                param.requires_grad = False

        features = list(vgg16.features.children())

        self.block1 = nn.Sequential(*features[0:6])   # [B, 64, 256, 256]
        self.block2 = nn.Sequential(*features[6:13])  # [B, 128, 128, 128]
        self.block3 = nn.Sequential(*features[13:23]) # [B, 256, 64, 64]
        self.block4 = nn.Sequential(*features[23:33]) # [B, 512, 32, 32]
        self.block5 = nn.Sequential(*features[33:43]) # [B, 512, 16, 16]

    def forward(self, x):
        x1 = self.block1(x)  # [B, 64, 256, 256]
        x2 = self.block2(x1) # [B, 128, 128, 128]
        x3 = self.block3(x2) # [B, 256, 64, 64]
        x4 = self.block4(x3) # [B, 512, 32, 32]
        x5 = self.block5(x4) # [B, 512, 16, 16]
        return x1, x2, x3, x4, x5

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SharpBlock(nn.Module):
    def __init__(self, channels):
        super(SharpBlock, self).__init__()
        self.sharpen = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False, groups=channels)
        with torch.no_grad():
            kernel = torch.tensor([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]], dtype=torch.float32)
            kernel = kernel.expand(channels, 1, 3, 3)
            self.sharpen.weight.copy_(kernel)

    def forward(self, x):
        return F.relu(self.sharpen(x))

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SharpUNetVGG16(nn.Module):
    def __init__(self, n_classes=1, freeze=True):
        super(SharpUNetVGG16, self).__init__()

        self.encoder = VGG16Encoder(freeze=freeze)

        # Sharpening
        self.sharp1 = SharpBlock(64)
        self.sharp2 = SharpBlock(128)
        self.sharp3 = SharpBlock(256)
        self.sharp4 = SharpBlock(512)
        self.sharp5 = SharpBlock(512)

        # Decoder
        self.up1 = Up(512, 512)   # 16 -> 32
        self.up2 = Up(512, 256)   # 32 -> 64
        self.up3 = Up(256, 128)   # 64 -> 128
        self.up4 = Up(128, 64)    # 128 -> 256

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)

        x1 = self.sharp1(x1)
        x2 = self.sharp2(x2)
        x3 = self.sharp3(x3)
        x4 = self.sharp4(x4)
        x5 = self.sharp5(x5)

        x = self.up1(x5, x4)   # 512 -> 512
        x = self.up2(x, x3)    # 512 -> 256
        x = self.up3(x, x2)    # 256 -> 128
        x = self.up4(x, x1)    # 128 -> 64

        logits = self.outc(x)  # [B, n_classes, 256, 256]
        return logits
