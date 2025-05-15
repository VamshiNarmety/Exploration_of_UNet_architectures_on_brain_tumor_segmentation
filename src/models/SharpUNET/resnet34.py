import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet34_Weights

class ResNet34Encoder(nn.Module):
    def __init__(self, freeze=True):
        super(ResNet34Encoder, self).__init__()
        resnet34_model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        
        # Modify first layer to maintain spatial dimensions
        resnet34_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        resnet34_model.maxpool = nn.Identity()  # Remove initial maxpool

        # Freeze all parameters if specified
        if freeze:
            for param in resnet34_model.parameters():
                param.requires_grad = False

        # Extract the different layers
        self.conv1 = nn.Sequential(
            resnet34_model.conv1,
            resnet34_model.bn1,
            resnet34_model.relu
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256->128
        self.layer1 = resnet34_model.layer1  # Output: 64 channels
        self.layer2 = resnet34_model.layer2  # Output: 128 channels
        self.layer3 = resnet34_model.layer3  # Output: 256 channels
        self.layer4 = resnet34_model.layer4  # Output: 512 channels

    def forward(self, x):
        x1 = self.conv1(x)       # [B, 64, 256, 256]
        x1_pool = self.pool1(x1) # [B, 64, 128, 128]
        x2 = self.layer1(x1_pool) # [B, 64, 128, 128]
        x3 = self.layer2(x2)     # [B, 128, 64, 64]
        x4 = self.layer3(x3)     # [B, 256, 32, 32]
        x5 = self.layer4(x4)     # [B, 512, 16, 16]
        return x1, x2, x3, x4, x5  # Return all features needed for decoder


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
                                 [-1, 8, -1],
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
        # Handle dimension mismatches
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SharpUNetResNet34(nn.Module):
    def __init__(self, n_classes=1, freeze=True):
        super(SharpUNetResNet34, self).__init__()
        
        # Encoder
        self.encoder = ResNet34Encoder(freeze=freeze)

        # Sharpening blocks
        self.sharp1 = SharpBlock(64)
        self.sharp2 = SharpBlock(64)
        self.sharp3 = SharpBlock(128)
        self.sharp4 = SharpBlock(256)
        self.sharp5 = SharpBlock(512)

        # Decoder
        self.up1 = Up(512, 256)  # 16->32
        self.up2 = Up(256, 128)  # 32->64
        self.up3 = Up(128, 64)   # 64->128
        self.up4 = Up(64, 64)    # 128->256

        # Final output
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1, x2, x3, x4, x5 = self.encoder(x)

        # Sharpening
        x1_sharp = self.sharp1(x1)
        x2_sharp = self.sharp2(x2)
        x3_sharp = self.sharp3(x3)
        x4_sharp = self.sharp4(x4)
        x5_sharp = self.sharp5(x5)

        # Decoder path
        x = self.up1(x5_sharp, x4_sharp)  # [B,256,32,32]
        x = self.up2(x, x3_sharp)         # [B,128,64,64]
        x = self.up3(x, x2_sharp)         # [B,64,128,128]
        x = self.up4(x, x1_sharp)         # [B,64,256,256]
        
        logits = self.outc(x)             # [B,1,256,256]
        return logits
