"""
Fixed Attention U-Net model for DR lesion segmentation.
The key fix is matching the feature dimensions to ResNet50's actual output channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import config

class ConvBlock(nn.Module):
    """
    Basic convolutional block with batch normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=True):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=not use_batchnorm),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class DoubleConvBlock(nn.Module):
    """
    Double convolutional block as used in U-Net.
    """
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super(DoubleConvBlock, self).__init__()
        
        self.double_conv = nn.Sequential(
            ConvBlock(in_channels, out_channels, use_batchnorm=use_batchnorm),
            ConvBlock(out_channels, out_channels, use_batchnorm=use_batchnorm)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class AttentionGate(nn.Module):
    """
    Attention gate for focusing on relevant features.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        # Gating signal convolution
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Skip connection convolution
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Output convolution
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # Apply convolutions
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Resize gating signal to match skip connection
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        # Combine signals and apply ReLU
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention
        return x * psi

class AttentionUNet(nn.Module):
    """
    Attention U-Net architecture for DR lesion segmentation.
    """
    def __init__(self, num_classes, in_channels=3, pretrained_encoder=True):
        super(AttentionUNet, self).__init__()
        self.num_classes = num_classes
        
        # ResNet50 as encoder
        resnet = models.resnet50(pretrained=pretrained_encoder)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1  # 64 -> 256 channels
        self.encoder2 = resnet.layer2  # 256 -> 512 channels
        self.encoder3 = resnet.layer3  # 512 -> 1024 channels
        self.encoder4 = resnet.layer4  # 1024 -> 2048 channels
        
        # FIXED: Use actual ResNet50 channel dimensions
        # ResNet50 outputs: [256, 512, 1024, 2048]
        encoder_channels = [256, 512, 1024, 2048]
        decoder_channels = [512, 256, 128, 64]
        
        # Bottleneck - FIXED: Use correct input channels (2048)
        self.bottleneck = DoubleConvBlock(encoder_channels[3], decoder_channels[0])
        
        # Attention gates - FIXED: Use correct channel dimensions
        self.attention1 = AttentionGate(F_g=decoder_channels[0], F_l=encoder_channels[2], F_int=decoder_channels[1])
        self.attention2 = AttentionGate(F_g=decoder_channels[1], F_l=encoder_channels[1], F_int=decoder_channels[2])
        self.attention3 = AttentionGate(F_g=decoder_channels[2], F_l=encoder_channels[0], F_int=decoder_channels[3])
        
        # Decoder - FIXED: Use correct channel dimensions
        self.upconv1 = nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], kernel_size=2, stride=2)
        self.decoder1 = DoubleConvBlock(decoder_channels[1] + encoder_channels[2], decoder_channels[1])
        
        self.upconv2 = nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2)
        self.decoder2 = DoubleConvBlock(decoder_channels[2] + encoder_channels[1], decoder_channels[2])
        
        self.upconv3 = nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2)
        self.decoder3 = DoubleConvBlock(decoder_channels[3] + encoder_channels[0], decoder_channels[3])
        
        # Deep supervision outputs
        self.dsv1 = nn.Conv2d(decoder_channels[1], num_classes, kernel_size=1)
        self.dsv2 = nn.Conv2d(decoder_channels[2], num_classes, kernel_size=1)
        self.dsv3 = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)
        
        # Final output
        self.final = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)
        
        # Special consideration for microaneurysms (small lesions)
        self.micro_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[3], decoder_channels[3] // 2, kernel_size=1),
            nn.BatchNorm2d(decoder_channels[3] // 2),
            nn.ReLU(inplace=True)
        )
        
        self.micro_output = nn.Conv2d(decoder_channels[3] // 2, 1, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.firstrelu(self.firstbn(self.firstconv(x)))
        x1 = self.firstmaxpool(x1)
        e1 = self.encoder1(x1)  # 256 channels
        e2 = self.encoder2(e1)  # 512 channels
        e3 = self.encoder3(e2)  # 1024 channels
        e4 = self.encoder4(e3)  # 2048 channels
        
        # Bottleneck
        b = self.bottleneck(e4)  # 2048 -> 512 channels
        
        # Apply attention and decoder
        # Stage 1
        a1 = self.attention1(b, e3)  # Apply attention to 1024-channel skip connection
        d1 = self.upconv1(b)  # 512 -> 256 channels
        d1 = torch.cat([d1, a1], dim=1)  # 256 + 1024 = 1280 channels
        d1 = self.decoder1(d1)  # 1280 -> 256 channels
        
        # Deep supervision 1
        dsv1 = self.dsv1(d1)
        dsv1 = F.interpolate(dsv1, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Stage 2
        a2 = self.attention2(d1, e2)  # Apply attention to 512-channel skip connection
        d2 = self.upconv2(d1)  # 256 -> 128 channels
        d2 = torch.cat([d2, a2], dim=1)  # 128 + 512 = 640 channels
        d2 = self.decoder2(d2)  # 640 -> 128 channels
        
        # Deep supervision 2
        dsv2 = self.dsv2(d2)
        dsv2 = F.interpolate(dsv2, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Stage 3
        a3 = self.attention3(d2, e1)  # Apply attention to 256-channel skip connection
        d3 = self.upconv3(d2)  # 128 -> 64 channels
        d3 = torch.cat([d3, a3], dim=1)  # 64 + 256 = 320 channels
        d3 = self.decoder3(d3)  # 320 -> 64 channels
        
        # Deep supervision 3
        dsv3 = self.dsv3(d3)
        dsv3 = F.interpolate(dsv3, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Final output
        out = self.final(d3)
        
        # Special branch for microaneurysms
        micro_features = self.micro_conv(d3)
        micro_out = self.micro_output(micro_features)
        
        # Replace the first channel (MA) with the specialized micro output
        if self.num_classes > 1:
            out[:, 0:1, :, :] = micro_out
        
        # Return results
        if self.training:
            return out, (dsv1, dsv2, dsv3)
        else:
            return out

def get_attention_unet(num_classes=len(config.LESION_TYPES), pretrained_encoder=True):
    """
    Factory function to create an Attention U-Net model.
    """
    model = AttentionUNet(
        num_classes=num_classes,
        pretrained_encoder=pretrained_encoder
    )
    return model