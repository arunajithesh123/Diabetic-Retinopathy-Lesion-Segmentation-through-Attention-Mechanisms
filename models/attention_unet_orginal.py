"""
Attention U-Net model for DR lesion segmentation.
Specifically designed to focus on small lesions like microaneurysms.

Key features:
1. Attention gates to focus on relevant regions
2. Multi-scale processing for different lesion sizes
3. Deep supervision for better gradient flow
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
    def __init__(self, num_classes, in_channels=3, features=[64, 128, 256, 512], pretrained_encoder=True):
        super(AttentionUNet, self).__init__()
        self.num_classes = num_classes
        
        # Always use pretrained encoder structure to match the checkpoint
        # ResNet34 as encoder
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        features = [64, 128, 256, 512]
        
        # Bottleneck
        self.bottleneck = DoubleConvBlock(features[3], features[3] * 2)
        
        # Attention gates
        self.attention1 = AttentionGate(F_g=features[3] * 2, F_l=features[2], F_int=features[2])
        self.attention2 = AttentionGate(F_g=features[2], F_l=features[1], F_int=features[1])
        self.attention3 = AttentionGate(F_g=features[1], F_l=features[0], F_int=features[0])
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.decoder1 = DoubleConvBlock(features[3] + features[2], features[2])
        
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = DoubleConvBlock(features[1] + features[1], features[1])
        
        self.upconv3 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder3 = DoubleConvBlock(features[0] + features[0], features[0])
        
        # Deep supervision outputs
        self.dsv1 = nn.Conv2d(features[2], num_classes, kernel_size=1)
        self.dsv2 = nn.Conv2d(features[1], num_classes, kernel_size=1)
        self.dsv3 = nn.Conv2d(features[0], num_classes, kernel_size=1)
        
        # Final output
        self.final = nn.Conv2d(features[0], num_classes, kernel_size=1)
        
        # Special consideration for microaneurysms (small lesions)
        # Extra filters with small receptive fields
        self.micro_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[0] // 2, kernel_size=1),
            nn.BatchNorm2d(features[0] // 2),
            nn.ReLU(inplace=True)
        )
        
        self.micro_output = nn.Conv2d(features[0] // 2, 1, kernel_size=1)  # Specific for MA detection
        
    def forward(self, x):
        # Encoder - Always use ResNet encoder path
        x1 = self.firstrelu(self.firstbn(self.firstconv(x)))
        x1 = self.firstmaxpool(x1)
        e1 = self.encoder1(x1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Apply attention and decoder
        # Stage 1
        a1 = self.attention1(b, e3)
        d1 = self.upconv1(b)
        d1 = torch.cat([d1, a1], dim=1)
        d1 = self.decoder1(d1)
        
        # Deep supervision 1
        dsv1 = self.dsv1(d1)
        dsv1 = F.interpolate(dsv1, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Stage 2
        a2 = self.attention2(d1, e2)
        d2 = self.upconv2(d1)
        d2 = torch.cat([d2, a2], dim=1)
        d2 = self.decoder2(d2)
        
        # Deep supervision 2
        dsv2 = self.dsv2(d2)
        dsv2 = F.interpolate(dsv2, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Stage 3
        a3 = self.attention3(d2, e1)
        d3 = self.upconv3(d2)
        d3 = torch.cat([d3, a3], dim=1)
        d3 = self.decoder3(d3)
        
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
        
        # Return results - simplify to just return out for testing
        if self.training:
            return out, (dsv1, dsv2, dsv3)
        else:
            return out

def get_attention_unet(num_classes=len(config.LESION_TYPES), pretrained_encoder=True):
    """
    Factory function to create an Attention U-Net model.
    """
    # FIX: Return the model instance, not the models module!
    model = AttentionUNet(
        num_classes=num_classes,
        pretrained_encoder=pretrained_encoder
    )
    return model  # Return the model instance, not models module