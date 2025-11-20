"""
PSPNet (Pyramid Scene Parsing Network) implementation for DR lesion segmentation.

PSPNet is excellent for medical segmentation as it captures multi-scale context
through pyramid pooling, which is crucial for detecting lesions of different sizes.

Reference: Zhao et al. "Pyramid Scene Parsing Network" (CVPR 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional

class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module for capturing multi-scale context.
    This is the key component that makes PSPNet effective for segmentation.
    """
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        
        # Each pooling branch reduces channels to 1/4 of input
        branch_channels = in_channels // 4
        
        self.branches = nn.ModuleList()
        for pool_size in pool_sizes:
            self.branches.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(in_channels, branch_channels, 1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final convolution to combine all branches
        self.conv_final = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * branch_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """Forward pass through pyramid pooling."""
        h, w = x.shape[2], x.shape[3]
        
        # Original feature map
        pyramid_features = [x]
        
        # Apply each pooling branch
        for branch in self.branches:
            pooled = branch(x)
            # Upsample back to original size
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            pyramid_features.append(upsampled)
        
        # Concatenate all features
        concat_features = torch.cat(pyramid_features, dim=1)
        
        # Final convolution
        output = self.conv_final(concat_features)
        
        return output

class ResNetBackbone(nn.Module):
    """
    ResNet backbone with dilated convolutions for PSPNet.
    Uses dilated convolutions in later stages to maintain spatial resolution.
    """
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super(ResNetBackbone, self).__init__()
        
        # Load pretrained ResNet
        if backbone_name == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.feature_channels = 512
        elif backbone_name == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.feature_channels = 512
        elif backbone_name == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.feature_channels = 2048
        elif backbone_name == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            self.feature_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Extract layers (excluding avgpool and fc)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Apply dilated convolutions to layer3 and layer4 for better spatial resolution
        self._make_layer_dilated(self.layer3, dilation=2)
        self._make_layer_dilated(self.layer4, dilation=4)
        
    def _make_layer_dilated(self, layer, dilation):
        """Apply dilated convolutions to a ResNet layer."""
        for module in layer.modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size[0] == 3:
                module.dilation = (dilation, dilation)
                module.padding = (dilation, dilation)
    
    def forward(self, x):
        """Forward pass through ResNet backbone."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

class PSPNet(nn.Module):
    """
    PSPNet model for semantic segmentation.
    
    Architecture:
    1. ResNet backbone with dilated convolutions
    2. Pyramid Pooling Module for multi-scale context
    3. Final classification head
    4. Auxiliary classifier for deep supervision (optional)
    """
    def __init__(self, 
                 num_classes, 
                 backbone='resnet50', 
                 pretrained=True,
                 aux_loss=True,
                 pool_sizes=[1, 2, 3, 6]):
        super(PSPNet, self).__init__()
        
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        
        # Backbone network
        self.backbone = ResNetBackbone(backbone, pretrained)
        backbone_channels = self.backbone.feature_channels
        
        # Pyramid Pooling Module
        self.ppm = PyramidPoolingModule(backbone_channels, pool_sizes)
        
        # Main classification head
        self.classifier = nn.Sequential(
            nn.Conv2d(backbone_channels, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, 1)
        )
        
        # Auxiliary classifier for deep supervision
        if aux_loss:
            # Use features from layer3 (before layer4)
            aux_channels = backbone_channels // 2 if 'resnet50' in backbone or 'resnet101' in backbone else 256
            self.aux_classifier = nn.Sequential(
                nn.Conv2d(aux_channels, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, num_classes, 1)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for non-backbone layers."""
        for module in [self.ppm, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        if self.aux_loss:
            for m in self.aux_classifier.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through PSPNet."""
        input_size = x.shape[2:]
        
        # Extract features at different stages
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        
        # Save layer3 features for auxiliary loss
        if self.aux_loss and self.training:
            aux_features = x
        
        x = self.backbone.layer4(x)
        
        # Apply Pyramid Pooling Module
        x = self.ppm(x)
        
        # Main classifier
        main_out = self.classifier(x)
        main_out = F.interpolate(main_out, size=input_size, mode='bilinear', align_corners=False)
        
        if self.aux_loss and self.training:
            # Auxiliary classifier
            aux_out = self.aux_classifier(aux_features)
            aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=False)
            return main_out, aux_out
        else:
            return main_out

class PSPNetLesionSegmentor(PSPNet):
    """
    PSPNet specialized for DR lesion segmentation with lesion-specific optimizations.
    """
    def __init__(self, 
                 lesion_types,
                 backbone='resnet50',
                 pretrained=True,
                 aux_loss=True):
        
        num_classes = len(lesion_types)
        super(PSPNetLesionSegmentor, self).__init__(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            aux_loss=aux_loss,
            pool_sizes=[1, 2, 3, 6]  # Good for medical images
        )
        
        self.lesion_types = lesion_types
        
        # Add lesion-specific attention modules
        self.lesion_attention = nn.ModuleList([
            self._make_attention_module(512) for _ in lesion_types
        ])
        
        # Lesion-specific refinement heads
        self.lesion_refinement = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 1, 1)
            ) for _ in lesion_types
        ])
        
    def _make_attention_module(self, channels):
        """Create channel attention module for lesion-specific features."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass with lesion-specific processing."""
        input_size = x.shape[2:]
        
        # Backbone forward pass
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        
        # Save layer3 features for auxiliary loss
        if self.aux_loss and self.training:
            aux_features = x
        
        x = self.backbone.layer4(x)
        
        # Apply Pyramid Pooling Module
        x = self.ppm(x)
        
        # Extract shared features
        shared_features = self.classifier[:-1](x)  # All layers except final conv
        
        # Lesion-specific processing
        lesion_outputs = []
        for i, (attention, refinement) in enumerate(zip(self.lesion_attention, self.lesion_refinement)):
            # Apply attention
            att_weights = attention(shared_features)
            attended_features = shared_features * att_weights
            
            # Lesion-specific refinement
            lesion_out = refinement(attended_features)
            lesion_out = F.interpolate(lesion_out, size=input_size, mode='bilinear', align_corners=False)
            lesion_outputs.append(lesion_out)
        
        # Combine lesion outputs
        main_out = torch.cat(lesion_outputs, dim=1)
        
        if self.aux_loss and self.training:
            # Auxiliary classifier
            aux_out = self.aux_classifier(aux_features)
            aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=False)
            return main_out, aux_out
        else:
            return main_out

def get_pspnet_model(num_classes, 
                    backbone='resnet50', 
                    pretrained=True, 
                    aux_loss=True,
                    lesion_specific=False,
                    lesion_types=None):
    """
    Factory function to create PSPNet models.
    
    Args:
        num_classes: Number of segmentation classes
        backbone: Backbone architecture ('resnet18', 'resnet34', 'resnet50', 'resnet101')
        pretrained: Whether to use ImageNet pretrained weights
        aux_loss: Whether to use auxiliary loss for deep supervision
        lesion_specific: Whether to use lesion-specific version
        lesion_types: List of lesion type names (required if lesion_specific=True)
    
    Returns:
        PSPNet model instance
    """
    if lesion_specific:
        if lesion_types is None:
            raise ValueError("lesion_types must be provided when lesion_specific=True")
        
        model = PSPNetLesionSegmentor(
            lesion_types=lesion_types,
            backbone=backbone,
            pretrained=pretrained,
            aux_loss=aux_loss
        )
        
        print(f"Created lesion-specific PSPNet:")
        print(f"  - Lesion types: {lesion_types}")
    else:
        model = PSPNet(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            aux_loss=aux_loss
        )
        
        print(f"Created standard PSPNet:")
        print(f"  - Classes: {num_classes}")
    
    print(f"  - Backbone: {backbone}")
    print(f"  - Pretrained: {pretrained}")
    print(f"  - Auxiliary loss: {aux_loss}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    return model

# Example usage and testing
if __name__ == "__main__":
    print("Testing PSPNet models...")
    
    # Test standard PSPNet
    model = get_pspnet_model(
        num_classes=4,
        backbone='resnet50',
        pretrained=False,  # Set to False for testing
        aux_loss=True
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    
    model.train()
    output = model(x)
    if isinstance(output, tuple):
        main_out, aux_out = output
        print(f"Training mode - Main output: {main_out.shape}, Aux output: {aux_out.shape}")
    else:
        print(f"Training mode - Output: {output.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"Eval mode - Output: {output.shape}")
    
    # Test lesion-specific PSPNet
    print("\nTesting lesion-specific PSPNet...")
    lesion_types = ['MA', 'EX', 'SE', 'HE']
    
    lesion_model = get_pspnet_model(
        num_classes=len(lesion_types),
        backbone='resnet50',
        pretrained=False,
        aux_loss=True,
        lesion_specific=True,
        lesion_types=lesion_types
    )
    
    lesion_model.train()
    output = lesion_model(x)
    if isinstance(output, tuple):
        main_out, aux_out = output
        print(f"Lesion-specific training - Main output: {main_out.shape}, Aux output: {aux_out.shape}")
    else:
        print(f"Lesion-specific training - Output: {output.shape}")
    
    print("PSPNet testing completed successfully!")