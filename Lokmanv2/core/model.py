"""
State-of-the-art 3D Attention U-Net for CT breast cancer metastasis detection.
Based on 2024 research combining attention mechanisms with deep supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math

class ConvBlock3D(nn.Module):
    """3D Convolutional block with normalization and activation"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 normalization: str = "batch", activation: str = "ReLU",
                 dropout_rate: float = 0.0):
        super().__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, 
                             stride, padding, bias=False)
        
        # Normalization
        if normalization.lower() == "batch":
            self.norm = nn.BatchNorm3d(out_channels)
        elif normalization.lower() == "instance":
            self.norm = nn.InstanceNorm3d(out_channels)
        elif normalization.lower() == "group":
            self.norm = nn.GroupNorm(8, out_channels)
        else:
            self.norm = nn.Identity()
        
        # Activation
        if activation == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "GELU":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class ResidualBlock3D(nn.Module):
    """3D Residual block for better gradient flow"""
    
    def __init__(self, channels: int, normalization: str = "batch", 
                 activation: str = "ReLU", dropout_rate: float = 0.0):
        super().__init__()
        
        self.conv1 = ConvBlock3D(channels, channels, normalization=normalization,
                                activation=activation, dropout_rate=dropout_rate)
        self.conv2 = ConvBlock3D(channels, channels, normalization=normalization,
                                activation="", dropout_rate=dropout_rate)
        
        if activation == "ReLU":
            self.final_activation = nn.ReLU(inplace=True)
        elif activation == "LeakyReLU":
            self.final_activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "GELU":
            self.final_activation = nn.GELU()
        else:
            self.final_activation = nn.Identity()
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        x = self.final_activation(x)
        return x

class AttentionGate3D(nn.Module):
    """3D Attention Gate for focusing on relevant features"""
    
    def __init__(self, gate_channels: int, in_channels: int, 
                 inter_channels: Optional[int] = None):
        super().__init__()
        
        if inter_channels is None:
            inter_channels = in_channels // 2
        
        self.W_gate = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm3d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm3d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate, x):
        """
        Args:
            gate: Gating signal from coarser scale
            x: Input feature map from finer scale
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(x)
        
        # Resize gate to match x if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class SpatialAttention3D(nn.Module):
    """3D Spatial attention mechanism"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        return x * attention

class ChannelAttention3D(nn.Module):
    """3D Channel attention mechanism"""
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        attention = avg_out + max_out
        attention = self.sigmoid(attention)
        return x * attention

class CBAM3D(nn.Module):
    """3D Convolutional Block Attention Module"""
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        
        self.channel_attention = ChannelAttention3D(in_channels, reduction)
        self.spatial_attention = SpatialAttention3D(in_channels)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class EncoderBlock3D(nn.Module):
    """3D Encoder block with attention"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 use_attention: bool = True, use_residual: bool = True,
                 normalization: str = "batch", activation: str = "ReLU",
                 dropout_rate: float = 0.0):
        super().__init__()
        
        self.conv1 = ConvBlock3D(in_channels, out_channels, 
                                normalization=normalization,
                                activation=activation, dropout_rate=dropout_rate)
        
        if use_residual:
            self.conv2 = ResidualBlock3D(out_channels, normalization=normalization,
                                       activation=activation, dropout_rate=dropout_rate)
        else:
            self.conv2 = ConvBlock3D(out_channels, out_channels,
                                   normalization=normalization,
                                   activation=activation, dropout_rate=dropout_rate)
        
        self.attention = CBAM3D(out_channels) if use_attention else nn.Identity()
        self.downsample = nn.MaxPool3d(2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        
        skip = x
        x = self.downsample(x)
        
        return x, skip

class DecoderBlock3D(nn.Module):
    """3D Decoder block with attention gate"""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 use_attention_gate: bool = True, use_residual: bool = True,
                 normalization: str = "batch", activation: str = "ReLU",
                 dropout_rate: float = 0.0):
        super().__init__()
        
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels // 2, 
                                          kernel_size=2, stride=2)
        
        self.attention_gate = AttentionGate3D(in_channels // 2, skip_channels) \
                             if use_attention_gate else nn.Identity()
        
        total_channels = (in_channels // 2) + skip_channels
        
        self.conv1 = ConvBlock3D(total_channels, out_channels,
                                normalization=normalization,
                                activation=activation, dropout_rate=dropout_rate)
        
        if use_residual:
            self.conv2 = ResidualBlock3D(out_channels, normalization=normalization,
                                       activation=activation, dropout_rate=dropout_rate)
        else:
            self.conv2 = ConvBlock3D(out_channels, out_channels,
                                   normalization=normalization,
                                   activation=activation, dropout_rate=dropout_rate)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Apply attention gate to skip connection
        if hasattr(self.attention_gate, 'W_gate'):  # Check if it's an actual attention gate
            skip = self.attention_gate(x, skip)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x

class AttentionUNet3D(nn.Module):
    """
    State-of-the-art 3D Attention U-Net for breast cancer metastasis detection.
    
    Features:
    - Multi-scale attention mechanisms
    - Deep supervision
    - Residual connections
    - Advanced normalization
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 base_channels: int = 32, depth: int = 5,
                 use_attention: bool = True, use_deep_supervision: bool = True,
                 normalization: str = "batch", activation: str = "ReLU",
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.depth = depth
        self.use_deep_supervision = use_deep_supervision
        
        # Calculate channel dimensions
        channels = [base_channels * (2 ** i) for i in range(depth)]
        
        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(depth - 1):
            in_ch = in_channels if i == 0 else channels[i-1]
            out_ch = channels[i]
            
            encoder = EncoderBlock3D(in_ch, out_ch, use_attention=use_attention,
                                   normalization=normalization, activation=activation,
                                   dropout_rate=dropout_rate)
            self.encoders.append(encoder)
        
        # Bottleneck
        bottleneck_in = channels[depth-2]
        bottleneck_out = channels[depth-1]
        self.bottleneck = nn.Sequential(
            ConvBlock3D(bottleneck_in, bottleneck_out, normalization=normalization,
                       activation=activation, dropout_rate=dropout_rate),
            ResidualBlock3D(bottleneck_out, normalization=normalization,
                           activation=activation, dropout_rate=dropout_rate),
            CBAM3D(bottleneck_out) if use_attention else nn.Identity()
        )
        
        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(depth - 1):
            in_ch = channels[depth - 1 - i]
            skip_ch = channels[depth - 2 - i]
            out_ch = channels[depth - 2 - i]
            
            decoder = DecoderBlock3D(in_ch, skip_ch, out_ch,
                                   use_attention_gate=use_attention,
                                   normalization=normalization, activation=activation,
                                   dropout_rate=dropout_rate)
            self.decoders.append(decoder)
        
        # Final output layers
        self.final_conv = nn.Conv3d(channels[0], out_channels, kernel_size=1)
        
        # Deep supervision outputs
        if use_deep_supervision:
            self.deep_supervision_outputs = nn.ModuleList()
            for i in range(depth - 2):  # Fixed indexing
                ds_conv = nn.Conv3d(channels[depth - 2 - i], out_channels, kernel_size=1)
                self.deep_supervision_outputs.append(ds_conv)
    
    def forward(self, x):
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder path
        for encoder in self.encoders:
            x, skip = encoder(x)
            encoder_outputs.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        deep_supervision_outputs = []
        
        for i, decoder in enumerate(self.decoders):
            skip = encoder_outputs[-(i+1)]  # Reverse order
            x = decoder(x, skip)
            
            # Deep supervision
            if self.use_deep_supervision and i < len(self.deep_supervision_outputs):
                ds_output = self.deep_supervision_outputs[i](x)
                # Upsample to original size if needed
                if ds_output.shape[2:] != encoder_outputs[0].shape[2:]:
                    ds_output = F.interpolate(ds_output, size=encoder_outputs[0].shape[2:],
                                            mode='trilinear', align_corners=True)
                deep_supervision_outputs.append(ds_output)
        
        # Final output
        main_output = self.final_conv(x)
        
        if self.training and self.use_deep_supervision:
            return main_output, deep_supervision_outputs
        else:
            return main_output
    
    def get_attention_maps(self, x):
        """Extract attention maps for visualization"""
        attention_maps = {}
        
        # Hook function to capture attention
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(module, (AttentionGate3D, CBAM3D, SpatialAttention3D, ChannelAttention3D)):
                    attention_maps[name] = output.detach()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in self.named_modules():
            if isinstance(module, (AttentionGate3D, CBAM3D, SpatialAttention3D, ChannelAttention3D)):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            output = self.forward(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return output, attention_maps

def create_model(config) -> AttentionUNet3D:
    """Create model from configuration"""
    return AttentionUNet3D(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_channels=config.model.base_channels,
        depth=config.model.depth,
        use_attention=config.model.use_attention,
        use_deep_supervision=config.model.use_deep_supervision,
        normalization=config.model.normalization,
        activation=config.model.activation,
        dropout_rate=config.model.dropout_rate
    )

# Test function
if __name__ == "__main__":
    # Test model
    model = AttentionUNet3D(in_channels=1, out_channels=4, base_channels=32, depth=4)
    
    # Create dummy input
    x = torch.randn(1, 1, 64, 64, 64)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / (1024**2):.2f} MB")