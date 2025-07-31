"""
StyleBezierFusionModule for FLUX BezierAdapter integration.

This module implements Adaptive Instance Normalization (AdaIN) style transfer
with Bézier curve guidance for precise font stylization. It fuses style embeddings
with spatial features using density-aware modulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from einops import rearrange

from .layers import QKNorm


class AdaINLayer(nn.Module):
    """
    Adaptive Instance Normalization layer for style transfer.
    
    Modulates spatial features using style statistics (mean, std) derived
    from style embeddings and Bézier density guidance.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
    def forward(self, content_features: Tensor, style_mean: Tensor, style_std: Tensor) -> Tensor:
        """
        Apply AdaIN normalization.
        
        Args:
            content_features: Content features (B, C, H, W) or (B, L, C)
            style_mean: Style mean parameters (B, C) or (B, 1, C)
            style_std: Style std parameters (B, C) or (B, 1, C)
            
        Returns:
            stylized_features: Style-transferred features
        """
        # Handle both spatial (B, C, H, W) and sequential (B, L, C) formats
        if content_features.dim() == 4:
            # Spatial format: (B, C, H, W)
            size = content_features.size()
            content_mean = content_features.view(size[0], size[1], -1).mean(dim=2, keepdim=True)
            content_std = content_features.view(size[0], size[1], -1).std(dim=2, keepdim=True) + self.eps
            
            # Normalize content
            normalized = (content_features.view(size[0], size[1], -1) - content_mean) / content_std
            
            # Apply style statistics
            stylized = normalized * style_std.unsqueeze(-1) + style_mean.unsqueeze(-1)
            stylized = stylized.view(size)
            
        else:
            # Sequential format: (B, L, C)
            content_mean = content_features.mean(dim=1, keepdim=True)  # (B, 1, C)
            content_std = content_features.std(dim=1, keepdim=True) + self.eps  # (B, 1, C)
            
            # Normalize and stylize
            normalized = (content_features - content_mean) / content_std
            stylized = normalized * style_std + style_mean
        
        return stylized


class DensityAwareStyleProjector(nn.Module):
    """
    Projects style and density information to AdaIN parameters.
    
    Combines style embeddings with density guidance to generate
    spatially-varying style transfer parameters.
    """
    
    def __init__(
        self, 
        style_dim: int = 1536, 
        spatial_dim: int = 768,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.style_dim = style_dim
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        
        # Style embedding projector
        self.style_projector = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Density guidance projector
        self.density_projector = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, hidden_dim // 2)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, spatial_dim * 2)  # Generate mean and std
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following FLUX patterns."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, style_embeddings: Tensor, density_weights: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Generate AdaIN parameters from style and density information.
        
        Args:
            style_embeddings: Style embeddings (B, style_dim)
            density_weights: Density map (B, H, W)
            
        Returns:
            style_mean: AdaIN mean parameters (B, 1, spatial_dim)
            style_std: AdaIN std parameters (B, 1, spatial_dim)
        """
        batch_size = style_embeddings.size(0)
        
        # Project style embeddings
        style_features = self.style_projector(style_embeddings)  # (B, hidden_dim)
        
        # Project density guidance
        density_input = density_weights.unsqueeze(1)  # (B, 1, H, W)
        density_features = self.density_projector(density_input)  # (B, hidden_dim//2)
        
        # Fuse style and density information
        combined_features = torch.cat([style_features, density_features], dim=1)
        adain_params = self.fusion_layer(combined_features)  # (B, spatial_dim * 2)
        
        # Split into mean and std parameters
        mean_params, std_params = adain_params.chunk(2, dim=1)  # Each: (B, spatial_dim)
        
        # Add small epsilon to std and apply softplus for positivity
        style_mean = mean_params.unsqueeze(1)  # (B, 1, spatial_dim)
        style_std = F.softplus(std_params).unsqueeze(1) + 1e-5  # (B, 1, spatial_dim)
        
        return style_mean, style_std


class StyleBezierFusionModule(nn.Module):
    """
    Style-Bézier fusion module with AdaIN-based style transfer.
    
    Integrates style embeddings with spatial features using Adaptive Instance
    Normalization, guided by Bézier curve density for precise font stylization.
    
    Args:
        spatial_dim: Input spatial feature dimension (default: 768)
        style_dim: Style embedding dimension (default: 1536)
        num_heads: Number of attention heads for cross-modal fusion (default: 8)
        dropout: Dropout rate (default: 0.1)
        use_cross_attention: Whether to use cross-attention fusion (default: True)
    """
    
    def __init__(
        self,
        spatial_dim: int = 768,
        style_dim: int = 1536,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_cross_attention: bool = True
    ):
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.style_dim = style_dim
        self.num_heads = num_heads
        self.use_cross_attention = use_cross_attention
        
        # AdaIN layer for style transfer
        self.adain_layer = AdaINLayer(spatial_dim)
        
        # Density-aware style projector
        self.style_projector = DensityAwareStyleProjector(
            style_dim=style_dim,
            spatial_dim=spatial_dim,
            hidden_dim=512
        )
        
        # Optional cross-attention for enhanced style-content fusion
        if use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=spatial_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # Layer normalization
            self.norm1 = nn.LayerNorm(spatial_dim)
            self.norm2 = nn.LayerNorm(spatial_dim)
            
        # Residual projection for skip connection
        self.residual_proj = nn.Linear(spatial_dim, spatial_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(spatial_dim, spatial_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_dim * 2, spatial_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following FLUX patterns."""
        for module in [self.residual_proj]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
        # Initialize output projection
        for module in self.output_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        spatial_features: Tensor, 
        style_embeddings: Tensor, 
        density_weights: Tensor
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass of StyleBezierFusionModule.
        
        Args:
            spatial_features: Input spatial features (B, L, spatial_dim)
            style_embeddings: Style embeddings (B, style_dim)
            density_weights: Density map from BezierParameterProcessor (B, H, W)
            
        Returns:
            stylized_features: Style-transferred spatial features (B, L, spatial_dim)
            attention_weights: Cross-attention weights if enabled (B, num_heads, L, 1)
        """
        batch_size, seq_len, feature_dim = spatial_features.shape
        
        # Validate input dimensions
        if feature_dim != self.spatial_dim:
            raise ValueError(f"Expected spatial_dim={self.spatial_dim}, got {feature_dim}")
        
        if style_embeddings.shape[-1] != self.style_dim:
            raise ValueError(f"Expected style_dim={self.style_dim}, got {style_embeddings.shape[-1]}")
        
        # Generate AdaIN parameters from style and density
        style_mean, style_std = self.style_projector(style_embeddings, density_weights)
        
        # Apply AdaIN style transfer
        stylized_features = self.adain_layer(spatial_features, style_mean, style_std)
        
        # Optional cross-attention fusion
        attention_weights = None
        if self.use_cross_attention:
            # Use style embeddings as keys/values for cross-attention
            style_kv = style_embeddings.unsqueeze(1)  # (B, 1, style_dim)
            
            # Project style to match spatial dimension for cross-attention
            if self.style_dim != self.spatial_dim:
                style_proj = nn.Linear(self.style_dim, self.spatial_dim, device=style_embeddings.device)
                style_kv = style_proj(style_kv)
            
            # Cross-attention: spatial features attend to style
            normed_features = self.norm1(stylized_features)
            attn_output, attention_weights = self.cross_attention(
                query=normed_features,
                key=style_kv,
                value=style_kv
            )
            
            # Residual connection
            stylized_features = stylized_features + attn_output
            stylized_features = self.norm2(stylized_features)
        
        # Residual connection with input
        residual = self.residual_proj(spatial_features)
        stylized_features = stylized_features + residual
        
        # Final output projection
        output_features = self.output_proj(stylized_features)
        
        return output_features, attention_weights
    
    def extra_repr(self) -> str:
        """String representation of module parameters."""
        return (
            f"spatial_dim={self.spatial_dim}, style_dim={self.style_dim}, "
            f"num_heads={self.num_heads}, use_cross_attention={self.use_cross_attention}"
        )