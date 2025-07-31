"""
SpatialAttentionFuser for FLUX BezierAdapter integration.

This module implements density-modulated spatial attention with transformer encoder-decoder
architecture. It fuses spatial features with condition embeddings using density guidance
from Bézier curves for precise spatial control.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple
from einops import rearrange
import math

from flux.math import attention, rope, apply_rope
from .layers import QKNorm, EmbedND


class DensityModulatedAttention(nn.Module):
    """
    Self-attention with density-based spatial modulation.
    
    Extends standard FLUX attention with density weighting to focus attention
    on regions with higher Bézier curve density.
    """
    
    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Standard QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Query and key normalization (FLUX pattern)
        self.norm = QKNorm(self.head_dim)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        
        # Density modulation parameters
        self.density_scale = nn.Parameter(torch.ones(1))
        self.density_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: Tensor, density_weights: Optional[Tensor] = None, pe: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass with optional density modulation.
        
        Args:
            x: Input features (B, L, D) where L = H*W (flattened spatial)
            density_weights: Density map (B, H, W) for spatial modulation
            pe: Position embeddings for RoPE compatibility
            
        Returns:
            output: Attended features (B, L, D)
        """
        B, L, D = x.shape
        
        # Standard QKV computation
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        
        # Apply QK normalization (FLUX pattern)
        q, k = self.norm(q, k, v)
        
        # Apply RoPE if position embeddings provided
        if pe is not None:
            q, k = apply_rope(q, k, pe)
        
        # Compute attention scores
        scores = torch.einsum('bhld,bhmd->bhlm', q, k) * self.scale
        
        # Apply density modulation if provided
        if density_weights is not None:
            # Resize density weights to match sequence length if needed
            B, H, W = density_weights.shape
            density_seq_len = H * W
            
            if density_seq_len != L:
                # Resize density weights to match actual sequence length
                import torch.nn.functional as F
                # Reshape to (B, 1, H, W) for interpolation
                density_reshaped = density_weights.unsqueeze(1)
                
                # Calculate target spatial dimensions
                target_h = target_w = int(L ** 0.5)
                if target_h * target_w != L:
                    # Handle non-square sequence lengths
                    target_h = int(L ** 0.5)
                    target_w = L // target_h
                    if target_h * target_w != L:
                        target_h, target_w = L, 1
                
                # Interpolate to target size
                density_resized = F.interpolate(
                    density_reshaped, 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                density_flat = density_resized.flatten(start_dim=2)  # (B, 1, L)
            else:
                density_flat = density_weights.flatten(start_dim=1).unsqueeze(1)  # (B, 1, L)
            
            # Create density modulation matrix
            # Each query position gets modulated by corresponding density value
            density_mod = self.density_scale * density_flat.unsqueeze(-1) + self.density_bias
            
            # Apply density modulation to attention scores
            scores = scores + density_mod
        
        # Apply softmax and compute attended values
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.einsum('bhlm,bhmd->bhld', attn_weights, v)
        
        # Reshape and project output
        out = rearrange(out, "B H L D -> B L (H D)")
        out = self.proj(out)
        
        return out


class TransformerLayer(nn.Module):
    """
    Transformer layer with density-modulated attention and MLP.
    """
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        
        # Density-modulated self-attention
        self.attention = DensityModulatedAttention(dim, num_heads)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP block
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        
    def forward(self, x: Tensor, density_weights: Optional[Tensor] = None, pe: Optional[Tensor] = None) -> Tensor:
        """Forward pass with residual connections."""
        # Self-attention with density modulation
        attn_out = self.attention(self.norm1(x), density_weights, pe)
        x = x + attn_out
        
        # MLP
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        
        return x


class SpatialAttentionFuser(nn.Module):
    """
    Spatial attention fuser with density-modulated attention mechanism.
    
    Implements a transformer encoder-decoder architecture that fuses spatial features
    with condition embeddings using density guidance from BezierParameterProcessor.
    
    Args:
        feature_dim: Input feature dimension (default: 768)
        num_heads: Number of attention heads (default: 12)
        num_layers: Number of transformer layers (default: 6)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        theta: RoPE theta parameter for position encoding (default: 10000)
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        theta: int = 10000
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.theta = theta
        
        # Input projection for spatial features
        self.spatial_proj = nn.Linear(feature_dim, feature_dim)
        
        # Condition injection layer
        self.condition_proj = nn.Linear(1536, feature_dim)  # From ConditionInjectionAdapter
        
        # Transformer layers with density modulation
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(feature_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        # Position embedding for spatial features (compatible with FLUX RoPE)
        # Use EmbedND like FLUX does for proper RoPE embeddings
        pe_dim = feature_dim // num_heads
        
        # For 2D spatial positions, ensure both axes are even for RoPE
        if pe_dim % 4 != 0:
            raise ValueError(f"pe_dim ({pe_dim}) must be divisible by 4 for 2D RoPE embeddings")
        
        # Split evenly between x and y axes
        x_dim = pe_dim // 2
        y_dim = pe_dim // 2
        axes_dim = [x_dim, y_dim]
        
        # Verify both dimensions are even
        assert x_dim % 2 == 0, f"x_dim ({x_dim}) must be even"
        assert y_dim % 2 == 0, f"y_dim ({y_dim}) must be even"
        assert sum(axes_dim) == pe_dim, f"axes_dim sum ({sum(axes_dim)}) != pe_dim ({pe_dim})"
        
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        
        # Note: spatial_ids will be created dynamically based on input sequence length
        
    def _create_spatial_ids(self, seq_len: int, device: torch.device) -> Tensor:
        """Create spatial position IDs dynamically based on sequence length."""
        # Infer spatial dimensions from sequence length (assume square)
        h = w = int(seq_len ** 0.5)
        if h * w != seq_len:
            # If not square, use closest square or fallback to 1D layout
            h = int(seq_len ** 0.5)
            w = seq_len // h
            if h * w != seq_len:
                # Fallback: treat as 1D sequence with dummy 2D positions
                h, w = seq_len, 1
        
        # Create 2D position indices for spatial dimensions (like FLUX img_ids)
        y_ids = torch.arange(h, dtype=torch.long, device=device).unsqueeze(1).repeat(1, w)
        x_ids = torch.arange(w, dtype=torch.long, device=device).unsqueeze(0).repeat(h, 1)
        
        # Stack to create position IDs and flatten
        # Shape: (H*W, 2) where each row is [x_id, y_id]
        spatial_ids = torch.stack([x_ids.flatten(), y_ids.flatten()], dim=-1)
        return spatial_ids
        
    def _get_position_embeddings(self, batch_size: int, seq_len: int, device: torch.device) -> Tensor:
        """Get RoPE position embeddings for spatial features."""
        # Create spatial IDs dynamically based on sequence length
        spatial_ids = self._create_spatial_ids(seq_len, device)  # (seq_len, 2)
        
        # Expand for batch dimension: (seq_len, 2) -> (B, seq_len, 2)
        batch_ids = spatial_ids.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Use EmbedND to generate position embeddings (like FLUX does)
        pe = self.pe_embedder(batch_ids)  # (B, 1, seq_len, pe_dim)
        
        return pe
        
    def forward(
        self, 
        spatial_features: Tensor, 
        density_weights: Tensor, 
        condition_embeddings: Tensor
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass of SpatialAttentionFuser.
        
        Args:
            spatial_features: Input spatial features (B, L, feature_dim) where L = H*W
            density_weights: Density map from BezierParameterProcessor (B, H, W)
            condition_embeddings: Unified conditions from ConditionInjectionAdapter (B, 1536)
            
        Returns:
            fused_features: Density-modulated spatial features (B, L, feature_dim)
            attention_maps: Optional attention visualization (B, num_heads, L, L)
        """
        batch_size, seq_len, _ = spatial_features.shape
        device = spatial_features.device
        
        # Validate input dimensions
        if spatial_features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {spatial_features.shape[-1]}")
        
        # Project spatial features
        x = self.spatial_proj(spatial_features)
        
        # Inject condition embeddings
        # Broadcast condition to all spatial locations
        condition_proj = self.condition_proj(condition_embeddings)  # (B, feature_dim)
        condition_spatial = condition_proj.unsqueeze(1).expand(-1, seq_len, -1)  # (B, L, feature_dim)
        x = x + condition_spatial
        
        # Get position embeddings
        pe = self._get_position_embeddings(batch_size, seq_len, device)
        
        # Apply transformer layers with gradient checkpointing for memory efficiency
        attention_maps = None
        for i, layer in enumerate(self.transformer_layers):
            if seq_len > 2048:  # Use checkpointing for large sequences
                x = checkpoint(
                    layer,
                    x,
                    density_weights,
                    pe,
                    use_reentrant=False
                )
            else:
                x = layer(x, density_weights, pe)
        
        # Output projection
        fused_features = self.output_proj(x)
        
        return fused_features, attention_maps
        
    def extra_repr(self) -> str:
        """String representation of module parameters."""
        return (
            f"feature_dim={self.feature_dim}, num_heads={self.num_heads}, "
            f"num_layers={self.num_layers}, theta={self.theta}"
        )