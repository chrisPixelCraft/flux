"""
ConditionInjectionAdapter for FLUX BezierAdapter integration.

This module implements a 4-branch multi-modal fusion architecture:
1. Style Branch: CLIP features → LoRA adaptation
2. Text Branch: T5 features → LoRA adaptation  
3. Mask Branch: VAE features → LoRA adaptation
4. Bézier Branch: Processed Bézier features → LoRA adaptation

All branches are fused through multi-head cross-attention for unified conditioning.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .lora import LinearLora
from .models import MultiModalCondition


class ConditionInjectionAdapter(nn.Module):
    """
    Multi-modal condition fusion adapter with LoRA efficiency.
    
    Processes different modalities (style, text, mask, Bézier) through dedicated branches
    and fuses them via multi-head cross-attention. Each branch uses LoRA adapters for
    parameter efficiency while maintaining the frozen FLUX backbone.
    
    Supports both standard FLUX (4 VAE channels) and FLUX.1-Fill-dev (320 channels)
    models with automatic adaptation of the mask processing branch.
    
    Args:
        clip_dim: CLIP feature dimension (default: 768)
        t5_dim: T5 text feature dimension (default: 4096)
        hidden_dim: Unified hidden dimension for fusion (default: 1536)
        lora_rank: LoRA rank for parameter efficiency (default: 64)
        num_heads: Number of attention heads for fusion (default: 8)
        dropout: Dropout rate for attention (default: 0.1)
        lora_scale: LoRA adaptation scale (default: 1.0)
        mask_channels: Number of mask/conditioning channels (4 for FLUX, 320 for Fill)
    """
    
    def __init__(
        self,
        clip_dim: int = 768,
        t5_dim: int = 4096,
        hidden_dim: int = 1536,
        lora_rank: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
        lora_scale: float = 1.0,
        mask_channels: int = 4  # VAE channels for standard FLUX, 320 for Fill models
    ):
        super().__init__()
        
        self.clip_dim = clip_dim
        self.t5_dim = t5_dim
        self.hidden_dim = hidden_dim
        self.lora_rank = lora_rank
        self.num_heads = num_heads
        self.mask_channels = mask_channels
        
        # Style branch: CLIP features (768) → hidden_dim + LoRA
        self.style_proj = nn.Linear(clip_dim, hidden_dim)
        self.style_lora = LinearLora(
            in_features=hidden_dim,
            out_features=hidden_dim,
            bias=True,
            rank=lora_rank,
            scale=lora_scale,
            dtype=torch.float32,
            device='cpu'  # Will be moved to correct device automatically
        )
        
        # Text branch: T5 features (4096) → hidden_dim + LoRA
        self.text_proj = nn.Linear(t5_dim, hidden_dim)
        self.text_lora = LinearLora(
            in_features=hidden_dim,
            out_features=hidden_dim,
            bias=True,
            rank=lora_rank,
            scale=lora_scale,
            dtype=torch.float32,
            device='cpu'
        )
        
        # Mask branch: VAE/Fill features → Conv2D → GlobalAvgPool → Linear + LoRA
        # Handles both standard VAE (4 channels) and Fill model (320 channels) conditioning
        # More memory-efficient approach using global average pooling
        
        # For Fill models with 320 channels, we use a larger intermediate representation
        conv_out_channels = 128 if mask_channels > 100 else 64
        
        self.mask_conv = nn.Conv2d(mask_channels, conv_out_channels, kernel_size=3, padding=1)
        self.mask_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.mask_proj = nn.Linear(conv_out_channels, hidden_dim)
        self.mask_lora = LinearLora(
            in_features=hidden_dim,
            out_features=hidden_dim,
            bias=True,
            rank=lora_rank,
            scale=lora_scale,
            dtype=torch.float32,
            device='cpu'
        )
        
        # Bézier branch: MLP for (x, y, density) triplet → hidden_dim + LoRA
        self.bezier_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        self.bezier_lora = LinearLora(
            in_features=hidden_dim,
            out_features=hidden_dim,
            bias=True,
            rank=lora_rank,
            scale=lora_scale,
            dtype=torch.float32,
            device='cpu'
        )
        
        # Multi-head cross-attention fusion
        # Critical for integrating multiple modalities
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization and residual connections
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following FLUX patterns."""
        # Initialize projection layers with Xavier uniform
        for module in [self.style_proj, self.text_proj, self.mask_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        # Initialize MLP layers
        for module in self.bezier_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize conv layer
        nn.init.kaiming_normal_(self.mask_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.mask_conv.bias is not None:
            nn.init.zeros_(self.mask_conv.bias)
            
    def _process_style_branch(self, style_features: Tensor) -> Tensor:
        """Process CLIP style features through style branch."""
        # Project to hidden dimension
        projected = self.style_proj(style_features)  # (batch, hidden_dim)
        
        # Apply LoRA adaptation
        adapted = self.style_lora(projected)  # (batch, hidden_dim)
        
        return adapted
        
    def _process_text_branch(self, text_features: Tensor) -> Tensor:
        """Process T5 text features through text branch."""
        # Project to hidden dimension
        projected = self.text_proj(text_features)  # (batch, hidden_dim)
        
        # Apply LoRA adaptation
        adapted = self.text_lora(projected)  # (batch, hidden_dim)
        
        return adapted
        
    def _process_mask_branch(self, mask_features: Tensor) -> Tensor:
        """Process VAE mask features through mask branch."""
        # Convolutional processing for spatial features
        conv_out = self.mask_conv(mask_features)  # (batch, 64, 64, 64)
        
        # Global average pooling to reduce spatial dimensions efficiently
        pooled = self.mask_pool(conv_out)  # (batch, 64, 1, 1)
        flattened = pooled.flatten(start_dim=1)  # (batch, 64)
        
        # Project to hidden dimension
        projected = self.mask_proj(flattened)  # (batch, hidden_dim)
        
        # Apply LoRA adaptation
        adapted = self.mask_lora(projected)  # (batch, hidden_dim)
        
        return adapted
        
    def _process_bezier_branch(self, bezier_features: Tensor) -> Tensor:
        """Process Bézier features through Bézier branch."""
        # MLP processing for (x, y, density) features
        mlp_out = self.bezier_mlp(bezier_features)  # (batch, hidden_dim)
        
        # Apply LoRA adaptation
        adapted = self.bezier_lora(mlp_out)  # (batch, hidden_dim)
        
        return adapted
        
    def forward(self, conditions: MultiModalCondition) -> Tensor:
        """
        Forward pass of ConditionInjectionAdapter.
        
        Args:
            conditions: Multi-modal condition inputs
            
        Returns:
            unified_conditions: Fused condition embeddings (batch, hidden_dim)
        """
        fusion_inputs = []
        
        # Process each available modality through its respective branch
        if conditions.style_features is not None:
            style_emb = self._process_style_branch(conditions.style_features)
            fusion_inputs.append(style_emb.unsqueeze(1))  # Add sequence dimension
            
        if conditions.text_features is not None:
            text_emb = self._process_text_branch(conditions.text_features)
            fusion_inputs.append(text_emb.unsqueeze(1))
            
        if conditions.mask_features is not None:
            mask_emb = self._process_mask_branch(conditions.mask_features)
            fusion_inputs.append(mask_emb.unsqueeze(1))
            
        if conditions.bezier_features is not None:
            bezier_emb = self._process_bezier_branch(conditions.bezier_features)
            fusion_inputs.append(bezier_emb.unsqueeze(1))
        
        # Validate that at least one modality is provided
        if not fusion_inputs:
            raise ValueError("At least one modality must be provided in conditions")
            
        # Concatenate all modalities for cross-attention fusion
        # Shape: (batch, num_modalities, hidden_dim)
        fused_sequence = torch.cat(fusion_inputs, dim=1)
        
        # Apply multi-head cross-attention fusion
        # Self-attention across modalities to create unified representation
        attended_output, attention_weights = self.fusion_attention(
            fused_sequence, fused_sequence, fused_sequence
        )
        
        # Apply dropout for regularization
        attended_output = self.dropout(attended_output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(attended_output + fused_sequence)
        
        # Return unified condition embedding (mean pooling across modalities)
        # Shape: (batch, hidden_dim)
        unified_conditions = output.mean(dim=1)
        
        return unified_conditions
        
    def get_lora_parameters(self):
        """Get all LoRA parameters for parameter-efficient training."""
        lora_params = []
        for lora_module in [self.style_lora, self.text_lora, self.mask_lora, self.bezier_lora]:
            lora_params.extend(list(lora_module.lora_A.parameters()))
            lora_params.extend(list(lora_module.lora_B.parameters()))
        return lora_params
        
    def set_lora_scale(self, scale: float):
        """Set LoRA scale for all branches."""
        for lora_module in [self.style_lora, self.text_lora, self.mask_lora, self.bezier_lora]:
            lora_module.set_scale(scale)
            
    def extra_repr(self) -> str:
        """String representation of module parameters."""
        return (
            f"clip_dim={self.clip_dim}, t5_dim={self.t5_dim}, "
            f"hidden_dim={self.hidden_dim}, lora_rank={self.lora_rank}, "
            f"num_heads={self.num_heads}"
        )