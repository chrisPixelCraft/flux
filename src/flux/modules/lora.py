"""
LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.

This module implements LoRA adapters for linear layers, enabling efficient
adaptation of large pre-trained models with minimal additional parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import math


class LinearLora(nn.Module):
    """
    LoRA adapter for linear layers.
    
    Implements low-rank adaptation by learning two smaller matrices (A and B)
    whose product approximates the weight updates. This reduces the number of
    trainable parameters from in_features * out_features to 
    (in_features + out_features) * rank.
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias term
        rank: Rank of the low-rank decomposition
        scale: Scaling factor for LoRA output (alpha/rank in original paper)
        dropout: Dropout probability for LoRA computation
        dtype: Data type for the layer
        device: Device to place the layer on
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 16,
        scale: float = 1.0,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = 'cpu'
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = scale
        self.dropout = dropout
        
        # Base linear layer (frozen during training)
        self.base_layer = nn.Linear(
            in_features, 
            out_features, 
            bias=bias,
            dtype=dtype,
            device=device
        )
        
        # LoRA matrices
        self.lora_A = nn.Parameter(
            torch.zeros(rank, in_features, dtype=dtype, device=device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, dtype=dtype, device=device)
        )
        
        # Dropout for LoRA path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize LoRA weights
        self._init_weights()
        
        # Freeze base layer by default
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def _init_weights(self):
        """Initialize LoRA weights using Kaiming initialization for A and zeros for B."""
        # Initialize A with Kaiming uniform (like standard linear layer)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # Initialize B with zeros (so initial LoRA contribution is zero)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Base layer forward pass
        base_output = self.base_layer(x)
        
        # LoRA forward pass: x @ A^T @ B^T
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        
        # Scale and combine
        return base_output + self.scale * lora_output
    
    def merge_weights(self):
        """
        Merge LoRA weights into the base layer.
        
        This is useful for inference when you want to eliminate the overhead
        of the separate LoRA computation.
        """
        if self.base_layer.weight.requires_grad:
            raise RuntimeError("Cannot merge LoRA weights when base layer is trainable")
        
        # Merge weights: W' = W + scale * B @ A
        with torch.no_grad():
            self.base_layer.weight.data += self.scale * self.lora_B @ self.lora_A
    
    def unmerge_weights(self):
        """
        Unmerge LoRA weights from the base layer.
        
        Reverses the effect of merge_weights().
        """
        with torch.no_grad():
            self.base_layer.weight.data -= self.scale * self.lora_B @ self.lora_A
    
    @property
    def weight(self):
        """Return the effective weight matrix (base + LoRA)."""
        return self.base_layer.weight + self.scale * self.lora_B @ self.lora_A
    
    @property
    def bias(self):
        """Return the bias term."""
        return self.base_layer.bias


def replace_linear_with_lora(
    model: nn.Module,
    max_rank: int = 16,
    scale: float = 1.0,
    dropout: float = 0.0,
    include_names: Optional[list] = None,
    exclude_names: Optional[list] = None
) -> int:
    """
    Replace Linear layers in a model with LoRA-adapted versions.
    
    Args:
        model: The model to modify
        max_rank: Maximum rank for LoRA adapters
        scale: Scaling factor for LoRA output
        dropout: Dropout probability for LoRA computation
        include_names: If provided, only replace layers whose names contain these strings
        exclude_names: If provided, skip layers whose names contain these strings
        
    Returns:
        Number of layers replaced
    """
    replaced_count = 0
    
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
            
        # Check inclusion/exclusion criteria
        if include_names and not any(inc in name for inc in include_names):
            continue
        if exclude_names and any(exc in name for exc in exclude_names):
            continue
        
        # Skip if already a LoRA layer
        if isinstance(module, LinearLora):
            continue
        
        # Calculate appropriate rank based on layer dimensions
        in_features = module.in_features
        out_features = module.out_features
        rank = min(max_rank, in_features // 2, out_features // 2)
        
        # Create LoRA replacement
        lora_layer = LinearLora(
            in_features=in_features,
            out_features=out_features,
            bias=module.bias is not None,
            rank=rank,
            scale=scale,
            dropout=dropout,
            dtype=module.weight.dtype,
            device=module.weight.device
        )
        
        # Copy original weights
        with torch.no_grad():
            lora_layer.base_layer.weight.copy_(module.weight)
            if module.bias is not None:
                lora_layer.base_layer.bias.copy_(module.bias)
        
        # Replace the module
        parent_module = model
        name_parts = name.split('.')
        for part in name_parts[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, name_parts[-1], lora_layer)
        
        replaced_count += 1
    
    return replaced_count


def get_lora_parameters(model: nn.Module):
    """
    Get only the LoRA parameters from a model.
    
    Useful for creating parameter groups in optimizers.
    
    Args:
        model: Model containing LoRA layers
        
    Returns:
        Generator yielding LoRA parameters
    """
    for module in model.modules():
        if isinstance(module, LinearLora):
            yield module.lora_A
            yield module.lora_B


def set_lora_scale(model: nn.Module, scale: float):
    """
    Set the scale factor for all LoRA layers in a model.
    
    Args:
        model: Model containing LoRA layers
        scale: New scale factor
    """
    for module in model.modules():
        if isinstance(module, LinearLora):
            module.scale = scale


def freeze_non_lora_parameters(model: nn.Module):
    """
    Freeze all parameters except LoRA parameters.
    
    Useful for LoRA-only fine-tuning.
    
    Args:
        model: Model to modify
    """
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Then unfreeze LoRA parameters
    for module in model.modules():
        if isinstance(module, LinearLora):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True