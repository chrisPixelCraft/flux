"""
Data models for BezierAdapter framework.

Provides Pydantic models for type safety and validation across all BezierAdapter components.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Tuple, Dict, Any
import torch
from torch import Tensor


class BezierControlPoints(BaseModel):
    """Bézier curve control points for font stylization."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    points: List[Tuple[float, float]] = Field(..., min_length=4, description="Bézier control points (x, y)")
    curve_type: str = Field("cubic", pattern="^(linear|quadratic|cubic)$")
    character: str = Field(..., min_length=1, max_length=1, description="Target character")
    font_size: float = Field(64.0, gt=0.0, description="Font size in pixels")


class DensityMapConfig(BaseModel):
    """Configuration for density map generation."""
    output_resolution: Tuple[int, int] = Field((64, 64), description="Target resolution matching FLUX latent space")
    kde_bandwidth: float = Field(0.1, gt=0.0, description="KDE bandwidth parameter")
    sigma_min: float = Field(0.01, gt=0.0, description="Minimum density value")
    sigma_max: float = Field(1.0, gt=0.0, description="Maximum density value")


class MultiModalCondition(BaseModel):
    """Multi-modal conditioning inputs for BezierAdapter."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    style_features: Optional[Tensor] = Field(None, description="CLIP style features (768,)")
    text_features: Optional[Tensor] = Field(None, description="T5 text features (4096,)")
    mask_features: Optional[Tensor] = Field(None, description="VAE mask features")
    bezier_features: Optional[Tensor] = Field(None, description="Processed Bézier features (3,)")


class TrainingConfig(BaseModel):
    """BezierAdapter training configuration."""
    learning_rate: float = Field(1e-4, gt=0.0)
    batch_size: int = Field(4, gt=0, description="Memory-optimized batch size")
    total_steps: int = Field(100000, gt=0)
    warmup_steps: int = Field(1000, gt=0)
    gradient_clipping: float = Field(1.0, gt=0.0)
    mixed_precision: bool = Field(True, description="Use FP16 for memory efficiency")
    trainable_params_only: bool = Field(True, description="Freeze FLUX backbone")
    
    # Multi-loss weights
    diffusion_loss_weight: float = Field(1.0, gt=0.0)
    density_loss_weight: float = Field(0.5, gt=0.0)
    style_loss_weight: float = Field(0.3, gt=0.0)


class BezierAdapterConfig(BaseModel):
    """Configuration for BezierAdapter components."""
    # BezierParameterProcessor config
    output_resolution: Tuple[int, int] = Field((64, 64))
    hidden_dim: int = Field(128, gt=0)
    
    # ConditionInjectionAdapter config
    clip_dim: int = Field(768, gt=0, description="CLIP feature dimension")
    t5_dim: int = Field(4096, gt=0, description="T5 feature dimension")
    fusion_dim: int = Field(1536, gt=0, description="Unified feature dimension")
    lora_rank: int = Field(64, gt=0, description="LoRA rank for parameter efficiency")
    
    # SpatialAttentionFuser config
    num_attention_heads: int = Field(12, gt=0)
    num_transformer_layers: int = Field(6, gt=0)
    
    # StyleBezierFusionModule config
    spatial_dim: int = Field(1280, gt=0, description="FLUX spatial feature dimension")
    style_fusion_heads: int = Field(8, gt=0)