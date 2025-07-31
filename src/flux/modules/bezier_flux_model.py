"""
FluxBezierAdapter - FLUX model with integrated BezierAdapter framework.

Extends the base FLUX model with BezierAdapter hooks at layers 7-12 for
Bézier-guided font stylization with density-aware spatial attention.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
from torch import Tensor, nn

from flux.model import Flux, FluxParams
from flux.modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)

from .bezier_processor import BezierParameterProcessor
from .condition_adapter import ConditionInjectionAdapter
from .spatial_fuser import SpatialAttentionFuser
from .style_fusion import StyleBezierFusionModule
from .models import MultiModalCondition, BezierControlPoints


@dataclass
class BezierAdapterConfig:
    """Configuration for BezierAdapter integration."""
    # Integration settings
    hook_layers: list[int] = None  # Layers to insert BezierAdapter hooks
    enable_bezier_guidance: bool = True
    enable_style_transfer: bool = True
    enable_density_attention: bool = True
    
    # BezierParameterProcessor config
    output_resolution: tuple[int, int] = (64, 64)
    kde_hidden_dim: int = 128
    
    # ConditionInjectionAdapter config
    clip_dim: int = 768
    t5_dim: int = 4096
    fusion_dim: int = 1536
    lora_rank: int = 64
    
    # SpatialAttentionFuser config
    num_attention_heads: int = 12
    num_transformer_layers: int = 6
    
    # StyleBezierFusionModule config
    use_cross_attention: bool = True
    style_fusion_heads: int = 8

    def __post_init__(self):
        if self.hook_layers is None:
            self.hook_layers = [7, 8, 9, 10, 11, 12]  # Default FLUX layers 7-12


class FluxBezierAdapter(Flux):
    """
    FLUX model with integrated BezierAdapter framework.
    
    Extends the base FLUX transformer with BezierAdapter components for
    Bézier-guided font stylization, density-aware attention, and style transfer.
    
    Supports both standard FLUX-dev (64 channels) and FLUX.1-Fill-dev (384 channels)
    models with automatic detection and appropriate conditioning handling.
    
    Key features:
    - BezierParameterProcessor: KDE-based density calculation from Bézier curves
    - ConditionInjectionAdapter: Multi-modal fusion with LoRA efficiency  
    - SpatialAttentionFuser: Density-modulated spatial attention
    - StyleBezierFusionModule: AdaIN-based style transfer
    - Fill Model Support: Seamless integration with FLUX.1-Fill-dev inpainting
    """
    
    def __init__(
        self, 
        flux_params: FluxParams, 
        bezier_config: Optional[BezierAdapterConfig] = None
    ):
        super().__init__(flux_params)
        
        # BezierAdapter configuration
        self.bezier_config = bezier_config or BezierAdapterConfig()
        self.hook_layers = set(self.bezier_config.hook_layers)
        
        # Detect model type based on input channels
        self.is_fill_model = flux_params.in_channels == 384
        self.is_standard_model = flux_params.in_channels == 64
        
        if not (self.is_fill_model or self.is_standard_model):
            raise ValueError(
                f"Unsupported input channels: {flux_params.in_channels}. "
                f"Expected 64 (standard FLUX) or 384 (FLUX Fill)"
            )
        
        # Validate hook layers
        max_layer = len(self.double_blocks) + len(self.single_blocks)
        for layer_idx in self.hook_layers:
            if layer_idx >= max_layer:
                raise ValueError(f"Hook layer {layer_idx} exceeds model depth {max_layer}")
        
        # Initialize BezierAdapter components
        self._init_bezier_components()
        
        # Track integration statistics
        self.bezier_stats = {
            "total_bezier_params": sum(p.numel() for p in self.get_bezier_parameters()),
            "hook_layers": sorted(list(self.hook_layers)),
            "integration_enabled": True,
            "model_type": "FLUX.1-Fill-dev" if self.is_fill_model else "FLUX-dev",
            "input_channels": flux_params.in_channels
        }
        
    def _init_bezier_components(self):
        """Initialize all BezierAdapter components."""
        config = self.bezier_config
        
        # BezierParameterProcessor - converts Bézier curves to density maps
        if config.enable_bezier_guidance:
            self.bezier_processor = BezierParameterProcessor(
                output_resolution=config.output_resolution,
                hidden_dim=config.kde_hidden_dim
            )
        else:
            self.bezier_processor = None
        
        # Enhanced ConditionInjectionAdapter - multi-modal fusion with LoRA
        # For Fill models, we need to handle additional mask conditioning
        mask_channels = 320 if self.is_fill_model else 0  # Additional channels from Fill conditioning
        
        self.condition_adapter = ConditionInjectionAdapter(
            clip_dim=config.clip_dim,
            t5_dim=config.t5_dim,
            hidden_dim=config.fusion_dim,
            lora_rank=config.lora_rank,
            # Extended mask features for Fill model
            mask_channels=mask_channels if self.is_fill_model else 4  # Standard VAE channels
        )
        
        # SpatialAttentionFuser - density-modulated attention
        # Hidden size remains the same after img_in projection
        if config.enable_density_attention:
            self.spatial_fuser = SpatialAttentionFuser(
                feature_dim=self.hidden_size,
                num_heads=self.num_heads,  # Use FLUX model's num_heads for compatibility
                num_layers=config.num_transformer_layers
            )
        else:
            self.spatial_fuser = None
            
        # StyleBezierFusionModule - AdaIN style transfer
        if config.enable_style_transfer:
            self.style_fusion = StyleBezierFusionModule(
                spatial_dim=self.hidden_size,
                style_dim=config.fusion_dim,
                num_heads=config.style_fusion_heads,
                use_cross_attention=config.use_cross_attention
            )
        else:
            self.style_fusion = None
    
    def get_bezier_parameters(self):
        """Get all BezierAdapter parameters for training."""
        bezier_params = []
        
        for component in [self.bezier_processor, self.condition_adapter, 
                         self.spatial_fuser, self.style_fusion]:
            if component is not None:
                bezier_params.extend(list(component.parameters()))
                
        return bezier_params
    
    def get_lora_parameters(self):
        """Get LoRA parameters specifically for parameter-efficient training."""
        if self.condition_adapter is not None:
            return self.condition_adapter.get_lora_parameters()
        return []
    
    def set_bezier_training_mode(self, enabled: bool = True):
        """Enable/disable training for BezierAdapter components only."""
        for component in [self.bezier_processor, self.condition_adapter,
                         self.spatial_fuser, self.style_fusion]:
            if component is not None:
                component.train(enabled)
                
        # Keep FLUX backbone frozen during BezierAdapter training
        if enabled:
            for param in self.parameters():
                param.requires_grad = False
            for param in self.get_bezier_parameters():
                param.requires_grad = True
    
    def _apply_bezier_hook(
        self, 
        layer_idx: int, 
        img_features: Tensor,
        bezier_conditions: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        """
        Apply BezierAdapter processing at specified layer.
        
        Args:
            layer_idx: Current layer index
            img_features: Image features from FLUX layer (B, L, D)
            bezier_conditions: BezierAdapter conditioning inputs
            
        Returns:
            processed_features: BezierAdapter-processed features
        """
        if bezier_conditions is None or layer_idx not in self.hook_layers:
            return img_features
            
        # Extract conditioning inputs
        conditions = bezier_conditions.get('conditions', None)
        bezier_points = bezier_conditions.get('bezier_points', None)
        
        if conditions is None:
            return img_features
        
        processed_features = img_features
        
        # Step 1: Process Bézier curves to density maps (if enabled)
        density_weights = None
        if self.bezier_processor is not None and bezier_points is not None:
            density_weights, _ = self.bezier_processor(bezier_points)
        
        # Step 2: Multi-modal condition fusion
        unified_conditions = self.condition_adapter(conditions)
        
        # Step 3: Density-modulated spatial attention (if enabled)
        if self.spatial_fuser is not None and density_weights is not None:
            processed_features, _ = self.spatial_fuser(
                spatial_features=processed_features,
                density_weights=density_weights,
                condition_embeddings=unified_conditions
            )
        
        # Step 4: Style transfer with Bézier guidance (if enabled)
        if self.style_fusion is not None and density_weights is not None:
            processed_features, _ = self.style_fusion(
                spatial_features=processed_features,
                style_embeddings=unified_conditions,
                density_weights=density_weights
            )
        
        return processed_features
    
    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        # BezierAdapter-specific inputs
        bezier_conditions: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """
        Forward pass with BezierAdapter integration.
        
        Supports both standard FLUX (64 channels) and FLUX Fill (384 channels) models
        with automatic detection and appropriate conditioning handling.
        
        Args:
            img: Input image tensor 
                - Standard FLUX: (B, L, 64) - latent image only
                - FLUX Fill: (B, L, 384) - latent image + mask + conditioning
            img_ids, txt, txt_ids: Standard FLUX inputs
            timesteps, y, guidance: Standard FLUX conditioning
            bezier_conditions: BezierAdapter conditioning dictionary containing:
                - conditions: MultiModalCondition object
                - bezier_points: Tensor of Bézier control points
                
        Returns:
            output: Generated image tensor with BezierAdapter guidance
        """
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # Validate input dimensions based on model type
        expected_channels = 384 if self.is_fill_model else 64
        if img.shape[-1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} input channels for "
                f"{'FLUX Fill' if self.is_fill_model else 'standard FLUX'} model, "
                f"got {img.shape[-1]}"
            )

        # Enhanced preprocessing for Fill models
        if self.is_fill_model and bezier_conditions is not None:
            # Extract Fill model conditioning (image + mask) and BezierAdapter conditioning
            img = self._preprocess_fill_conditioning(img, bezier_conditions)
        
        # Standard FLUX preprocessing - img_in handles channel projection
        img = self.img_in(img)  # Projects to hidden_size regardless of input channels
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # Double stream blocks with BezierAdapter hooks
        for layer_idx, block in enumerate(self.double_blocks):
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
            
            # Apply BezierAdapter hook at specified layers
            if layer_idx in self.hook_layers:
                img = self._apply_bezier_hook(layer_idx, img, bezier_conditions)

        # Single stream blocks with BezierAdapter hooks
        img = torch.cat((txt, img), 1)
        txt_len = txt.shape[1]
        
        for layer_idx, block in enumerate(self.single_blocks):
            layer_idx += len(self.double_blocks)  # Adjust for global layer index
            img = block(img, vec=vec, pe=pe)
            
            # Apply BezierAdapter hook at specified layers
            if layer_idx in self.hook_layers:
                img = self._apply_bezier_hook(layer_idx, img, bezier_conditions)
        
        # Extract image features and final layer
        img = img[:, txt_len:, ...]
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        
        return img
    
    def _preprocess_fill_conditioning(self, img: Tensor, bezier_conditions: Dict[str, Any]) -> Tensor:
        """
        Preprocess Fill model conditioning to integrate BezierAdapter inputs.
        
        For FLUX Fill models, the input tensor contains:
        - img[:, :, :64]: Base latent image
        - img[:, :, 64:]: Fill conditioning (image + mask)
        
        This method can modify the conditioning to include BezierAdapter guidance.
        
        Args:
            img: Fill model input tensor (B, L, 384)
            bezier_conditions: BezierAdapter conditioning
            
        Returns:
            processed_img: Enhanced conditioning tensor (B, L, 384)
        """
        if bezier_conditions is None:
            return img
        
        # Split Fill conditioning
        base_latent = img[:, :, :64]           # Base image latent
        fill_conditioning = img[:, :, 64:]     # Image + mask conditioning (320 channels)
        
        # For now, pass through unchanged - could enhance conditioning here
        # Future enhancement: inject BezierAdapter density guidance into fill_conditioning
        
        return img  # Return unchanged for now
    
    def enable_bezier_guidance(self, enabled: bool = True):
        """Enable/disable BezierAdapter guidance."""
        self.bezier_config.enable_bezier_guidance = enabled
        self.bezier_config.enable_density_attention = enabled
        self.bezier_config.enable_style_transfer = enabled
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get statistics about BezierAdapter integration."""
        total_flux_params = sum(p.numel() for p in super().parameters())
        total_bezier_params = sum(p.numel() for p in self.get_bezier_parameters())
        
        return {
            "total_flux_params": total_flux_params,
            "total_bezier_params": total_bezier_params,
            "total_params": total_flux_params + total_bezier_params,
            "bezier_param_ratio": total_bezier_params / (total_flux_params + total_bezier_params),
            "hook_layers": sorted(list(self.hook_layers)),
            "components_enabled": {
                "bezier_processor": self.bezier_processor is not None,
                "condition_adapter": self.condition_adapter is not None,
                "spatial_fuser": self.spatial_fuser is not None,
                "style_fusion": self.style_fusion is not None,
            }
        }
    
    def extra_repr(self) -> str:
        """String representation of FluxBezierAdapter."""
        stats = self.get_integration_stats()
        return (
            f"FluxBezierAdapter(flux_params={self.params}, "
            f"bezier_params={stats['total_bezier_params']:,}, "
            f"hook_layers={stats['hook_layers']}, "
            f"param_ratio={stats['bezier_param_ratio']:.1%})"
        )


class FluxBezierLoraWrapper(FluxBezierAdapter):
    """
    FluxBezierAdapter with additional LoRA adaptation.
    
    Combines BezierAdapter framework with LoRA adaptation for maximum
    parameter efficiency while maintaining full expressiveness.
    """
    
    def __init__(
        self,
        flux_params: FluxParams,
        bezier_config: Optional[BezierAdapterConfig] = None,
        lora_rank: int = 128,
        lora_scale: float = 1.0,
        apply_lora_to_flux: bool = False,
    ):
        super().__init__(flux_params, bezier_config)
        
        self.lora_rank = lora_rank
        self.lora_scale = lora_scale
        
        # Optionally apply LoRA to FLUX backbone (for additional adaptation)
        if apply_lora_to_flux:
            from flux.modules.lora import replace_linear_with_lora
            replace_linear_with_lora(
                self,
                max_rank=lora_rank,
                scale=lora_scale,
            )
    
    def set_lora_scale(self, scale: float) -> None:
        """Set LoRA scale for all components."""
        # Set scale for BezierAdapter LoRA components
        if self.condition_adapter is not None:
            self.condition_adapter.set_lora_scale(scale)
            
        # Set scale for FLUX LoRA components (if enabled)
        from flux.modules.lora import LinearLora
        for module in self.modules():
            if isinstance(module, LinearLora):
                module.set_scale(scale=scale)