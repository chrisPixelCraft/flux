"""
BezierFluxPipeline - Simplified pipeline for BezierAdapter-enhanced FLUX models.

This pipeline provides an easy-to-use interface for generating images with
BezierAdapter guidance, building on the existing FLUX infrastructure.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
from PIL import Image
from torch import Tensor

from ..model import Flux
from ..modules.bezier_flux_model import FluxBezierAdapter
from ..modules.models import MultiModalCondition, BezierControlPoints
from ..modules.autoencoder import AutoEncoder
from ..modules.conditioner import HFEmbedder
from ..sampling import (
    get_noise, prepare, prepare_fill, get_schedule, denoise, unpack,
    time_shift, get_lin_function
)


class BezierFluxPipeline:
    """
    Simplified pipeline for BezierAdapter-enhanced FLUX models.
    
    This pipeline provides an easy interface for generating images with both
    standard FLUX and FLUX.1-Fill-dev models, enhanced with BezierAdapter
    capabilities for precise font control and style transfer.
    
    Args:
        model: FluxBezierAdapter model instance
        ae: AutoEncoder for VAE encoding/decoding
        t5: T5 text encoder
        clip: CLIP text encoder
        device: Target device for computation
    """
    
    def __init__(
        self,
        model: FluxBezierAdapter,
        ae: AutoEncoder,
        t5: HFEmbedder,
        clip: HFEmbedder,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.ae = ae.to(device)
        self.t5 = t5.to(device)
        self.clip = clip.to(device)
        self.device = device
        
        # Model capabilities
        self.is_fill_model = model.is_fill_model
        self.is_standard_model = model.is_standard_model
        self.model_type = model.bezier_stats.get('model_type', 'Unknown')
        
        print(f"Initialized BezierFluxPipeline:")
        print(f"  Model type: {self.model_type}")
        print(f"  Fill model: {self.is_fill_model}")
        print(f"  Device: {device}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "flux-dev-fill",
        bezier_config: Optional[Dict] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        """
        Load a pre-trained BezierFluxPipeline.
        
        Args:
            model_name: FLUX model variant ("flux-dev", "flux-dev-fill", etc.)
            bezier_config: BezierAdapter configuration dictionary
            device: Target device
            torch_dtype: Model precision
            
        Returns:
            BezierFluxPipeline instance
        """
        from ..util import load_flow_model, load_ae, load_t5, load_clip, configs
        from ..modules.bezier_flux_model import BezierAdapterConfig
        
        print(f"Loading {model_name} with BezierAdapter...")
        
        # Load FLUX model
        flux_model = load_flow_model(model_name, device=device)
        
        # Convert to BezierAdapter
        if bezier_config is None:
            bezier_config = {}
        
        adapter_config = BezierAdapterConfig(**bezier_config)
        bezier_model = FluxBezierAdapter(flux_model.params, adapter_config)
        
        # Copy weights from original FLUX
        bezier_model.load_state_dict(flux_model.state_dict(), strict=False)
        
        # Load other components
        ae = load_ae(model_name, device=device)
        t5 = load_t5(device=device)
        clip = load_clip(device=device)
        
        return cls(
            model=bezier_model,
            ae=ae,
            t5=t5,
            clip=clip,
            device=device
        )
    
    def load_bezier_curves(
        self, 
        bezier_source: Union[str, List, Dict, BezierControlPoints]
    ) -> Tensor:
        """
        Load Bézier curves from various sources.
        
        Args:
            bezier_source: Can be:
                - String path to JSON file
                - List of (x, y) coordinate pairs
                - Dictionary with 'control_points' key
                - BezierControlPoints object
                
        Returns:
            Tensor of Bézier control points (1, N, 2)
        """
        if isinstance(bezier_source, str):
            # Load from JSON file
            with open(bezier_source, 'r') as f:
                data = json.load(f)
            if 'control_points' in data:
                points = data['control_points']
            else:
                points = data
        elif isinstance(bezier_source, dict):
            points = bezier_source.get('control_points', bezier_source)
        elif isinstance(bezier_source, BezierControlPoints):
            points = bezier_source.points
        else:
            points = bezier_source
        
        # Convert to tensor
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        
        # Ensure proper shape (batch, num_points, 2)
        if points.dim() == 2:
            points = points.unsqueeze(0)
        
        return points.to(self.device)
    
    def prepare_conditions(
        self,
        prompt: str,
        style_image: Optional[Image.Image] = None,
        bezier_curves: Optional[Union[str, List, Dict]] = None,
        mask_image: Optional[Image.Image] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare multi-modal conditioning for BezierAdapter.
        
        Args:
            prompt: Text prompt
            style_image: Style reference image
            bezier_curves: Bézier curve specification
            mask_image: Mask image (for Fill models)
            
        Returns:
            Dictionary with BezierAdapter conditioning
        """
        if bezier_curves is None and style_image is None:
            return None
        
        conditions = {}
        
        # Process text through CLIP
        with torch.no_grad():
            clip_features = self.clip([prompt])  # Shape: (1, 768)
            conditions['style_features'] = clip_features
        
        # Process text through T5
        with torch.no_grad():
            t5_features = self.t5([prompt])  # Shape: (1, seq_len, 4096)
            # Take mean pooling for conditioning
            conditions['text_features'] = t5_features.mean(dim=1)  # Shape: (1, 4096)
        
        # Process style image (simplified - would use CLIP vision encoder in practice)
        if style_image is not None:
            # Placeholder: In practice, would encode through CLIP vision model
            conditions['style_features'] = torch.randn(1, 768, device=self.device)
        
        # Process mask image for Fill models
        if mask_image is not None and self.is_fill_model:
            # Convert mask to tensor and resize
            mask_array = np.array(mask_image.convert('L'))
            mask_tensor = torch.from_numpy(mask_array).float() / 255.0
            
            # Resize to latent resolution (64x64) and add batch/channel dims
            mask_resized = torch.nn.functional.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0),
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            )
            
            # For Fill models, we need 320 channels - pad with zeros for now
            if self.is_fill_model:
                mask_features = torch.zeros(1, 320, 64, 64, device=self.device)
                mask_features[:, 0:1] = mask_resized.to(self.device)
            else:
                mask_features = mask_resized.repeat(1, 4, 1, 1).to(self.device)  # VAE channels
            
            conditions['mask_features'] = mask_features
        
        # Process Bézier curves
        bezier_points = None
        if bezier_curves is not None:
            bezier_points = self.load_bezier_curves(bezier_curves)
            
            # Create (x, y, density) features
            num_points = bezier_points.shape[1]
            densities = torch.ones(1, num_points, 1, device=self.device) / num_points
            bezier_features = torch.cat([bezier_points, densities], dim=-1)
            
            conditions['bezier_features'] = bezier_features
        
        # Create MultiModalCondition
        multimodal_condition = MultiModalCondition(
            style_features=conditions.get('style_features'),
            text_features=conditions.get('text_features'),
            mask_features=conditions.get('mask_features'),
            bezier_features=conditions.get('bezier_features')
        )
        
        return {
            'conditions': multimodal_condition,
            'bezier_points': bezier_points
        }
    
    def generate(
        self,
        prompt: str,
        # Image parameters
        width: int = 1024,
        height: int = 1024,
        num_steps: int = 50,
        guidance: float = 4.0,
        seed: Optional[int] = None,
        
        # BezierAdapter parameters
        bezier_curves: Optional[Union[str, List, Dict]] = None,
        style_image: Optional[Image.Image] = None,
        
        # Fill model parameters (if applicable)
        image: Optional[Image.Image] = None,
        mask_image: Optional[Image.Image] = None,
    ) -> Image.Image:
        """
        Generate an image with BezierAdapter guidance.
        
        Args:
            prompt: Text description
            width, height: Output dimensions
            num_steps: Number of denoising steps
            guidance: Guidance scale
            seed: Random seed for reproducibility
            bezier_curves: Bézier curve specification for font control
            style_image: Style reference for font transfer
            image: Input image (for Fill models)
            mask_image: Mask image (for Fill models)
            
        Returns:
            Generated PIL Image
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Prepare BezierAdapter conditions
        bezier_conditions = self.prepare_conditions(
            prompt=prompt,
            style_image=style_image,
            bezier_curves=bezier_curves,
            mask_image=mask_image
        )
        
        # Get noise
        noise = get_noise(
            num_samples=1,
            height=height,
            width=width,
            device=torch.device(self.device),
            dtype=torch.bfloat16,
            seed=seed or 0
        )
        
        # Prepare inputs based on model type
        if self.is_fill_model and image is not None and mask_image is not None:
            # Fill model: prepare image + mask conditioning
            inputs = prepare_fill(
                t5=self.t5,
                clip=self.clip,
                img=noise,
                prompt=prompt,
                ae=self.ae,
                img_cond_path=None,  # Would need to save image temporarily
                mask_path=None       # Would need to save mask temporarily
            )
            
            # For now, create dummy img_cond for testing
            inputs["img_cond"] = torch.randn(1, 4096, 320, device=self.device)  # Placeholder
            
        else:
            # Standard model: prepare basic inputs
            inputs = prepare(
                t5=self.t5,
                clip=self.clip,
                img=noise,
                prompt=prompt
            )
        
        # Get timesteps
        timesteps = get_schedule(
            num_steps=num_steps,
            image_seq_len=inputs["img"].shape[1],
            shift=True
        )
        
        # Denoise with BezierAdapter
        with torch.no_grad():
            # Custom denoising loop that includes BezierAdapter conditioning
            img = inputs["img"]
            
            for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
                t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
                guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
                
                # Model forward pass with BezierAdapter
                pred = self.model(
                    img=torch.cat((img, inputs.get("img_cond", torch.zeros_like(img))), dim=-1) if self.is_fill_model else img,
                    img_ids=inputs["img_ids"],
                    txt=inputs["txt"],
                    txt_ids=inputs["txt_ids"],
                    y=inputs["vec"],
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    bezier_conditions=bezier_conditions
                )
                
                # Update image
                img = img + (t_prev - t_curr) * pred
        
        # Unpack and decode
        img = unpack(img, height, width)
        
        with torch.no_grad():
            img = self.ae.decode(img)
        
        # Convert to PIL
        img = img.clamp(-1, 1)
        img = (img + 1) / 2  # [-1, 1] -> [0, 1]
        img = img.permute(0, 2, 3, 1)  # NCHW -> NHWC
        img = (img * 255).cpu().numpy().astype(np.uint8)
        
        return Image.fromarray(img[0])
    
    def generate_font_character(
        self,
        character: str,
        font_style: str = "elegant serif",
        bezier_curves: Optional[Union[str, List, Dict]] = None,
        width: int = 512,
        height: int = 512,
        **kwargs
    ) -> Image.Image:
        """
        Generate a specific font character with Bézier guidance.
        
        Args:
            character: Character to generate
            font_style: Style description
            bezier_curves: Bézier curve specification
            width, height: Output dimensions
            **kwargs: Additional generation parameters
            
        Returns:
            Generated character image
        """
        prompt = f"{font_style} typography character '{character}'"
        
        return self.generate(
            prompt=prompt,
            bezier_curves=bezier_curves,
            width=width,
            height=height,
            **kwargs
        )
    
    def transfer_font_style(
        self,
        source_image: Image.Image,
        target_style_image: Image.Image,
        mask_image: Image.Image,
        prompt: str = "font style transfer",
        **kwargs
    ) -> Image.Image:
        """
        Transfer style from one font to another using Fill model.
        
        Args:
            source_image: Original character/font image
            target_style_image: Style reference image
            mask_image: Mask defining regions to modify
            prompt: Text guidance
            **kwargs: Additional generation parameters
            
        Returns:
            Style-transferred image
        """
        if not self.is_fill_model:
            raise ValueError("Font style transfer requires FLUX.1-Fill-dev model")
        
        return self.generate(
            prompt=prompt,
            image=source_image,
            mask_image=mask_image,
            style_image=target_style_image,
            **kwargs
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        stats = self.model.get_integration_stats()
        
        return {
            "model_type": self.model_type,
            "is_fill_model": self.is_fill_model,
            "device": self.device,
            "total_parameters": stats["total_params"],
            "bezier_parameters": stats["total_bezier_params"],
            "parameter_efficiency": f"{stats['bezier_param_ratio']:.2%}",
            "hook_layers": stats["hook_layers"],
            "components_enabled": stats["components_enabled"]
        }