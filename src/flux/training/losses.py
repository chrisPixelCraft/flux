"""
Multi-loss training system for FLUX BezierAdapter.

Implements coordinated optimization across diffusion quality, density accuracy,
and style transfer objectives with configurable loss weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass

from .config import TrainingConfig


@dataclass
class LossOutputs:
    """Container for multi-loss training outputs."""
    total_loss: Tensor
    diffusion_loss: Tensor
    density_loss: Optional[Tensor] = None
    style_loss: Optional[Tensor] = None
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class DiffusionLoss(nn.Module):
    """
    Standard FLUX diffusion loss for image generation quality.
    
    Implements flow matching loss with noise prediction for the rectified flow model.
    Compatible with FLUX's rectified flow formulation.
    """
    
    def __init__(self, 
                 prediction_type: str = "flow",
                 loss_type: str = "mse",
                 weighting_scheme: str = "uniform"):
        super().__init__()
        
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.weighting_scheme = weighting_scheme
        
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "huber":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, 
                model_output: Tensor, 
                target: Tensor, 
                timesteps: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Compute diffusion loss.
        
        Args:
            model_output: Model predictions (B, L, D)
            target: Target values (B, L, D) 
            timesteps: Timestep values (B,) for weighting
            mask: Optional mask for valid regions (B, L)
            
        Returns:
            Dictionary with loss components
        """
        # Basic loss computation
        if mask is not None:
            # Apply mask to focus on valid regions
            model_output = model_output * mask.unsqueeze(-1)
            target = target * mask.unsqueeze(-1)
        
        loss = self.loss_fn(model_output, target)
        
        # Apply timestep weighting if provided
        if timesteps is not None and self.weighting_scheme != "uniform":
            weights = self._compute_timestep_weights(timesteps)
            if weights.dim() != loss.dim():
                weights = weights.view(-1, *([1] * (loss.dim() - 1)))
            loss = loss * weights
            loss = loss.mean()
        
        # Compute additional metrics
        with torch.no_grad():
            mse = F.mse_loss(model_output, target)
            mae = F.l1_loss(model_output, target)
        
        return {
            "loss": loss,
            "mse": mse,
            "mae": mae
        }
    
    def _compute_timestep_weights(self, timesteps: Tensor) -> Tensor:
        """Compute per-timestep loss weights."""
        if self.weighting_scheme == "uniform":
            return torch.ones_like(timesteps)
        elif self.weighting_scheme == "snr":
            # Signal-to-noise ratio weighting (common in diffusion)
            # Higher weights for harder (noisier) examples
            snr = 1.0 / (timesteps + 1e-8)
            return snr / snr.mean()
        elif self.weighting_scheme == "min_snr":
            # Min-SNR weighting to prevent very high weights
            snr = 1.0 / (timesteps + 1e-8)
            snr_clipped = torch.clamp(snr, max=5.0)
            return snr_clipped / snr_clipped.mean()
        else:
            return torch.ones_like(timesteps)


class DensityLoss(nn.Module):
    """
    KDE-based density loss for Bézier curve accuracy.
    
    Ensures that generated density maps accurately reflect the spatial
    distribution of Bézier control points through differentiable KDE matching.
    """
    
    def __init__(self,
                 kde_bandwidth: float = 0.1,
                 loss_type: str = "kl_div",
                 spatial_weight: float = 1.0):
        super().__init__()
        
        self.kde_bandwidth = kde_bandwidth
        self.loss_type = loss_type
        self.spatial_weight = spatial_weight
        
        if loss_type == "kl_div":
            self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "js_div":
            # Jensen-Shannon divergence (symmetric KL)
            self.loss_fn = self._js_divergence
        else:
            raise ValueError(f"Unknown density loss type: {loss_type}")
    
    def forward(self,
                predicted_density: Tensor,
                target_bezier_points: Tensor,
                mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Compute density matching loss.
        
        Args:
            predicted_density: Predicted density maps (B, H, W)
            target_bezier_points: Target Bézier points (B, N, 2)
            mask: Optional validity mask (B,)
            
        Returns:
            Dictionary with loss components
        """
        batch_size = predicted_density.shape[0]
        
        # Generate target density maps from Bézier points
        target_density = self._generate_target_density(
            target_bezier_points, 
            predicted_density.shape[-2:],
            predicted_density.device
        )
        
        # Apply mask if provided
        if mask is not None:
            predicted_density = predicted_density * mask.view(-1, 1, 1)
            target_density = target_density * mask.view(-1, 1, 1)
        
        # Normalize densities to probability distributions
        pred_normalized = self._normalize_density(predicted_density)
        target_normalized = self._normalize_density(target_density)
        
        # Compute density loss
        if self.loss_type == "kl_div":
            # KL divergence expects log probabilities for first argument
            pred_log = torch.log(pred_normalized + 1e-8)
            loss = self.loss_fn(pred_log, target_normalized)
        else:
            loss = self.loss_fn(pred_normalized, target_normalized)
        
        # Compute additional metrics
        with torch.no_grad():
            # Earth Mover's Distance approximation
            emd_loss = self._compute_emd_loss(pred_normalized, target_normalized)
            
            # Peak density accuracy
            pred_peaks = self._find_density_peaks(pred_normalized)
            target_peaks = self._find_density_peaks(target_normalized)
            peak_accuracy = self._compute_peak_accuracy(pred_peaks, target_peaks)
        
        return {
            "loss": loss,
            "emd_loss": emd_loss,
            "peak_accuracy": peak_accuracy
        }
    
    def _generate_target_density(self, 
                                bezier_points: Tensor, 
                                spatial_size: Tuple[int, int],
                                device: torch.device) -> Tensor:
        """Generate target density maps from Bézier points using KDE."""
        batch_size, num_points, _ = bezier_points.shape
        H, W = spatial_size
        
        # Create spatial grid
        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_points = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        
        density_maps = []
        
        for b in range(batch_size):
            points = bezier_points[b]  # (N, 2)
            
            # Compute KDE
            densities = torch.zeros(H, W, device=device)
            
            for i in range(num_points):
                point = points[i:i+1]  # (1, 2)
                
                # Gaussian kernel
                diff = grid_points - point  # (H, W, 2)
                dist_sq = (diff ** 2).sum(dim=-1)  # (H, W)
                kernel = torch.exp(-dist_sq / (2 * self.kde_bandwidth ** 2))
                
                densities += kernel
            
            density_maps.append(densities)
        
        return torch.stack(density_maps, dim=0)  # (B, H, W)
    
    def _normalize_density(self, density: Tensor) -> Tensor:
        """Normalize density maps to probability distributions."""
        # Flatten spatial dimensions for normalization
        batch_size = density.shape[0]
        density_flat = density.view(batch_size, -1)
        
        # Add small epsilon for numerical stability
        density_flat = density_flat + 1e-8
        
        # Normalize to sum to 1
        density_normalized = density_flat / density_flat.sum(dim=1, keepdim=True)
        
        # Reshape back to spatial dimensions
        return density_normalized.view_as(density)
    
    def _js_divergence(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute Jensen-Shannon divergence."""
        # Average of the two distributions
        m = 0.5 * (pred + target)
        
        # KL divergences
        kl1 = F.kl_div(torch.log(pred + 1e-8), m, reduction='batchmean')
        kl2 = F.kl_div(torch.log(target + 1e-8), m, reduction='batchmean')
        
        # JS divergence
        return 0.5 * (kl1 + kl2)
    
    def _compute_emd_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute approximate Earth Mover's Distance."""
        # Simple approximation using cumulative distributions
        # For 2D, we compute along both dimensions
        
        # Cumulative along height
        pred_cum_h = torch.cumsum(pred, dim=1)
        target_cum_h = torch.cumsum(target, dim=1)
        emd_h = (pred_cum_h - target_cum_h).abs().mean()
        
        # Cumulative along width  
        pred_cum_w = torch.cumsum(pred, dim=2)
        target_cum_w = torch.cumsum(target, dim=2)
        emd_w = (pred_cum_w - target_cum_w).abs().mean()
        
        return (emd_h + emd_w) / 2
    
    def _find_density_peaks(self, density: Tensor) -> Tensor:
        """Find peak locations in density maps."""
        batch_size, H, W = density.shape
        density_flat = density.view(batch_size, -1)
        
        # Find maximum density locations
        max_indices = torch.argmax(density_flat, dim=1)
        
        # Convert to 2D coordinates
        peak_y = max_indices // W
        peak_x = max_indices % W
        
        # Normalize to [0, 1]
        peak_coords = torch.stack([
            peak_x.float() / (W - 1),
            peak_y.float() / (H - 1)
        ], dim=1)
        
        return peak_coords  # (B, 2)
    
    def _compute_peak_accuracy(self, pred_peaks: Tensor, target_peaks: Tensor) -> Tensor:
        """Compute accuracy of peak detection."""
        # Distance between predicted and target peaks
        peak_distances = (pred_peaks - target_peaks).norm(dim=1)
        
        # Accuracy based on distance threshold
        threshold = 0.1  # 10% of image size
        accuracy = (peak_distances < threshold).float().mean()
        
        return accuracy


class StyleLoss(nn.Module):
    """
    Perceptual style loss for style transfer quality.
    
    Uses VGG features to compare style characteristics between generated
    and target style images, ensuring high-quality style transfer.
    """
    
    def __init__(self,
                 style_layers: list[str] = None,
                 content_layers: list[str] = None,
                 style_weight: float = 1.0,
                 content_weight: float = 0.1):
        super().__init__()
        
        if style_layers is None:
            style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        if content_layers is None:
            content_layers = ['conv4_2']
            
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.style_weight = style_weight
        self.content_weight = content_weight
        
        # Load VGG network for feature extraction
        self.vgg = self._load_vgg_network()
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self,
                generated_images: Tensor,
                style_images: Tensor,
                content_images: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Compute perceptual style loss.
        
        Args:
            generated_images: Generated images (B, C, H, W)
            style_images: Style reference images (B, C, H, W)
            content_images: Optional content images (B, C, H, W)
            
        Returns:
            Dictionary with loss components
        """
        # Extract features
        gen_features = self._extract_features(generated_images)
        style_features = self._extract_features(style_images)
        
        # Compute style loss
        style_loss = 0.0
        for layer in self.style_layers:
            gen_gram = self._gram_matrix(gen_features[layer])
            style_gram = self._gram_matrix(style_features[layer])
            style_loss += F.mse_loss(gen_gram, style_gram)
        
        style_loss = style_loss * self.style_weight
        
        # Compute content loss if content images provided
        content_loss = 0.0
        if content_images is not None:
            content_features = self._extract_features(content_images)
            
            for layer in self.content_layers:
                content_loss += F.mse_loss(gen_features[layer], content_features[layer])
            
            content_loss = content_loss * self.content_weight
        
        total_loss = style_loss + content_loss
        
        return {
            "loss": total_loss,
            "style_loss": style_loss,
            "content_loss": content_loss
        }
    
    def _load_vgg_network(self):
        """Load pretrained VGG network for feature extraction."""
        try:
            import torchvision.models as models
            vgg = models.vgg19(pretrained=True).features
            vgg.eval()
            return vgg
        except ImportError:
            # Fallback to simple conv layers if torchvision not available
            return self._create_simple_feature_extractor()
    
    def _create_simple_feature_extractor(self):
        """Create simple feature extractor as fallback."""
        layers = []
        in_channels = 3
        
        # Simple conv layers
        for out_channels in [64, 128, 256, 512]:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _extract_features(self, images: Tensor) -> Dict[str, Tensor]:
        """Extract VGG features from images."""
        features = {}
        x = images
        
        # Normalize for VGG (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        # Extract features at different layers
        layer_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
        
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            
            if i < len(layer_names):
                features[layer_names[i]] = x
        
        return features
    
    def _gram_matrix(self, features: Tensor) -> Tensor:
        """Compute Gram matrix for style representation."""
        b, c, h, w = features.shape
        features = features.view(b, c, h * w)
        
        # Compute Gram matrix
        gram = torch.bmm(features, features.transpose(1, 2))
        
        # Normalize by number of elements
        gram = gram / (c * h * w)
        
        return gram


class InpaintingLoss(nn.Module):
    """
    Specialized loss for inpainting tasks with FLUX.1-Fill-dev.
    
    Adds mask-aware loss computation and region-specific objectives
    for high-quality inpainting and style transfer.
    """
    
    def __init__(self, 
                 mask_weight: float = 2.0,
                 boundary_weight: float = 1.5):
        super().__init__()
        
        self.mask_weight = mask_weight
        self.boundary_weight = boundary_weight
    
    def forward(self,
                model_output: Tensor,
                target: Tensor,
                mask: Optional[Tensor] = None,
                **kwargs) -> Dict[str, Tensor]:
        """
        Compute inpainting-aware loss.
        
        Args:
            model_output: Model predictions (B, C, H, W)
            target: Target values (B, C, H, W)
            mask: Inpainting mask (B, 1, H, W) - 1 for regions to inpaint
            
        Returns:
            Dictionary with loss components
        """
        if mask is None:
            # Fallback to standard MSE if no mask provided
            loss = F.mse_loss(model_output, target)
            return {"loss": loss, "masked_loss": loss, "unmasked_loss": loss}
        
        # Ensure mask is binary and in correct shape
        mask = (mask > 0.5).float()
        if mask.shape[1] != 1:
            mask = mask.mean(dim=1, keepdim=True)
        
        # Compute losses for masked and unmasked regions
        masked_region = mask
        unmasked_region = 1.0 - mask
        
        # Masked region loss (regions to inpaint) - higher weight
        masked_loss = F.mse_loss(
            model_output * masked_region,
            target * masked_region
        ) * self.mask_weight
        
        # Unmasked region loss (preserve existing content) - lower weight
        unmasked_loss = F.mse_loss(
            model_output * unmasked_region,
            target * unmasked_region
        )
        
        # Boundary smoothness loss
        boundary_loss = self._compute_boundary_loss(model_output, target, mask)
        
        # Combined loss
        total_loss = masked_loss + unmasked_loss + boundary_loss * self.boundary_weight
        
        return {
            "loss": total_loss,
            "masked_loss": masked_loss,
            "unmasked_loss": unmasked_loss,
            "boundary_loss": boundary_loss
        }
    
    def _compute_boundary_loss(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Compute smoothness loss at mask boundaries."""
        # Compute gradient of mask to find boundaries
        mask_grad_x = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1])
        mask_grad_y = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :])
        
        # Pad gradients to match original size
        mask_grad_x = F.pad(mask_grad_x, (0, 1), "constant", 0)
        mask_grad_y = F.pad(mask_grad_y, (1, 0), "constant", 0)
        
        # Boundary regions (where mask changes)
        boundary_mask = (mask_grad_x + mask_grad_y).clamp(0, 1)
        
        if boundary_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Compute smoothness at boundaries
        pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        
        target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        
        # Pad gradients
        pred_grad_x = F.pad(pred_grad_x, (0, 1), "constant", 0)
        pred_grad_y = F.pad(pred_grad_y, (1, 0), "constant", 0)
        target_grad_x = F.pad(target_grad_x, (0, 1), "constant", 0)
        target_grad_y = F.pad(target_grad_y, (1, 0), "constant", 0)
        
        # Loss at boundaries
        boundary_loss_x = F.mse_loss(
            (pred_grad_x * boundary_mask),
            (target_grad_x * boundary_mask)
        )
        boundary_loss_y = F.mse_loss(
            (pred_grad_y * boundary_mask),
            (target_grad_y * boundary_mask)
        )
        
        return (boundary_loss_x + boundary_loss_y) / 2


class MultiLossTrainer(nn.Module):
    """
    Coordinated multi-objective optimization for BezierAdapter training.
    
    Combines diffusion, density, and style losses with configurable weights
    and adaptive scheduling for balanced optimization.
    Enhanced for FLUX.1-Fill-dev with inpainting capabilities.
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        
        self.config = config
        self.is_fill_model = getattr(config.model, 'is_fill_model', False)
        
        # Initialize loss functions
        self.diffusion_loss = DiffusionLoss()
        self.density_loss = DensityLoss()
        self.style_loss = StyleLoss()
        
        # Add inpainting loss for Fill model
        if self.is_fill_model:
            self.inpainting_loss = InpaintingLoss()
        
        # Loss weights (can be scheduled)
        self.register_buffer('diffusion_weight', torch.tensor(config.diffusion_loss_weight))
        self.register_buffer('density_weight', torch.tensor(config.density_loss_weight))
        self.register_buffer('style_weight', torch.tensor(config.style_loss_weight))
        
        # Training statistics
        self.step_count = 0
        self.loss_history = {
            'total': [],
            'diffusion': [],
            'density': [],
            'style': []
        }
    
    def forward(self,
                model_outputs: Dict[str, Tensor],
                targets: Dict[str, Tensor],
                **kwargs) -> LossOutputs:
        """
        Compute multi-objective loss.
        
        Args:
            model_outputs: Dictionary with model predictions
            targets: Dictionary with target values
            
        Returns:
            LossOutputs with total loss and components
        """
        losses = {}
        metrics = {}
        
        # Compute diffusion loss
        if 'diffusion_output' in model_outputs and 'diffusion_target' in targets:
            diffusion_results = self.diffusion_loss(
                model_outputs['diffusion_output'],
                targets['diffusion_target'],
                kwargs.get('timesteps'),
                kwargs.get('mask')
            )
            losses['diffusion'] = diffusion_results['loss']
            metrics.update({f'diffusion_{k}': v for k, v in diffusion_results.items()})
        
        # Compute density loss
        if 'density_output' in model_outputs and 'bezier_points' in targets:
            density_results = self.density_loss(
                model_outputs['density_output'],
                targets['bezier_points'],
                kwargs.get('density_mask')
            )
            losses['density'] = density_results['loss']
            metrics.update({f'density_{k}': v for k, v in density_results.items()})
        
        # Compute style loss
        if 'generated_images' in model_outputs and 'style_images' in targets:
            style_results = self.style_loss(
                model_outputs['generated_images'],
                targets['style_images'],
                targets.get('content_images')
            )
            losses['style'] = style_results['loss']
            metrics.update({f'style_{k}': v for k, v in style_results.items()})
        
        # Compute inpainting loss for Fill model
        if self.is_fill_model and 'diffusion_output' in model_outputs and 'diffusion_target' in targets:
            inpainting_results = self.inpainting_loss(
                model_outputs['diffusion_output'],
                targets['diffusion_target'],
                targets.get('inpainting_mask'),
                **kwargs
            )
            # Enhance diffusion loss with inpainting-specific components
            if 'diffusion' in losses:
                losses['diffusion'] = losses['diffusion'] + inpainting_results['loss'] * 0.5
            else:
                losses['diffusion'] = inpainting_results['loss']
            
            metrics.update({f'inpainting_{k}': v for k, v in inpainting_results.items()})
        
        # Combine losses with weights
        total_loss = 0.0
        weighted_losses = {}
        
        if 'diffusion' in losses:
            weighted_losses['diffusion'] = self.diffusion_weight * losses['diffusion']
            total_loss += weighted_losses['diffusion']
        
        if 'density' in losses:
            weighted_losses['density'] = self.density_weight * losses['density']
            total_loss += weighted_losses['density']
        
        if 'style' in losses:
            weighted_losses['style'] = self.style_weight * losses['style']
            total_loss += weighted_losses['style']
        
        # Update statistics
        self.step_count += 1
        self._update_loss_history(total_loss, weighted_losses)
        
        return LossOutputs(
            total_loss=total_loss,
            diffusion_loss=weighted_losses.get('diffusion'),
            density_loss=weighted_losses.get('density'),
            style_loss=weighted_losses.get('style'),
            metrics=metrics
        )
    
    def update_loss_weights(self, step: int):
        """Update loss weights based on training schedule."""
        # Adaptive weighting based on training progress
        if hasattr(self.config, 'loss_weight_schedule'):
            schedule = self.config.loss_weight_schedule
            
            if 'diffusion' in schedule:
                self.diffusion_weight = torch.tensor(self._get_scheduled_weight(
                    schedule['diffusion'], step
                ))
            
            if 'density' in schedule:
                self.density_weight = torch.tensor(self._get_scheduled_weight(
                    schedule['density'], step
                ))
            
            if 'style' in schedule:
                self.style_weight = torch.tensor(self._get_scheduled_weight(
                    schedule['style'], step
                ))
    
    def _get_scheduled_weight(self, schedule: Dict[str, Any], step: int) -> float:
        """Get weight value according to schedule."""
        if isinstance(schedule, dict):
            if 'type' in schedule:
                if schedule['type'] == 'linear':
                    start_weight = schedule['start']
                    end_weight = schedule['end']
                    start_step = schedule.get('start_step', 0)
                    end_step = schedule.get('end_step', self.config.total_steps)
                    
                    if step <= start_step:
                        return start_weight
                    elif step >= end_step:
                        return end_weight
                    else:
                        progress = (step - start_step) / (end_step - start_step)
                        return start_weight + progress * (end_weight - start_weight)
                
                elif schedule['type'] == 'exponential':
                    base_weight = schedule['base']
                    decay_rate = schedule.get('decay', 0.99)
                    return base_weight * (decay_rate ** step)
        
        return float(schedule)  # Constant weight
    
    def _update_loss_history(self, total_loss: Tensor, component_losses: Dict[str, Tensor]):
        """Update loss history for monitoring."""
        self.loss_history['total'].append(total_loss.item())
        
        for component, loss in component_losses.items():
            if component in self.loss_history:
                self.loss_history[component].append(loss.item())
        
        # Keep only recent history to avoid memory growth
        max_history = 1000
        for key in self.loss_history:
            if len(self.loss_history[key]) > max_history:
                self.loss_history[key] = self.loss_history[key][-max_history:]
    
    def get_loss_statistics(self) -> Dict[str, float]:
        """Get current loss statistics."""
        stats = {}
        
        for component, history in self.loss_history.items():
            if history:
                stats[f'{component}_mean'] = np.mean(history[-100:])  # Last 100 steps
                stats[f'{component}_std'] = np.std(history[-100:])
                stats[f'{component}_min'] = np.min(history)
                stats[f'{component}_max'] = np.max(history)
        
        return stats