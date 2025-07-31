"""
BezierParameterProcessor for FLUX BezierAdapter integration.

This module processes Bézier curve control points through:
1. Point embedding MLP
2. Differentiable KDE calculation with learnable bandwidth
3. Density mapping with sigmoid normalization
4. Spatial interpolation for target resolution scaling
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple
import math

from sklearn.neighbors import KernelDensity
import numpy as np


class BezierParameterProcessor(nn.Module):
    """
    Processes Bézier control points to generate density maps and control fields.
    
    This module converts Bézier curve control points into:
    1. Density maps representing curve density at each spatial location
    2. Control fields encoding spatial features for attention modulation
    
    Args:
        output_resolution: Target spatial resolution (H, W) matching FLUX latent space (64, 64)
        hidden_dim: Hidden dimension for point embedding MLP (default: 128)
        sigma_min: Minimum density value for normalization (default: 0.01)
        sigma_max: Maximum density value for normalization (default: 1.0)
        initial_bandwidth: Initial value for learnable KDE bandwidth (default: 0.1)
    """
    
    def __init__(
        self, 
        output_resolution: Tuple[int, int] = (64, 64),
        hidden_dim: int = 128,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        initial_bandwidth: float = 0.1
    ):
        super().__init__()
        
        self.output_resolution = output_resolution
        self.hidden_dim = hidden_dim
        
        # Point embedding MLP: (2) → 64 → 128 → 128
        # Follows FLUX MLPEmbedder pattern but with ReLU activation
        self.point_embedder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128), 
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        
        # Learnable KDE bandwidth parameter - critical for differentiable density calculation
        self.kde_bandwidth = nn.Parameter(torch.tensor(initial_bandwidth))
        
        # Density normalization parameters as buffers (not trainable)
        self.register_buffer('sigma_min', torch.tensor(sigma_min))
        self.register_buffer('sigma_max', torch.tensor(sigma_max))
        
        # Create spatial grid for density computation (computed once, cached)
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(0, 1, output_resolution[0]),
            torch.linspace(0, 1, output_resolution[1]),
            indexing='ij'
        )
        # Register as buffer to automatically move to correct device
        self.register_buffer('grid_points', torch.stack([x_coords.flatten(), y_coords.flatten()], dim=-1))
        
    def _compute_kde_density(self, bezier_points: Tensor, grid_points: Tensor, bandwidth: Tensor) -> Tensor:
        """
        Compute differentiable KDE density using Gaussian kernels.
        
        Args:
            bezier_points: Control points (batch_size, num_points, 2)
            grid_points: Spatial grid points (H*W, 2) 
            bandwidth: Learnable bandwidth parameter (scalar)
        
        Returns:
            density_map: Density values at grid points (batch_size, H*W)
        """
        batch_size, num_points, _ = bezier_points.shape
        grid_size = grid_points.shape[0]
        
        # Ensure bandwidth is positive
        bandwidth = torch.clamp(bandwidth, min=1e-6)
        
        # Expand dimensions for broadcasting
        # bezier_points: (batch_size, num_points, 1, 2)
        # grid_points: (1, 1, grid_size, 2)
        bezier_expanded = bezier_points.unsqueeze(2)  # (batch_size, num_points, 1, 2)
        grid_expanded = grid_points.unsqueeze(0).unsqueeze(0)  # (1, 1, grid_size, 2)
        
        # Compute squared distances: (batch_size, num_points, grid_size)
        distances_sq = torch.sum((bezier_expanded - grid_expanded) ** 2, dim=-1)
        
        # Gaussian kernel: exp(-0.5 * d^2 / bandwidth^2)
        gaussians = torch.exp(-0.5 * distances_sq / (bandwidth ** 2))
        
        # Sum over all control points and normalize
        # Shape: (batch_size, grid_size)
        density = torch.sum(gaussians, dim=1) / (num_points * bandwidth * math.sqrt(2 * math.pi))
        
        return density
        
    def _generate_control_field(self, embedded_points: Tensor, grid_points: Tensor) -> Tensor:
        """
        Generate spatial control field from embedded points.
        
        Uses attention-like mechanism to create spatial feature maps.
        
        Args:
            embedded_points: Point embeddings (batch_size, num_points, hidden_dim)
            grid_points: Spatial grid points (H*W, 2)
            
        Returns:
            field_map: Spatial control features (batch_size, hidden_dim, H*W)
        """
        batch_size, num_points, hidden_dim = embedded_points.shape
        grid_size = grid_points.shape[0]
        
        # Simple spatial interpolation using distance-based weighting
        # This creates a field where each grid point is influenced by nearby control points
        
        # Compute spatial attention weights based on distance
        # This is a simplified version - could be enhanced with learned attention
        
        # For now, use mean pooling across points as a baseline
        # Shape: (batch_size, hidden_dim)
        mean_embedding = torch.mean(embedded_points, dim=1)
        
        # Broadcast to all grid points
        # Shape: (batch_size, hidden_dim, grid_size)
        field_map = mean_embedding.unsqueeze(-1).expand(-1, -1, grid_size)
        
        return field_map
        
    def forward(self, bezier_points: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of BezierParameterProcessor.
        
        Args:
            bezier_points: Bézier control points (batch_size, num_points, 2)
                          Coordinates should be normalized to [0, 1] range
            mask: Optional mask for valid points (batch_size, num_points)
            
        Returns:
            density_map: Density values (batch_size, H, W)
            field_map: Control field features (batch_size, hidden_dim, H, W)
        """
        batch_size = bezier_points.shape[0]
        
        # Validate input dimensions
        if bezier_points.dim() != 3 or bezier_points.shape[-1] != 2:
            raise ValueError(f"Expected bezier_points shape (batch, num_points, 2), got {bezier_points.shape}")
        
        # Apply mask if provided
        if mask is not None:
            # Zero out masked points
            bezier_points = bezier_points * mask.unsqueeze(-1)
        
        # 1. Embed control points through MLP
        embedded_points = self.point_embedder(bezier_points)  # (batch, num_points, hidden_dim)
        
        # 2. Compute KDE density using differentiable implementation
        # Use gradient checkpointing for memory efficiency with large point clouds
        if bezier_points.shape[1] > 100:  # Large point cloud threshold
            density_map = checkpoint(
                self._compute_kde_density,
                bezier_points, 
                self.grid_points, 
                self.kde_bandwidth,
                use_reentrant=False
            )
        else:
            density_map = self._compute_kde_density(bezier_points, self.grid_points, self.kde_bandwidth)
        
        # Reshape to spatial dimensions
        density_map = density_map.view(batch_size, *self.output_resolution)
        
        # 3. Normalize density with learnable parameters
        density_map = torch.sigmoid(density_map)
        density_map = self.sigma_min + (self.sigma_max - self.sigma_min) * density_map
        
        # 4. Generate control field from embedded points
        if embedded_points.shape[1] > 100:  # Large point cloud threshold
            field_map = checkpoint(
                self._generate_control_field,
                embedded_points,
                self.grid_points,
                use_reentrant=False
            )
        else:
            field_map = self._generate_control_field(embedded_points, self.grid_points)
        
        # Reshape field map to spatial dimensions
        field_map = field_map.view(batch_size, self.hidden_dim, *self.output_resolution)
        
        return density_map, field_map
        
    def extra_repr(self) -> str:
        """String representation of module parameters."""
        return (
            f"output_resolution={self.output_resolution}, "
            f"hidden_dim={self.hidden_dim}, "
            f"kde_bandwidth={self.kde_bandwidth.item():.4f}"
        )