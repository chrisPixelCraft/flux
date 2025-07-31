#!/usr/bin/env python3
"""
Unit tests for BezierParameterProcessor module.

Tests KDE density calculation, point embedding MLP, gradient checkpointing,
and various edge cases for Bézier curve processing.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add flux modules to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from flux.modules.bezier_processor import BezierParameterProcessor
from flux.modules.models import BezierControlPoints


class TestBezierParameterProcessor:
    """Test suite for BezierParameterProcessor."""
    
    @pytest.fixture
    def device(self):
        """Get test device (CUDA if available)."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def processor(self, device):
        """Create a BezierParameterProcessor instance."""
        return BezierParameterProcessor(
            output_resolution=(32, 32),  # Smaller for faster tests
            hidden_dim=64,  # Smaller for faster tests
            sigma_min=0.01,
            sigma_max=1.0,
            initial_bandwidth=0.1
        ).to(device)
    
    def test_initialization(self, processor):
        """Test proper initialization of the processor."""
        assert processor.output_resolution == (32, 32)
        assert processor.hidden_dim == 64
        assert processor.sigma_min == 0.01
        assert processor.sigma_max == 1.0
        
        # Check MLP architecture
        assert isinstance(processor.point_embedding_mlp, nn.Sequential)
        assert len(processor.point_embedding_mlp) == 6  # 3 Linear + 3 ReLU layers
        
        # Check learnable parameters
        assert hasattr(processor, 'log_bandwidth')
        assert processor.log_bandwidth.requires_grad
        
        # Check parameter count
        total_params = sum(p.numel() for p in processor.parameters())
        assert total_params > 0, "Processor should have trainable parameters"
    
    def test_forward_basic(self, processor, device):
        """Test basic forward pass with simple input."""
        batch_size = 2
        num_points = 4  # Cubic Bézier curve
        
        # Create test Bézier points
        bezier_points = torch.rand(batch_size, num_points, 2, device=device)
        
        # Forward pass
        density_map, field_map = processor(bezier_points)
        
        # Check output shapes
        assert density_map.shape == (batch_size, 32, 32)
        assert field_map.shape == (batch_size, 32, 32, 2)
        
        # Check output ranges
        assert torch.all(density_map >= 0) and torch.all(density_map <= 1)
        assert torch.all(torch.isfinite(density_map))
        assert torch.all(torch.isfinite(field_map))
    
    def test_kde_density_calculation(self, processor, device):
        """Test KDE density calculation correctness."""
        batch_size = 1
        
        # Create points concentrated in center
        center_points = torch.tensor([
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
        ], device=device)
        
        density_map, _ = processor(center_points)
        
        # Density should be highest at center
        center_idx = 16  # For 32x32 grid
        center_density = density_map[0, center_idx, center_idx]
        corner_density = density_map[0, 0, 0]
        
        assert center_density > corner_density, "Center should have higher density"
        
        # Test with spread out points
        spread_points = torch.tensor([
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ], device=device)
        
        density_map_spread, _ = processor(spread_points)
        
        # Density should be more uniform
        density_std = torch.std(density_map_spread)
        center_std = torch.std(density_map)
        
        assert density_std < center_std, "Spread points should have more uniform density"
    
    def test_learnable_bandwidth(self, processor, device):
        """Test that bandwidth parameter is learnable."""
        # Create test data
        bezier_points = torch.rand(2, 4, 2, device=device, requires_grad=True)
        
        # Get initial bandwidth
        initial_bandwidth = processor._get_bandwidth().item()
        
        # Forward pass
        density_map, _ = processor(bezier_points)
        
        # Compute loss and backward
        loss = density_map.mean()
        loss.backward()
        
        # Check gradient exists
        assert processor.log_bandwidth.grad is not None
        assert not torch.allclose(processor.log_bandwidth.grad, torch.zeros_like(processor.log_bandwidth.grad))
        
        # Update bandwidth
        with torch.no_grad():
            processor.log_bandwidth -= 0.1 * processor.log_bandwidth.grad
        
        # Check bandwidth changed
        new_bandwidth = processor._get_bandwidth().item()
        assert new_bandwidth != initial_bandwidth
    
    def test_gradient_checkpointing(self, processor, device):
        """Test gradient checkpointing for memory efficiency."""
        # Large number of points to trigger checkpointing
        batch_size = 1
        num_points = 200  # > 100 threshold
        
        bezier_points = torch.rand(batch_size, num_points, 2, device=device, requires_grad=True)
        
        # Forward pass (should use checkpointing)
        density_map, field_map = processor(bezier_points)
        
        # Backward pass
        loss = density_map.mean() + field_map.mean()
        loss.backward()
        
        # Check gradients exist
        assert bezier_points.grad is not None
        assert torch.all(torch.isfinite(bezier_points.grad))
    
    def test_edge_cases(self, processor, device):
        """Test edge cases and error handling."""
        # Test with single point
        single_point = torch.rand(1, 1, 2, device=device)
        density_single, field_single = processor(single_point)
        assert density_single.shape == (1, 32, 32)
        assert torch.all(torch.isfinite(density_single))
        
        # Test with many points
        many_points = torch.rand(1, 1000, 2, device=device)
        density_many, field_many = processor(many_points)
        assert density_many.shape == (1, 32, 32)
        assert torch.all(torch.isfinite(density_many))
        
        # Test with points outside [0, 1] range
        out_of_range = torch.tensor([
            [[-0.5, -0.5], [1.5, 1.5], [0.5, 0.5], [0.5, 0.5]]
        ], device=device)
        density_oor, _ = processor(out_of_range)
        assert torch.all(torch.isfinite(density_oor))
    
    def test_mask_application(self, processor, device):
        """Test mask application functionality."""
        batch_size = 2
        bezier_points = torch.rand(batch_size, 4, 2, device=device)
        
        # Create mask (second sample invalid)
        mask = torch.tensor([1.0, 0.0], device=device)
        
        # Forward pass with mask
        density_map, field_map = processor(bezier_points, mask=mask)
        
        # Second sample should have zero density
        assert torch.allclose(density_map[1], torch.zeros_like(density_map[1]))
        assert torch.allclose(field_map[1], torch.zeros_like(field_map[1]))
        
        # First sample should be non-zero
        assert not torch.allclose(density_map[0], torch.zeros_like(density_map[0]))
    
    def test_output_resolution_scaling(self, device):
        """Test different output resolutions."""
        resolutions = [(16, 16), (32, 32), (64, 64), (128, 128)]
        
        for resolution in resolutions:
            processor = BezierParameterProcessor(
                output_resolution=resolution,
                hidden_dim=32
            ).to(device)
            
            bezier_points = torch.rand(1, 4, 2, device=device)
            density_map, field_map = processor(bezier_points)
            
            assert density_map.shape == (1, resolution[0], resolution[1])
            assert field_map.shape == (1, resolution[0], resolution[1], 2)
    
    def test_batch_processing(self, processor, device):
        """Test batch processing efficiency."""
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            bezier_points = torch.rand(batch_size, 4, 2, device=device)
            density_map, field_map = processor(bezier_points)
            
            assert density_map.shape[0] == batch_size
            assert field_map.shape[0] == batch_size
            
            # Check each sample is different
            if batch_size > 1:
                for i in range(1, batch_size):
                    assert not torch.allclose(density_map[0], density_map[i])
    
    def test_gradient_flow(self, processor, device):
        """Test gradient flow through the entire module."""
        bezier_points = torch.rand(2, 4, 2, device=device, requires_grad=True)
        
        # Forward pass
        density_map, field_map = processor(bezier_points)
        
        # Create target and loss
        target_density = torch.rand_like(density_map)
        target_field = torch.rand_like(field_map)
        
        loss = nn.MSELoss()(density_map, target_density) + nn.MSELoss()(field_map, target_field)
        loss.backward()
        
        # Check gradients
        assert bezier_points.grad is not None
        assert torch.all(torch.isfinite(bezier_points.grad))
        assert not torch.allclose(bezier_points.grad, torch.zeros_like(bezier_points.grad))
        
        # Check MLP gradients
        for param in processor.point_embedding_mlp.parameters():
            if param.grad is not None:
                assert torch.all(torch.isfinite(param.grad))


@pytest.mark.benchmark
class TestBezierProcessorPerformance:
    """Performance benchmarks for BezierParameterProcessor."""
    
    def test_inference_speed(self, benchmark):
        """Benchmark inference speed."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        processor = BezierParameterProcessor().to(device)
        processor.eval()
        
        bezier_points = torch.rand(4, 4, 2, device=device)
        
        def inference():
            with torch.no_grad():
                return processor(bezier_points)
        
        # Run benchmark
        result = benchmark(inference)
    
    def test_memory_usage(self):
        """Test memory usage with different batch sizes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory testing")
        
        device = torch.device('cuda')
        processor = BezierParameterProcessor().to(device)
        
        # Test increasing batch sizes
        for batch_size in [1, 4, 8, 16, 32]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            bezier_points = torch.rand(batch_size, 4, 2, device=device)
            
            # Measure memory
            start_memory = torch.cuda.memory_allocated()
            density_map, field_map = processor(bezier_points)
            peak_memory = torch.cuda.max_memory_allocated()
            
            memory_mb = (peak_memory - start_memory) / 1024 / 1024
            print(f"Batch {batch_size}: {memory_mb:.2f} MB")
            
            # Ensure reasonable memory usage
            assert memory_mb < 100 * batch_size  # Less than 100MB per sample


if __name__ == "__main__":
    pytest.main([__file__, "-v"])