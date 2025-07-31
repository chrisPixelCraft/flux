#!/usr/bin/env python3
"""
Unit tests for SpatialAttentionFuser module.

Tests density-modulated attention, FLUX RoPE compatibility, dynamic sequence handling,
and gradient checkpointing for the spatial attention fusion mechanism.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# Add flux modules to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from flux.modules.spatial_fuser import SpatialAttentionFuser, DensityModulatedAttention, TransformerLayer
from flux.modules.models import MultiModalCondition


class TestDensityModulatedAttention:
    """Test suite for DensityModulatedAttention component."""
    
    @pytest.fixture
    def device(self):
        """Get test device (CUDA if available)."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def attention(self, device):
        """Create a DensityModulatedAttention instance."""
        return DensityModulatedAttention(
            dim=768,
            num_heads=12,
            qkv_bias=False
        ).to(device)
    
    def test_attention_initialization(self, attention):
        """Test proper initialization of attention module."""
        assert attention.num_heads == 12
        assert attention.head_dim == 64  # 768 / 12
        assert attention.scale == 64 ** -0.5
        
        # Check learnable density parameters
        assert hasattr(attention, 'density_scale')
        assert hasattr(attention, 'density_bias')
        assert attention.density_scale.requires_grad
        assert attention.density_bias.requires_grad
    
    def test_attention_forward(self, attention, device):
        """Test attention forward pass."""
        batch_size = 2
        seq_len = 256
        dim = 768
        
        x = torch.randn(batch_size, seq_len, dim, device=device)
        density_weights = torch.rand(batch_size, 16, 16, device=device)  # 16x16 = 256
        
        output = attention(x, density_weights)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert torch.all(torch.isfinite(output))
    
    def test_density_modulation(self, attention, device):
        """Test that density weights actually modulate attention."""
        batch_size = 1
        seq_len = 64
        dim = 768
        
        x = torch.randn(batch_size, seq_len, dim, device=device)
        
        # High density in center
        density_high = torch.zeros(batch_size, 8, 8, device=device)
        density_high[:, 3:5, 3:5] = 1.0
        
        # Uniform density
        density_uniform = torch.ones(batch_size, 8, 8, device=device) * 0.5
        
        output_high = attention(x, density_high)
        output_uniform = attention(x, density_uniform)
        
        # Outputs should be different
        assert not torch.allclose(output_high, output_uniform)
    
    def test_density_resizing(self, attention, device):
        """Test automatic density resizing for mismatched dimensions."""
        batch_size = 2
        seq_len = 256  # 16x16
        dim = 768
        
        x = torch.randn(batch_size, seq_len, dim, device=device)
        
        # Density with different resolution (64x64)
        density_weights = torch.rand(batch_size, 64, 64, device=device)
        
        # Should resize automatically
        output = attention(x, density_weights)
        assert output.shape == (batch_size, seq_len, dim)


class TestSpatialAttentionFuser:
    """Test suite for SpatialAttentionFuser."""
    
    @pytest.fixture
    def device(self):
        """Get test device (CUDA if available)."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def fuser(self, device):
        """Create a SpatialAttentionFuser instance."""
        return SpatialAttentionFuser(
            feature_dim=768,
            num_heads=12,
            num_layers=6,
            mlp_ratio=4.0,
            theta=10000
        ).to(device)
    
    def test_initialization(self, fuser):
        """Test proper initialization of the fuser."""
        assert fuser.feature_dim == 768
        assert fuser.num_heads == 12
        assert fuser.num_layers == 6
        assert fuser.theta == 10000
        
        # Check transformer layers
        assert len(fuser.transformer_layers) == 6
        for layer in fuser.transformer_layers:
            assert isinstance(layer, TransformerLayer)
        
        # Check position embedding setup
        assert hasattr(fuser, 'pe_embedder')
        pe_dim = 768 // 12  # 64
        assert fuser.pe_embedder.dim == pe_dim
    
    def test_rope_compatibility(self, fuser):
        """Test FLUX RoPE compatibility with even dimensions."""
        pe_dim = fuser.feature_dim // fuser.num_heads
        axes_dim = fuser.pe_embedder.axes_dim
        
        # Both axes dimensions must be even for RoPE
        assert all(dim % 2 == 0 for dim in axes_dim)
        assert sum(axes_dim) == pe_dim
    
    def test_forward_basic(self, fuser, device):
        """Test basic forward pass."""
        batch_size = 2
        seq_len = 256  # 16x16
        feature_dim = 768
        
        spatial_features = torch.randn(batch_size, seq_len, feature_dim, device=device)
        density_weights = torch.rand(batch_size, 16, 16, device=device)
        condition_embeddings = torch.randn(batch_size, 1536, device=device)
        
        fused_features, attention_maps = fuser(
            spatial_features=spatial_features,
            density_weights=density_weights,
            condition_embeddings=condition_embeddings
        )
        
        assert fused_features.shape == (batch_size, seq_len, feature_dim)
        assert attention_maps is None  # Not implemented in current version
        assert torch.all(torch.isfinite(fused_features))
    
    def test_dynamic_sequence_length(self, fuser, device):
        """Test handling of different sequence lengths."""
        batch_size = 2
        feature_dim = 768
        
        # Different sequence lengths (must be square numbers for spatial)
        seq_lengths = [64, 256, 1024, 4096]  # 8x8, 16x16, 32x32, 64x64
        
        for seq_len in seq_lengths:
            spatial_dim = int(seq_len ** 0.5)
            
            spatial_features = torch.randn(batch_size, seq_len, feature_dim, device=device)
            density_weights = torch.rand(batch_size, spatial_dim, spatial_dim, device=device)
            condition_embeddings = torch.randn(batch_size, 1536, device=device)
            
            fused_features, _ = fuser(
                spatial_features=spatial_features,
                density_weights=density_weights,
                condition_embeddings=condition_embeddings
            )
            
            assert fused_features.shape == (batch_size, seq_len, feature_dim)
    
    def test_position_embeddings(self, fuser, device):
        """Test position embedding generation."""
        batch_size = 2
        seq_len = 256
        
        # Get position embeddings
        pe = fuser._get_position_embeddings(batch_size, seq_len, device)
        
        # Check shape: (B, 1, seq_len, pe_dim, 2, 2)
        pe_dim = fuser.feature_dim // fuser.num_heads
        expected_shape = (batch_size, 1, seq_len, pe_dim // 2, 2, 2)
        assert pe.shape == expected_shape
        
        # Check different positions have different embeddings
        assert not torch.allclose(pe[:, :, 0], pe[:, :, 1])
    
    def test_condition_injection(self, fuser, device):
        """Test condition embedding injection."""
        batch_size = 2
        seq_len = 256
        feature_dim = 768
        
        spatial_features = torch.randn(batch_size, seq_len, feature_dim, device=device)
        density_weights = torch.rand(batch_size, 16, 16, device=device)
        
        # Different condition embeddings
        condition_1 = torch.ones(batch_size, 1536, device=device)
        condition_2 = torch.zeros(batch_size, 1536, device=device)
        
        output_1, _ = fuser(spatial_features, density_weights, condition_1)
        output_2, _ = fuser(spatial_features, density_weights, condition_2)
        
        # Different conditions should produce different outputs
        assert not torch.allclose(output_1, output_2)
    
    def test_gradient_checkpointing(self, fuser, device):
        """Test gradient checkpointing for large sequences."""
        batch_size = 1
        seq_len = 3000  # > 2048 threshold
        feature_dim = 768
        
        # Adjust density for non-square sequence
        density_h = density_w = int(seq_len ** 0.5) + 1
        
        spatial_features = torch.randn(batch_size, seq_len, feature_dim, device=device, requires_grad=True)
        density_weights = torch.rand(batch_size, density_h, density_w, device=device)
        condition_embeddings = torch.randn(batch_size, 1536, device=device)
        
        # Forward pass (should use checkpointing)
        fused_features, _ = fuser(
            spatial_features=spatial_features,
            density_weights=density_weights,
            condition_embeddings=condition_embeddings
        )
        
        # Backward pass
        loss = fused_features.mean()
        loss.backward()
        
        # Check gradients exist
        assert spatial_features.grad is not None
        assert torch.all(torch.isfinite(spatial_features.grad))
    
    def test_transformer_layers(self, fuser, device):
        """Test individual transformer layers."""
        batch_size = 2
        seq_len = 64
        dim = 768
        
        x = torch.randn(batch_size, seq_len, dim, device=device)
        density_weights = torch.rand(batch_size, 8, 8, device=device)
        pe = fuser._get_position_embeddings(batch_size, seq_len, device)
        
        # Test each layer
        for i, layer in enumerate(fuser.transformer_layers):
            output = layer(x, density_weights, pe)
            assert output.shape == x.shape
            assert torch.all(torch.isfinite(output))
            
            # Residual connections should prevent identical input/output
            assert not torch.allclose(x, output)
            x = output  # Use for next layer
    
    def test_edge_cases(self, fuser, device):
        """Test edge cases and error handling."""
        batch_size = 2
        feature_dim = 768
        
        # Test with mismatched dimensions
        spatial_features = torch.randn(batch_size, 256, 512, device=device)  # Wrong feature dim
        density_weights = torch.rand(batch_size, 16, 16, device=device)
        condition_embeddings = torch.randn(batch_size, 1536, device=device)
        
        with pytest.raises(ValueError, match="Expected feature_dim"):
            fuser(spatial_features, density_weights, condition_embeddings)
        
        # Test with 1D sequence (non-square)
        seq_len = 100  # Not a perfect square
        spatial_features = torch.randn(batch_size, seq_len, feature_dim, device=device)
        density_weights = torch.rand(batch_size, 10, 10, device=device)
        
        # Should handle gracefully
        output, _ = fuser(spatial_features, density_weights, condition_embeddings)
        assert output.shape == (batch_size, seq_len, feature_dim)
    
    def test_output_projection(self, fuser, device):
        """Test output projection functionality."""
        batch_size = 2
        seq_len = 256
        feature_dim = 768
        
        # Check output projection exists and has correct dimensions
        assert hasattr(fuser, 'output_proj')
        assert isinstance(fuser.output_proj, nn.Linear)
        assert fuser.output_proj.in_features == feature_dim
        assert fuser.output_proj.out_features == feature_dim
        
        # Test projection preserves dimensions
        x = torch.randn(batch_size, seq_len, feature_dim, device=device)
        projected = fuser.output_proj(x)
        assert projected.shape == x.shape
    
    def test_memory_efficiency(self, fuser, device):
        """Test memory usage with different sequence lengths."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory testing")
        
        feature_dim = 768
        
        for seq_len in [256, 1024, 4096]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            spatial_dim = int(seq_len ** 0.5)
            spatial_features = torch.randn(1, seq_len, feature_dim, device=device)
            density_weights = torch.rand(1, spatial_dim, spatial_dim, device=device)
            condition_embeddings = torch.randn(1, 1536, device=device)
            
            start_memory = torch.cuda.memory_allocated()
            output, _ = fuser(spatial_features, density_weights, condition_embeddings)
            peak_memory = torch.cuda.max_memory_allocated()
            
            memory_mb = (peak_memory - start_memory) / 1024 / 1024
            print(f"Sequence {seq_len}: {memory_mb:.2f} MB")
            
            # Memory should scale reasonably with sequence length
            assert memory_mb < seq_len * 0.1  # Less than 0.1MB per position


@pytest.mark.parametrize("num_heads,num_layers", [
    (8, 4),
    (12, 6),
    (16, 8),
])
def test_different_architectures(num_heads, num_layers, device):
    """Test fuser with different architectural configurations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_dim = 768  # Must be divisible by num_heads
    
    if feature_dim % num_heads != 0:
        pytest.skip(f"Feature dim {feature_dim} not divisible by {num_heads}")
    
    fuser = SpatialAttentionFuser(
        feature_dim=feature_dim,
        num_heads=num_heads,
        num_layers=num_layers
    ).to(device)
    
    # Test forward pass
    spatial_features = torch.randn(1, 256, feature_dim, device=device)
    density_weights = torch.rand(1, 16, 16, device=device)
    condition_embeddings = torch.randn(1, 1536, device=device)
    
    output, _ = fuser(spatial_features, density_weights, condition_embeddings)
    assert output.shape == spatial_features.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])