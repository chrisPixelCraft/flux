#!/usr/bin/env python3
"""
Unit tests for StyleBezierFusionModule.

Tests AdaIN style transfer, density-aware projection, cross-attention fusion,
and various style transfer configurations.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# Add flux modules to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from flux.modules.style_fusion import (
    StyleBezierFusionModule, 
    AdaINLayer, 
    DensityAwareStyleProjector
)


class TestAdaINLayer:
    """Test suite for AdaINLayer component."""
    
    @pytest.fixture
    def device(self):
        """Get test device (CUDA if available)."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def adain(self, device):
        """Create an AdaINLayer instance."""
        return AdaINLayer(num_features=768, eps=1e-5).to(device)
    
    def test_adain_initialization(self, adain):
        """Test proper initialization of AdaIN layer."""
        assert adain.num_features == 768
        assert adain.eps == 1e-5
    
    def test_adain_sequential_format(self, adain, device):
        """Test AdaIN with sequential format (B, L, C)."""
        batch_size = 2
        seq_len = 100
        feature_dim = 768
        
        content_features = torch.randn(batch_size, seq_len, feature_dim, device=device)
        style_mean = torch.randn(batch_size, 1, feature_dim, device=device)
        style_std = torch.abs(torch.randn(batch_size, 1, feature_dim, device=device)) + 0.1
        
        stylized = adain(content_features, style_mean, style_std)
        
        assert stylized.shape == content_features.shape
        assert torch.all(torch.isfinite(stylized))
        
        # Check normalization worked
        # Mean should be close to style_mean
        content_mean = stylized.mean(dim=1, keepdim=True)
        assert torch.allclose(content_mean, style_mean, atol=0.1)
    
    def test_adain_spatial_format(self, adain, device):
        """Test AdaIN with spatial format (B, C, H, W)."""
        batch_size = 2
        channels = 768
        height = width = 16
        
        content_features = torch.randn(batch_size, channels, height, width, device=device)
        style_mean = torch.randn(batch_size, channels, device=device)
        style_std = torch.abs(torch.randn(batch_size, channels, device=device)) + 0.1
        
        stylized = adain(content_features, style_mean, style_std)
        
        assert stylized.shape == content_features.shape
        assert torch.all(torch.isfinite(stylized))
    
    def test_adain_style_transfer_property(self, adain, device):
        """Test that AdaIN actually transfers style statistics."""
        batch_size = 2
        seq_len = 100
        feature_dim = 768
        
        # Create content with specific statistics
        content_features = torch.randn(batch_size, seq_len, feature_dim, device=device) * 2 + 1
        
        # Target style statistics
        target_mean = torch.zeros(batch_size, 1, feature_dim, device=device)
        target_std = torch.ones(batch_size, 1, feature_dim, device=device) * 0.5
        
        stylized = adain(content_features, target_mean, target_std)
        
        # Check transferred statistics
        actual_mean = stylized.mean(dim=1, keepdim=True)
        actual_std = stylized.std(dim=1, keepdim=True)
        
        assert torch.allclose(actual_mean, target_mean, atol=0.1)
        assert torch.allclose(actual_std, target_std, atol=0.1)


class TestDensityAwareStyleProjector:
    """Test suite for DensityAwareStyleProjector component."""
    
    @pytest.fixture
    def device(self):
        """Get test device (CUDA if available)."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def projector(self, device):
        """Create a DensityAwareStyleProjector instance."""
        return DensityAwareStyleProjector(
            style_dim=1536,
            spatial_dim=768,
            hidden_dim=512
        ).to(device)
    
    def test_projector_initialization(self, projector):
        """Test proper initialization of projector."""
        assert projector.style_dim == 1536
        assert projector.spatial_dim == 768
        assert projector.hidden_dim == 512
        
        # Check components exist
        assert hasattr(projector, 'style_projector')
        assert hasattr(projector, 'density_projector')
        assert hasattr(projector, 'fusion_layer')
    
    def test_projector_forward(self, projector, device):
        """Test projector forward pass."""
        batch_size = 2
        
        style_embeddings = torch.randn(batch_size, 1536, device=device)
        density_weights = torch.rand(batch_size, 64, 64, device=device)
        
        style_mean, style_std = projector(style_embeddings, density_weights)
        
        assert style_mean.shape == (batch_size, 1, 768)
        assert style_std.shape == (batch_size, 1, 768)
        
        # Style std should be positive
        assert torch.all(style_std > 0)
    
    def test_density_influence(self, projector, device):
        """Test that density weights influence the output."""
        batch_size = 2
        style_embeddings = torch.randn(batch_size, 1536, device=device)
        
        # Different density patterns
        density_uniform = torch.ones(batch_size, 64, 64, device=device) * 0.5
        density_concentrated = torch.zeros(batch_size, 64, 64, device=device)
        density_concentrated[:, 30:34, 30:34] = 1.0
        
        mean_uniform, std_uniform = projector(style_embeddings, density_uniform)
        mean_concentrated, std_concentrated = projector(style_embeddings, density_concentrated)
        
        # Different density patterns should produce different parameters
        assert not torch.allclose(mean_uniform, mean_concentrated)
        assert not torch.allclose(std_uniform, std_concentrated)


class TestStyleBezierFusionModule:
    """Test suite for StyleBezierFusionModule."""
    
    @pytest.fixture
    def device(self):
        """Get test device (CUDA if available)."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def fusion_module(self, device):
        """Create a StyleBezierFusionModule instance."""
        return StyleBezierFusionModule(
            spatial_dim=768,
            style_dim=1536,
            num_heads=8,
            dropout=0.1,
            use_cross_attention=True
        ).to(device)
    
    def test_initialization(self, fusion_module):
        """Test proper initialization of fusion module."""
        assert fusion_module.spatial_dim == 768
        assert fusion_module.style_dim == 1536
        assert fusion_module.num_heads == 8
        assert fusion_module.use_cross_attention == True
        
        # Check components
        assert isinstance(fusion_module.adain_layer, AdaINLayer)
        assert isinstance(fusion_module.style_projector, DensityAwareStyleProjector)
        assert hasattr(fusion_module, 'cross_attention')
        assert hasattr(fusion_module, 'residual_proj')
        assert hasattr(fusion_module, 'output_proj')
    
    def test_forward_with_cross_attention(self, fusion_module, device):
        """Test forward pass with cross-attention enabled."""
        batch_size = 2
        seq_len = 256
        spatial_dim = 768
        style_dim = 1536
        
        spatial_features = torch.randn(batch_size, seq_len, spatial_dim, device=device)
        style_embeddings = torch.randn(batch_size, style_dim, device=device)
        density_weights = torch.rand(batch_size, 16, 16, device=device)
        
        stylized_features, attention_weights = fusion_module(
            spatial_features=spatial_features,
            style_embeddings=style_embeddings,
            density_weights=density_weights
        )
        
        assert stylized_features.shape == (batch_size, seq_len, spatial_dim)
        assert attention_weights is not None
        assert attention_weights.shape == (batch_size, seq_len, 1)
        assert torch.all(torch.isfinite(stylized_features))
    
    def test_forward_without_cross_attention(self, device):
        """Test forward pass with cross-attention disabled."""
        fusion_module = StyleBezierFusionModule(
            spatial_dim=768,
            style_dim=1536,
            use_cross_attention=False
        ).to(device)
        
        batch_size = 2
        seq_len = 256
        
        spatial_features = torch.randn(batch_size, seq_len, 768, device=device)
        style_embeddings = torch.randn(batch_size, 1536, device=device)
        density_weights = torch.rand(batch_size, 16, 16, device=device)
        
        stylized_features, attention_weights = fusion_module(
            spatial_features=spatial_features,
            style_embeddings=style_embeddings,
            density_weights=density_weights
        )
        
        assert stylized_features.shape == (batch_size, seq_len, 768)
        assert attention_weights is None
    
    def test_style_transfer_effectiveness(self, fusion_module, device):
        """Test that style transfer actually modifies features."""
        batch_size = 2
        seq_len = 256
        spatial_dim = 768
        
        spatial_features = torch.randn(batch_size, seq_len, spatial_dim, device=device)
        density_weights = torch.rand(batch_size, 16, 16, device=device)
        
        # Different style embeddings
        style_1 = torch.randn(batch_size, 1536, device=device)
        style_2 = torch.randn(batch_size, 1536, device=device) * 2
        
        output_1, _ = fusion_module(spatial_features, style_1, density_weights)
        output_2, _ = fusion_module(spatial_features, style_2, density_weights)
        
        # Different styles should produce different outputs
        assert not torch.allclose(output_1, output_2)
        
        # But outputs shouldn't be completely different from input (residual connection)
        diff_1 = (output_1 - spatial_features).abs().mean()
        diff_2 = (output_2 - spatial_features).abs().mean()
        assert diff_1 < spatial_features.abs().mean()
        assert diff_2 < spatial_features.abs().mean()
    
    def test_dimension_mismatch_handling(self, fusion_module, device):
        """Test handling of dimension mismatches."""
        batch_size = 2
        seq_len = 256
        
        # Wrong spatial dimension
        spatial_features = torch.randn(batch_size, seq_len, 512, device=device)
        style_embeddings = torch.randn(batch_size, 1536, device=device)
        density_weights = torch.rand(batch_size, 16, 16, device=device)
        
        with pytest.raises(ValueError, match="Expected spatial_dim"):
            fusion_module(spatial_features, style_embeddings, density_weights)
        
        # Wrong style dimension
        spatial_features = torch.randn(batch_size, seq_len, 768, device=device)
        style_embeddings = torch.randn(batch_size, 1024, device=device)
        
        with pytest.raises(ValueError, match="Expected style_dim"):
            fusion_module(spatial_features, style_embeddings, density_weights)
    
    def test_gradient_flow(self, fusion_module, device):
        """Test gradient flow through the module."""
        batch_size = 2
        seq_len = 256
        
        spatial_features = torch.randn(batch_size, seq_len, 768, device=device, requires_grad=True)
        style_embeddings = torch.randn(batch_size, 1536, device=device, requires_grad=True)
        density_weights = torch.rand(batch_size, 16, 16, device=device, requires_grad=True)
        
        output, _ = fusion_module(spatial_features, style_embeddings, density_weights)
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist
        assert spatial_features.grad is not None
        assert style_embeddings.grad is not None
        assert density_weights.grad is not None
        
        # Check gradients are non-zero
        assert not torch.allclose(spatial_features.grad, torch.zeros_like(spatial_features.grad))
        assert not torch.allclose(style_embeddings.grad, torch.zeros_like(style_embeddings.grad))
    
    def test_training_eval_mode(self, fusion_module, device):
        """Test behavior in training vs eval mode."""
        batch_size = 1
        seq_len = 100
        
        spatial_features = torch.randn(batch_size, seq_len, 768, device=device)
        style_embeddings = torch.randn(batch_size, 1536, device=device)
        density_weights = torch.rand(batch_size, 10, 10, device=device)
        
        # Training mode - outputs should vary due to dropout
        fusion_module.train()
        outputs_train = []
        for _ in range(5):
            output, _ = fusion_module(spatial_features, style_embeddings, density_weights)
            outputs_train.append(output)
        
        # Check variation
        for i in range(1, 5):
            assert not torch.allclose(outputs_train[0], outputs_train[i])
        
        # Eval mode - outputs should be identical
        fusion_module.eval()
        outputs_eval = []
        for _ in range(5):
            output, _ = fusion_module(spatial_features, style_embeddings, density_weights)
            outputs_eval.append(output)
        
        # Check consistency
        for i in range(1, 5):
            assert torch.allclose(outputs_eval[0], outputs_eval[i])
    
    def test_cross_attention_mechanism(self, fusion_module, device):
        """Test cross-attention mechanism specifically."""
        batch_size = 2
        seq_len = 100
        
        spatial_features = torch.randn(batch_size, seq_len, 768, device=device)
        style_embeddings = torch.randn(batch_size, 1536, device=device)
        density_weights = torch.rand(batch_size, 10, 10, device=device)
        
        # Get attention weights
        _, attention_weights = fusion_module(spatial_features, style_embeddings, density_weights)
        
        # Attention weights should sum to 1 across spatial positions
        attention_sum = attention_weights.sum(dim=1)
        assert torch.allclose(attention_sum, torch.ones_like(attention_sum))
    
    def test_memory_efficiency(self, fusion_module, device):
        """Test memory usage of the module."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory testing")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        batch_size = 4
        seq_len = 1024
        
        spatial_features = torch.randn(batch_size, seq_len, 768, device=device)
        style_embeddings = torch.randn(batch_size, 1536, device=device)
        density_weights = torch.rand(batch_size, 32, 32, device=device)
        
        start_memory = torch.cuda.memory_allocated()
        output, _ = fusion_module(spatial_features, style_embeddings, density_weights)
        peak_memory = torch.cuda.max_memory_allocated()
        
        memory_mb = (peak_memory - start_memory) / 1024 / 1024
        print(f"Style fusion memory usage: {memory_mb:.2f} MB")
        
        # Should be reasonable
        assert memory_mb < 500  # Less than 500MB


@pytest.mark.parametrize("spatial_dim,style_dim,use_cross_attention", [
    (512, 1024, True),
    (768, 1536, True),
    (768, 1536, False),
    (1024, 2048, True),
])
def test_different_configurations(spatial_dim, style_dim, use_cross_attention, device):
    """Test fusion module with different configurations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    fusion_module = StyleBezierFusionModule(
        spatial_dim=spatial_dim,
        style_dim=style_dim,
        num_heads=8,
        use_cross_attention=use_cross_attention
    ).to(device)
    
    # Test forward pass
    spatial_features = torch.randn(1, 100, spatial_dim, device=device)
    style_embeddings = torch.randn(1, style_dim, device=device)
    density_weights = torch.rand(1, 10, 10, device=device)
    
    output, attention = fusion_module(spatial_features, style_embeddings, density_weights)
    
    assert output.shape == (1, 100, spatial_dim)
    if use_cross_attention:
        assert attention is not None
    else:
        assert attention is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])