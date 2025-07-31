#!/usr/bin/env python3
"""
Unit tests for ConditionInjectionAdapter module.

Tests multi-modal fusion, LoRA adaptation, cross-attention mechanism,
and various input combinations for the 4-branch architecture.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add flux modules to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from flux.modules.condition_adapter import ConditionInjectionAdapter
from flux.modules.models import MultiModalCondition
from flux.modules.lora import LinearLora


class TestConditionInjectionAdapter:
    """Test suite for ConditionInjectionAdapter."""
    
    @pytest.fixture
    def device(self):
        """Get test device (CUDA if available)."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def adapter(self, device):
        """Create a ConditionInjectionAdapter instance."""
        return ConditionInjectionAdapter(
            clip_dim=768,
            t5_dim=4096,
            hidden_dim=1536,
            lora_rank=64,
            num_heads=8,
            dropout=0.1,
            lora_scale=1.0
        ).to(device)
    
    @pytest.fixture
    def full_conditions(self, device):
        """Create full multi-modal conditions."""
        batch_size = 2
        return MultiModalCondition(
            style_features=torch.randn(batch_size, 768, device=device),
            text_features=torch.randn(batch_size, 4096, device=device),
            mask_features=torch.randn(batch_size, 4, 64, 64, device=device),
            bezier_features=torch.randn(batch_size, 3, device=device)
        )
    
    def test_initialization(self, adapter):
        """Test proper initialization of the adapter."""
        assert adapter.clip_dim == 768
        assert adapter.t5_dim == 4096
        assert adapter.hidden_dim == 1536
        assert adapter.lora_rank == 64
        assert adapter.num_heads == 8
        
        # Check branches exist
        assert hasattr(adapter, 'style_proj')
        assert hasattr(adapter, 'style_lora')
        assert hasattr(adapter, 'text_proj')
        assert hasattr(adapter, 'text_lora')
        assert hasattr(adapter, 'mask_conv')
        assert hasattr(adapter, 'mask_proj')
        assert hasattr(adapter, 'mask_lora')
        assert hasattr(adapter, 'bezier_mlp')
        assert hasattr(adapter, 'bezier_lora')
        
        # Check fusion components
        assert isinstance(adapter.fusion_attention, nn.MultiheadAttention)
        assert isinstance(adapter.layer_norm, nn.LayerNorm)
    
    def test_style_branch(self, adapter, device):
        """Test style branch processing."""
        batch_size = 2
        style_features = torch.randn(batch_size, 768, device=device)
        
        # Process through style branch
        style_output = adapter._process_style_branch(style_features)
        
        assert style_output.shape == (batch_size, 1536)
        assert torch.all(torch.isfinite(style_output))
    
    def test_text_branch(self, adapter, device):
        """Test text branch processing."""
        batch_size = 2
        text_features = torch.randn(batch_size, 4096, device=device)
        
        # Process through text branch
        text_output = adapter._process_text_branch(text_features)
        
        assert text_output.shape == (batch_size, 1536)
        assert torch.all(torch.isfinite(text_output))
    
    def test_mask_branch(self, adapter, device):
        """Test mask branch processing with spatial features."""
        batch_size = 2
        mask_features = torch.randn(batch_size, 4, 64, 64, device=device)
        
        # Process through mask branch
        mask_output = adapter._process_mask_branch(mask_features)
        
        assert mask_output.shape == (batch_size, 1536)
        assert torch.all(torch.isfinite(mask_output))
        
        # Test global average pooling worked correctly
        # Output should be different for different spatial patterns
        mask_features_2 = torch.zeros_like(mask_features)
        mask_features_2[:, :, 32:, 32:] = 1.0  # Different pattern
        mask_output_2 = adapter._process_mask_branch(mask_features_2)
        
        assert not torch.allclose(mask_output, mask_output_2)
    
    def test_bezier_branch(self, adapter, device):
        """Test Bézier branch processing."""
        batch_size = 2
        bezier_features = torch.randn(batch_size, 3, device=device)  # (x, y, density)
        
        # Process through Bézier branch
        bezier_output = adapter._process_bezier_branch(bezier_features)
        
        assert bezier_output.shape == (batch_size, 1536)
        assert torch.all(torch.isfinite(bezier_output))
    
    def test_full_forward(self, adapter, full_conditions):
        """Test full forward pass with all modalities."""
        # Forward pass
        unified_conditions = adapter(full_conditions)
        
        assert unified_conditions.shape == (2, 1536)
        assert torch.all(torch.isfinite(unified_conditions))
    
    def test_partial_modalities(self, adapter, device):
        """Test with missing modalities."""
        batch_size = 2
        
        # Only style and text
        conditions_partial = MultiModalCondition(
            style_features=torch.randn(batch_size, 768, device=device),
            text_features=torch.randn(batch_size, 4096, device=device),
            mask_features=None,
            bezier_features=None
        )
        
        output_partial = adapter(conditions_partial)
        assert output_partial.shape == (batch_size, 1536)
        
        # Only Bézier
        conditions_bezier_only = MultiModalCondition(
            style_features=None,
            text_features=None,
            mask_features=None,
            bezier_features=torch.randn(batch_size, 3, device=device)
        )
        
        output_bezier = adapter(conditions_bezier_only)
        assert output_bezier.shape == (batch_size, 1536)
    
    def test_no_modalities_error(self, adapter, device):
        """Test error when no modalities provided."""
        empty_conditions = MultiModalCondition(
            style_features=None,
            text_features=None,
            mask_features=None,
            bezier_features=None
        )
        
        with pytest.raises(ValueError, match="At least one modality"):
            adapter(empty_conditions)
    
    def test_lora_parameters(self, adapter):
        """Test LoRA parameter functionality."""
        # Get LoRA parameters
        lora_params = adapter.get_lora_parameters()
        
        # Should have 8 LoRA parameters (4 branches × 2 matrices each)
        assert len(lora_params) == 8
        
        # Check all are trainable
        for param in lora_params:
            assert param.requires_grad
        
        # Count LoRA vs total parameters
        lora_count = sum(p.numel() for p in lora_params)
        total_count = sum(p.numel() for p in adapter.parameters())
        
        print(f"LoRA parameters: {lora_count:,}")
        print(f"Total parameters: {total_count:,}")
        print(f"LoRA ratio: {lora_count / total_count:.2%}")
        
        # LoRA should be a significant portion but not all
        assert 0.1 < lora_count / total_count < 0.5
    
    def test_lora_scale_setting(self, adapter, device, full_conditions):
        """Test LoRA scale adjustment."""
        # Get output with scale 1.0
        adapter.set_lora_scale(1.0)
        output_scale_1 = adapter(full_conditions)
        
        # Get output with scale 0.5
        adapter.set_lora_scale(0.5)
        output_scale_half = adapter(full_conditions)
        
        # Get output with scale 0.0 (effectively no LoRA)
        adapter.set_lora_scale(0.0)
        output_scale_0 = adapter(full_conditions)
        
        # Outputs should be different
        assert not torch.allclose(output_scale_1, output_scale_half)
        assert not torch.allclose(output_scale_1, output_scale_0)
        assert not torch.allclose(output_scale_half, output_scale_0)
    
    def test_cross_attention_fusion(self, adapter, device):
        """Test multi-head cross-attention fusion mechanism."""
        batch_size = 2
        
        # Create conditions with distinct patterns
        conditions = MultiModalCondition(
            style_features=torch.ones(batch_size, 768, device=device),
            text_features=torch.zeros(batch_size, 4096, device=device),
            mask_features=None,
            bezier_features=None
        )
        
        # Forward pass
        output = adapter(conditions)
        
        # Cross-attention should blend the modalities
        assert not torch.allclose(output, torch.ones_like(output))
        assert not torch.allclose(output, torch.zeros_like(output))
    
    def test_gradient_flow(self, adapter, device, full_conditions):
        """Test gradient flow through all branches."""
        # Enable gradients
        full_conditions.style_features.requires_grad = True
        full_conditions.text_features.requires_grad = True
        full_conditions.mask_features.requires_grad = True
        full_conditions.bezier_features.requires_grad = True
        
        # Forward pass
        output = adapter(full_conditions)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist for all inputs
        assert full_conditions.style_features.grad is not None
        assert full_conditions.text_features.grad is not None
        assert full_conditions.mask_features.grad is not None
        assert full_conditions.bezier_features.grad is not None
        
        # Check gradients are non-zero
        assert not torch.allclose(full_conditions.style_features.grad, torch.zeros_like(full_conditions.style_features.grad))
    
    def test_dropout_training_eval(self, adapter, device, full_conditions):
        """Test dropout behavior in training vs eval mode."""
        # Training mode
        adapter.train()
        outputs_train = []
        for _ in range(5):
            outputs_train.append(adapter(full_conditions))
        
        # Outputs should vary due to dropout
        for i in range(1, 5):
            assert not torch.allclose(outputs_train[0], outputs_train[i])
        
        # Eval mode
        adapter.eval()
        outputs_eval = []
        for _ in range(5):
            outputs_eval.append(adapter(full_conditions))
        
        # Outputs should be identical
        for i in range(1, 5):
            assert torch.allclose(outputs_eval[0], outputs_eval[i])
    
    def test_batch_independence(self, adapter, device):
        """Test that batch samples are processed independently."""
        batch_size = 4
        
        # Create conditions where each sample is different
        conditions = MultiModalCondition(
            style_features=torch.randn(batch_size, 768, device=device),
            text_features=torch.randn(batch_size, 4096, device=device),
            mask_features=torch.randn(batch_size, 4, 64, 64, device=device),
            bezier_features=torch.randn(batch_size, 3, device=device)
        )
        
        output = adapter(conditions)
        
        # Each output should be different
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                assert not torch.allclose(output[i], output[j])
    
    def test_memory_efficiency(self, adapter, device):
        """Test memory efficiency of mask branch with global pooling."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory testing")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Large spatial input
        batch_size = 4
        mask_features = torch.randn(batch_size, 4, 128, 128, device=device)
        
        start_memory = torch.cuda.memory_allocated()
        output = adapter._process_mask_branch(mask_features)
        peak_memory = torch.cuda.max_memory_allocated()
        
        memory_mb = (peak_memory - start_memory) / 1024 / 1024
        print(f"Mask branch memory usage: {memory_mb:.2f} MB")
        
        # Should be efficient due to global pooling
        assert memory_mb < 100  # Less than 100MB


@pytest.mark.parametrize("clip_dim,t5_dim,hidden_dim", [
    (512, 2048, 1024),
    (768, 4096, 1536),
    (1024, 4096, 2048),
])
def test_different_dimensions(clip_dim, t5_dim, hidden_dim, device):
    """Test adapter with different dimension configurations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    adapter = ConditionInjectionAdapter(
        clip_dim=clip_dim,
        t5_dim=t5_dim,
        hidden_dim=hidden_dim,
        lora_rank=32
    ).to(device)
    
    conditions = MultiModalCondition(
        style_features=torch.randn(1, clip_dim, device=device),
        text_features=torch.randn(1, t5_dim, device=device),
        mask_features=None,
        bezier_features=None
    )
    
    output = adapter(conditions)
    assert output.shape == (1, hidden_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])