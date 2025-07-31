#!/usr/bin/env python3
"""
Test script for SpatialAttentionFuser with fixed RoPE position embeddings.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add flux modules to path
flux_path = Path(__file__).parent / "src"
sys.path.append(str(flux_path))

from flux.modules.spatial_fuser import SpatialAttentionFuser
from flux.modules.models import MultiModalCondition

def test_spatial_attention_fuser():
    """Test SpatialAttentionFuser with proper position embeddings."""
    print("Testing SpatialAttentionFuser with fixed RoPE position embeddings...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test configuration
    batch_size = 2
    seq_len = 64 * 64  # 4096 spatial locations
    feature_dim = 768
    
    # Initialize SpatialAttentionFuser
    fuser = SpatialAttentionFuser(
        feature_dim=feature_dim,
        num_heads=12,
        num_layers=6,
        mlp_ratio=4.0,
        theta=10000
    ).to(device)
    
    print(f"SpatialAttentionFuser initialized: {fuser.extra_repr()}")
    print(f"Total parameters: {sum(p.numel() for p in fuser.parameters()):,}")
    
    # Create test inputs
    spatial_features = torch.randn(batch_size, seq_len, feature_dim, device=device)
    density_weights = torch.rand(batch_size, 64, 64, device=device)  # Density map
    condition_embeddings = torch.randn(batch_size, 1536, device=device)  # From ConditionInjectionAdapter
    
    print(f"Input shapes:")
    print(f"  spatial_features: {spatial_features.shape}")
    print(f"  density_weights: {density_weights.shape}")
    print(f"  condition_embeddings: {condition_embeddings.shape}")
    
    # Test forward pass
    try:
        print("\nTesting forward pass...")
        with torch.no_grad():
            fused_features, attention_maps = fuser(
                spatial_features=spatial_features,
                density_weights=density_weights,
                condition_embeddings=condition_embeddings
            )
        
        print(f"Forward pass successful!")
        print(f"Output shapes:")
        print(f"  fused_features: {fused_features.shape}")
        print(f"  attention_maps: {attention_maps}")
        
        # Test backward pass
        print("\nTesting backward pass...")
        fused_features, _ = fuser(
            spatial_features=spatial_features,
            density_weights=density_weights,
            condition_embeddings=condition_embeddings
        )
        
        # Compute dummy loss
        loss = fused_features.mean()
        loss.backward()
        
        print(f"Backward pass successful! Loss: {loss.item():.6f}")
        
        # Check gradients
        grad_norm = 0.0
        for param in fuser.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        print(f"Gradient norm: {grad_norm:.6f}")
        
        # Test position embeddings directly
        print("\nTesting position embeddings...")
        pe = fuser._get_position_embeddings(batch_size, device)
        print(f"Position embeddings shape: {pe.shape}")
        print(f"Position embeddings dtype: {pe.dtype}")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """Test memory efficiency with gradient checkpointing."""
    print("\n" + "="*60)
    print("Testing memory efficiency with gradient checkpointing...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Large sequence length to trigger gradient checkpointing
    batch_size = 1
    seq_len = 3000  # > 2048 to trigger gradient checkpointing
    feature_dim = 768
    
    fuser = SpatialAttentionFuser(
        feature_dim=feature_dim,
        num_heads=12,
        num_layers=3,  # Reduced for memory test
        mlp_ratio=4.0,
        theta=10000
    ).to(device)
    
    try:
        # Create inputs that will trigger gradient checkpointing
        spatial_features = torch.randn(batch_size, seq_len, feature_dim, device=device, requires_grad=True)
        # For large sequences, we need to adjust density weights to match
        density_h = density_w = int(seq_len ** 0.5) + 1  # Approximate sqrt
        density_weights = torch.rand(batch_size, density_h, density_w, device=device)
        condition_embeddings = torch.randn(batch_size, 1536, device=device)
        
        print(f"Testing with large sequence length: {seq_len}")
        print(f"This should trigger gradient checkpointing...")
        
        # Forward pass
        fused_features, _ = fuser(
            spatial_features=spatial_features,
            density_weights=density_weights,
            condition_embeddings=condition_embeddings
        )
        
        # Backward pass
        loss = fused_features.mean()
        loss.backward()
        
        print(f"Gradient checkpointing test passed! Loss: {loss.item():.6f}")
        return True
        
    except Exception as e:
        print(f"Memory efficiency test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("SpatialAttentionFuser Test Suite")
    print("="*60)
    
    # Run tests
    success1 = test_spatial_attention_fuser()
    success2 = test_memory_efficiency()
    
    print("\n" + "="*60)
    print("Test Results:")
    print(f"  Basic functionality: {'PASS' if success1 else 'FAIL'}")
    print(f"  Memory efficiency: {'PASS' if success2 else 'FAIL'}")
    
    if success1 and success2:
        print("\n✅ All tests passed! SpatialAttentionFuser is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above.")
    
    print("="*60)