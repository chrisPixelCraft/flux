#!/usr/bin/env python3
"""
Test script for StyleBezierFusionModule implementation.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add flux modules to path
flux_path = Path(__file__).parent / "src"
sys.path.append(str(flux_path))

from flux.modules.style_fusion import StyleBezierFusionModule, AdaINLayer, DensityAwareStyleProjector

def test_adain_layer():
    """Test AdaIN layer implementation."""
    print("Testing AdaIN layer...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with sequential format (B, L, C)
    adain = AdaINLayer(num_features=768).to(device)
    
    batch_size = 2
    seq_len = 100
    feature_dim = 768
    
    content_features = torch.randn(batch_size, seq_len, feature_dim, device=device)
    style_mean = torch.randn(batch_size, 1, feature_dim, device=device)
    style_std = torch.abs(torch.randn(batch_size, 1, feature_dim, device=device)) + 0.1
    
    stylized = adain(content_features, style_mean, style_std)
    
    print(f"  Input shape: {content_features.shape}")
    print(f"  Output shape: {stylized.shape}")
    print(f"  AdaIN layer test: {'PASS' if stylized.shape == content_features.shape else 'FAIL'}")
    
    return stylized.shape == content_features.shape

def test_density_aware_style_projector():
    """Test DensityAwareStyleProjector implementation."""
    print("\nTesting DensityAwareStyleProjector...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    projector = DensityAwareStyleProjector(
        style_dim=1536,
        spatial_dim=768,
        hidden_dim=512
    ).to(device)
    
    batch_size = 2
    style_embeddings = torch.randn(batch_size, 1536, device=device)
    density_weights = torch.rand(batch_size, 64, 64, device=device)
    
    style_mean, style_std = projector(style_embeddings, density_weights)
    
    print(f"  Style embeddings shape: {style_embeddings.shape}")
    print(f"  Density weights shape: {density_weights.shape}")
    print(f"  Output mean shape: {style_mean.shape}")
    print(f"  Output std shape: {style_std.shape}")
    
    expected_shape = (batch_size, 1, 768)
    success = (style_mean.shape == expected_shape and style_std.shape == expected_shape)
    print(f"  DensityAwareStyleProjector test: {'PASS' if success else 'FAIL'}")
    
    return success

def test_style_bezier_fusion_module():
    """Test StyleBezierFusionModule implementation."""
    print("\nTesting StyleBezierFusionModule...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test configuration
    batch_size = 2
    seq_len = 4096  # 64x64 spatial locations
    spatial_dim = 768
    style_dim = 1536
    
    # Initialize StyleBezierFusionModule
    fusion_module = StyleBezierFusionModule(
        spatial_dim=spatial_dim,
        style_dim=style_dim,
        num_heads=8,
        dropout=0.1,
        use_cross_attention=True
    ).to(device)
    
    print(f"StyleBezierFusionModule initialized: {fusion_module.extra_repr()}")
    print(f"Total parameters: {sum(p.numel() for p in fusion_module.parameters()):,}")
    
    # Create test inputs
    spatial_features = torch.randn(batch_size, seq_len, spatial_dim, device=device)
    style_embeddings = torch.randn(batch_size, style_dim, device=device)
    density_weights = torch.rand(batch_size, 64, 64, device=device)
    
    print(f"Input shapes:")
    print(f"  spatial_features: {spatial_features.shape}")
    print(f"  style_embeddings: {style_embeddings.shape}")
    print(f"  density_weights: {density_weights.shape}")
    
    # Test forward pass
    try:
        print("\nTesting forward pass...")
        with torch.no_grad():
            stylized_features, attention_weights = fusion_module(
                spatial_features=spatial_features,
                style_embeddings=style_embeddings,
                density_weights=density_weights
            )
        
        print(f"Forward pass successful!")
        print(f"Output shapes:")
        print(f"  stylized_features: {stylized_features.shape}")
        print(f"  attention_weights: {attention_weights.shape if attention_weights is not None else None}")
        
        # Test backward pass
        print("\nTesting backward pass...")
        stylized_features, _ = fusion_module(
            spatial_features=spatial_features,
            style_embeddings=style_embeddings,
            density_weights=density_weights
        )
        
        # Compute dummy loss
        loss = stylized_features.mean()
        loss.backward()
        
        print(f"Backward pass successful! Loss: {loss.item():.6f}")
        
        # Check gradients
        grad_norm = 0.0
        for param in fusion_module.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        print(f"Gradient norm: {grad_norm:.6f}")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_without_cross_attention():
    """Test StyleBezierFusionModule without cross-attention."""
    print("\n" + "="*60)
    print("Testing StyleBezierFusionModule without cross-attention...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize without cross-attention
    fusion_module = StyleBezierFusionModule(
        spatial_dim=768,
        style_dim=1536,
        num_heads=8,
        dropout=0.1,
        use_cross_attention=False
    ).to(device)
    
    batch_size = 2
    seq_len = 1000  # Smaller for faster testing
    
    spatial_features = torch.randn(batch_size, seq_len, 768, device=device)
    style_embeddings = torch.randn(batch_size, 1536, device=device)
    density_weights = torch.rand(batch_size, 64, 64, device=device)
    
    try:
        stylized_features, attention_weights = fusion_module(
            spatial_features=spatial_features,
            style_embeddings=style_embeddings,
            density_weights=density_weights
        )
        
        print(f"Output shape: {stylized_features.shape}")
        print(f"Attention weights: {attention_weights}")  # Should be None
        
        # Test backward pass
        loss = stylized_features.mean()
        loss.backward()
        
        print(f"Test without cross-attention: PASS")
        return True
        
    except Exception as e:
        print(f"Test without cross-attention: FAIL - {e}")
        return False

def test_style_transfer_properties():
    """Test style transfer properties of the module."""
    print("\n" + "="*60)
    print("Testing style transfer properties...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    fusion_module = StyleBezierFusionModule(
        spatial_dim=256,  # Smaller for testing
        style_dim=512,
        use_cross_attention=False
    ).to(device)
    
    batch_size = 2
    seq_len = 100
    
    # Create content and two different styles
    content_features = torch.randn(batch_size, seq_len, 256, device=device)
    style1 = torch.randn(1, 512, device=device)
    style2 = torch.randn(1, 512, device=device)
    density = torch.rand(1, 32, 32, device=device)  # Dummy density
    
    with torch.no_grad():
        # Apply different styles to same content
        stylized1, _ = fusion_module(content_features[:1], style1, density)
        stylized2, _ = fusion_module(content_features[:1], style2, density)
        
        # Check that different styles produce different outputs
        style_difference = (stylized1 - stylized2).abs().mean().item()
        
        print(f"Style difference: {style_difference:.6f}")
        
        if style_difference > 1e-6:
            print("Style transfer properties: PASS")
            return True
        else:
            print("Style transfer properties: FAIL - outputs too similar")
            return False

if __name__ == "__main__":
    print("="*60)
    print("StyleBezierFusionModule Test Suite")
    print("="*60)
    
    # Run tests
    success1 = test_adain_layer()
    success2 = test_density_aware_style_projector()
    success3 = test_style_bezier_fusion_module()
    success4 = test_without_cross_attention()
    success5 = test_style_transfer_properties()
    
    print("\n" + "="*60)
    print("Test Results:")
    print(f"  AdaIN layer: {'PASS' if success1 else 'FAIL'}")
    print(f"  DensityAwareStyleProjector: {'PASS' if success2 else 'FAIL'}")
    print(f"  StyleBezierFusionModule (full): {'PASS' if success3 else 'FAIL'}")
    print(f"  Without cross-attention: {'PASS' if success4 else 'FAIL'}")
    print(f"  Style transfer properties: {'PASS' if success5 else 'FAIL'}")
    
    total_success = success1 and success2 and success3 and success4 and success5
    
    if total_success:
        print("\n✅ All tests passed! StyleBezierFusionModule is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above.")
    
    print("="*60)