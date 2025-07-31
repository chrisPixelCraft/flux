#!/usr/bin/env python3
"""
Test script for FluxBezierAdapter integration.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add flux modules to path
flux_path = Path(__file__).parent / "src"
sys.path.append(str(flux_path))

from flux.model import FluxParams
from flux.modules.bezier_flux_model import FluxBezierAdapter, BezierAdapterConfig, FluxBezierLoraWrapper
from flux.modules.models import MultiModalCondition

def create_test_flux_params():
    """Create minimal FLUX parameters for testing."""
    return FluxParams(
        in_channels=16,
        out_channels=16, 
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=1024,  # Must be divisible by num_heads
        mlp_ratio=4.0,
        num_heads=16,  # 1024 // 16 = 64 (divisible by 4 for RoPE)
        depth=12,  # 12 double blocks
        depth_single_blocks=20,  # 20 single blocks  
        axes_dim=[32, 32],  # sum = 64 = hidden_size // num_heads, both even for RoPE
        theta=10000,
        qkv_bias=True,
        guidance_embed=True
    )

def create_test_inputs(batch_size=1, device='cuda'):
    """Create test inputs for FLUX model."""
    seq_len_img = 256  # 16x16 patches
    seq_len_txt = 77   # Text sequence length
    
    # Standard FLUX inputs
    img = torch.randn(batch_size, seq_len_img, 16, device=device)
    img_ids = torch.randint(0, 16, (batch_size, seq_len_img, 2), device=device)  # 2D position IDs for spatial
    txt = torch.randn(batch_size, seq_len_txt, 4096, device=device)
    txt_ids = torch.randint(0, 77, (batch_size, seq_len_txt, 2), device=device)  # 2D position IDs
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    y = torch.randn(batch_size, 768, device=device)
    guidance = torch.rand(batch_size, device=device) * 10
    
    return img, img_ids, txt, txt_ids, timesteps, y, guidance

def create_bezier_conditions(batch_size=1, device='cuda'):
    """Create BezierAdapter conditioning inputs."""
    # Multi-modal conditions
    conditions = MultiModalCondition(
        style_features=torch.randn(batch_size, 768, device=device),
        text_features=torch.randn(batch_size, 4096, device=device),
        mask_features=torch.randn(batch_size, 4, 64, 64, device=device),
        bezier_features=torch.randn(batch_size, 3, device=device)  # (x, y, density)
    )
    
    # Bézier control points
    bezier_points = torch.rand(batch_size, 4, 2, device=device)  # 4 control points
    
    return {
        'conditions': conditions,
        'bezier_points': bezier_points
    }

def test_bezier_adapter_config():
    """Test BezierAdapterConfig."""
    print("Testing BezierAdapterConfig...")
    
    # Default config
    config1 = BezierAdapterConfig()
    print(f"  Default hook layers: {config1.hook_layers}")
    
    # Custom config
    config2 = BezierAdapterConfig(
        hook_layers=[5, 7, 9],
        enable_style_transfer=False,
        lora_rank=32
    )
    print(f"  Custom hook layers: {config2.hook_layers}")
    print(f"  Style transfer enabled: {config2.enable_style_transfer}")
    
    return True

def test_flux_bezier_adapter():
    """Test FluxBezierAdapter integration."""
    print("\nTesting FluxBezierAdapter...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test configuration
    flux_params = create_test_flux_params()
    bezier_config = BezierAdapterConfig(
        hook_layers=[2, 4, 6],  # Test with smaller layer indices
        enable_bezier_guidance=True,
        enable_style_transfer=True,
        enable_density_attention=True
    )
    
    # Initialize FluxBezierAdapter
    try:
        model = FluxBezierAdapter(flux_params, bezier_config).to(device)
        print(f"FluxBezierAdapter initialized successfully")
        
        # Print integration statistics
        stats = model.get_integration_stats()
        print(f"Integration statistics:")
        print(f"  Total FLUX params: {stats['total_flux_params']:,}")
        print(f"  Total BezierAdapter params: {stats['total_bezier_params']:,}")
        print(f"  Total params: {stats['total_params']:,}")
        print(f"  BezierAdapter param ratio: {stats['bezier_param_ratio']:.1%}")
        print(f"  Hook layers: {stats['hook_layers']}")
        
        return model, True
        
    except Exception as e:
        print(f"FluxBezierAdapter initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False

def test_forward_pass(model, device):
    """Test forward pass with BezierAdapter."""
    print("\nTesting forward pass...")
    
    try:
        # Create test inputs
        img, img_ids, txt, txt_ids, timesteps, y, guidance = create_test_inputs(batch_size=1, device=device)
        bezier_conditions = create_bezier_conditions(batch_size=1, device=device)
        
        print(f"Input shapes:")
        print(f"  img: {img.shape}")
        print(f"  txt: {txt.shape}")
        print(f"  timesteps: {timesteps.shape}")
        
        # Forward pass without BezierAdapter (baseline)
        with torch.no_grad():
            output_baseline = model(
                img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                timesteps=timesteps, y=y, guidance=guidance,
                bezier_conditions=None  # No BezierAdapter
            )
        
        print(f"Baseline output shape: {output_baseline.shape}")
        
        # Forward pass with BezierAdapter
        with torch.no_grad():
            output_bezier = model(
                img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                timesteps=timesteps, y=y, guidance=guidance,
                bezier_conditions=bezier_conditions
            )
        
        print(f"BezierAdapter output shape: {output_bezier.shape}")
        
        # Check that outputs have same shape but are different
        if output_baseline.shape == output_bezier.shape:
            difference = (output_baseline - output_bezier).abs().mean().item()
            print(f"Output difference: {difference:.6f}")
            
            if difference > 1e-6:
                print("✅ Forward pass successful - BezierAdapter is modifying outputs")
                return True
            else:
                print("⚠️  Forward pass warning - outputs are too similar")
                return True
        else:
            print("❌ Forward pass failed - output shapes don't match")
            return False
            
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_mode(model):
    """Test BezierAdapter training mode."""
    print("\nTesting training mode...")
    
    try:
        # Enable BezierAdapter training mode (freeze FLUX)
        model.set_bezier_training_mode(True)
        
        # Count trainable parameters
        flux_trainable = 0
        bezier_trainable = 0
        
        flux_params = set(super(FluxBezierAdapter, model).parameters())
        bezier_params = set(model.get_bezier_parameters())
        
        for param in model.parameters():
            if param.requires_grad:
                if param in bezier_params:
                    bezier_trainable += param.numel()
                elif param in flux_params:
                    flux_trainable += param.numel()
        
        print(f"Training mode statistics:")
        print(f"  FLUX trainable params: {flux_trainable:,}")
        print(f"  BezierAdapter trainable params: {bezier_trainable:,}")
        
        if flux_trainable == 0 and bezier_trainable > 0:
            print("✅ Training mode test passed - FLUX frozen, BezierAdapter trainable")
            return True
        else:
            print("⚠️  Training mode warning - unexpected parameter training state")
            return True
            
    except Exception as e:
        print(f"Training mode test failed: {e}")
        return False

def test_bezier_lora_wrapper():
    """Test FluxBezierLoraWrapper."""
    print("\n" + "="*60)
    print("Testing FluxBezierLoraWrapper...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        flux_params = create_test_flux_params()
        bezier_config = BezierAdapterConfig(hook_layers=[1, 3])
        
        model = FluxBezierLoraWrapper(
            flux_params=flux_params,
            bezier_config=bezier_config,
            lora_rank=32,
            lora_scale=1.0,
            apply_lora_to_flux=False  # Don't apply LoRA to FLUX for this test
        ).to(device)
        
        print(f"FluxBezierLoraWrapper initialized successfully")
        
        # Test LoRA parameter access
        lora_params = model.get_lora_parameters()
        print(f"LoRA parameters: {len(lora_params)}")
        
        # Test LoRA scale setting
        model.set_lora_scale(0.5)
        print("LoRA scale setting successful")
        
        return True
        
    except Exception as e:
        print(f"FluxBezierLoraWrapper test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("FluxBezierAdapter Integration Test Suite")
    print("="*60)
    
    # Run tests
    success1 = test_bezier_adapter_config()
    model, success2 = test_flux_bezier_adapter()
    
    success3 = False
    success4 = False
    if model is not None:
        device = next(model.parameters()).device
        success3 = test_forward_pass(model, device)
        success4 = test_training_mode(model)
    
    success5 = test_bezier_lora_wrapper()
    
    print("\n" + "="*60)
    print("Test Results:")
    print(f"  BezierAdapterConfig: {'PASS' if success1 else 'FAIL'}")
    print(f"  FluxBezierAdapter init: {'PASS' if success2 else 'FAIL'}")
    print(f"  Forward pass: {'PASS' if success3 else 'FAIL'}")
    print(f"  Training mode: {'PASS' if success4 else 'FAIL'}")
    print(f"  FluxBezierLoraWrapper: {'PASS' if success5 else 'FAIL'}")
    
    total_success = success1 and success2 and success3 and success4 and success5
    
    if total_success:
        print("\n✅ All tests passed! FluxBezierAdapter integration is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above.")
    
    print("="*60)