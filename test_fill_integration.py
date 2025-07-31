#!/usr/bin/env python3
"""
Test script for FLUX.1-Fill-dev BezierAdapter integration.

This script validates that our FluxBezierAdapter works correctly with
FLUX.1-Fill-dev parameters and can handle 384-channel inputs.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from flux.model import FluxParams
from flux.modules.bezier_flux_model import FluxBezierAdapter, BezierAdapterConfig
from flux.modules.models import MultiModalCondition
from flux.util import configs


def test_flux_fill_bezier_adapter():
    """Test FluxBezierAdapter with FLUX.1-Fill-dev configuration."""
    print("🧪 Testing FLUX.1-Fill-dev BezierAdapter Integration")
    print("=" * 60)
    
    try:
        # Test 1: Load FLUX.1-Fill-dev configuration
        print("📋 Test 1: Loading FLUX.1-Fill-dev configuration...")
        fill_config = configs["flux-dev-fill"]
        flux_params = fill_config.params
        
        print(f"✅ Configuration loaded:")
        print(f"   Input channels: {flux_params.in_channels}")
        print(f"   Output channels: {flux_params.out_channels}")
        print(f"   Hidden size: {flux_params.hidden_size}")
        print(f"   Num heads: {flux_params.num_heads}")
        print(f"   Depth: {flux_params.depth}")
        print()
        
        # Test 2: Initialize BezierAdapter with Fill model
        print("🔧 Test 2: Initializing FluxBezierAdapter with Fill model...")
        bezier_config = BezierAdapterConfig(
            enable_bezier_guidance=True,
            enable_style_transfer=True,
            enable_density_attention=True
        )
        
        model = FluxBezierAdapter(flux_params, bezier_config)
        
        print(f"✅ FluxBezierAdapter initialized:")
        print(f"   Model type: {model.bezier_stats['model_type']}")
        print(f"   Input channels: {model.bezier_stats['input_channels']}")
        print(f"   Is Fill model: {model.is_fill_model}")
        print(f"   Hook layers: {model.bezier_stats['hook_layers']}")
        print()
        
        # Test 3: Check parameter counts
        print("📊 Test 3: Checking parameter counts...")
        stats = model.get_integration_stats()
        
        print(f"✅ Parameter statistics:")
        print(f"   Total FLUX params: {stats['total_flux_params']:,}")
        print(f"   Total BezierAdapter params: {stats['total_bezier_params']:,}")
        print(f"   Total params: {stats['total_params']:,}")
        print(f"   BezierAdapter ratio: {stats['bezier_param_ratio']:.2%}")
        print()
        
        # Test 4: Test forward pass with Fill model inputs
        print("🚀 Test 4: Testing forward pass with 384-channel inputs...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        batch_size = 1
        seq_len = 64 * 64  # 64x64 latent resolution
        
        # Create Fill model inputs (384 channels)
        img = torch.randn(batch_size, seq_len, 384, device=device)  # Fill model input
        img_ids = torch.randn(batch_size, seq_len, 3, device=device)
        txt = torch.randn(batch_size, 512, 4096, device=device)  # T5 features
        txt_ids = torch.randn(batch_size, 512, 3, device=device)
        timesteps = torch.randn(batch_size, device=device)
        y = torch.randn(batch_size, 768, device=device)  # CLIP features
        guidance = torch.randn(batch_size, device=device)
        
        print(f"✅ Input tensors created:")
        print(f"   img shape: {img.shape} (384 channels for Fill model)")
        print(f"   txt shape: {txt.shape}")
        print(f"   timesteps shape: {timesteps.shape}")
        print()
        
        # Test 5: Forward pass without BezierAdapter conditioning
        print("🔄 Test 5: Forward pass without BezierAdapter conditioning...")
        
        with torch.no_grad():
            output = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                y=y,
                guidance=guidance
            )
        
        print(f"✅ Forward pass successful:")
        print(f"   Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        print()
        
        # Test 6: Forward pass with BezierAdapter conditioning
        print("🎨 Test 6: Forward pass with BezierAdapter conditioning...")
        
        # Create BezierAdapter conditioning
        bezier_conditions = {
            'conditions': MultiModalCondition(
                style_features=torch.randn(batch_size, 768, device=device),
                text_features=torch.randn(batch_size, 4096, device=device),
                mask_features=torch.randn(batch_size, 320, 64, 64, device=device),  # Fill model mask
                bezier_features=torch.randn(batch_size, 3, device=device)
            ),
            'bezier_points': torch.randn(batch_size, 16, 2, device=device)  # 16 Bézier points
        }
        
        with torch.no_grad():
            output_bezier = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                y=y,
                guidance=guidance,
                bezier_conditions=bezier_conditions
            )
        
        print(f"✅ BezierAdapter forward pass successful:")
        print(f"   Output shape: {output_bezier.shape}")
        print(f"   Output dtype: {output_bezier.dtype}")
        print(f"   Output differs from standard: {not torch.allclose(output, output_bezier, atol=1e-6)}")
        print()
        
        # Test 7: Component compatibility check
        print("🔍 Test 7: Checking component compatibility...")
        
        components_status = stats['components_enabled']
        print(f"✅ Component status:")
        for component, enabled in components_status.items():
            print(f"   {component}: {'✅ Enabled' if enabled else '❌ Disabled'}")
        print()
        
        # Test 8: Memory usage check
        print("💾 Test 8: Memory usage check...")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
            
            print(f"✅ GPU Memory usage:")
            print(f"   Allocated: {memory_allocated:.2f} GB")
            print(f"   Reserved: {memory_reserved:.2f} GB")
        else:
            print("ℹ️  CPU mode - memory check skipped")
        print()
        
        print("🎉 All tests passed! FLUX.1-Fill-dev BezierAdapter integration successful!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_standard_flux_compatibility():
    """Test that standard FLUX still works."""
    print("\n🔄 Testing backward compatibility with standard FLUX...")
    print("=" * 60)
    
    try:
        # Load standard FLUX configuration
        standard_config = configs["flux-dev"]
        flux_params = standard_config.params
        
        print(f"📋 Standard FLUX configuration:")
        print(f"   Input channels: {flux_params.in_channels}")
        print(f"   Model type: {'Standard FLUX' if flux_params.in_channels == 64 else 'Unknown'}")
        print()
        
        # Initialize with standard FLUX
        bezier_config = BezierAdapterConfig()
        model = FluxBezierAdapter(flux_params, bezier_config)
        
        print(f"✅ Standard FLUX BezierAdapter:")
        print(f"   Model type: {model.bezier_stats['model_type']}")
        print(f"   Is Fill model: {model.is_fill_model}")
        print(f"   Is Standard model: {model.is_standard_model}")
        print()
        
        # Quick forward pass test
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        batch_size = 1
        seq_len = 64 * 64
        
        img = torch.randn(batch_size, seq_len, 64, device=device)    # Standard 64 channels
        img_ids = torch.randn(batch_size, seq_len, 3, device=device)
        txt = torch.randn(batch_size, 512, 4096, device=device)
        txt_ids = torch.randn(batch_size, 512, 3, device=device)
        timesteps = torch.randn(batch_size, device=device)
        y = torch.randn(batch_size, 768, device=device)
        guidance = torch.randn(batch_size, device=device)
        
        with torch.no_grad():
            output = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                y=y,
                guidance=guidance
            )
        
        print(f"✅ Standard FLUX forward pass successful:")
        print(f"   Output shape: {output.shape}")
        print()
        
        print("🎉 Backward compatibility confirmed!")
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 FLUX.1-Fill-dev BezierAdapter Integration Test Suite")
    print("=" * 60)
    print()
    
    # Run tests
    fill_test_passed = test_flux_fill_bezier_adapter()
    standard_test_passed = test_standard_flux_compatibility()
    
    print("\n" + "=" * 60)
    print("📋 Test Results Summary:")
    print(f"   FLUX.1-Fill-dev Integration: {'✅ PASSED' if fill_test_passed else '❌ FAILED'}")
    print(f"   Standard FLUX Compatibility: {'✅ PASSED' if standard_test_passed else '❌ FAILED'}")
    print()
    
    if fill_test_passed and standard_test_passed:
        print("🎉 All integration tests passed! Ready for BezierFluxFillPipeline implementation.")
        exit(0)
    else:
        print("❌ Some tests failed. Please check the implementation.")
        exit(1)