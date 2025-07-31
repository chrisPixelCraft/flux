#!/usr/bin/env python3
"""Simple test for FLUX.1-Fill-dev integration."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_integration():
    """Test basic integration without heavy computation."""
    try:
        from flux.model import FluxParams
        from flux.modules.bezier_flux_model import FluxBezierAdapter, BezierAdapterConfig
        from flux.util import configs
        
        print("✅ Imports successful")
        
        # Test FLUX.1-Fill-dev configuration
        fill_config = configs["flux-dev-fill"]
        flux_params = fill_config.params
        
        print(f"✅ FLUX.1-Fill-dev config loaded:")
        print(f"   Input channels: {flux_params.in_channels}")
        print(f"   Expected: 384")
        
        # Test model initialization
        bezier_config = BezierAdapterConfig()
        model = FluxBezierAdapter(flux_params, bezier_config)
        
        print(f"✅ FluxBezierAdapter initialized:")
        print(f"   Model type: {model.bezier_stats['model_type']}")
        print(f"   Is Fill model: {model.is_fill_model}")
        print(f"   Input channels: {model.bezier_stats['input_channels']}")
        
        # Test parameter counting
        stats = model.get_integration_stats()
        print(f"✅ Parameter stats:")
        print(f"   Total BezierAdapter params: {stats['total_bezier_params']:,}")
        
        # Test standard FLUX compatibility
        standard_config = configs["flux-dev"]
        standard_params = standard_config.params
        standard_model = FluxBezierAdapter(standard_params, bezier_config)
        
        print(f"✅ Standard FLUX compatibility:")
        print(f"   Model type: {standard_model.bezier_stats['model_type']}")
        print(f"   Is Fill model: {standard_model.is_fill_model}")
        print(f"   Input channels: {standard_model.bezier_stats['input_channels']}")
        
        print("🎉 Basic integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_integration()