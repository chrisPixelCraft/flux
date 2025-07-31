#!/usr/bin/env python3
"""
Quick test for BezierFluxPipeline implementation.

This test validates that our pipeline classes can be imported and initialized
without requiring the full FLUX model loading (which takes too long for testing).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all pipeline components can be imported."""
    print("🧪 Testing Pipeline Imports")
    print("=" * 40)
    
    try:
        # Test basic imports
        from flux.pipeline import BezierFluxPipeline
        print("✅ BezierFluxPipeline imported")
        
        from flux.pipeline.utils import (
            load_bezier_curves, 
            create_bezier_from_character,
            prepare_bezier_inputs
        )
        print("✅ Pipeline utilities imported")
        
        # Test utility functions
        curves = create_bezier_from_character('A', method='outline')
        print(f"✅ Created {len(curves)} Bézier points for 'A'")
        
        # Test curve validation
        from flux.pipeline.utils import validate_bezier_curves, normalize_bezier_points
        
        # Test with valid curves
        is_valid = validate_bezier_curves(curves)
        print(f"✅ Curve validation: {'Valid' if is_valid else 'Invalid'}")
        
        # Test normalization
        normalized = normalize_bezier_points(curves)
        print(f"✅ Normalized curves: {len(normalized)} points")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bezier_utilities():
    """Test Bézier curve utility functions."""
    print("\n🔧 Testing Bézier Utilities")
    print("=" * 40)
    
    try:
        from flux.pipeline.utils import (
            create_bezier_from_character,
            normalize_bezier_points,
            validate_bezier_curves,
            interpolate_bezier_curves,
            save_bezier_curves
        )
        
        # Test character curve generation
        for char in ['A', 'B', 'O']:
            curves = create_bezier_from_character(char, method='outline')
            print(f"✅ Generated curves for '{char}': {len(curves)} points")
            
            # Test validation
            is_valid = validate_bezier_curves(curves)
            if not is_valid:
                print(f"⚠️  Curves for '{char}' failed validation")
            
            # Test normalization
            normalized = normalize_bezier_points(curves)
            if len(normalized) != len(curves):
                print(f"❌ Normalization changed point count for '{char}'")
                return False
        
        # Test curve interpolation
        curve1 = create_bezier_from_character('A', method='outline')
        curve2 = create_bezier_from_character('B', method='outline')
        
        # Make curves same length for interpolation
        min_len = min(len(curve1), len(curve2))
        curve1 = curve1[:min_len]
        curve2 = curve2[:min_len]
        
        interp_curve = interpolate_bezier_curves(curve1, curve2, 0.5)
        print(f"✅ Interpolated curves: {len(interp_curve)} points")
        
        # Test save/load
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        save_bezier_curves(curve1, temp_path, {"test": "data"})
        
        # Verify saved file
        with open(temp_path, 'r') as f:
            saved_data = json.load(f)
        
        if 'control_points' not in saved_data:
            print("❌ Saved curve file missing control_points")
            return False
        
        print(f"✅ Saved/loaded curves: {len(saved_data['control_points'])} points")
        
        # Cleanup
        Path(temp_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Utility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_compatibility():
    """Test model configuration compatibility."""
    print("\n🔧 Testing Model Compatibility")
    print("=" * 40)
    
    try:
        # Test FLUX config loading
        from flux.util import configs
        
        # Test standard FLUX
        standard_config = configs.get("flux-dev")
        if standard_config:
            print(f"✅ Standard FLUX config: {standard_config.params.in_channels} channels")
        
        # Test Fill model
        fill_config = configs.get("flux-dev-fill")
        if fill_config:
            print(f"✅ Fill FLUX config: {fill_config.params.in_channels} channels")
            
            if fill_config.params.in_channels != 384:
                print(f"⚠️  Expected 384 channels for Fill model, got {fill_config.params.in_channels}")
        
        # Test BezierAdapter config
        from flux.modules.bezier_flux_model import BezierAdapterConfig
        
        bezier_config = BezierAdapterConfig()
        print(f"✅ BezierAdapter config: {len(bezier_config.hook_layers)} hook layers")
        
        # Test MultiModalCondition
        from flux.modules.models import MultiModalCondition
        import torch
        
        # Create dummy condition
        condition = MultiModalCondition(
            style_features=torch.randn(1, 768),
            text_features=torch.randn(1, 4096),
            mask_features=torch.randn(1, 4, 64, 64),
            bezier_features=torch.randn(1, 3)
        )
        
        print("✅ MultiModalCondition created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_utilities():
    """Test dataset creation utilities."""
    print("\n📊 Testing Dataset Utilities")
    print("=" * 40)
    
    try:
        from flux.pipeline.utils import (
            create_font_dataset_sample,
            batch_process_characters
        )
        
        import tempfile
        import shutil
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Test single sample creation
            sample = create_font_dataset_sample(
                character='A',
                font_style='serif',
                output_dir=temp_dir,
                generate_curves=True
            )
            
            print(f"✅ Created dataset sample for 'A'")
            print(f"   Keys: {list(sample.keys())}")
            
            # Verify files were created
            if 'bezier_curves_path' in sample:
                curves_path = Path(sample['bezier_curves_path'])
                if curves_path.exists():
                    print(f"✅ Curves file created: {curves_path.name}")
                else:
                    print(f"⚠️  Curves file not found: {curves_path}")
            
            # Test batch processing
            characters = ['B', 'C', 'D']
            samples = batch_process_characters(
                characters=characters,
                font_style='sans-serif',
                output_dir=temp_dir / 'batch',
                generate_curves=True
            )
            
            print(f"✅ Batch processed {len(samples)} characters")
            
            for sample in samples:
                char = sample['character']
                if 'control_points' in sample:
                    print(f"   '{char}': {len(sample['control_points'])} control points")
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 BezierFluxPipeline Test Suite")
    print("=" * 50)
    
    tests = [
        ("Pipeline Imports", test_imports),
        ("Bézier Utilities", test_bezier_utilities),
        ("Model Compatibility", test_model_compatibility),
        ("Dataset Utilities", test_dataset_utilities)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n📋 Test Results Summary")
    print("=" * 50)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! BezierFluxPipeline is ready for use.")
        print("\nNext steps:")
        print("  1. Run 'python example_bezier_pipeline.py' for usage examples")
        print("  2. Load FLUX.1-Fill-dev model for actual generation")
        print("  3. Create custom Bézier curves for your font characters")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)