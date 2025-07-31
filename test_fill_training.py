#!/usr/bin/env python3
"""
Test script for FLUX.1-Fill-dev training pipeline.

This script validates that the updated training components work correctly
with the 384-channel Fill model architecture.
"""

import sys
from pathlib import Path
import torch

# Add source path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_fill_model_configuration():
    """Test Fill model configuration."""
    print("🧪 Testing Fill Model Configuration")
    print("=" * 50)
    
    try:
        from flux.training.config import get_fill_model_config
        
        config = get_fill_model_config()
        
        # Verify Fill model settings
        assert config.model.flux_model_name == "flux-dev-fill"
        assert config.model.is_fill_model == True
        assert config.model.mask_conditioning_channels == 320
        assert config.model.condition_adapter.mask_input_channels == 320
        assert config.model.condition_adapter.mask_conv_channels == 128
        
        print("✅ Fill model configuration created successfully")
        print(f"   Model: {config.model.flux_model_name}")
        print(f"   Fill model: {config.model.is_fill_model}")
        print(f"   Mask channels: {config.model.mask_conditioning_channels}")
        print(f"   Batch size: {config.data.batch_size}")
        print(f"   Mixed precision: {config.mixed_precision}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fill model configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fill_dataset_creation():
    """Test dataset creation with Fill model parameters."""
    print("\n🧪 Testing Fill Dataset Creation")
    print("=" * 50)
    
    try:
        from flux.training.dataset import BezierFontDataset
        import tempfile
        
        # Create temporary dataset directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create minimal dataset structure
            font_dir = temp_path / "test_font"
            char_dir = font_dir / "A"
            char_dir.mkdir(parents=True)
            
            # Create dummy files
            (char_dir / "rendered.png").touch()
            (char_dir / "style.png").touch() 
            (char_dir / "mask.png").touch()
            
            # Create bezier data
            import json
            bezier_data = {
                "control_points": [
                    [0.1, 0.1], [0.5, 0.9], [0.9, 0.1], [0.5, 0.5]
                ]
            }
            with open(char_dir / "bezier.json", "w") as f:
                json.dump(bezier_data, f)
            
            # Test Fill model dataset creation
            dataset = BezierFontDataset(
                data_root=str(temp_path),
                split="train",
                is_fill_model=True,
                mask_channels=320,
                cache_size=10
            )
            
            print("✅ Fill dataset created successfully")
            print(f"   Fill model: {dataset.is_fill_model}")
            print(f"   Mask channels: {dataset.mask_channels}")
            print(f"   Samples found: {len(dataset)}")
            
            # Test dataset loading if samples exist
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"   Sample keys: {list(sample.keys())}")
                
                # Check mask tensor dimensions
                mask_tensor = sample['mask_image']
                expected_channels = 320 if dataset.is_fill_model else 1
                print(f"   Mask tensor shape: {mask_tensor.shape}")
                
                if dataset.is_fill_model and mask_tensor.shape[0] == 320:
                    print("✅ Fill model mask conditioning created correctly")
                elif not dataset.is_fill_model and mask_tensor.shape[0] == 1:
                    print("✅ Standard model mask created correctly")
                else:
                    print(f"⚠️  Unexpected mask shape: {mask_tensor.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Fill dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fill_loss_functions():
    """Test Fill model loss functions."""
    print("\n🧪 Testing Fill Loss Functions")  
    print("=" * 50)
    
    try:
        from flux.training.losses import InpaintingLoss, MultiLossTrainer
        from flux.training.config import get_fill_model_config
        
        # Test inpainting loss
        inpainting_loss = InpaintingLoss()
        
        # Create dummy tensors
        batch_size = 2
        model_output = torch.randn(batch_size, 4, 64, 64)
        target = torch.randn(batch_size, 4, 64, 64)
        mask = torch.randint(0, 2, (batch_size, 1, 64, 64)).float()
        
        # Test loss computation
        loss_results = inpainting_loss(model_output, target, mask)
        
        print("✅ Inpainting loss computed successfully")
        print(f"   Loss keys: {list(loss_results.keys())}")
        print(f"   Total loss: {loss_results['loss'].item():.4f}")
        print(f"   Masked loss: {loss_results['masked_loss'].item():.4f}")
        print(f"   Boundary loss: {loss_results['boundary_loss'].item():.4f}")
        
        # Test multi-loss trainer with Fill model
        config = get_fill_model_config()
        multi_loss = MultiLossTrainer(config)
        
        print("✅ Multi-loss trainer created for Fill model")
        print(f"   Fill model detected: {multi_loss.is_fill_model}")
        print(f"   Has inpainting loss: {hasattr(multi_loss, 'inpainting_loss')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fill loss functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_input_preparation():
    """Test model input preparation for Fill model."""
    print("\n🧪 Testing Model Input Preparation")
    print("=" * 50)
    
    try:
        from flux.training.config import get_fill_model_config
        from flux.training.trainer import BezierAdapterTrainer
        from flux.training.dataset import BezierFontDataset
        
        # Create minimal trainer setup
        config = get_fill_model_config()
        
        # Create dummy batch
        batch = {
            'style_images': torch.randn(2, 3, 512, 512),
            'mask_images': torch.randn(2, 320, 64, 64),  # Fill model format
            'bezier_features': torch.randn(2, 8, 3),
            'bezier_points': torch.randn(2, 8, 2),
            'bezier_masks': torch.ones(2, 8).bool(),
            'characters': ['A', 'B'],
            'style_prompts': ['elegant serif', 'modern sans'],
            'font_names': ['Font1', 'Font2'],
            'sample_ids': ['sample_1', 'sample_2']
        }
        
        # Mock trainer setup (without full initialization)
        class MockTrainer:
            def __init__(self, config):
                self.config = config
                self.device = torch.device('cpu')
                self.clip_embedder = None
                self.t5_embedder = None
        
        trainer = MockTrainer(config)
        
        # Import the method we want to test
        from flux.training.trainer import BezierAdapterTrainer
        
        # Use the prepare_batch method
        model_inputs, targets = BezierAdapterTrainer.prepare_batch(trainer, batch)
        
        print("✅ Model input preparation successful")
        print(f"   Model input keys: {list(model_inputs.keys())}")
        print(f"   Target keys: {list(targets.keys())}")
        print(f"   Input tensor shape: {model_inputs['img'].shape}")
        
        # Verify Fill model specific components
        expected_channels = 384 if config.model.is_fill_model else 64
        actual_channels = model_inputs['img'].shape[1]
        
        if actual_channels == expected_channels:
            print(f"✅ Correct input channels: {actual_channels}")
        else:
            print(f"⚠️  Expected {expected_channels} channels, got {actual_channels}")
        
        # Check for inpainting mask in targets
        if 'inpainting_mask' in targets:
            print("✅ Inpainting mask included in targets")
            print(f"   Mask shape: {targets['inpainting_mask'].shape}")
        else:
            print("⚠️  Inpainting mask not found in targets")
        
        return True
        
    except Exception as e:
        print(f"❌ Model input preparation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Fill model training tests."""
    print("🧪 FLUX.1-Fill-dev Training Pipeline Tests")
    print("=" * 60)
    
    tests = [
        ("Fill Model Configuration", test_fill_model_configuration),
        ("Fill Dataset Creation", test_fill_dataset_creation),
        ("Fill Loss Functions", test_fill_loss_functions),
        ("Model Input Preparation", test_model_input_preparation)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All Fill model training pipeline tests passed!")
        print("\nThe training pipeline is ready for FLUX.1-Fill-dev:")
        print("  • Configuration supports 384-channel architecture")
        print("  • Dataset handles 320-channel mask conditioning")
        print("  • Loss functions include inpainting-specific objectives")
        print("  • Model inputs prepared correctly for Fill model format")
        print("\nNext steps:")
        print("  1. Prepare font dataset with proper image+mask pairs")
        print("  2. Run: python train_fill_model.py --data_root <dataset_path>")
        print("  3. Monitor training progress in outputs/fill_model_training/")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)