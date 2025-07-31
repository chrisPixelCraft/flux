#!/usr/bin/env python3
"""
Download script for FLUX.1-Fill-dev and FLUX.1-schnell models.

This script downloads the required model checkpoints from HuggingFace
for the BezierAdapter-FLUX integration.
"""

import os
import sys
from pathlib import Path
import argparse

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("❌ huggingface-hub not installed. Installing...")
    os.system(f"{sys.executable} -m pip install huggingface-hub")
    from huggingface_hub import snapshot_download


def download_model(repo_id: str, local_dir: str, model_name: str):
    """Download a model from HuggingFace."""
    print(f"\n📥 Downloading {model_name}...")
    print(f"   Repository: {repo_id}")
    print(f"   Local directory: {local_dir}")
    
    try:
        # Create directory if it doesn't exist
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        
        # Download the model
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=["*.safetensors", "*.json"],
            ignore_patterns=["*.md", ".gitattributes"],
            resume_download=True  # Resume if interrupted
        )
        
        print(f"✅ {model_name} downloaded successfully!")
        
        # Verify downloaded files
        verify_download(local_dir, model_name)
        
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Ensure you have sufficient disk space (~24GB per model)")
        print("3. If authentication is required, run: huggingface-cli login")
        return False
    
    return True


def verify_download(local_dir: str, model_name: str):
    """Verify that model files were downloaded correctly."""
    print(f"\n🔍 Verifying {model_name} download...")
    
    expected_files = {
        "FLUX.1-Fill-dev": [
            ("flux1-fill-dev.safetensors", 23.8e9),  # ~23.8GB
            ("ae.safetensors", 335e6),  # ~335MB
            ("config.json", None),
            ("flux_fill_config.json", None)
        ],
        "FLUX.1-schnell": [
            ("flux1-schnell.safetensors", 23.8e9),  # ~23.8GB
            ("ae.safetensors", 335e6),  # ~335MB
            ("config.json", None)
        ]
    }
    
    files_to_check = expected_files.get(model_name, [])
    all_good = True
    
    for filename, expected_size in files_to_check:
        filepath = Path(local_dir) / filename
        
        if filepath.exists():
            actual_size = filepath.stat().st_size
            size_str = f"{actual_size / 1e9:.2f}GB" if actual_size > 1e9 else f"{actual_size / 1e6:.2f}MB"
            
            if expected_size and abs(actual_size - expected_size) / expected_size > 0.1:
                print(f"   ⚠️  {filename}: {size_str} (expected ~{expected_size / 1e9:.1f}GB)")
                all_good = False
            else:
                print(f"   ✅ {filename}: {size_str}")
        else:
            print(f"   ❌ {filename}: Missing")
            all_good = False
    
    if all_good:
        print(f"   ✅ All files verified for {model_name}")
    else:
        print(f"   ⚠️  Some files may be missing or incomplete for {model_name}")
    
    return all_good


def check_disk_space(required_gb: float = 50):
    """Check if there's enough disk space."""
    import shutil
    
    stat = shutil.disk_usage(".")
    available_gb = stat.free / (1024**3)
    
    print(f"\n💾 Disk Space Check:")
    print(f"   Available: {available_gb:.1f}GB")
    print(f"   Required: {required_gb:.1f}GB")
    
    if available_gb < required_gb:
        print(f"   ❌ Insufficient disk space! Need at least {required_gb}GB")
        return False
    else:
        print(f"   ✅ Sufficient disk space available")
        return True


def main():
    parser = argparse.ArgumentParser(description="Download FLUX models for BezierAdapter")
    parser.add_argument("--models", nargs="+", 
                       choices=["fill", "schnell", "all"],
                       default=["fill"],
                       help="Which models to download (default: fill)")
    parser.add_argument("--checkpoint-dir", type=str,
                       default="checkpoints",
                       help="Directory to save checkpoints (default: checkpoints)")
    parser.add_argument("--skip-space-check", action="store_true",
                       help="Skip disk space check")
    
    args = parser.parse_args()
    
    print("🚀 FLUX Model Downloader for BezierAdapter")
    print("=" * 50)
    
    # Check disk space
    if not args.skip_space_check:
        required_space = 50 if "all" in args.models else 25
        if not check_disk_space(required_space):
            print("\n⚠️  Proceeding anyway... (use --skip-space-check to bypass)")
            response = input("Continue? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                return
    
    # Determine which models to download
    models_to_download = []
    
    if "all" in args.models:
        models_to_download = [
            ("black-forest-labs/FLUX.1-Fill-dev", "FLUX.1-Fill-dev"),
            ("black-forest-labs/FLUX.1-schnell", "FLUX.1-schnell")
        ]
    else:
        if "fill" in args.models:
            models_to_download.append(
                ("black-forest-labs/FLUX.1-Fill-dev", "FLUX.1-Fill-dev")
            )
        if "schnell" in args.models:
            models_to_download.append(
                ("black-forest-labs/FLUX.1-schnell", "FLUX.1-schnell")
            )
    
    # Download models
    success_count = 0
    for repo_id, model_name in models_to_download:
        local_dir = os.path.join(args.checkpoint_dir, repo_id.replace("/", "_"))
        if download_model(repo_id, local_dir, model_name):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Download Summary:")
    print(f"   Models requested: {len(models_to_download)}")
    print(f"   Successfully downloaded: {success_count}")
    
    if success_count == len(models_to_download):
        print("\n✅ All models downloaded successfully!")
        print("\n🎯 Next steps:")
        print("1. Activate conda environment: conda activate easycontrol")
        print("2. Run examples: python example_bezier_pipeline.py")
        print("3. Start training: python train_fill_model.py --data_root <path>")
    else:
        print("\n⚠️  Some downloads failed. Please check the errors above.")
    
    # Test model loading
    if success_count > 0:
        print("\n🧪 Testing model configuration...")
        try:
            sys.path.insert(0, str(Path(__file__).parent / "src"))
            from flux.util import configs
            
            if "fill" in args.models:
                config = configs.get("flux-dev-fill")
                print(f"   ✅ FLUX.1-Fill-dev config loaded: {config.params.in_channels} channels")
            
            if "schnell" in args.models:
                config = configs.get("flux-schnell")
                print(f"   ✅ FLUX.1-schnell config loaded: {config.params.in_channels} channels")
                
        except Exception as e:
            print(f"   ⚠️  Could not test model loading: {e}")


if __name__ == "__main__":
    main()