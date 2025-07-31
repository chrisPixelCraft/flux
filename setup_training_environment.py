#!/usr/bin/env python3
"""
Setup script for FLUX Fill training environment.

This script helps you set up the complete environment for training
BezierAdapter with FLUX.1-Fill-dev, including checks and preparations.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import shutil

def check_system_requirements():
    """Check if system meets training requirements."""
    print("🔍 Checking System Requirements...")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"✓ PyTorch {torch.__version__}")
            print(f"✓ CUDA Available: {gpu_count} GPU(s)")
            print(f"✓ Primary GPU: {gpu_name}")
            print(f"✓ GPU Memory: {memory_gb:.1f} GB")
            
            if memory_gb < 12:
                print("⚠️  Warning: GPU has less than 12GB memory. Training may be slow or fail.")
                print("   Consider reducing batch size or using gradient accumulation.")
            
        else:
            print("❌ CUDA not available. GPU required for training.")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Check disk space
    disk_usage = shutil.disk_usage(os.getcwd())
    free_gb = disk_usage.free / (1024**3)
    print(f"✓ Available disk space: {free_gb:.1f} GB")
    
    if free_gb < 50:
        print("⚠️  Warning: Less than 50GB free space. Consider freeing up space.")
    
    return True

def check_model_files():
    """Check if required model files are downloaded."""
    print("\n📦 Checking Model Files...")
    print("=" * 50)
    
    model_dir = Path("checkpoints/black-forest-labs_FLUX.1-Fill-dev")
    required_files = [
        "flux1-fill-dev.safetensors",
        "ae.safetensors", 
        "config.json",
        "flux_fill_config.json"
    ]
    
    all_present = True
    for file in required_files:
        file_path = model_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"✓ {file} ({size_mb:.1f} MB)")
        else:
            print(f"❌ Missing: {file}")
            all_present = False
    
    if not all_present:
        print("\n📥 To download FLUX.1-Fill-dev model:")
        print("1. Install huggingface-hub: pip install huggingface-hub")
        print("2. Run download script:")
        print("   python download_models.py")
        print("   OR manually download from HuggingFace")
        
    return all_present

def create_sample_dataset():
    """Create a minimal sample dataset for testing."""
    print("\n📁 Creating Sample Dataset...")
    print("=" * 50)
    
    dataset_dir = Path("sample_dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    # Create directory structure
    dirs = ["fonts/serif/A", "fonts/sans-serif/A", "fonts/script/A"]
    for dir_path in dirs:
        (dataset_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create sample data files
    for font_style in ["serif", "sans-serif", "script"]:
        char_dir = dataset_dir / "fonts" / font_style / "A"
        
        # Sample character data
        sample_data = {
            "character": "A",
            "font_style": font_style,
            "image_path": str(char_dir / "rendered.png"),
            "style_image_path": str(char_dir / "style.png"),
            "mask_path": str(char_dir / "mask.png"),
            "bezier_curves": [
                [[0.5, 0.1], [0.2, 0.9], [0.8, 0.9], [0.3, 0.5]],  # Triangle shape for 'A'
                [[0.3, 0.5], [0.7, 0.5], [0.65, 0.6], [0.35, 0.6]]  # Crossbar
            ],
            "style_prompt": f"{font_style} typography letter A, clean and elegant"
        }
        
        # Save metadata
        with open(char_dir / "metadata.json", 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"✓ Created sample data for {font_style} 'A'")
    
    # Create dataset manifest
    manifest = {
        "dataset_name": "Sample Font Dataset",
        "version": "1.0",
        "characters": ["A"],
        "font_styles": ["serif", "sans-serif", "script"],
        "total_samples": 3,
        "format": "bezier_font_v1"
    }
    
    with open(dataset_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Dataset manifest created")
    print(f"✓ Sample dataset ready at: {dataset_dir}")
    return str(dataset_dir)

def create_training_configs():
    """Create training configuration files."""
    print("\n⚙️  Creating Training Configurations...")
    print("=" * 50)
    
    configs_dir = Path("training_configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Development config (fast testing)
    dev_config = {
        "experiment_name": "bezier_adapter_dev_test",
        "total_steps": 100,
        "model": {
            "flux_model_name": "flux-dev-fill",
            "is_fill_model": True,
            "condition_adapter": {
                "mask_input_channels": 320
            }
        },
        "data": {
            "batch_size": 1,
            "image_size": 512
        },
        "optimization": {
            "learning_rate": 1e-4
        },
        "validate_every": 25,
        "save_every": 50,
        "log_every": 10
    }
    
    with open(configs_dir / "development.json", 'w') as f:
        json.dump(dev_config, f, indent=2)
    
    # Production config
    prod_config = {
        "experiment_name": "bezier_adapter_production",
        "total_steps": 50000,
        "model": {
            "flux_model_name": "flux-dev-fill",
            "is_fill_model": True,
            "mixed_precision": True,
            "use_gradient_checkpointing": True
        },
        "data": {
            "batch_size": 2,
            "image_size": 512
        },
        "optimization": {
            "learning_rate": 1e-4,
            "warmup_steps": 1000
        },
        "validate_every": 500,
        "save_every": 2500,
        "log_every": 50
    }
    
    with open(configs_dir / "production.json", 'w') as f:
        json.dump(prod_config, f, indent=2)
    
    print("✓ Development config: training_configs/development.json")
    print("✓ Production config: training_configs/production.json")
    
    return str(configs_dir)

def print_training_commands(dataset_path, configs_path):
    """Print example training commands."""
    print("\n🚀 Training Commands...")
    print("=" * 50)
    
    print("1. Development Training (Quick Test):")
    print(f"   python train_fill_model.py \\")
    print(f"     --data_root {dataset_path} \\")
    print(f"     --config_type custom \\")
    print(f"     --config_file {configs_path}/development.json \\")
    print(f"     --output_dir outputs/dev_test")
    print()
    
    print("2. Production Training (Full Scale):")
    print(f"   python train_fill_model.py \\")
    print(f"     --data_root {dataset_path} \\")
    print(f"     --config_type custom \\")
    print(f"     --config_file {configs_path}/production.json \\")
    print(f"     --output_dir outputs/production_training \\")
    print(f"     --mixed_precision")
    print()
    
    print("3. Resume Training:")
    print(f"   python train_fill_model.py \\")
    print(f"     --data_root {dataset_path} \\")
    print(f"     --config_type custom \\")
    print(f"     --config_file {configs_path}/production.json \\")
    print(f"     --resume_from outputs/production_training/checkpoints/best_model.pt")
    print()

def main():
    """Main setup function."""
    print("🔧 FLUX Fill Training Environment Setup")
    print("=" * 60)
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met. Please install required components.")
        return 1
    
    # Check model files
    models_ready = check_model_files()
    
    # Create sample dataset
    dataset_path = create_sample_dataset()
    
    # Create training configs
    configs_path = create_training_configs()
    
    # Print training commands
    print_training_commands(dataset_path, configs_path)
    
    # Summary
    print("\n✅ Setup Summary:")
    print("=" * 50)
    print("✓ System requirements checked")
    if models_ready:
        print("✓ Model files present")
    else:
        print("⚠️  Model files need to be downloaded")
    print("✓ Sample dataset created")
    print("✓ Training configurations created")
    print("✓ Ready to start training!")
    
    if not models_ready:
        print("\n⚠️  Next Step: Download FLUX.1-Fill-dev model files")
        print("   Run: python download_models.py")
    else:
        print("\n🚀 Next Step: Start training with the commands above")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)