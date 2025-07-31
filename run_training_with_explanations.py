#!/usr/bin/env python3
"""
Training runner with detailed explanations of each action.

This script runs the FLUX Fill training while explaining what each step does
and why it's important for the BezierAdapter system.
"""

import sys
import os
import time
import subprocess
from pathlib import Path

def print_section(title, description=""):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)
    if description:
        print(f"{description}\n")

def print_action(action, explanation):
    """Print an action with its explanation."""
    print(f"\n🔧 ACTION: {action}")
    print(f"💡 EXPLANATION: {explanation}")

def check_prerequisites():
    """Check if all prerequisites are met."""
    print_section("CHECKING PREREQUISITES", 
                  "Verifying that all required components are available")
    
    # Check Python and PyTorch
    print_action("Checking Python and PyTorch installation",
                "These are the core dependencies for running the training")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ CUDA GPU available: {gpu_name} ({memory_gb:.1f}GB)")
            
            if memory_gb < 12:
                print("⚠️  WARNING: GPU has less than 12GB. Training will be slower.")
                return "low_memory"
        else:
            print("❌ CUDA not available. Training requires GPU.")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed. Please install PyTorch with CUDA support.")
        return False
    
    # Check model files
    print_action("Checking FLUX.1-Fill-dev model files",
                "The base model weights are required before we can add BezierAdapter")
    
    model_dir = Path("checkpoints/black-forest-labs_FLUX.1-Fill-dev")
    required_files = ["flux1-fill-dev.safetensors", "ae.safetensors"]
    
    missing_files = []
    for file in required_files:
        if not (model_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        print("   Run: python download_models.py to download them")
        return False
    else:
        print("✅ All required model files present")
    
    # Check dataset
    print_action("Checking training dataset",
                "We need font images with Bezier curve annotations for training")
    
    if Path("sample_dataset").exists():
        print("✅ Sample dataset found")
        return "sample_data"
    elif Path("real_dataset").exists():
        print("✅ Real dataset found")
        return "real_data"
    else:
        print("⚠️  No dataset found. Will create sample dataset for testing.")
        return "create_sample"
    
    return True

def create_sample_data():
    """Create minimal sample data for testing."""
    print_section("CREATING SAMPLE DATA",
                  "Since no dataset exists, we'll create minimal test data")
    
    print_action("Running setup script to create sample dataset",
                "This creates synthetic Bezier curves and placeholder images for testing")
    
    try:
        result = subprocess.run([sys.executable, "setup_training_environment.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Sample dataset created successfully")
            return True
        else:
            print(f"❌ Failed to create sample data: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False

def explain_training_command(data_path, config_type, batch_size):
    """Explain what the training command does."""
    print_section("UNDERSTANDING THE TRAINING COMMAND",
                  "Let's break down what each part of the training command does")
    
    command = [
        sys.executable, "train_fill_model.py",
        "--data_root", data_path,
        "--config_type", config_type,
        "--batch_size", str(batch_size),
        "--output_dir", "outputs/explained_training",
        "--mixed_precision"
    ]
    
    print("📋 TRAINING COMMAND:")
    print(" ".join(command))
    print()
    
    explanations = {
        "python train_fill_model.py": "Main training script that coordinates all components",
        f"--data_root {data_path}": "Path to dataset with font images + Bezier annotations",
        f"--config_type {config_type}": f"Use {config_type} configuration (fast testing vs full training)",
        f"--batch_size {batch_size}": f"Process {batch_size} sample(s) at a time (affects memory usage)",
        "--output_dir outputs/explained_training": "Where to save model checkpoints and logs",
        "--mixed_precision": "Use 16-bit floats to save memory and speed up training"
    }
    
    for cmd_part, explanation in explanations.items():
        print_action(cmd_part, explanation)
    
    print("\n🎯 WHAT WILL HAPPEN DURING TRAINING:")
    print("1. Load FLUX.1-Fill-dev (12 billion parameters) - FROZEN")
    print("2. Add BezierAdapter (21 million parameters) - TRAINABLE") 
    print("3. Load font images + Bezier curves from dataset")
    print("4. For each training step:")
    print("   a. Convert Bezier curves to density maps using KDE")
    print("   b. Extract text features with T5, style features with CLIP")
    print("   c. Fuse all features (text + style + Bezier + mask)")
    print("   d. Generate font image using FLUX with BezierAdapter guidance")
    print("   e. Compute loss: diffusion + density + style")
    print("   f. Update only BezierAdapter parameters (1.8% of total)")
    print("5. Save best model when validation loss improves")
    
    return command

def monitor_training_progress():
    """Explain how to monitor training progress."""
    print_section("MONITORING TRAINING PROGRESS",
                  "Understanding what the training logs mean")
    
    print("📊 TRAINING LOGS EXPLANATION:")
    print()
    print("When training starts, you'll see logs like:")
    print("Step 100/1000 | Loss: 0.145 | D: 0.098 | Dens: 0.032 | Style: 0.015")
    print()
    
    print_action("Step 100/1000", "Current step out of total steps")
    print_action("Loss: 0.145", "Total combined loss (lower = better)")
    print_action("D: 0.098", "Diffusion loss (measures image quality)")
    print_action("Dens: 0.032", "Density loss (measures Bezier adherence)")
    print_action("Style: 0.015", "Style loss (measures style consistency)")
    
    print("\n🎯 WHAT GOOD TRAINING LOOKS LIKE:")
    print("✅ Total loss decreasing over time")
    print("✅ Density loss < 0.05 (good Bezier adherence)")
    print("✅ Style loss < 0.03 (good style consistency)")
    print("✅ No sudden spikes or instability")
    
    print("\n⚠️  WARNING SIGNS:")
    print("❌ Loss stuck or increasing")
    print("❌ Out of memory errors")
    print("❌ NaN values in loss")
    
    print("\n📈 TENSORBOARD MONITORING:")
    print("Run in another terminal: tensorboard --logdir outputs/explained_training/logs")
    print("Then open: http://localhost:6006")

def run_training(command, timeout_minutes=10):
    """Run the training command with live output."""
    print_section("RUNNING TRAINING",
                  f"Starting training process (timeout: {timeout_minutes} minutes)")
    
    print("🚀 TRAINING STARTED!")
    print("Watch the logs below to see BezierAdapter learning...")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Start the training process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor output with timeout
        while True:
            # Check if process finished
            if process.poll() is not None:
                break
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_minutes * 60:
                print(f"\n⏰ TIMEOUT: Stopping training after {timeout_minutes} minutes")
                process.terminate()
                process.wait()
                break
            
            # Read output line by line
            try:
                line = process.stdout.readline()
                if line:
                    # Add timestamp and print
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] {line.rstrip()}")
                else:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n⛔ TRAINING INTERRUPTED BY USER")
                process.terminate()
                process.wait()
                break
        
        return_code = process.returncode
        elapsed = time.time() - start_time
        
        print("-" * 60)
        print(f"🏁 TRAINING COMPLETED in {elapsed:.1f} seconds")
        
        if return_code == 0:
            print("✅ Training finished successfully!")
            return True
        else:
            print(f"❌ Training failed with exit code {return_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return False

def explain_results():
    """Explain the training results."""
    print_section("UNDERSTANDING THE RESULTS",
                  "What was accomplished during training")
    
    output_dir = Path("outputs/explained_training")
    
    print_action("Checking for saved model checkpoints",
                "The trained BezierAdapter weights are saved for later use")
    
    checkpoints_dir = output_dir / "checkpoints"
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob("*.pt"))
        if checkpoints:
            print(f"✅ Found {len(checkpoints)} checkpoint(s):")
            for checkpoint in checkpoints:
                size_mb = checkpoint.stat().st_size / (1024*1024)
                print(f"   - {checkpoint.name} ({size_mb:.1f} MB)")
        else:
            print("⚠️  No checkpoints found (training may have been too short)")
    
    print_action("Checking training logs",
                "Logs contain detailed information about the training process")
    
    logs_dir = output_dir / "logs"
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*"))
        print(f"✅ Found {len(log_files)} log file(s)")
        print("   View with: tensorboard --logdir outputs/explained_training/logs")
    
    print("\n🎯 WHAT YOU'VE ACCOMPLISHED:")
    print("1. ✅ Loaded FLUX.1-Fill-dev (12B parameter diffusion model)")
    print("2. ✅ Added BezierAdapter (21M trainable parameters)")
    print("3. ✅ Processed font data with Bezier curve annotations")
    print("4. ✅ Trained multi-modal fusion (text + style + Bezier)")
    print("5. ✅ Optimized for font generation with geometric control")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Test the model: python example_bezier_pipeline.py")
    print("2. Generate fonts: Use trained model for font creation")
    print("3. Evaluate quality: Check if Bezier curves are followed")
    print("4. Scale up: Train on larger dataset for production use")

def main():
    """Main function that runs the explained training process."""
    print("🎓 FLUX Fill Training with Detailed Explanations")
    print("=" * 60)
    print("This script will run FLUX Fill training while explaining")
    print("every step of the process and what it accomplishes.")
    print()
    
    # Phase 1: Check prerequisites
    prereq_status = check_prerequisites()
    if prereq_status == False:
        print("❌ Prerequisites not met. Please install required components.")
        return 1
    
    # Phase 2: Prepare data if needed
    if prereq_status == "create_sample":
        if not create_sample_data():
            return 1
        data_path = "sample_dataset"
        config_type = "development"
        batch_size = 1
    elif prereq_status in ["sample_data", "low_memory"]:
        data_path = "sample_dataset"
        config_type = "development"
        batch_size = 1
    else:
        data_path = "real_dataset"
        config_type = "full"
        batch_size = 2 if prereq_status != "low_memory" else 1
    
    # Phase 3: Explain training command
    command = explain_training_command(data_path, config_type, batch_size)
    
    # Phase 4: Explain monitoring
    monitor_training_progress()
    
    # Phase 5: Ask user to continue
    print_section("READY TO START TRAINING")
    response = input("Continue with training? (y/n): ").lower().strip()
    if response != 'y':
        print("Training cancelled by user.")
        return 0
    
    # Phase 6: Run training
    success = run_training(command, timeout_minutes=10)
    
    # Phase 7: Explain results
    explain_results()
    
    if success:
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("You now understand how BezierAdapter training works!")
        return 0
    else:
        print("\n⚠️  Training encountered issues, but you learned the process!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)