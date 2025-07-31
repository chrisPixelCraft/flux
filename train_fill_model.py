#!/usr/bin/env python3
"""
Training script for BezierAdapter with FLUX.1-Fill-dev model.

This script demonstrates how to train the BezierAdapter framework with the enhanced
FLUX.1-Fill-dev model for font generation and style transfer with inpainting capabilities.
"""

import sys
from pathlib import Path
import argparse
import logging

# Add source path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from flux.util import configs
from flux.modules.bezier_flux_model import FluxBezierAdapter, BezierAdapterConfig
from flux.modules.conditioner import HFEmbedder
from flux.training.config import get_fill_model_config, get_development_config, TrainingConfig
from flux.training.dataset import BezierFontDataset, split_dataset
from flux.training.trainer import BezierAdapterTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train BezierAdapter with FLUX.1-Fill-dev")
    
    parser.add_argument("--data_root", type=str, required=True,
                       help="Path to preprocessed font dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/fill_model_training",
                       help="Output directory for training artifacts")
    parser.add_argument("--config_type", type=str, choices=["development", "full", "custom"],
                       default="development", help="Training configuration type")
    parser.add_argument("--config_file", type=str, default=None,
                       help="Path to custom config file (if config_type=custom)")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="flux-dev-fill",
                       help="FLUX model variant to use")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Override batch size")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Override learning rate")
    parser.add_argument("--total_steps", type=int, default=None,
                       help="Override total training steps")
    
    # System arguments
    parser.add_argument("--distributed", action="store_true",
                       help="Use distributed training")
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                       help="Use mixed precision training")
    parser.add_argument("--compile_model", action="store_true",
                       help="Use torch.compile for optimization")
    
    return parser.parse_args()


def setup_logging(output_dir: str):
    """Setup logging configuration."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_model(model_name: str, device: str = "cuda") -> FluxBezierAdapter:
    """
    Load FLUX model with BezierAdapter integration.
    
    Args:
        model_name: FLUX model variant name
        device: Target device
        
    Returns:
        FluxBezierAdapter model instance
    """
    # Load FLUX configuration
    flux_config = configs.get(model_name)
    if flux_config is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create BezierAdapter configuration
    bezier_config = BezierAdapterConfig(
        # Enhanced configuration for Fill model
        hook_layers=[7, 8, 9, 10, 11, 12],  # Integration layers
        bezier_output_dim=1536,  # Match condition adapter hidden dim
        density_output_resolution=(64, 64),  # VAE latent resolution
        style_transfer_enabled=True,  # Enable style fusion for Fill model
        kde_bandwidth_init=0.1,
        use_gradient_checkpointing=True,
        lora_rank=64
    )
    
    # Create model
    model = FluxBezierAdapter(flux_config.params, bezier_config)
    
    print(f"Loaded {model_name} model:")
    print(f"  Input channels: {flux_config.params.in_channels}")
    print(f"  Fill model: {model.is_fill_model}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    bezier_params = sum(p.numel() for p in model.bezier_adapter.parameters()) if hasattr(model, 'bezier_adapter') else 0
    print(f"  BezierAdapter parameters: {bezier_params:,}")
    
    return model.to(device)


def create_datasets(data_root: str, config: TrainingConfig):
    """
    Create training and validation datasets.
    
    Args:
        data_root: Path to dataset root
        config: Training configuration
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Check if data directory exists
    data_path = Path(data_root)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_root}")
    
    # Create datasets with Fill model configuration
    train_dataset, val_dataset, _ = split_dataset(
        data_root=data_root,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,  # Skip test split for training
        seed=42,
        is_fill_model=config.model.is_fill_model,
        mask_channels=config.model.condition_adapter.mask_input_channels
    )
    
    print(f"Created datasets:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Fill model: {train_dataset.is_fill_model}")
    print(f"  Mask channels: {train_dataset.mask_channels}")
    
    return train_dataset, val_dataset


def create_text_encoders(config: TrainingConfig, device: str = "cuda"):
    """
    Create text encoders for multi-modal conditioning.
    
    Args:
        config: Training configuration
        device: Target device
        
    Returns:
        Tuple of (clip_embedder, t5_embedder)
    """
    # CLIP embedder for style features
    clip_embedder = HFEmbedder(
        config.clip_model_name,
        max_length=config.max_text_length,
        device=device
    )
    
    # T5 embedder for text features
    t5_embedder = HFEmbedder(
        config.t5_model_name,
        max_length=512,  # Longer context for detailed descriptions
        device=device
    )
    
    print(f"Created text encoders:")
    print(f"  CLIP: {config.clip_model_name}")
    print(f"  T5: {config.t5_model_name}")
    
    return clip_embedder, t5_embedder


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("Starting BezierAdapter FLUX.1-Fill-dev training")
    
    # Load configuration
    if args.config_type == "development":
        config = get_development_config()
    elif args.config_type == "full":
        config = get_fill_model_config()
    elif args.config_type == "custom":
        if args.config_file is None:
            raise ValueError("Custom config file required when config_type=custom")
        config = TrainingConfig.from_file(args.config_file)
    else:
        raise ValueError(f"Unknown config type: {args.config_type}")
    
    # Apply command line overrides
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.optimization.learning_rate = args.learning_rate
    if args.total_steps is not None:
        config.total_steps = args.total_steps
    
    # Update system configuration
    config.distributed = args.distributed
    config.mixed_precision = args.mixed_precision
    config.compile_model = args.compile_model
    config.output_dir = args.output_dir
    
    # Validate configuration
    issues = config.validate()
    if issues:
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return 1
    
    # Print configuration summary
    config.print_summary()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA not available, using CPU (training will be very slow)")
        config.mixed_precision = False
    
    logger.info(f"Using device: {device}")
    
    try:
        # Load model
        logger.info("Loading FLUX.1-Fill-dev model...")
        model = load_model(args.model_name, device)
        
        # Create datasets
        logger.info("Creating datasets...")
        train_dataset, val_dataset = create_datasets(args.data_root, config)
        
        # Create text encoders
        logger.info("Loading text encoders...")
        clip_embedder, t5_embedder = create_text_encoders(config, device)
        
        # Create trainer
        logger.info("Initializing trainer...")
        trainer = BezierAdapterTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            clip_embedder=clip_embedder,
            t5_embedder=t5_embedder,
            output_dir=args.output_dir,
            resume_from=args.resume_from,
            distributed=args.distributed
        )
        
        # Compile model if requested
        if args.compile_model:
            logger.info("Compiling model with torch.compile...")
            trainer.model = torch.compile(trainer.model)
        
        # Start training
        logger.info("Starting training loop...")
        trainer.train()
        
        logger.info("Training completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)