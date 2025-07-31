#!/usr/bin/env python3
"""
Main training entry point for BezierAdapter-FLUX integration.

This script provides a complete training pipeline with configuration management,
distributed training support, and comprehensive logging.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from flux.model import Flux
from flux.util import load_flow_model, load_clip, load_t5, configs
from flux.modules.bezier_flux_model import FluxBezierAdapter
from flux.training.config import TrainingConfig, get_development_config, get_full_training_config, get_distributed_config
from flux.training.trainer import BezierAdapterTrainer
from flux.training.dataset import BezierFontDataset, split_dataset


def setup_logging(output_dir: Path, log_level: str = "INFO"):
    """Setup logging configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "training.log")
    ]
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    # Set specific loggers
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def setup_distributed():
    """Setup distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        
        # Set device
        torch.cuda.set_device(local_rank)
        
        return True, rank, world_size, local_rank
    
    return False, 0, 1, 0


def load_pretrained_flux(model_name: str, device: str) -> Flux:
    """Load pretrained FLUX model."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading pretrained FLUX model: {model_name}")
    
    # Load model using FLUX utilities
    flux_model = load_flow_model(model_name, device=device, verbose=True)
    
    logger.info(f"FLUX model loaded successfully")
    logger.info(f"Total parameters: {sum(p.numel() for p in flux_model.parameters()):,}")
    
    return flux_model


def create_bezier_adapter_model(flux_model: Flux, config: TrainingConfig, device: str) -> FluxBezierAdapter:
    """Create BezierAdapter model from pretrained FLUX."""
    logger = logging.getLogger(__name__)
    
    logger.info("Creating BezierAdapter model...")
    
    # Get FLUX parameters
    flux_params = flux_model.params
    
    # Create BezierAdapter configuration
    bezier_config = config.get_model_config_dict()
    
    # Initialize BezierAdapter model
    bezier_model = FluxBezierAdapter(flux_params, bezier_config=bezier_config)
    
    # Copy FLUX weights
    bezier_model.load_state_dict(flux_model.state_dict(), strict=False)
    
    # Move to device
    bezier_model = bezier_model.to(device)
    
    logger.info("BezierAdapter model created successfully")
    logger.info(f"Total parameters: {sum(p.numel() for p in bezier_model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in bezier_model.parameters() if p.requires_grad):,}")
    
    return bezier_model


def load_datasets(config: TrainingConfig) -> tuple:
    """Load and split datasets."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading datasets from: {config.data.data_root}")
    
    # Check if data exists
    data_root = Path(config.data.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        str(data_root),
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )
    
    logger.info(f"Dataset loaded: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train BezierAdapter-FLUX integration")
    
    # Configuration arguments
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--config-type", type=str, choices=["dev", "full", "distributed"], 
                       default="dev", help="Predefined configuration type")
    parser.add_argument("--output-dir", type=str, help="Output directory override")
    parser.add_argument("--data-root", type=str, help="Data root directory override")
    
    # Model arguments
    parser.add_argument("--flux-model", type=str, default="flux-dev",
                       choices=list(configs.keys()), help="FLUX model variant")
    parser.add_argument("--resume-from", type=str, help="Resume training from checkpoint")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, help="Batch size override")
    parser.add_argument("--learning-rate", type=float, help="Learning rate override")
    parser.add_argument("--total-steps", type=int, help="Total training steps override")
    
    # System arguments
    parser.add_argument("--device", type=str, default="cuda", help="Training device")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for distributed training")
    parser.add_argument("--mixed-precision", action="store_true", default=True, help="Use mixed precision training")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for optimization")
    
    # Logging arguments
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    
    args = parser.parse_args()
    
    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()
    
    # Load configuration
    if args.config:
        config = TrainingConfig.from_file(args.config)
    elif args.config_type == "dev":
        config = get_development_config()
    elif args.config_type == "full":
        config = get_full_training_config()
    elif args.config_type == "distributed":
        config = get_distributed_config(args.num_gpus)
    else:
        config = TrainingConfig()
    
    # Apply argument overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.data_root:
        config.data.data_root = args.data_root
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.learning_rate:
        config.optimization.learning_rate = args.learning_rate
    if args.total_steps:
        config.total_steps = args.total_steps
    if args.resume_from:
        config.resume_from = args.resume_from
    
    config.device = args.device
    config.mixed_precision = args.mixed_precision
    config.compile_model = args.compile
    config.distributed = is_distributed
    config.num_gpus = world_size if is_distributed else args.num_gpus
    
    # Setup logging (only on main process for distributed)
    if rank == 0:
        logger = setup_logging(Path(config.output_dir), args.log_level if not args.quiet else "WARNING")
        logger.info("Starting BezierAdapter training...")
        config.print_summary()
        
        # Save configuration
        config.to_file(Path(config.output_dir) / "config.json")
        
        # Validate configuration
        issues = config.validate()
        if issues:
            logger.error("Configuration validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return 1
    else:
        logger = logging.getLogger(__name__)
    
    try:
        # Set device
        if config.device == "cuda":
            if is_distributed:
                device = f"cuda:{local_rank}"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = config.device
        
        if rank == 0:
            logger.info(f"Using device: {device}")
        
        # Load pretrained FLUX model
        flux_model = load_pretrained_flux(args.flux_model, device)
        
        # Create BezierAdapter model
        model = create_bezier_adapter_model(flux_model, config, device)
        
        # Compile model if requested (PyTorch 2.0+)
        if config.compile_model and hasattr(torch, 'compile'):
            if rank == 0:
                logger.info("Compiling model with torch.compile...")
            model = torch.compile(model)
        
        # Load text encoders
        if rank == 0:
            logger.info("Loading text encoders...")
        
        clip_embedder = load_clip(device)
        t5_embedder = load_t5(device, max_length=config.max_text_length)
        
        # Load datasets
        train_dataset, val_dataset, test_dataset = load_datasets(config)
        
        # Initialize trainer
        trainer = BezierAdapterTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            clip_embedder=clip_embedder,
            t5_embedder=t5_embedder,
            output_dir=config.output_dir,
            resume_from=config.resume_from,
            distributed=is_distributed
        )
        
        # Start training
        if rank == 0:
            logger.info("Starting training loop...")
        
        trainer.train()
        
        # Run test evaluation if requested
        if config.run_test_after_training and rank == 0:
            logger.info("Running test evaluation...")
            # TODO: Implement test evaluation
        
        if rank == 0:
            logger.info("Training completed successfully!")
        
        return 0
        
    except Exception as e:
        if rank == 0:
            logger.error(f"Training failed with error: {e}")
            logger.exception("Full traceback:")
        return 1
    
    finally:
        # Cleanup distributed training
        if is_distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())