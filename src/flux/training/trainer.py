"""
BezierAdapterTrainer for multi-phase training pipeline.

Implements comprehensive training infrastructure with multi-loss optimization,
parameter-efficient training, and memory optimization for large-scale models.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import asdict
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..modules.models import MultiModalCondition
from .config import TrainingConfig
from ..modules.bezier_flux_model import FluxBezierAdapter
from ..modules.conditioner import HFEmbedder
from .losses import MultiLossTrainer, LossOutputs
from .dataset import BezierFontDataset, create_dataloader


class BezierAdapterTrainer:
    """
    Comprehensive trainer for BezierAdapter-FLUX integration.
    
    Handles multi-loss optimization, parameter-efficient training,
    memory optimization, and distributed training capabilities.
    """
    
    def __init__(self,
                 model: FluxBezierAdapter,
                 config: TrainingConfig,
                 train_dataset: BezierFontDataset,
                 val_dataset: Optional[BezierFontDataset] = None,
                 clip_embedder: Optional[HFEmbedder] = None,
                 t5_embedder: Optional[HFEmbedder] = None,
                 output_dir: str = "outputs",
                 resume_from: Optional[str] = None,
                 distributed: bool = False):
        """
        Initialize BezierAdapterTrainer.
        
        Args:
            model: FluxBezierAdapter model instance
            config: Training configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            clip_embedder: CLIP text embedder for style features
            t5_embedder: T5 text embedder for text features
            output_dir: Directory for saving outputs
            resume_from: Path to checkpoint for resuming training
            distributed: Whether to use distributed training
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.clip_embedder = clip_embedder
        self.t5_embedder = t5_embedder
        self.output_dir = Path(output_dir)
        self.distributed = distributed
        
        # Setup directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize training components
        self._setup_device()
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        self._setup_data_loaders()
        
        # Mixed precision training
        self.scaler = GradScaler(enabled=config.mixed_precision)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Tensorboard logging
        self.writer = SummaryWriter(self.output_dir / "logs")
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        self.logger.info(f"Initialized BezierAdapterTrainer")
        self.logger.info(f"Trainable parameters: {self.count_trainable_parameters():,}")
        self.logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_device(self):
        """Setup device and distributed training."""
        if self.distributed:
            # Initialize distributed training
            dist.init_process_group(backend='nccl')
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.local_rank = 0
            self.world_size = 1
        
        self.logger.info(f"Using device: {self.device}")
    
    def _setup_model(self):
        """Setup model for training."""
        self.model = self.model.to(self.device)
        
        # Verify Fill model configuration
        if hasattr(self.config.model, 'is_fill_model') and self.config.model.is_fill_model:
            if not hasattr(self.model, 'is_fill_model') or not self.model.is_fill_model:
                self.logger.warning("Config specifies Fill model but model doesn't support it")
        
        # Freeze FLUX backbone if specified
        if getattr(self.config.model, 'freeze_flux_backbone', True):
            self._freeze_flux_backbone()
        
        # Wrap model for distributed training
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
    
    def _freeze_flux_backbone(self):
        """Freeze FLUX backbone parameters, keeping only BezierAdapter trainable."""
        frozen_count = 0
        trainable_count = 0
        
        for name, param in self.model.named_parameters():
            if 'bezier' in name.lower() or 'adapter' in name.lower():
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()
        
        self.logger.info(f"Frozen parameters: {frozen_count:,}")
        self.logger.info(f"Trainable parameters: {trainable_count:,}")
    
    def _setup_optimizer(self):
        """Setup optimizer for trainable parameters."""
        # Only optimize parameters that require gradients
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                # Linear warmup
                return step / self.config.warmup_steps
            else:
                # Cosine annealing
                progress = (step - self.config.warmup_steps) / (self.config.total_steps - self.config.warmup_steps)
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _setup_loss_function(self):
        """Setup multi-loss training function."""
        self.loss_function = MultiLossTrainer(self.config)
    
    def _setup_data_loaders(self):
        """Setup training and validation data loaders."""
        # Training data loader
        self.train_loader = create_dataloader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Validation data loader
        if self.val_dataset:
            self.val_loader = create_dataloader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
        else:
            self.val_loader = None
    
    def count_trainable_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def prepare_batch(self, batch: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Prepare batch for training by encoding text and organizing inputs.
        
        Args:
            batch: Batch from DataLoader
            
        Returns:
            Tuple of (model_inputs, targets) dictionaries
        """
        # Move tensors to device
        style_images = batch['style_images'].to(self.device)
        mask_images = batch['mask_images'].to(self.device)
        bezier_features = batch['bezier_features'].to(self.device)
        bezier_masks = batch['bezier_masks'].to(self.device)
        
        batch_size = style_images.shape[0]
        
        # Encode text features if embedders available
        style_features = None
        text_features = None
        
        if self.clip_embedder:
            with torch.no_grad():
                style_features = self.clip_embedder(batch['style_prompts'])
                style_features = style_features.to(self.device)
        
        if self.t5_embedder:
            with torch.no_grad():
                # Create detailed text descriptions
                text_prompts = [
                    f"Generate the character '{char}' in {style}" 
                    for char, style in zip(batch['characters'], batch['style_prompts'])
                ]
                text_features = self.t5_embedder(text_prompts)
                text_features = text_features.to(self.device)
        
        # Handle Fill model's extended input format
        is_fill_model = getattr(self.config.model, 'is_fill_model', False)
        
        if is_fill_model:
            # Fill model expects 384-channel input (64 base + 320 conditioning)
            base_latents = torch.randn(batch_size, 64, 64, 64, device=self.device)  # Base FLUX latents
            
            # Use the enhanced mask conditioning from dataset (320 channels)
            if mask_images.shape[1] == 320:
                # Mask images already prepared as 320-channel conditioning
                extended_conditioning = mask_images
            else:
                # Fallback: expand mask to 320 channels
                extended_conditioning = mask_images.repeat(1, 320//mask_images.shape[1], 1, 1)[:, :320]
            
            # Combine base + conditioning = 384 channels
            combined_input = torch.cat([base_latents, extended_conditioning], dim=1)
            target_images = combined_input
        else:
            # Standard FLUX with 64 channels
            target_images = torch.randn(batch_size, 64, 64, 64, device=self.device)  # Standard VAE latent space
        
        # Create multi-modal conditions
        conditions = MultiModalCondition(
            style_features=style_features,
            text_features=text_features,
            mask_features=mask_images,
            bezier_features=bezier_features
        )
        
        # Create model inputs
        model_inputs = {
            'img': target_images + torch.randn_like(target_images) * 0.1,  # Add noise
            'img_ids': torch.zeros(batch_size, 64*64, 3, device=self.device),
            'txt': text_features if text_features is not None else torch.zeros(batch_size, 512, 4096, device=self.device),
            'txt_ids': torch.zeros(batch_size, 512, 3, device=self.device),
            'bezier_conditions': conditions,
            'timesteps': torch.rand(batch_size, device=self.device),
            'guidance': torch.full((batch_size,), 4.0, device=self.device)
        }
        
        # Create targets for multi-loss training
        targets = {
            'diffusion_target': target_images,
            'bezier_points': batch['bezier_points'].to(self.device),
            'style_images': style_images,
            'generated_images': target_images  # Will be replaced with model output
        }
        
        # Add inpainting mask for Fill model
        if is_fill_model:
            # Extract binary mask from extended conditioning for inpainting loss
            if mask_images.shape[1] == 320:
                # Use the last 4 channels as the binary mask (standard VAE mask format)
                inpainting_mask = mask_images[:, -4:].mean(dim=1, keepdim=True)
                targets['inpainting_mask'] = (inpainting_mask > 0.5).float()
            else:
                # Use the mask directly if it's simple format
                targets['inpainting_mask'] = (mask_images > 0.5).float()
        
        return model_inputs, targets
    
    def training_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Execute a single training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary of loss values and metrics
        """
        # Prepare batch
        model_inputs, targets = self.prepare_batch(batch)
        
        # Forward pass with mixed precision
        with autocast(device_type='cuda', enabled=self.config.mixed_precision):
            # Model forward pass
            model_output = self.model(**model_inputs)
            
            # Prepare outputs for loss calculation
            model_outputs = {
                'diffusion_output': model_output,
                'generated_images': model_output,
                # Add other outputs from BezierAdapter components
            }
            
            # Compute multi-loss
            loss_outputs = self.loss_function(model_outputs, targets)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss_outputs.total_loss).backward()
        
        # Gradient clipping
        if self.config.gradient_clipping > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.gradient_clipping
            )
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        # Update learning rate
        self.scheduler.step()
        
        # Convert tensors to floats for logging
        metrics = {
            'total_loss': loss_outputs.total_loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        if loss_outputs.diffusion_loss is not None:
            metrics['diffusion_loss'] = loss_outputs.diffusion_loss.item()
        if loss_outputs.density_loss is not None:
            metrics['density_loss'] = loss_outputs.density_loss.item()
        if loss_outputs.style_loss is not None:
            metrics['style_loss'] = loss_outputs.style_loss.item()
        
        # Add custom metrics
        metrics.update({k: v for k, v in loss_outputs.metrics.items() if isinstance(v, (int, float))})
        
        return metrics
    
    def validation_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Execute a single validation step.
        
        Args:
            batch: Validation batch
            
        Returns:
            Dictionary of validation metrics
        """
        with torch.no_grad():
            model_inputs, targets = self.prepare_batch(batch)
            
            # Forward pass
            model_output = self.model(**model_inputs)
            
            # Prepare outputs for loss calculation
            model_outputs = {
                'diffusion_output': model_output,
                'generated_images': model_output,
            }
            
            # Compute loss
            loss_outputs = self.loss_function(model_outputs, targets)
            
            # Convert to metrics
            metrics = {
                'val_total_loss': loss_outputs.total_loss.item(),
            }
            
            if loss_outputs.diffusion_loss is not None:
                metrics['val_diffusion_loss'] = loss_outputs.diffusion_loss.item()
            if loss_outputs.density_loss is not None:
                metrics['val_density_loss'] = loss_outputs.density_loss.item()
            if loss_outputs.style_loss is not None:
                metrics['val_style_loss'] = loss_outputs.style_loss.item()
        
        return metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_metrics = {}
        num_batches = len(self.train_loader)
        
        for step, batch in enumerate(self.train_loader):
            start_time = time.time()
            
            # Training step
            step_metrics = self.training_step(batch)
            
            # Update global step
            self.global_step += 1
            
            # Accumulate metrics
            for key, value in step_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
            
            # Log metrics
            step_time = time.time() - start_time
            
            if step % 10 == 0:  # Log every 10 steps
                self.logger.info(
                    f"Epoch {self.epoch}, Step {step}/{num_batches}, "
                    f"Loss: {step_metrics['total_loss']:.4f}, "
                    f"LR: {step_metrics['learning_rate']:.2e}, "
                    f"Time: {step_time:.3f}s"
                )
                
                # Tensorboard logging
                for key, value in step_metrics.items():
                    self.writer.add_scalar(f"train/{key}", value, self.global_step)
            
            # Update loss weights dynamically
            self.loss_function.update_loss_weights(self.global_step)
        
        # Average metrics over epoch
        averaged_metrics = {
            key: sum(values) / len(values) 
            for key, values in epoch_metrics.items()
        }
        
        return averaged_metrics
    
    def validate(self) -> Dict[str, float]:
        """Run validation loop."""
        if not self.val_loader:
            return {}
            
        self.model.eval()
        
        val_metrics = {}
        num_batches = len(self.val_loader)
        
        for step, batch in enumerate(self.val_loader):
            step_metrics = self.validation_step(batch)
            
            # Accumulate metrics
            for key, value in step_metrics.items():
                if key not in val_metrics:
                    val_metrics[key] = []
                val_metrics[key].append(value)
        
        # Average metrics
        averaged_metrics = {
            key: sum(values) / len(values)
            for key, values in val_metrics.items()
        }
        
        return averaged_metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': asdict(self.config),
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint at epoch {self.epoch}")
        
        # Keep only last 3 checkpoints to save space
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > 3:
            for old_checkpoint in checkpoints[:-3]:
                old_checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Resumed training from epoch {self.epoch}, step {self.global_step}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.epoch, self.config.total_steps // len(self.train_loader)):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Log epoch results
            self.logger.info(f"Epoch {epoch} completed:")
            self.logger.info(f"  Train Loss: {train_metrics.get('total_loss', 0.0):.4f}")
            if val_metrics:
                self.logger.info(f"  Val Loss: {val_metrics.get('val_total_loss', 0.0):.4f}")
            
            # Tensorboard logging
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"epoch/{key}", value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"epoch/{key}", value, epoch)
            
            # Save checkpoint
            is_best = False
            if val_metrics and val_metrics.get('val_total_loss', float('inf')) < self.best_val_loss:
                self.best_val_loss = val_metrics['val_total_loss']
                is_best = True
            
            if epoch % 5 == 0 or is_best:  # Save every 5 epochs or if best
                self.save_checkpoint(is_best)
            
            # Early stopping check (optional)
            if self.global_step >= self.config.total_steps:
                self.logger.info("Reached maximum training steps. Stopping.")
                break
        
        self.logger.info("Training completed!")
        self.writer.close()