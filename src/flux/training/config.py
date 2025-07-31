"""
Training configuration for BezierAdapter-FLUX integration.

Provides comprehensive configuration classes for all aspects of training,
including model parameters, loss weights, optimization settings, and scheduling.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import torch


@dataclass
class BezierProcessorConfig:
    """Configuration for BezierParameterProcessor."""
    output_resolution: tuple[int, int] = (64, 64)
    hidden_dim: int = 128
    kde_bandwidth_init: float = 0.1
    sigma_min: float = 0.01
    sigma_max: float = 1.0
    use_gradient_checkpointing: bool = True
    
    # Point embedding MLP configuration
    point_embed_dims: List[int] = field(default_factory=lambda: [2, 64, 128, 128, 128])
    dropout_rate: float = 0.1


@dataclass
class ConditionAdapterConfig:
    """Configuration for ConditionInjectionAdapter."""
    clip_dim: int = 768
    t5_dim: int = 4096
    hidden_dim: int = 1536
    lora_rank: int = 64
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # Branch-specific configurations
    style_branch_dims: List[int] = field(default_factory=lambda: [768, 1536])
    text_branch_dims: List[int] = field(default_factory=lambda: [4096, 1536])
    mask_conv_channels: int = 128  # Increased for Fill model (320->128 compression)
    bezier_mlp_dims: List[int] = field(default_factory=lambda: [3, 64, 256, 1536])
    
    # Fill model specific configurations
    mask_input_channels: int = 320  # Fill model mask conditioning channels


@dataclass
class SpatialFuserConfig:
    """Configuration for SpatialAttentionFuser."""
    feature_dim: int = 768
    num_heads: int = 12
    num_layers: int = 6
    dropout_rate: float = 0.1
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    
    # Position encoding configuration
    max_sequence_length: int = 4096  # 64x64 spatial resolution
    position_encoding_type: str = "rope"  # Compatible with FLUX


@dataclass
class StyleFusionConfig:
    """Configuration for StyleBezierFusionModule."""
    spatial_dim: int = 768
    style_dim: int = 1280
    num_attention_heads: int = 8
    adain_epsilon: float = 1e-5
    
    # Integration layers configuration
    integration_layers: List[int] = field(default_factory=lambda: [7, 8, 9, 10, 11, 12])
    residual_scale: float = 1.0


@dataclass
class LossConfig:
    """Configuration for multi-loss training."""
    # Loss weights
    diffusion_loss_weight: float = 1.0
    density_loss_weight: float = 0.5
    style_loss_weight: float = 0.3
    
    # Diffusion loss configuration
    diffusion_prediction_type: str = "flow"  # FLUX uses rectified flow
    diffusion_loss_type: str = "mse"
    diffusion_weighting_scheme: str = "uniform"
    
    # Density loss configuration
    density_kde_bandwidth: float = 0.1
    density_loss_type: str = "kl_div"  # "kl_div", "mse", "js_div"
    density_spatial_weight: float = 1.0
    
    # Style loss configuration
    style_layers: List[str] = field(default_factory=lambda: [
        'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'
    ])
    content_layers: List[str] = field(default_factory=lambda: ['conv4_2'])
    style_loss_weight: float = 1.0
    content_loss_weight: float = 0.1
    
    # Loss weight scheduling
    use_loss_scheduling: bool = True
    loss_weight_schedule: Dict[str, Any] = field(default_factory=lambda: {
        'diffusion': {'type': 'constant', 'value': 1.0},
        'density': {
            'type': 'linear',
            'start': 0.1,
            'end': 0.5,
            'start_step': 0,
            'end_step': 10000
        },
        'style': {
            'type': 'exponential',
            'base': 0.3,
            'decay': 0.999
        }
    })


@dataclass
class OptimizationConfig:
    """Configuration for optimization settings."""
    # Optimizer configuration
    optimizer_type: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Learning rate scheduling
    use_lr_scheduling: bool = True
    scheduler_type: str = "cosine_with_warmup"
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.01
    
    # Gradient configuration
    gradient_clipping: float = 1.0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


@dataclass
class ModelConfig:
    """Configuration for BezierAdapter model components."""
    # Component configurations
    bezier_processor: BezierProcessorConfig = field(default_factory=BezierProcessorConfig)
    condition_adapter: ConditionAdapterConfig = field(default_factory=ConditionAdapterConfig)
    spatial_fuser: SpatialFuserConfig = field(default_factory=SpatialFuserConfig)
    style_fusion: StyleFusionConfig = field(default_factory=StyleFusionConfig)
    
    # FLUX integration settings
    flux_model_name: str = "flux-dev-fill"  # "flux-dev", "flux-dev-fill", or "flux-schnell"
    freeze_flux_backbone: bool = True
    trainable_layer_pattern: str = "bezier|adapter"  # Regex pattern for trainable parameters
    
    # Fill model specific settings
    is_fill_model: bool = True  # Whether using FLUX.1-Fill-dev (384 channels)
    mask_conditioning_channels: int = 320  # Extended mask conditioning for Fill model
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    mixed_precision: bool = True
    precision_dtype: str = "fp16"  # "fp16" or "bf16"


@dataclass
class DataConfig:
    """Configuration for dataset and data loading."""
    # Dataset paths
    data_root: str = "data/bezier_fonts"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    
    # Data processing
    image_size: int = 512
    latent_size: int = 64  # FLUX VAE latent resolution
    max_bezier_points: int = 32
    min_bezier_points: int = 4
    
    # Data loading
    batch_size: int = 4  # Memory-optimized batch size
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_config: Dict[str, Any] = field(default_factory=lambda: {
        'horizontal_flip_prob': 0.3,
        'color_jitter': {
            'brightness': 0.1,
            'contrast': 0.1,
            'saturation': 0.1,
            'hue': 0.05
        },
        'rotation_degrees': 2,
        'bezier_perturbation': 0.05,
        'style_transforms': True
    })
    
    # Caching
    use_cache: bool = True
    cache_size: int = 1000


@dataclass
class TrainingConfig:
    """Main training configuration combining all components."""
    # Training parameters
    total_steps: int = 100000
    max_epochs: int = 1000
    validate_every: int = 1000
    save_every: int = 5000
    log_every: int = 100
    
    # Component configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    # Output and logging
    output_dir: str = "outputs/bezier_adapter_training"
    experiment_name: str = "bezier_adapter_flux"
    log_level: str = "INFO"
    
    # Checkpointing
    resume_from: Optional[str] = None
    save_top_k: int = 3
    monitor_metric: str = "val_total_loss"
    monitor_mode: str = "min"
    
    # Hardware configuration
    device: str = "cuda"
    distributed: bool = False
    num_gpus: int = 1
    mixed_precision: bool = True
    compile_model: bool = False  # torch.compile for optimization
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Evaluation and testing
    run_test_after_training: bool = True
    test_batch_size: int = 8
    num_test_samples: int = 100
    
    # Text encoder settings
    clip_model_name: str = "openai/clip-vit-large-patch14"
    t5_model_name: str = "google/t5-v1_1-xxl"
    max_text_length: int = 77
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate device configuration
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            self.mixed_precision = False
            print("Warning: CUDA not available, falling back to CPU")
        
        # Adjust batch size for distributed training
        if self.distributed and self.num_gpus > 1:
            self.data.batch_size = self.data.batch_size // self.num_gpus
        
        # Validate model configuration consistency
        if self.model.mixed_precision != self.mixed_precision:
            self.model.mixed_precision = self.mixed_precision
    
    @classmethod
    def from_file(cls, config_path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        import json
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def to_file(self, config_path: str):
        """Save configuration to JSON file."""
        import json
        from dataclasses import asdict
        
        config_dict = asdict(self)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_model_config_dict(self) -> Dict[str, Any]:
        """Get model configuration as dictionary for model initialization."""
        return {
            'bezier_processor': {
                'output_resolution': self.model.bezier_processor.output_resolution,
                'hidden_dim': self.model.bezier_processor.hidden_dim,
                'kde_bandwidth_init': self.model.bezier_processor.kde_bandwidth_init,
                'sigma_min': self.model.bezier_processor.sigma_min,
                'sigma_max': self.model.bezier_processor.sigma_max,
                'use_gradient_checkpointing': self.model.bezier_processor.use_gradient_checkpointing
            },
            'condition_adapter': {
                'clip_dim': self.model.condition_adapter.clip_dim,
                't5_dim': self.model.condition_adapter.t5_dim,
                'hidden_dim': self.model.condition_adapter.hidden_dim,
                'lora_rank': self.model.condition_adapter.lora_rank,
                'num_attention_heads': self.model.condition_adapter.num_attention_heads
            },
            'spatial_fuser': {
                'feature_dim': self.model.spatial_fuser.feature_dim,
                'num_heads': self.model.spatial_fuser.num_heads,
                'num_layers': self.model.spatial_fuser.num_layers,
                'use_flash_attention': self.model.spatial_fuser.use_flash_attention,
                'use_gradient_checkpointing': self.model.spatial_fuser.use_gradient_checkpointing
            },
            'style_fusion': {
                'spatial_dim': self.model.style_fusion.spatial_dim,
                'style_dim': self.model.style_fusion.style_dim,
                'num_attention_heads': self.model.style_fusion.num_attention_heads,
                'integration_layers': self.model.style_fusion.integration_layers
            }
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check batch size constraints
        if self.data.batch_size < 1:
            issues.append("Batch size must be at least 1")
        
        # Check learning rate bounds
        if self.optimization.learning_rate <= 0:
            issues.append("Learning rate must be positive")
        
        # Check gradient clipping
        if self.optimization.gradient_clipping <= 0:
            issues.append("Gradient clipping must be positive")
        
        # Check loss weights
        if self.loss.diffusion_loss_weight < 0:
            issues.append("Diffusion loss weight must be non-negative")
        if self.loss.density_loss_weight < 0:
            issues.append("Density loss weight must be non-negative")
        if self.loss.style_loss_weight < 0:
            issues.append("Style loss weight must be non-negative")
        
        # Check total steps
        if self.total_steps <= 0:
            issues.append("Total steps must be positive")
        
        # Check warmup steps
        if self.optimization.warmup_steps >= self.total_steps:
            issues.append("Warmup steps must be less than total steps")
        
        return issues
    
    def print_summary(self):
        """Print configuration summary."""
        print("=" * 60)
        print("BezierAdapter Training Configuration Summary")
        print("=" * 60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Output Dir: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.mixed_precision}")
        print(f"Distributed: {self.distributed} ({self.num_gpus} GPUs)")
        print()
        
        print("Training Parameters:")
        print(f"  Total Steps: {self.total_steps:,}")
        print(f"  Batch Size: {self.data.batch_size}")
        print(f"  Learning Rate: {self.optimization.learning_rate:.2e}")
        print(f"  Gradient Clipping: {self.optimization.gradient_clipping}")
        print()
        
        print("Model Configuration:")
        print(f"  FLUX Model: {self.model.flux_model_name}")
        print(f"  Freeze Backbone: {self.model.freeze_flux_backbone}")
        print(f"  Hidden Dim: {self.model.condition_adapter.hidden_dim}")
        print(f"  LoRA Rank: {self.model.condition_adapter.lora_rank}")
        print()
        
        print("Loss Configuration:")
        print(f"  Diffusion Weight: {self.loss.diffusion_loss_weight}")
        print(f"  Density Weight: {self.loss.density_loss_weight}")
        print(f"  Style Weight: {self.loss.style_loss_weight}")
        print()
        
        print("Data Configuration:")
        print(f"  Data Root: {self.data.data_root}")
        print(f"  Image Size: {self.data.image_size}")
        print(f"  Max Bézier Points: {self.data.max_bezier_points}")
        print(f"  Augmentation: {self.data.use_augmentation}")
        print("=" * 60)


# Predefined configurations for different scenarios
def get_development_config() -> TrainingConfig:
    """Get configuration optimized for development and testing."""
    config = TrainingConfig()
    config.total_steps = 1000
    config.data.batch_size = 2
    config.validate_every = 100
    config.save_every = 500
    config.log_every = 10
    config.experiment_name = "bezier_adapter_dev"
    return config


def get_full_training_config() -> TrainingConfig:
    """Get configuration for full-scale training."""
    config = TrainingConfig()
    config.total_steps = 100000
    config.data.batch_size = 4
    config.validate_every = 1000
    config.save_every = 5000
    config.log_every = 100
    config.experiment_name = "bezier_adapter_full"
    return config


def get_distributed_config(num_gpus: int = 4) -> TrainingConfig:
    """Get configuration for distributed training."""
    config = get_full_training_config()
    config.distributed = True
    config.num_gpus = num_gpus
    config.data.batch_size = 16  # Will be divided by num_gpus
    config.data.num_workers = 8
    config.experiment_name = f"bezier_adapter_distributed_{num_gpus}gpu"
    return config


def get_fill_model_config() -> TrainingConfig:
    """Get configuration optimized for FLUX.1-Fill-dev training."""
    config = TrainingConfig()
    
    # Model settings for Fill model
    config.model.flux_model_name = "flux-dev-fill"
    config.model.is_fill_model = True
    config.model.mask_conditioning_channels = 320
    
    # Condition adapter settings for Fill model
    config.model.condition_adapter.mask_input_channels = 320
    config.model.condition_adapter.mask_conv_channels = 128
    
    # Data settings optimized for inpainting
    config.data.image_size = 512  # Fill model native resolution
    config.data.latent_size = 64   # VAE latent size
    config.data.batch_size = 2     # Reduced for memory (384 channel model)
    
    # Loss configuration for inpainting
    config.loss.diffusion_loss_weight = 1.0
    config.loss.density_loss_weight = 0.3   # Reduced for inpainting focus
    config.loss.style_loss_weight = 0.5     # Increased for style transfer
    
    # Training parameters
    config.total_steps = 50000  # Focused training for Fill model
    config.validate_every = 500
    config.save_every = 2500
    config.log_every = 50
    
    # Memory optimization for larger model
    config.mixed_precision = True
    config.model.mixed_precision = True
    config.model.precision_dtype = "bf16"  # Better for Fill model
    config.model.use_gradient_checkpointing = True
    
    # Experiment naming
    config.experiment_name = "bezier_adapter_flux_fill"
    config.output_dir = "outputs/bezier_adapter_fill_training"
    
    return config