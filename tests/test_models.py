#!/usr/bin/env python3
"""
Unit tests for Pydantic data models.

Tests validation, serialization, and type checking for all BezierAdapter data models.
"""

import pytest
import torch
import json
from pathlib import Path
import sys
from pydantic import ValidationError

# Add flux modules to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from flux.modules.models import (
    BezierControlPoints,
    DensityMapConfig,
    MultiModalCondition,
    TrainingConfig,
    BezierAdapterConfig
)


class TestBezierControlPoints:
    """Test suite for BezierControlPoints model."""
    
    def test_valid_bezier_points(self):
        """Test valid Bézier control points."""
        points = BezierControlPoints(
            points=[(0.0, 0.0), (0.33, 0.5), (0.67, 0.5), (1.0, 1.0)],
            curve_type="cubic",
            character="A",
            font_size=64.0
        )
        
        assert len(points.points) == 4
        assert points.curve_type == "cubic"
        assert points.character == "A"
        assert points.font_size == 64.0
    
    def test_invalid_points_count(self):
        """Test validation with too few points."""
        with pytest.raises(ValidationError, match="at least 4"):
            BezierControlPoints(
                points=[(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)],  # Only 3 points
                curve_type="cubic",
                character="A"
            )
    
    def test_invalid_curve_type(self):
        """Test validation with invalid curve type."""
        with pytest.raises(ValidationError, match="pattern"):
            BezierControlPoints(
                points=[(0.0, 0.0), (0.33, 0.5), (0.67, 0.5), (1.0, 1.0)],
                curve_type="invalid_type",
                character="A"
            )
    
    def test_invalid_character(self):
        """Test validation with invalid character."""
        with pytest.raises(ValidationError, match="at most 1"):
            BezierControlPoints(
                points=[(0.0, 0.0), (0.33, 0.5), (0.67, 0.5), (1.0, 1.0)],
                curve_type="cubic",
                character="AB"  # More than 1 character
            )
    
    def test_serialization(self):
        """Test JSON serialization/deserialization."""
        points = BezierControlPoints(
            points=[(0.0, 0.0), (0.33, 0.5), (0.67, 0.5), (1.0, 1.0)],
            curve_type="quadratic",
            character="中",
            font_size=128.0
        )
        
        # Serialize to dict
        data = points.model_dump()
        assert isinstance(data, dict)
        assert data["character"] == "中"
        
        # Serialize to JSON
        json_str = points.model_dump_json()
        assert isinstance(json_str, str)
        
        # Deserialize from JSON
        loaded = BezierControlPoints.model_validate_json(json_str)
        assert loaded.character == "中"
        assert loaded.font_size == 128.0


class TestDensityMapConfig:
    """Test suite for DensityMapConfig model."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DensityMapConfig()
        
        assert config.output_resolution == (64, 64)
        assert config.kde_bandwidth == 0.1
        assert config.sigma_min == 0.01
        assert config.sigma_max == 1.0
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = DensityMapConfig(
            output_resolution=(128, 128),
            kde_bandwidth=0.2,
            sigma_min=0.001,
            sigma_max=2.0
        )
        
        assert config.output_resolution == (128, 128)
        assert config.kde_bandwidth == 0.2
    
    def test_invalid_values(self):
        """Test validation with invalid values."""
        # Negative bandwidth
        with pytest.raises(ValidationError, match="greater than 0"):
            DensityMapConfig(kde_bandwidth=-0.1)
        
        # Negative sigma
        with pytest.raises(ValidationError, match="greater than 0"):
            DensityMapConfig(sigma_min=-0.01)


class TestMultiModalCondition:
    """Test suite for MultiModalCondition model."""
    
    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_full_condition(self, device):
        """Test condition with all modalities."""
        condition = MultiModalCondition(
            style_features=torch.randn(2, 768, device=device),
            text_features=torch.randn(2, 4096, device=device),
            mask_features=torch.randn(2, 4, 64, 64, device=device),
            bezier_features=torch.randn(2, 3, device=device)
        )
        
        assert condition.style_features is not None
        assert condition.text_features is not None
        assert condition.mask_features is not None
        assert condition.bezier_features is not None
    
    def test_partial_condition(self, device):
        """Test condition with some modalities."""
        condition = MultiModalCondition(
            style_features=torch.randn(2, 768, device=device),
            text_features=None,
            mask_features=None,
            bezier_features=torch.randn(2, 3, device=device)
        )
        
        assert condition.style_features is not None
        assert condition.text_features is None
        assert condition.bezier_features is not None
    
    def test_empty_condition(self):
        """Test empty condition creation."""
        condition = MultiModalCondition(
            style_features=None,
            text_features=None,
            mask_features=None,
            bezier_features=None
        )
        
        assert all(getattr(condition, field) is None for field in condition.model_fields)
    
    def test_tensor_validation(self, device):
        """Test that tensors are properly handled."""
        # Create tensors
        style_tensor = torch.randn(1, 768, device=device)
        
        condition = MultiModalCondition(
            style_features=style_tensor,
            text_features=None,
            mask_features=None,
            bezier_features=None
        )
        
        # Should preserve tensor properties
        assert isinstance(condition.style_features, torch.Tensor)
        assert condition.style_features.device == device


class TestTrainingConfig:
    """Test suite for TrainingConfig model."""
    
    def test_default_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        
        assert config.learning_rate == 1e-4
        assert config.batch_size == 4
        assert config.total_steps == 100000
        assert config.warmup_steps == 1000
        assert config.gradient_clipping == 1.0
        assert config.mixed_precision == True
        assert config.trainable_params_only == True
        
        # Multi-loss weights
        assert config.diffusion_loss_weight == 1.0
        assert config.density_loss_weight == 0.5
        assert config.style_loss_weight == 0.3
    
    def test_custom_config(self):
        """Test custom training configuration."""
        config = TrainingConfig(
            learning_rate=5e-5,
            batch_size=8,
            total_steps=50000,
            mixed_precision=False,
            diffusion_loss_weight=0.8,
            density_loss_weight=0.7,
            style_loss_weight=0.5
        )
        
        assert config.learning_rate == 5e-5
        assert config.batch_size == 8
        assert config.mixed_precision == False
        
        # Check loss weights
        assert config.diffusion_loss_weight == 0.8
        assert config.density_loss_weight == 0.7
        assert config.style_loss_weight == 0.5
    
    def test_invalid_config(self):
        """Test validation with invalid values."""
        # Negative learning rate
        with pytest.raises(ValidationError, match="greater than 0"):
            TrainingConfig(learning_rate=-1e-4)
        
        # Zero batch size
        with pytest.raises(ValidationError, match="greater than 0"):
            TrainingConfig(batch_size=0)
        
        # Negative loss weight
        with pytest.raises(ValidationError, match="greater than 0"):
            TrainingConfig(diffusion_loss_weight=-0.5)


class TestBezierAdapterConfig:
    """Test suite for BezierAdapterConfig model."""
    
    def test_default_config(self):
        """Test default adapter configuration."""
        config = BezierAdapterConfig()
        
        # Check defaults
        assert config.output_resolution == (64, 64)
        assert config.hidden_dim == 128
        assert config.clip_dim == 768
        assert config.t5_dim == 4096
        assert config.fusion_dim == 1536
        assert config.lora_rank == 64
        assert config.num_attention_heads == 12
        assert config.num_transformer_layers == 6
        assert config.spatial_dim == 1280
        assert config.style_fusion_heads == 8
    
    def test_custom_config(self):
        """Test custom adapter configuration."""
        config = BezierAdapterConfig(
            output_resolution=(128, 128),
            hidden_dim=256,
            clip_dim=1024,
            lora_rank=128,
            num_attention_heads=16
        )
        
        assert config.output_resolution == (128, 128)
        assert config.hidden_dim == 256
        assert config.clip_dim == 1024
        assert config.lora_rank == 128
        assert config.num_attention_heads == 16
    
    def test_invalid_dimensions(self):
        """Test validation with invalid dimensions."""
        # Zero hidden dimension
        with pytest.raises(ValidationError, match="greater than 0"):
            BezierAdapterConfig(hidden_dim=0)
        
        # Zero LoRA rank
        with pytest.raises(ValidationError, match="greater than 0"):
            BezierAdapterConfig(lora_rank=0)
    
    def test_serialization_deserialization(self):
        """Test config serialization and deserialization."""
        config = BezierAdapterConfig(
            output_resolution=(32, 32),
            hidden_dim=64,
            lora_rank=32
        )
        
        # Serialize to dict
        config_dict = config.model_dump()
        assert config_dict["output_resolution"] == [32, 32]  # Tuples become lists
        assert config_dict["hidden_dim"] == 64
        
        # Serialize to JSON
        json_str = config.model_dump_json()
        
        # Deserialize from JSON
        loaded_config = BezierAdapterConfig.model_validate_json(json_str)
        assert loaded_config.output_resolution == (32, 32)
        assert loaded_config.hidden_dim == 64
        assert loaded_config.lora_rank == 32
    
    def test_config_compatibility(self):
        """Test configuration compatibility checks."""
        config = BezierAdapterConfig()
        
        # CLIP dimension should match standard CLIP output
        assert config.clip_dim in [512, 768, 1024]
        
        # T5 dimension should match T5-XXL output
        assert config.t5_dim == 4096
        
        # Fusion dimension should be reasonable
        assert config.fusion_dim >= config.clip_dim
        assert config.fusion_dim <= config.t5_dim


class TestModelIntegration:
    """Test integration between different models."""
    
    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_bezier_to_condition_flow(self, device):
        """Test flow from Bézier points to multi-modal condition."""
        # Create Bézier control points
        bezier_points = BezierControlPoints(
            points=[(0.1, 0.1), (0.4, 0.6), (0.6, 0.6), (0.9, 0.9)],
            curve_type="cubic",
            character="字",
            font_size=96.0
        )
        
        # Process to tensor (simulated)
        num_points = len(bezier_points.points)
        bezier_tensor = torch.tensor(bezier_points.points, device=device)
        
        # Create condition with Bézier features
        condition = MultiModalCondition(
            style_features=None,
            text_features=None,
            mask_features=None,
            bezier_features=bezier_tensor.mean(dim=0).unsqueeze(0)  # Simplified
        )
        
        assert condition.bezier_features is not None
        assert condition.bezier_features.shape == (1, 2)
    
    def test_config_to_model_params(self):
        """Test configuration to model parameter mapping."""
        bezier_config = BezierAdapterConfig(
            hidden_dim=256,
            lora_rank=128,
            num_attention_heads=16
        )
        
        training_config = TrainingConfig(
            learning_rate=2e-4,
            batch_size=8,
            mixed_precision=True
        )
        
        # Simulate model creation
        total_params = (
            bezier_config.hidden_dim * bezier_config.lora_rank * 2 +  # LoRA params
            bezier_config.num_attention_heads * 100  # Attention params (simplified)
        )
        
        assert total_params > 0
        assert training_config.learning_rate == 2e-4
        assert training_config.batch_size == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])