#!/usr/bin/env python3
"""
Comprehensive integration tests for FLUX BezierAdapter.

Tests full pipeline, different configurations, memory usage, training workflow,
and error handling across the entire BezierAdapter framework.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import time
import gc

# Add flux modules to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from flux.model import FluxParams
from flux.modules.bezier_flux_model import FluxBezierAdapter, BezierAdapterConfig, FluxBezierLoraWrapper
from flux.modules.bezier_processor import BezierParameterProcessor
from flux.modules.condition_adapter import ConditionInjectionAdapter
from flux.modules.spatial_fuser import SpatialAttentionFuser
from flux.modules.style_fusion import StyleBezierFusionModule
from flux.modules.models import MultiModalCondition, TrainingConfig


class TestFullPipeline:
    """Test complete BezierAdapter pipeline from input to output."""
    
    @pytest.fixture
    def device(self):
        """Get test device (CUDA if available)."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def flux_params(self):
        """Create test FLUX parameters."""
        return FluxParams(
            in_channels=16,
            out_channels=16,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=768,  # Smaller for testing
            mlp_ratio=4.0,
            num_heads=12,
            depth=6,  # Fewer layers for testing
            depth_single_blocks=8,
            axes_dim=[32, 32],  # 768 // 12 = 64, split evenly
            theta=10000,
            qkv_bias=True,
            guidance_embed=True
        )
    
    @pytest.fixture
    def bezier_config(self):
        """Create test BezierAdapter configuration."""
        return BezierAdapterConfig(
            hook_layers=[2, 3, 4],  # Middle layers
            enable_bezier_guidance=True,
            enable_style_transfer=True,
            enable_density_attention=True,
            output_resolution=(32, 32),  # Smaller for testing
            hidden_dim=64,
            lora_rank=32
        )
    
    def test_full_forward_pass(self, device, flux_params, bezier_config):
        """Test complete forward pass through FluxBezierAdapter."""
        # Initialize model
        model = FluxBezierAdapter(flux_params, bezier_config).to(device)
        model.eval()
        
        batch_size = 2
        seq_len_img = 256
        seq_len_txt = 77
        
        # Create inputs
        img = torch.randn(batch_size, seq_len_img, 16, device=device)
        img_ids = torch.randint(0, 16, (batch_size, seq_len_img, 2), device=device)
        txt = torch.randn(batch_size, seq_len_txt, 4096, device=device)
        txt_ids = torch.randint(0, 77, (batch_size, seq_len_txt, 2), device=device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)
        y = torch.randn(batch_size, 768, device=device)
        guidance = torch.rand(batch_size, device=device) * 10
        
        # Create BezierAdapter conditions
        conditions = MultiModalCondition(
            style_features=torch.randn(batch_size, 768, device=device),
            text_features=torch.randn(batch_size, 4096, device=device),
            mask_features=torch.randn(batch_size, 4, 32, 32, device=device),
            bezier_features=torch.randn(batch_size, 3, device=device)
        )
        
        bezier_points = torch.rand(batch_size, 4, 2, device=device)
        
        bezier_conditions = {
            'conditions': conditions,
            'bezier_points': bezier_points
        }
        
        # Forward pass
        with torch.no_grad():
            output = model(
                img=img, img_ids=img_ids,
                txt=txt, txt_ids=txt_ids,
                timesteps=timesteps, y=y, guidance=guidance,
                bezier_conditions=bezier_conditions
            )
        
        # Validate output
        assert output.shape == (batch_size, seq_len_img, 16)
        assert torch.all(torch.isfinite(output))
        
        # Test without BezierAdapter
        output_baseline = model(
            img=img, img_ids=img_ids,
            txt=txt, txt_ids=txt_ids,
            timesteps=timesteps, y=y, guidance=guidance,
            bezier_conditions=None
        )
        
        # Outputs should be different
        assert not torch.allclose(output, output_baseline)
    
    def test_component_integration(self, device):
        """Test integration between individual BezierAdapter components."""
        batch_size = 2
        
        # Initialize components
        bezier_processor = BezierParameterProcessor(
            output_resolution=(32, 32),
            hidden_dim=64
        ).to(device)
        
        condition_adapter = ConditionInjectionAdapter(
            clip_dim=768,
            t5_dim=4096,
            hidden_dim=1536,
            lora_rank=32
        ).to(device)
        
        spatial_fuser = SpatialAttentionFuser(
            feature_dim=768,
            num_heads=12,
            num_layers=2  # Fewer layers for testing
        ).to(device)
        
        style_fusion = StyleBezierFusionModule(
            spatial_dim=768,
            style_dim=1536
        ).to(device)
        
        # Test data flow
        # Step 1: Process Bézier curves
        bezier_points = torch.rand(batch_size, 4, 2, device=device)
        density_map, field_map = bezier_processor(bezier_points)
        assert density_map.shape == (batch_size, 32, 32)
        
        # Step 2: Process conditions
        conditions = MultiModalCondition(
            style_features=torch.randn(batch_size, 768, device=device),
            text_features=torch.randn(batch_size, 4096, device=device),
            mask_features=torch.randn(batch_size, 4, 32, 32, device=device),
            bezier_features=field_map.mean(dim=(1, 2))  # Aggregate field map
        )
        unified_conditions = condition_adapter(conditions)
        assert unified_conditions.shape == (batch_size, 1536)
        
        # Step 3: Spatial attention fusion
        spatial_features = torch.randn(batch_size, 256, 768, device=device)
        fused_features, _ = spatial_fuser(
            spatial_features=spatial_features,
            density_weights=density_map,
            condition_embeddings=unified_conditions
        )
        assert fused_features.shape == spatial_features.shape
        
        # Step 4: Style fusion
        stylized_features, _ = style_fusion(
            spatial_features=fused_features,
            style_embeddings=unified_conditions,
            density_weights=density_map
        )
        assert stylized_features.shape == fused_features.shape
    
    def test_different_hook_configurations(self, device, flux_params):
        """Test different hook layer configurations."""
        hook_configs = [
            [0, 1, 2],      # Early layers
            [2, 3, 4],      # Middle layers
            [4, 5],         # Late layers
            [0, 2, 4, 5],   # Mixed layers
        ]
        
        batch_size = 1
        seq_len_img = 64  # Smaller for faster testing
        
        # Create test inputs
        img = torch.randn(batch_size, seq_len_img, 16, device=device)
        img_ids = torch.randint(0, 8, (batch_size, seq_len_img, 2), device=device)
        txt = torch.randn(batch_size, 20, 4096, device=device)
        txt_ids = torch.randint(0, 20, (batch_size, 20, 2), device=device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)
        y = torch.randn(batch_size, 768, device=device)
        
        conditions = MultiModalCondition(
            style_features=torch.randn(batch_size, 768, device=device),
            text_features=None,
            mask_features=None,
            bezier_features=None
        )
        
        outputs = []
        for hook_layers in hook_configs:
            config = BezierAdapterConfig(
                hook_layers=hook_layers,
                enable_bezier_guidance=False,  # Faster
                enable_style_transfer=True,
                enable_density_attention=False
            )
            
            model = FluxBezierAdapter(flux_params, config).to(device)
            model.eval()
            
            with torch.no_grad():
                output = model(
                    img=img, img_ids=img_ids,
                    txt=txt, txt_ids=txt_ids,
                    timesteps=timesteps, y=y,
                    bezier_conditions={'conditions': conditions, 'bezier_points': None}
                )
            
            outputs.append(output)
        
        # Different hook configurations should produce different outputs
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not torch.allclose(outputs[i], outputs[j])


class TestTrainingWorkflow:
    """Test training workflow and parameter updates."""
    
    @pytest.fixture
    def small_model(self, device):
        """Create small model for training tests."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        flux_params = FluxParams(
            in_channels=8,
            out_channels=8,
            vec_in_dim=256,
            context_in_dim=512,
            hidden_size=256,
            mlp_ratio=2.0,
            num_heads=8,
            depth=2,
            depth_single_blocks=2,
            axes_dim=[16, 16],
            theta=10000,
            qkv_bias=True,
            guidance_embed=False
        )
        
        bezier_config = BezierAdapterConfig(
            hook_layers=[1],
            enable_bezier_guidance=True,
            enable_style_transfer=False,
            enable_density_attention=False,
            output_resolution=(16, 16),
            hidden_dim=32,
            lora_rank=16
        )
        
        return FluxBezierAdapter(flux_params, bezier_config).to(device)
    
    def test_training_mode_switching(self, small_model):
        """Test switching between training modes."""
        # Enable BezierAdapter training
        small_model.set_bezier_training_mode(True)
        
        # Check FLUX is frozen
        flux_params = [p for name, p in small_model.named_parameters() 
                      if not any(comp in name for comp in ['bezier', 'condition', 'spatial', 'style'])]
        bezier_params = list(small_model.get_bezier_parameters())
        
        for param in flux_params:
            assert not param.requires_grad
        
        for param in bezier_params:
            assert param.requires_grad
        
        # Switch to full training
        small_model.set_bezier_training_mode(False)
        for param in small_model.parameters():
            param.requires_grad = True
        
        # All parameters should be trainable
        for param in small_model.parameters():
            assert param.requires_grad
    
    def test_gradient_flow_training(self, small_model, device):
        """Test gradient flow during training."""
        device = next(small_model.parameters()).device
        small_model.set_bezier_training_mode(True)
        
        # Create mini batch
        batch_size = 2
        img = torch.randn(batch_size, 16, 8, device=device)
        img_ids = torch.randint(0, 4, (batch_size, 16, 2), device=device)
        txt = torch.randn(batch_size, 10, 512, device=device)
        txt_ids = torch.randint(0, 10, (batch_size, 10, 2), device=device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)
        y = torch.randn(batch_size, 256, device=device)
        
        conditions = MultiModalCondition(
            style_features=torch.randn(batch_size, 768, device=device),
            text_features=None,
            mask_features=None,
            bezier_features=None
        )
        
        bezier_points = torch.rand(batch_size, 4, 2, device=device)
        
        # Forward pass
        output = small_model(
            img=img, img_ids=img_ids,
            txt=txt, txt_ids=txt_ids,
            timesteps=timesteps, y=y,
            bezier_conditions={'conditions': conditions, 'bezier_points': bezier_points}
        )
        
        # Compute loss
        target = torch.randn_like(output)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist for BezierAdapter parameters
        bezier_params = list(small_model.get_bezier_parameters())
        grad_norms = []
        
        for param in bezier_params:
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        assert len(grad_norms) > 0
        assert all(norm > 0 for norm in grad_norms)
    
    def test_lora_fine_tuning(self, device):
        """Test LoRA fine-tuning workflow."""
        # Create model with LoRA
        flux_params = FluxParams(
            in_channels=8, out_channels=8, vec_in_dim=256,
            context_in_dim=512, hidden_size=256, mlp_ratio=2.0,
            num_heads=8, depth=2, depth_single_blocks=2,
            axes_dim=[16, 16], theta=10000, qkv_bias=True,
            guidance_embed=False
        )
        
        bezier_config = BezierAdapterConfig(
            hook_layers=[0], lora_rank=16
        )
        
        model = FluxBezierLoraWrapper(
            flux_params=flux_params,
            bezier_config=bezier_config,
            lora_rank=32,
            lora_scale=1.0,
            apply_lora_to_flux=False
        ).to(device)
        
        # Get LoRA parameters
        lora_params = model.get_lora_parameters()
        assert len(lora_params) > 0
        
        # Test scale adjustment
        model.set_lora_scale(0.5)
        
        # Create optimizer for LoRA parameters only
        optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
        
        # Training step
        batch_size = 1
        img = torch.randn(batch_size, 16, 8, device=device)
        img_ids = torch.randint(0, 4, (batch_size, 16, 2), device=device)
        txt = torch.randn(batch_size, 10, 512, device=device)
        txt_ids = torch.randint(0, 10, (batch_size, 10, 2), device=device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)
        y = torch.randn(batch_size, 256, device=device)
        
        output = model(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                      timesteps=timesteps, y=y)
        
        loss = output.mean()
        loss.backward()
        
        # Update LoRA parameters
        optimizer.step()
        optimizer.zero_grad()
        
        # Check parameters were updated
        assert all(param.grad is None or torch.allclose(param.grad, torch.zeros_like(param.grad)) 
                  for param in lora_params)


class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_memory_scaling(self, device):
        """Test memory usage with different batch sizes."""
        device = torch.device('cuda')
        
        flux_params = FluxParams(
            in_channels=16, out_channels=16, vec_in_dim=768,
            context_in_dim=4096, hidden_size=768, mlp_ratio=4.0,
            num_heads=12, depth=4, depth_single_blocks=4,
            axes_dim=[32, 32], theta=10000, qkv_bias=True,
            guidance_embed=False
        )
        
        bezier_config = BezierAdapterConfig(
            hook_layers=[1, 2],
            output_resolution=(32, 32)
        )
        
        model = FluxBezierAdapter(flux_params, bezier_config).to(device)
        model.eval()
        
        batch_sizes = [1, 2, 4]
        memory_usage = []
        
        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Create inputs
            img = torch.randn(batch_size, 64, 16, device=device)
            img_ids = torch.randint(0, 8, (batch_size, 64, 2), device=device)
            txt = torch.randn(batch_size, 20, 4096, device=device)
            txt_ids = torch.randint(0, 20, (batch_size, 20, 2), device=device)
            timesteps = torch.randint(0, 1000, (batch_size,), device=device)
            y = torch.randn(batch_size, 768, device=device)
            
            start_memory = torch.cuda.memory_allocated()
            
            with torch.no_grad():
                output = model(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                             timesteps=timesteps, y=y)
            
            peak_memory = torch.cuda.max_memory_allocated()
            memory_mb = (peak_memory - start_memory) / 1024 / 1024
            memory_usage.append(memory_mb)
            
            print(f"Batch size {batch_size}: {memory_mb:.2f} MB")
        
        # Memory should scale roughly linearly with batch size
        for i in range(1, len(memory_usage)):
            ratio = memory_usage[i] / memory_usage[0]
            expected_ratio = batch_sizes[i] / batch_sizes[0]
            assert 0.8 * expected_ratio <= ratio <= 1.5 * expected_ratio
    
    def test_inference_speed(self, device):
        """Test inference speed with and without BezierAdapter."""
        flux_params = FluxParams(
            in_channels=16, out_channels=16, vec_in_dim=768,
            context_in_dim=4096, hidden_size=768, mlp_ratio=4.0,
            num_heads=12, depth=2, depth_single_blocks=2,
            axes_dim=[32, 32], theta=10000, qkv_bias=True,
            guidance_embed=False
        )
        
        bezier_config = BezierAdapterConfig(hook_layers=[1])
        
        model = FluxBezierAdapter(flux_params, bezier_config).to(device)
        model.eval()
        
        # Test inputs
        batch_size = 1
        img = torch.randn(batch_size, 64, 16, device=device)
        img_ids = torch.randint(0, 8, (batch_size, 64, 2), device=device)
        txt = torch.randn(batch_size, 20, 4096, device=device)
        txt_ids = torch.randint(0, 20, (batch_size, 20, 2), device=device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)
        y = torch.randn(batch_size, 768, device=device)
        
        conditions = MultiModalCondition(
            style_features=torch.randn(batch_size, 768, device=device),
            text_features=None, mask_features=None, bezier_features=None
        )
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                         timesteps=timesteps, y=y)
        
        # Time baseline (no BezierAdapter)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        for _ in range(10):
            with torch.no_grad():
                _ = model(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                         timesteps=timesteps, y=y, bezier_conditions=None)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        baseline_time = time.time() - start_time
        
        # Time with BezierAdapter
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        for _ in range(10):
            with torch.no_grad():
                _ = model(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                         timesteps=timesteps, y=y,
                         bezier_conditions={'conditions': conditions, 'bezier_points': None})
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        bezier_time = time.time() - start_time
        
        overhead = (bezier_time - baseline_time) / baseline_time
        print(f"Baseline: {baseline_time:.3f}s, BezierAdapter: {bezier_time:.3f}s")
        print(f"Overhead: {overhead:.1%}")
        
        # Overhead should be reasonable (less than 50%)
        assert overhead < 0.5


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_hook_layers(self, device):
        """Test error handling for invalid hook layers."""
        flux_params = FluxParams(
            in_channels=16, out_channels=16, vec_in_dim=768,
            context_in_dim=4096, hidden_size=768, mlp_ratio=4.0,
            num_heads=12, depth=4, depth_single_blocks=4,
            axes_dim=[32, 32], theta=10000, qkv_bias=True,
            guidance_embed=False
        )
        
        # Hook layer exceeds model depth
        bezier_config = BezierAdapterConfig(hook_layers=[100])
        
        with pytest.raises(ValueError, match="exceeds model depth"):
            FluxBezierAdapter(flux_params, bezier_config)
    
    def test_missing_required_inputs(self, device):
        """Test handling of missing required inputs."""
        flux_params = FluxParams(
            in_channels=16, out_channels=16, vec_in_dim=768,
            context_in_dim=4096, hidden_size=768, mlp_ratio=4.0,
            num_heads=12, depth=2, depth_single_blocks=2,
            axes_dim=[32, 32], theta=10000, qkv_bias=True,
            guidance_embed=True  # Requires guidance
        )
        
        model = FluxBezierAdapter(flux_params).to(device)
        
        # Missing guidance when required
        img = torch.randn(1, 64, 16, device=device)
        img_ids = torch.randint(0, 8, (1, 64, 2), device=device)
        txt = torch.randn(1, 20, 4096, device=device)
        txt_ids = torch.randint(0, 20, (1, 20, 2), device=device)
        timesteps = torch.randint(0, 1000, (1,), device=device)
        y = torch.randn(1, 768, device=device)
        
        with pytest.raises(ValueError, match="guidance strength"):
            model(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids,
                 timesteps=timesteps, y=y, guidance=None)
    
    def test_component_enable_disable(self, device):
        """Test enabling/disabling individual components."""
        flux_params = FluxParams(
            in_channels=16, out_channels=16, vec_in_dim=768,
            context_in_dim=4096, hidden_size=768, mlp_ratio=4.0,
            num_heads=12, depth=2, depth_single_blocks=2,
            axes_dim=[32, 32], theta=10000, qkv_bias=True,
            guidance_embed=False
        )
        
        # Test different component combinations
        configs = [
            {'enable_bezier_guidance': True, 'enable_style_transfer': False, 'enable_density_attention': False},
            {'enable_bezier_guidance': False, 'enable_style_transfer': True, 'enable_density_attention': False},
            {'enable_bezier_guidance': False, 'enable_style_transfer': False, 'enable_density_attention': True},
            {'enable_bezier_guidance': False, 'enable_style_transfer': False, 'enable_density_attention': False},
        ]
        
        for config_dict in configs:
            bezier_config = BezierAdapterConfig(hook_layers=[1], **config_dict)
            model = FluxBezierAdapter(flux_params, bezier_config).to(device)
            
            # Check components are created/not created as expected
            if config_dict['enable_bezier_guidance']:
                assert model.bezier_processor is not None
            else:
                assert model.bezier_processor is None
                
            if config_dict['enable_style_transfer']:
                assert model.style_fusion is not None
            else:
                assert model.style_fusion is None
                
            if config_dict['enable_density_attention']:
                assert model.spatial_fuser is not None
            else:
                assert model.spatial_fuser is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])