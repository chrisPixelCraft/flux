#!/usr/bin/env python3
"""
Utility functions and helpers for BezierAdapter testing.

Provides common test utilities, data generators, assertion helpers,
and performance measurement tools.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
import time
import psutil
import GPUtil
from pathlib import Path
import json
import random


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the appropriate device for testing.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        torch.device: Selected device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def set_random_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class TestDataGenerator:
    """Generate test data for BezierAdapter components."""
    
    @staticmethod
    def generate_bezier_points(
        batch_size: int,
        num_points: int = 4,
        device: torch.device = None,
        normalized: bool = True
    ) -> torch.Tensor:
        """
        Generate random Bézier control points.
        
        Args:
            batch_size: Number of samples
            num_points: Number of control points per curve
            device: Target device
            normalized: Whether to normalize points to [0, 1]
            
        Returns:
            Tensor of shape (batch_size, num_points, 2)
        """
        if device is None:
            device = get_device()
            
        if normalized:
            points = torch.rand(batch_size, num_points, 2, device=device)
        else:
            points = torch.randn(batch_size, num_points, 2, device=device)
            
        return points
    
    @staticmethod
    def generate_multi_modal_conditions(
        batch_size: int,
        device: torch.device = None,
        include_all: bool = True,
        clip_dim: int = 768,
        t5_dim: int = 4096,
        mask_resolution: Tuple[int, int] = (64, 64)
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Generate multi-modal condition inputs.
        
        Args:
            batch_size: Number of samples
            device: Target device
            include_all: Whether to include all modalities
            clip_dim: CLIP feature dimension
            t5_dim: T5 feature dimension
            mask_resolution: Spatial resolution for mask
            
        Returns:
            Dictionary with condition tensors
        """
        if device is None:
            device = get_device()
        
        conditions = {}
        
        if include_all or random.random() > 0.3:
            conditions['style_features'] = torch.randn(batch_size, clip_dim, device=device)
        else:
            conditions['style_features'] = None
            
        if include_all or random.random() > 0.3:
            conditions['text_features'] = torch.randn(batch_size, t5_dim, device=device)
        else:
            conditions['text_features'] = None
            
        if include_all or random.random() > 0.3:
            conditions['mask_features'] = torch.randn(
                batch_size, 4, mask_resolution[0], mask_resolution[1], device=device
            )
        else:
            conditions['mask_features'] = None
            
        if include_all or random.random() > 0.3:
            conditions['bezier_features'] = torch.randn(batch_size, 3, device=device)
        else:
            conditions['bezier_features'] = None
            
        return conditions
    
    @staticmethod
    def generate_flux_inputs(
        batch_size: int,
        seq_len_img: int = 256,
        seq_len_txt: int = 77,
        in_channels: int = 16,
        context_dim: int = 4096,
        vec_dim: int = 768,
        device: torch.device = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate FLUX model inputs.
        
        Returns:
            Dictionary with all required FLUX inputs
        """
        if device is None:
            device = get_device()
            
        # Calculate spatial dimensions
        img_h = img_w = int(seq_len_img ** 0.5)
        if img_h * img_w != seq_len_img:
            img_h = seq_len_img
            img_w = 1
            
        return {
            'img': torch.randn(batch_size, seq_len_img, in_channels, device=device),
            'img_ids': torch.randint(0, min(img_h, 64), (batch_size, seq_len_img, 2), device=device),
            'txt': torch.randn(batch_size, seq_len_txt, context_dim, device=device),
            'txt_ids': torch.randint(0, seq_len_txt, (batch_size, seq_len_txt, 2), device=device),
            'timesteps': torch.randint(0, 1000, (batch_size,), device=device),
            'y': torch.randn(batch_size, vec_dim, device=device),
            'guidance': torch.rand(batch_size, device=device) * 10
        }


class TensorAssertions:
    """Custom assertions for tensor comparisons."""
    
    @staticmethod
    def assert_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], name: str = "tensor"):
        """Assert tensor has expected shape."""
        assert tensor.shape == expected_shape, \
            f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}"
    
    @staticmethod
    def assert_finite(tensor: torch.Tensor, name: str = "tensor"):
        """Assert all tensor values are finite."""
        assert torch.all(torch.isfinite(tensor)), \
            f"{name} contains non-finite values (inf or nan)"
    
    @staticmethod
    def assert_range(tensor: torch.Tensor, min_val: float, max_val: float, name: str = "tensor"):
        """Assert tensor values are within range."""
        assert torch.all(tensor >= min_val) and torch.all(tensor <= max_val), \
            f"{name} values outside range [{min_val}, {max_val}]"
    
    @staticmethod
    def assert_close(
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        name: str = "tensors"
    ):
        """Assert two tensors are close."""
        assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol), \
            f"{name} not close enough (rtol={rtol}, atol={atol})"
    
    @staticmethod
    def assert_not_close(
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        min_diff: float = 1e-6,
        name: str = "tensors"
    ):
        """Assert two tensors are sufficiently different."""
        max_diff = (tensor1 - tensor2).abs().max().item()
        assert max_diff > min_diff, \
            f"{name} too similar (max diff={max_diff:.2e} <= {min_diff:.2e})"
    
    @staticmethod
    def assert_gradients_exist(model: nn.Module, check_non_zero: bool = True):
        """Assert model has gradients after backward pass."""
        grad_params = []
        zero_grad_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    raise AssertionError(f"Parameter {name} has no gradient")
                    
                if check_non_zero:
                    if torch.allclose(param.grad, torch.zeros_like(param.grad)):
                        zero_grad_params.append(name)
                    else:
                        grad_params.append(name)
        
        if check_non_zero and len(zero_grad_params) == len(grad_params) + len(zero_grad_params):
            raise AssertionError("All gradients are zero")
            
        return grad_params, zero_grad_params


class PerformanceMonitor:
    """Monitor performance metrics during tests."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.start_memory = None
    
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated()
        else:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss
    
    def stop(self, name: str = "default"):
        """Stop monitoring and record metrics."""
        if self.start_time is None:
            return
            
        elapsed_time = time.time() - self.start_time
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = peak_memory - self.start_memory
        else:
            process = psutil.Process()
            memory_used = process.memory_info().rss - self.start_memory
        
        self.metrics[name] = {
            'time_seconds': elapsed_time,
            'memory_bytes': memory_used,
            'memory_mb': memory_used / 1024 / 1024
        }
        
        self.start_time = None
        self.start_memory = None
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get recorded metrics."""
        return self.metrics
    
    def print_summary(self):
        """Print performance summary."""
        print("\nPerformance Summary:")
        print("-" * 50)
        
        for name, metrics in self.metrics.items():
            print(f"{name}:")
            print(f"  Time: {metrics['time_seconds']:.3f} seconds")
            print(f"  Memory: {metrics['memory_mb']:.2f} MB")
        
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                print(f"\nGPU Status:")
                print(f"  Name: {gpu.name}")
                print(f"  Memory Used: {gpu.memoryUsed} MB")
                print(f"  Memory Total: {gpu.memoryTotal} MB")
                print(f"  GPU Utilization: {gpu.load * 100:.1f}%")


def count_parameters(model: nn.Module, only_trainable: bool = False) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        only_trainable: Whether to count only trainable parameters
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
    
    if only_trainable:
        return {'trainable': trainable_params}
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def create_test_config(
    test_name: str,
    config_updates: Dict[str, any] = None
) -> Dict[str, any]:
    """
    Create test configuration with defaults.
    
    Args:
        test_name: Name of the test
        config_updates: Updates to default config
        
    Returns:
        Test configuration dictionary
    """
    default_config = {
        'batch_size': 2,
        'seq_len': 256,
        'feature_dim': 768,
        'num_heads': 12,
        'num_layers': 6,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dtype': torch.float32,
        'seed': 42,
        'test_name': test_name
    }
    
    if config_updates:
        default_config.update(config_updates)
        
    return default_config


def save_test_results(
    results: Dict[str, any],
    output_dir: Union[str, Path] = "test_results"
):
    """Save test results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_name = results.get('test_name', 'test')
    filename = f"{test_name}_{timestamp}.json"
    
    output_path = output_dir / filename
    
    # Convert non-serializable objects
    def convert_value(v):
        if isinstance(v, torch.Tensor):
            return v.tolist()
        elif isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, torch.device):
            return str(v)
        return v
    
    serializable_results = {
        k: convert_value(v) for k, v in results.items()
    }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Test results saved to: {output_path}")


def compare_model_outputs(
    model: nn.Module,
    baseline_inputs: Dict[str, torch.Tensor],
    modified_inputs: Dict[str, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> Dict[str, float]:
    """
    Compare model outputs with different inputs.
    
    Returns:
        Dictionary with comparison metrics
    """
    model.eval()
    
    with torch.no_grad():
        baseline_output = model(**baseline_inputs)
        modified_output = model(**modified_inputs)
    
    # Calculate differences
    abs_diff = (baseline_output - modified_output).abs()
    rel_diff = abs_diff / (baseline_output.abs() + 1e-8)
    
    metrics = {
        'max_abs_diff': abs_diff.max().item(),
        'mean_abs_diff': abs_diff.mean().item(),
        'max_rel_diff': rel_diff.max().item(),
        'mean_rel_diff': rel_diff.mean().item(),
        'outputs_close': torch.allclose(baseline_output, modified_output, rtol=rtol, atol=atol)
    }
    
    return metrics


if __name__ == "__main__":
    # Example usage
    print("BezierAdapter Test Utilities")
    
    # Test data generation
    generator = TestDataGenerator()
    device = get_device()
    
    bezier_points = generator.generate_bezier_points(4, device=device)
    print(f"Generated Bézier points: {bezier_points.shape}")
    
    conditions = generator.generate_multi_modal_conditions(4, device=device)
    print(f"Generated conditions: {list(conditions.keys())}")
    
    # Test performance monitoring
    monitor = PerformanceMonitor()
    
    monitor.start()
    time.sleep(0.1)  # Simulate work
    monitor.stop("test_operation")
    
    monitor.print_summary()