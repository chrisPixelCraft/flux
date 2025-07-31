"""
Pytest configuration and shared fixtures for BezierAdapter tests.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add flux modules to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from flux.model import FluxParams
from flux.modules.bezier_flux_model import BezierAdapterConfig
from tests.test_utils import set_random_seed, get_device


@pytest.fixture(scope="session")
def device():
    """Get test device for the entire test session."""
    return get_device(prefer_cuda=True)


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed before each test."""
    set_random_seed(42)


@pytest.fixture
def small_flux_params():
    """Create small FLUX parameters for fast testing."""
    return FluxParams(
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


@pytest.fixture
def test_bezier_config():
    """Create test BezierAdapter configuration."""
    return BezierAdapterConfig(
        hook_layers=[1],
        enable_bezier_guidance=True,
        enable_style_transfer=True,
        enable_density_attention=True,
        output_resolution=(32, 32),
        hidden_dim=64,
        lora_rank=32
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "benchmark: mark test as a benchmark test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "memory: mark test as a memory usage test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add slow marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add memory marker to memory tests
        if "memory" in item.name.lower():
            item.add_marker(pytest.mark.memory)
        
        # Add benchmark marker to performance/benchmark tests
        if "benchmark" in item.name.lower() or "performance" in item.name.lower():
            item.add_marker(pytest.mark.benchmark)