# BezierAdapter Test Suite

Comprehensive unit and integration tests for the FLUX BezierAdapter framework.

## Test Structure

```
tests/
├── conftest.py                         # Pytest configuration and fixtures
├── test_utils.py                       # Testing utilities and helpers
├── test_bezier_processor.py           # BezierParameterProcessor unit tests
├── test_condition_adapter.py          # ConditionInjectionAdapter unit tests
├── test_spatial_fuser.py              # SpatialAttentionFuser unit tests
├── test_style_fusion.py               # StyleBezierFusionModule unit tests
├── test_models.py                      # Pydantic models unit tests
├── test_integration_comprehensive.py  # Full integration tests
└── README.md                          # This file
```

## Running Tests

### Prerequisites

1. **Install Dependencies**:
```bash
conda activate easycontrol
pip install pytest pytest-cov pytest-benchmark
```

2. **Set PYTHONPATH** (if needed):
```bash
export PYTHONPATH=/root/context-engineering-intro/flux/src:$PYTHONPATH
```

### Basic Test Execution

**Run all tests**:
```bash
cd /root/context-engineering-intro/flux
python -m pytest tests/ -v
```

**Run specific test file**:
```bash
python -m pytest tests/test_bezier_processor.py -v
```

**Run specific test**:
```bash
python -m pytest tests/test_bezier_processor.py::TestBezierParameterProcessor::test_forward_basic -v
```

### Test Categories

**Unit tests only** (fast):
```bash
python -m pytest tests/ -v -m "not integration and not slow"
```

**Integration tests**:
```bash
python -m pytest tests/test_integration_comprehensive.py -v
```

**Memory tests** (requires CUDA):
```bash
python -m pytest tests/ -v -m memory
```

**Benchmark tests**:
```bash
python -m pytest tests/ -v -m benchmark --benchmark-only
```

### Test Coverage

**Run with coverage**:
```bash
python -m pytest tests/ --cov=flux.modules --cov-report=html --cov-report=term
```

**View coverage report**:
```bash
open htmlcov/index.html  # or equivalent browser command
```

## Test Details

### 1. BezierParameterProcessor Tests (`test_bezier_processor.py`)

**Coverage**: KDE density calculation, point embedding, gradient checkpointing

**Key Tests**:
- `test_initialization`: Module setup and parameters
- `test_forward_basic`: Basic forward pass functionality
- `test_kde_density_calculation`: KDE correctness validation
- `test_learnable_bandwidth`: Bandwidth parameter learning
- `test_gradient_checkpointing`: Memory optimization
- `test_edge_cases`: Boundary conditions (single point, many points, out-of-range)

**Expected Results**:
```
✅ 10+ tests passing
✅ Parameter count: 25,025
✅ Memory efficient for large point sets
✅ Learnable bandwidth updates during training
```

### 2. ConditionInjectionAdapter Tests (`test_condition_adapter.py`)

**Coverage**: Multi-modal fusion, LoRA adaptation, cross-attention

**Key Tests**:
- `test_style_branch`: CLIP feature processing
- `test_text_branch`: T5 feature processing
- `test_mask_branch`: VAE spatial feature processing with global pooling
- `test_bezier_branch`: Bézier feature processing
- `test_lora_parameters`: LoRA parameter efficiency
- `test_partial_modalities`: Missing modality handling

**Expected Results**:
```
✅ 15+ tests passing
✅ LoRA parameter ratio: 36.99%
✅ Handles partial modalities gracefully
✅ Memory efficient mask processing
```

### 3. SpatialAttentionFuser Tests (`test_spatial_fuser.py`)

**Coverage**: Density-modulated attention, RoPE compatibility, dynamic sequences

**Key Tests**:
- `test_rope_compatibility`: FLUX RoPE dimension validation
- `test_forward_basic`: Attention mechanism functionality
- `test_dynamic_sequence_length`: Various spatial resolutions
- `test_density_modulation`: Density weighting effects
- `test_gradient_checkpointing`: Large sequence handling

**Expected Results**:
```
✅ 12+ tests passing
✅ RoPE dimensions even and compatible
✅ Dynamic sequence lengths (64-4096)
✅ Density modulation functional
```

### 4. StyleBezierFusionModule Tests (`test_style_fusion.py`)

**Coverage**: AdaIN style transfer, density-aware projection, cross-attention

**Key Tests**:
- `test_adain_sequential_format`: AdaIN normalization correctness
- `test_style_transfer_effectiveness`: Style transfer validation
- `test_cross_attention_mechanism`: Attention weight validation
- `test_training_eval_mode`: Dropout behavior consistency

**Expected Results**:
```
✅ 15+ tests passing
✅ AdaIN normalization working correctly
✅ Style transfer produces different outputs
✅ Cross-attention weights sum to 1
```

### 5. Pydantic Models Tests (`test_models.py`)

**Coverage**: Data validation, serialization, type checking

**Key Tests**:
- `test_bezier_control_points`: Bézier curve validation
- `test_multi_modal_condition`: Tensor handling
- `test_training_config`: Parameter validation
- `test_serialization`: JSON serialization/deserialization

**Expected Results**:
```
✅ 12+ tests passing
✅ Input validation working
✅ Serialization preserves data
✅ Type checking functional
```

### 6. Integration Tests (`test_integration_comprehensive.py`)

**Coverage**: Full pipeline, training workflow, memory usage, error handling

**Key Tests**:
- `test_full_forward_pass`: Complete BezierAdapter pipeline
- `test_component_integration`: Inter-component data flow
- `test_training_workflow`: Parameter updates and gradients
- `test_memory_scaling`: Memory usage with batch sizes
- `test_error_handling`: Edge cases and validation

**Expected Results**:
```
✅ 20+ tests passing
✅ Full pipeline functional
✅ Memory scales linearly with batch size
✅ Training mode switching works
✅ Error handling robust
```

## Performance Benchmarks

### Expected Performance Metrics

| Component | Forward Time | Memory Usage | Parameters |
|-----------|-------------|--------------|------------|
| BezierParameterProcessor | <5ms | <50MB | 25K |
| ConditionInjectionAdapter | <10ms | <100MB | 27.6M |
| SpatialAttentionFuser | <20ms | <200MB | 44.9M |
| StyleBezierFusionModule | <8ms | <80MB | 7.6M |
| **Full Integration** | **<50ms** | **<500MB** | **118.9M** |

### Memory Usage (CUDA)

| Batch Size | Sequence Length | Memory Usage |
|------------|----------------|--------------|
| 1 | 256 | ~400MB |
| 2 | 256 | ~600MB |
| 4 | 256 | ~1GB |
| 1 | 1024 | ~800MB |
| 1 | 4096 | ~2GB |

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**:
```bash
# Run with smaller batch sizes
python -m pytest tests/ -v -k "not memory"

# Or run CPU-only
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -v
```

**2. Import Errors**:
```bash
# Add src to Python path
export PYTHONPATH=/root/context-engineering-intro/flux/src:$PYTHONPATH

# Or run from flux directory
cd /root/context-engineering-intro/flux
python -m pytest tests/ -v
```

**3. RoPE Dimension Errors**:
- Ensure `feature_dim` is divisible by `num_heads`
- Ensure `pe_dim = feature_dim // num_heads` is divisible by 4
- Check `axes_dim` sums to `pe_dim` and all values are even

**4. Slow Tests**:
```bash
# Skip slow tests
python -m pytest tests/ -v -m "not slow"

# Run parallel with pytest-xdist
pip install pytest-xdist
python -m pytest tests/ -v -n auto
```

### Debug Mode

**Enable verbose output**:
```bash
python -m pytest tests/ -v -s --tb=long
```

**Run single test with debugging**:
```bash
python -c "
import sys
sys.path.append('src')
from tests.test_bezier_processor import TestBezierParameterProcessor
test = TestBezierParameterProcessor()
# Set breakpoints and debug
"
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: BezierAdapter Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.10
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        cd flux
        python -m pytest tests/ --cov=flux.modules --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## Contributing

### Adding New Tests

1. **Create test file**: Follow naming convention `test_<component>.py`
2. **Use fixtures**: Leverage shared fixtures from `conftest.py`
3. **Add docstrings**: Document test purpose and expected behavior
4. **Include edge cases**: Test boundary conditions and error cases
5. **Add markers**: Use appropriate pytest markers (`@pytest.mark.slow`, etc.)

### Test Quality Guidelines

- **Coverage**: Aim for >90% code coverage
- **Independence**: Tests should not depend on each other
- **Reproducibility**: Use fixed seeds and deterministic inputs
- **Performance**: Include performance benchmarks for critical paths
- **Documentation**: Clear docstrings and comments

---

**Total Test Count**: ~80 tests across 6 test files  
**Coverage Target**: >90% of BezierAdapter modules  
**Execution Time**: <5 minutes (unit tests), <15 minutes (full suite)