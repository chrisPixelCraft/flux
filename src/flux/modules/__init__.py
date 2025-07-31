"""
BezierAdapter modules for FLUX integration.
"""

# Core modules
from .bezier_processor import BezierParameterProcessor
from .condition_adapter import ConditionInjectionAdapter
from .spatial_fuser import SpatialAttentionFuser
from .style_fusion import StyleBezierFusionModule
# FluxBezierAdapter imported separately to avoid circular imports

# Data models
from .models import (
    BezierControlPoints,
    DensityMapConfig,
    MultiModalCondition,
    BezierAdapterConfig
)

# LoRA utilities
from .lora import (
    LinearLora,
    replace_linear_with_lora,
    get_lora_parameters,
    set_lora_scale,
    freeze_non_lora_parameters
)

__all__ = [
    # Core modules
    'BezierParameterProcessor',
    'ConditionInjectionAdapter',
    'SpatialAttentionFuser',
    'StyleBezierFusionModule',
    
    # Data models
    'BezierControlPoints',
    'DensityMapConfig',
    'MultiModalCondition',
    'BezierAdapterConfig',
    
    # LoRA utilities
    'LinearLora',
    'replace_linear_with_lora',
    'get_lora_parameters',
    'set_lora_scale',
    'freeze_non_lora_parameters'
]