"""
Utility functions for BezierFlux pipelines.

This module provides helper functions for loading Bézier curves,
preprocessing inputs, and other common pipeline operations.
"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
from PIL import Image
from torch import Tensor

from ..modules.models import BezierControlPoints


def load_bezier_curves(
    source: Union[str, Path, Dict, List, BezierControlPoints]
) -> List[Tuple[float, float]]:
    """
    Load Bézier curves from various sources.
    
    Args:
        source: Can be:
            - String/Path to JSON file
            - Dict with 'control_points' or direct points
            - List of (x, y) tuples
            - BezierControlPoints object
    
    Returns:
        List of (x, y) control point tuples
    """
    if isinstance(source, (str, Path)):
        # Load from JSON file
        with open(source, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            if 'control_points' in data:
                return data['control_points']
            elif 'curves' in data and isinstance(data['curves'], list):
                # Extract points from multiple curves
                points = []
                for curve in data['curves']:
                    if 'points' in curve:
                        points.extend(curve['points'])
                return points
            else:
                # Assume the dict contains direct coordinate data
                return [(data.get('x', []), data.get('y', []))]
        else:
            return data
    
    elif isinstance(source, dict):
        if 'control_points' in source:
            return source['control_points']
        elif 'points' in source:
            return source['points']
        else:
            # Try to extract x, y coordinates
            x_coords = source.get('x', [])
            y_coords = source.get('y', [])
            if len(x_coords) == len(y_coords):
                return list(zip(x_coords, y_coords))
            return []
    
    elif isinstance(source, BezierControlPoints):
        return source.points
    
    elif isinstance(source, list):
        # Validate that it's a list of coordinate pairs
        if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in source):
            return [(float(x), float(y)) for x, y in source]
        return source
    
    else:
        raise ValueError(f"Unsupported bezier curve source type: {type(source)}")


def normalize_bezier_points(
    points: List[Tuple[float, float]],
    source_width: Optional[int] = None,
    source_height: Optional[int] = None,
    target_width: int = 1,
    target_height: int = 1
) -> List[Tuple[float, float]]:
    """
    Normalize Bézier points to [0, 1] coordinate space.
    
    Args:
        points: List of (x, y) coordinate pairs
        source_width: Original coordinate space width (auto-detect if None)
        source_height: Original coordinate space height (auto-detect if None)
        target_width: Target coordinate space width (typically 1)
        target_height: Target coordinate space height (typically 1)
    
    Returns:
        List of normalized (x, y) coordinate pairs
    """
    if not points:
        return points
    
    # Extract x and y coordinates
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # Auto-detect source dimensions if not provided
    if source_width is None:
        source_width = max(x_coords) - min(x_coords)
        if source_width == 0:
            source_width = 1
    
    if source_height is None:
        source_height = max(y_coords) - min(y_coords)
        if source_height == 0:
            source_height = 1
    
    # Normalize coordinates
    min_x, min_y = min(x_coords), min(y_coords)
    
    normalized_points = []
    for x, y in points:
        norm_x = ((x - min_x) / source_width) * target_width
        norm_y = ((y - min_y) / source_height) * target_height
        normalized_points.append((norm_x, norm_y))
    
    return normalized_points


def validate_bezier_curves(
    points: List[Tuple[float, float]],
    min_points: int = 3,
    max_points: int = 32
) -> bool:
    """
    Validate Bézier curve points.
    
    Args:
        points: List of (x, y) coordinate pairs
        min_points: Minimum required points
        max_points: Maximum allowed points
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(points, list):
        return False
    
    if len(points) < min_points or len(points) > max_points:
        return False
    
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return False
        
        x, y = point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return False
        
        # Check if coordinates are in reasonable range
        if not (0 <= x <= 1) or not (0 <= y <= 1):
            return False
    
    return True


def create_bezier_from_character(
    character: str,
    font_size: int = 64,
    method: str = "simple"
) -> List[Tuple[float, float]]:
    """
    Create approximate Bézier curves for a character.
    
    This is a simplified method for testing. In practice, you would
    extract actual Bézier curves from font files using libraries like
    fonttools or by tracing character outlines.
    
    Args:
        character: Character to create curves for
        font_size: Font size for scaling
        method: Curve generation method ("simple", "outline", "skeleton")
    
    Returns:
        List of approximate Bézier control points
    """
    # This is a placeholder implementation
    # In practice, you would use font rendering and outline extraction
    
    if method == "simple":
        # Create simple rectangular outline for any character
        return [
            (0.1, 0.1),  # Bottom-left
            (0.1, 0.9),  # Top-left
            (0.9, 0.9),  # Top-right
            (0.9, 0.1),  # Bottom-right
        ]
    
    elif method == "outline":
        # Create more complex character-specific outlines
        char_outlines = {
            'A': [
                (0.5, 0.1),   # Bottom center
                (0.2, 0.9),   # Top left
                (0.8, 0.9),   # Top right
                (0.3, 0.5),   # Left crossbar
                (0.7, 0.5),   # Right crossbar
            ],
            'O': [
                (0.5, 0.1),   # Bottom
                (0.1, 0.3),   # Bottom-left
                (0.1, 0.7),   # Top-left
                (0.5, 0.9),   # Top
                (0.9, 0.7),   # Top-right
                (0.9, 0.3),   # Bottom-right
            ],
            'B': [
                (0.1, 0.1),   # Bottom-left
                (0.1, 0.9),   # Top-left
                (0.6, 0.9),   # Top-right curve start
                (0.8, 0.7),   # Top-right curve
                (0.6, 0.5),   # Middle
                (0.8, 0.3),   # Bottom-right curve
                (0.6, 0.1),   # Bottom-right curve end
            ]
        }
        
        return char_outlines.get(character.upper(), char_outlines['O'])
    
    else:
        # Fallback to simple method
        return create_bezier_from_character(character, font_size, "simple")


def prepare_bezier_inputs(
    bezier_curves: Optional[Union[str, List, Dict, BezierControlPoints]],
    normalize: bool = True,
    validate: bool = True,
    device: str = "cuda"
) -> Optional[Tensor]:
    """
    Prepare Bézier curve inputs for the pipeline.
    
    Args:
        bezier_curves: Bézier curve specification
        normalize: Whether to normalize coordinates to [0, 1]
        validate: Whether to validate curve points
        device: Target device for tensor
    
    Returns:
        Tensor of shape (1, N, 2) or None if no curves provided
    """
    if bezier_curves is None:
        return None
    
    # Load curves
    points = load_bezier_curves(bezier_curves)
    
    if not points:
        return None
    
    # Normalize if requested
    if normalize:
        points = normalize_bezier_points(points)
    
    # Validate if requested
    if validate and not validate_bezier_curves(points):
        raise ValueError("Invalid Bézier curve points")
    
    # Convert to tensor
    tensor = torch.tensor(points, dtype=torch.float32, device=device)
    
    # Add batch dimension
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    
    return tensor


def save_bezier_curves(
    points: List[Tuple[float, float]],
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save Bézier curves to JSON file.
    
    Args:
        points: List of (x, y) control points
        output_path: Output file path
        metadata: Optional metadata to include
    """
    data = {
        "control_points": points,
        "num_points": len(points),
        "format": "bezier_curves_v1"
    }
    
    if metadata:
        data["metadata"] = metadata
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def interpolate_bezier_curves(
    curve1: List[Tuple[float, float]],
    curve2: List[Tuple[float, float]],
    t: float
) -> List[Tuple[float, float]]:
    """
    Interpolate between two Bézier curves.
    
    Args:
        curve1: First curve control points
        curve2: Second curve control points
        t: Interpolation factor (0.0 = curve1, 1.0 = curve2)
    
    Returns:
        Interpolated curve control points
    """
    if len(curve1) != len(curve2):
        raise ValueError("Curves must have the same number of control points")
    
    interpolated = []
    for (x1, y1), (x2, y2) in zip(curve1, curve2):
        x_interp = x1 + t * (x2 - x1)
        y_interp = y1 + t * (y2 - y1)
        interpolated.append((x_interp, y_interp))
    
    return interpolated


def create_font_dataset_sample(
    character: str,
    font_style: str = "serif",
    output_dir: Union[str, Path] = "font_samples",
    generate_curves: bool = True
) -> Dict[str, Any]:
    """
    Create a sample dataset entry for font training.
    
    Args:
        character: Character to create sample for
        font_style: Font style description
        output_dir: Output directory for files
        generate_curves: Whether to generate Bézier curves
    
    Returns:
        Dataset sample dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample = {
        "character": character,
        "font_style": font_style,
        "image_path": str(output_dir / f"{character}.png"),
        "style_prompt": f"{font_style} typography character '{character}'"
    }
    
    if generate_curves:
        # Generate curves for the character
        curves = create_bezier_from_character(character, method="outline")
        curve_path = output_dir / f"{character}_curves.json"
        save_bezier_curves(curves, curve_path, {"character": character, "style": font_style})
        sample["bezier_curves_path"] = str(curve_path)
        sample["control_points"] = curves
    
    return sample


def batch_process_characters(
    characters: List[str],
    font_style: str = "serif",
    output_dir: Union[str, Path] = "font_batch",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Batch process multiple characters for dataset creation.
    
    Args:
        characters: List of characters to process
        font_style: Font style description
        output_dir: Output directory
        **kwargs: Additional arguments for create_font_dataset_sample
    
    Returns:
        List of dataset sample dictionaries
    """
    samples = []
    
    for char in characters:
        sample = create_font_dataset_sample(
            character=char,
            font_style=font_style,
            output_dir=Path(output_dir) / font_style,
            **kwargs
        )
        samples.append(sample)
    
    return samples