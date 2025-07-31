#!/usr/bin/env python3
"""
Demo: Bezier Curve Extraction to Font Generation

This script demonstrates the complete workflow:
1. Extract Bezier curves from a calligraphy image
2. Process and visualize the curves
3. Generate new font styles using the extracted curves (requires FLUX model)

Usage:
    python demo_bezier_to_font.py [image_path]
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import cv2

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import bezier extraction
from bezier_extraction import BezierCurveExtractor

# Try to import FLUX components (optional)
try:
    import torch
    from flux.pipeline.utils import (
        normalize_bezier_points,
        save_bezier_curves,
        create_bezier_from_character
    )
    FLUX_AVAILABLE = True
except ImportError:
    FLUX_AVAILABLE = False
    print("⚠️  FLUX components not available. Install PyTorch for full functionality.")


def extract_and_visualize_bezier(image_path, output_dir="demo_output"):
    """Extract Bezier curves from an image and create visualizations."""
    print(f"\n📊 Extracting Bezier curves from: {image_path}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor with high-detail parameters
    extractor = BezierCurveExtractor(
        smoothing_factor=0.001,
        max_points=400,
        min_contour_area=25,
        max_segments=50,
        visualization_alpha=0.7
    )
    
    # Extract Bezier curves
    try:
        bezier_data = extractor.extract_character_bezier(image_path)
        print(f"✅ Extracted {len(bezier_data['characters'])} characters")
        
        # Save Bezier data
        json_path = output_dir / "bezier_curves.json"
        extractor.save_bezier_data(bezier_data, str(json_path))
        print(f"💾 Saved Bezier data to: {json_path}")
        
        # Create visualization
        viz_path = output_dir / "bezier_visualization.jpg"
        visualization = extractor.visualize_bezier_curves(image_path, bezier_data, str(viz_path))
        print(f"🎨 Created visualization: {viz_path}")
        
        # Print statistics
        total_curves = sum(len(char['bezier_curves']) for char in bezier_data['characters'])
        total_points = sum(char['original_contour_points'] for char in bezier_data['characters'])
        
        print(f"\n📈 Extraction Statistics:")
        print(f"  - Total characters: {len(bezier_data['characters'])}")
        print(f"  - Total Bezier segments: {total_curves}")
        print(f"  - Total original points: {total_points}")
        print(f"  - Compression ratio: {total_points / (total_curves * 4):.1f}:1")
        
        return bezier_data
        
    except Exception as e:
        print(f"❌ Error extracting Bezier curves: {e}")
        return None


def analyze_bezier_complexity(bezier_data):
    """Analyze the complexity of extracted Bezier curves."""
    print("\n🔍 Analyzing Bezier Curve Complexity:")
    
    for i, char_data in enumerate(bezier_data['characters']):
        print(f"\nCharacter {i + 1}:")
        print(f"  - Contour area: {char_data['contour_area']:.1f} pixels²")
        print(f"  - Bounding box: {char_data['bounding_box']}")
        print(f"  - Original points: {char_data['original_contour_points']}")
        print(f"  - Bezier segments: {len(char_data['bezier_curves'])}")
        
        # Analyze curve distribution
        curve_lengths = []
        for curve in char_data['bezier_curves']:
            # Calculate approximate curve length
            points = np.array(curve)
            diffs = np.diff(points, axis=0)
            length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
            curve_lengths.append(length)
        
        if curve_lengths:
            print(f"  - Average segment length: {np.mean(curve_lengths):.1f}")
            print(f"  - Length std deviation: {np.std(curve_lengths):.1f}")


def create_font_samples(bezier_data, output_dir="demo_output"):
    """Create font sample visualizations from Bezier curves."""
    if not FLUX_AVAILABLE:
        print("\n⚠️  Skipping font generation (PyTorch not available)")
        return
    
    print("\n🎨 Creating Font Samples:")
    output_dir = Path(output_dir)
    
    # Create samples for different styles
    styles = ["serif", "sans-serif", "script", "modern"]
    
    for style in styles:
        print(f"\n  Creating {style} style sample...")
        
        # For demo, we'll create placeholder visualizations
        # In production, this would use the FLUX pipeline
        style_dir = output_dir / "font_styles" / style
        style_dir.mkdir(parents=True, exist_ok=True)
        
        # Save style configuration
        style_config = {
            "style": style,
            "bezier_source": str(output_dir / "bezier_curves.json"),
            "generation_params": {
                "guidance_scale": 7.5,
                "num_inference_steps": 50,
                "style_strength": 0.8
            }
        }
        
        with open(style_dir / "config.json", 'w') as f:
            json.dump(style_config, f, indent=2)
        
        print(f"    ✅ Saved configuration for {style}")


def demonstrate_bezier_manipulation(bezier_data, output_dir="demo_output"):
    """Demonstrate Bezier curve manipulation capabilities."""
    print("\n🔧 Demonstrating Bezier Manipulation:")
    
    if not bezier_data['characters']:
        print("  ❌ No characters found for manipulation")
        return
    
    # Take the first character's curves
    char_data = bezier_data['characters'][0]
    curves = char_data['bezier_curves']
    
    if not curves:
        print("  ❌ No curves found for manipulation")
        return
    
    output_dir = Path(output_dir) / "manipulations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Simplification - reduce number of segments
    print("\n  1. Curve Simplification:")
    simplified_curves = curves[::2]  # Take every other curve
    print(f"    - Original segments: {len(curves)}")
    print(f"    - Simplified segments: {len(simplified_curves)}")
    
    # 2. Smoothing - interpolate between control points
    print("\n  2. Curve Smoothing:")
    if len(curves) > 0 and FLUX_AVAILABLE:
        first_curve = np.array(curves[0])
        smoothed_curve = []
        for i in range(len(first_curve) - 1):
            smoothed_curve.append(first_curve[i].tolist())
            # Add intermediate point
            mid_point = (first_curve[i] + first_curve[i+1]) / 2
            smoothed_curve.append(mid_point.tolist())
        smoothed_curve.append(first_curve[-1].tolist())
        print(f"    - Original control points: {len(first_curve)}")
        print(f"    - Smoothed control points: {len(smoothed_curve)}")
    
    # 3. Transformation - scale and rotate
    print("\n  3. Geometric Transformations:")
    if FLUX_AVAILABLE:
        # Normalize curves to [0, 1] range
        all_points = []
        for curve in curves:
            all_points.extend(curve)
        
        normalized = normalize_bezier_points(all_points)
        print(f"    - Normalized {len(normalized)} control points to [0,1] range")
        
        # Save normalized curves
        save_bezier_curves(
            normalized,
            output_dir / "normalized_curves.json",
            metadata={"transformation": "normalized", "original_count": len(all_points)}
        )
        print(f"    - Saved normalized curves")


def create_demo_report(bezier_data, output_dir="demo_output"):
    """Create a comprehensive demo report."""
    print("\n📝 Creating Demo Report:")
    
    output_dir = Path(output_dir)
    report_path = output_dir / "demo_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Bezier Curve Extraction Demo Report\n\n")
        
        f.write("## Overview\n")
        f.write(f"- Total characters extracted: {len(bezier_data['characters'])}\n")
        f.write(f"- Image path: {bezier_data['image_path']}\n\n")
        
        f.write("## Character Analysis\n")
        for i, char in enumerate(bezier_data['characters']):
            f.write(f"\n### Character {i + 1}\n")
            f.write(f"- Area: {char['contour_area']:.1f} pixels²\n")
            f.write(f"- Bounding box: {char['bounding_box']}\n")
            f.write(f"- Original contour points: {char['original_contour_points']}\n")
            f.write(f"- Bezier segments: {len(char['bezier_curves'])}\n")
            f.write(f"- Points per segment: 4 (cubic Bezier)\n")
            f.write(f"- Total control points: {len(char['bezier_curves']) * 4}\n")
        
        f.write("\n## Next Steps\n")
        f.write("1. **Training**: Use the extracted curves to train BezierAdapter\n")
        f.write("2. **Generation**: Generate new font styles with FLUX\n")
        f.write("3. **Style Transfer**: Transfer styles between different fonts\n")
        f.write("4. **Refinement**: Fine-tune curves for specific applications\n")
        
        f.write("\n## Technical Details\n")
        f.write("- Extraction method: OpenCV contour detection + B-spline smoothing\n")
        f.write("- Curve fitting: Least-squares Bezier approximation\n")
        f.write("- Segmentation: Adaptive based on contour complexity\n")
        f.write("- Output format: JSON with hierarchical structure\n")
    
    print(f"✅ Report saved to: {report_path}")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Demo: Bezier curve extraction to font generation"
    )
    parser.add_argument(
        "image_path",
        nargs='?',
        default=None,
        help="Path to input image (optional)"
    )
    parser.add_argument(
        "--output-dir",
        default="demo_output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--character",
        default="A",
        help="Character to use for synthetic demo"
    )
    
    args = parser.parse_args()
    
    print("🚀 Bezier Curve Extraction to Font Generation Demo")
    print("=" * 60)
    
    # If no image provided, create synthetic demo
    if args.image_path is None:
        print("\n📝 No image provided. Creating synthetic demo...")
        
        # Create synthetic Bezier data
        if FLUX_AVAILABLE:
            curves = create_bezier_from_character(args.character, method="outline")
            bezier_data = {
                "image_path": f"synthetic_{args.character}.png",
                "characters": [{
                    "character_id": 0,
                    "contour_area": 1000.0,
                    "bounding_box": [0, 0, 100, 100],
                    "bezier_curves": [curves[i:i+4] for i in range(0, len(curves)-3, 3)],
                    "original_contour_points": len(curves) * 10
                }]
            }
            
            # Save synthetic data
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / "bezier_curves.json", 'w') as f:
                json.dump(bezier_data, f, indent=2)
            
            print(f"✅ Created synthetic Bezier data for character '{args.character}'")
        else:
            print("❌ Cannot create synthetic data without PyTorch")
            return
    else:
        # Extract from real image
        bezier_data = extract_and_visualize_bezier(args.image_path, args.output_dir)
        if bezier_data is None:
            return
    
    # Analyze complexity
    analyze_bezier_complexity(bezier_data)
    
    # Create font samples
    create_font_samples(bezier_data, args.output_dir)
    
    # Demonstrate manipulation
    demonstrate_bezier_manipulation(bezier_data, args.output_dir)
    
    # Create report
    create_demo_report(bezier_data, args.output_dir)
    
    print("\n✅ Demo completed successfully!")
    print(f"📁 Results saved to: {args.output_dir}/")
    print("\n🎯 Next Steps:")
    print("1. Review the extracted Bezier curves in bezier_curves.json")
    print("2. Check the visualization in bezier_visualization.jpg")
    print("3. Read the detailed report in demo_report.md")
    
    if FLUX_AVAILABLE:
        print("4. Train BezierAdapter with: python train_fill_model.py")
        print("5. Generate fonts with: python example_bezier_pipeline.py")
    else:
        print("4. Install PyTorch to enable font generation features")
        print("5. Download FLUX model checkpoints for full functionality")


if __name__ == "__main__":
    main()