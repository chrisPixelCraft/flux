#!/usr/bin/env python3
"""
Example usage of BezierFluxPipeline for font generation and style transfer.

This script demonstrates how to use the BezierFluxPipeline for:
1. Basic font character generation
2. Bézier-guided font control
3. Style transfer between fonts (Fill model)
4. Batch character generation
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from flux.pipeline import BezierFluxPipeline, load_bezier_curves, prepare_bezier_inputs
from flux.pipeline.utils import create_bezier_from_character, create_font_dataset_sample
from PIL import Image


def example_basic_font_generation():
    """Example 1: Basic font character generation with Bézier guidance."""
    print("🎨 Example 1: Basic Font Character Generation")
    print("=" * 50)
    
    try:
        # Load pipeline with FLUX.1-Fill-dev
        print("Loading BezierFluxPipeline...")
        pipeline = BezierFluxPipeline.from_pretrained(
            model_name="flux-dev-fill",  # Use Fill model for font generation
            device="cuda"
        )
        
        # Print model info
        info = pipeline.get_model_info()
        print("Pipeline loaded successfully!")
        print(f"Model: {info['model_type']}")
        print(f"Parameters: {info['total_parameters']:,} total, {info['bezier_parameters']:,} BezierAdapter")
        print(f"Efficiency: {info['parameter_efficiency']}")
        print()
        
        # Create Bézier curves for letter 'A'
        print("Creating Bézier curves for letter 'A'...")
        bezier_curves = create_bezier_from_character('A', method='outline')
        print(f"Generated {len(bezier_curves)} control points")
        
        # Save curves for inspection
        curves_file = "example_A_curves.json"
        with open(curves_file, 'w') as f:
            json.dump({"control_points": bezier_curves, "character": "A"}, f, indent=2)
        print(f"Saved curves to {curves_file}")
        print()
        
        # Generate character with Bézier guidance
        print("Generating character 'A' with Bézier guidance...")
        image = pipeline.generate_font_character(
            character='A',
            font_style='elegant serif',
            bezier_curves=bezier_curves,
            width=512,
            height=512,
            num_steps=20,  # Reduced for faster generation
            guidance=7.5,
            seed=42
        )
        
        # Save result
        output_path = "generated_character_A.png"
        image.save(output_path)
        print(f"✅ Generated character saved to {output_path}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic font generation: {e}")
        return False


def example_batch_character_generation():
    """Example 2: Batch generation of multiple characters."""
    print("🔤 Example 2: Batch Character Generation")
    print("=" * 50)
    
    try:
        # Load pipeline
        pipeline = BezierFluxPipeline.from_pretrained(
            model_name="flux-dev-fill",
            device="cuda"
        )
        
        # Generate multiple characters
        characters = ['B', 'C', 'D']
        font_style = 'modern sans-serif'
        
        print(f"Generating {len(characters)} characters in {font_style} style...")
        
        results = []
        for char in characters:
            print(f"  Generating '{char}'...")
            
            # Create character-specific curves
            curves = create_bezier_from_character(char, method='outline')
            
            # Generate image
            image = pipeline.generate_font_character(
                character=char,
                font_style=font_style,
                bezier_curves=curves,
                width=256,  # Smaller for batch processing
                height=256,
                num_steps=15,
                seed=42 + ord(char)  # Different seed per character
            )
            
            # Save result
            output_path = f"generated_character_{char}.png"
            image.save(output_path)
            results.append(output_path)
            
            print(f"    ✅ Saved to {output_path}")
        
        print(f"✅ Batch generation completed! Generated {len(results)} characters.")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in batch generation: {e}")
        return False


def example_style_transfer():
    """Example 3: Font style transfer using Fill model."""
    print("🎭 Example 3: Font Style Transfer")
    print("=" * 50)
    
    try:
        # Load pipeline
        pipeline = BezierFluxPipeline.from_pretrained(
            model_name="flux-dev-fill",
            device="cuda"
        )
        
        if not pipeline.is_fill_model:
            print("⚠️  Style transfer requires FLUX.1-Fill-dev model")
            return False
        
        print("Creating example images for style transfer...")
        
        # Create a simple source image (placeholder - in practice you'd load real images)
        source_image = Image.new('RGB', (512, 512), color='white')
        # In practice, this would be an image of text/character to be restyled
        
        # Create a style reference image (placeholder)
        style_image = Image.new('RGB', (512, 512), color='lightgray')
        # In practice, this would be an example of the target font style
        
        # Create a mask defining areas to modify
        mask_image = Image.new('L', (512, 512), color=128)  # Gray mask
        # In practice, this would define which parts of the source to modify
        
        print("Performing style transfer...")
        result = pipeline.transfer_font_style(
            source_image=source_image,
            target_style_image=style_image,
            mask_image=mask_image,
            prompt="elegant serif font style transfer",
            num_steps=20,
            guidance=15.0,  # Higher guidance for style transfer
            seed=42
        )
        
        # Save result
        output_path = "style_transfer_result.png"
        result.save(output_path)
        print(f"✅ Style transfer result saved to {output_path}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in style transfer: {e}")
        return False


def example_custom_bezier_curves():
    """Example 4: Using custom Bézier curves from file."""
    print("📐 Example 4: Custom Bézier Curves")
    print("=" * 50)
    
    try:
        # Create custom Bézier curves
        custom_curves = [
            (0.1, 0.2),   # Start point
            (0.3, 0.8),   # Control point 1
            (0.7, 0.8),   # Control point 2
            (0.9, 0.2),   # End point
            (0.5, 0.1),   # Additional detail point
        ]
        
        # Save to file
        curves_file = "custom_curves.json"
        with open(curves_file, 'w') as f:
            json.dump({
                "control_points": custom_curves,
                "description": "Custom stylized character curve",
                "author": "BezierFluxPipeline Example"
            }, f, indent=2)
        
        print(f"Created custom curves file: {curves_file}")
        
        # Load pipeline
        pipeline = BezierFluxPipeline.from_pretrained(
            model_name="flux-dev-fill",
            device="cuda"
        )
        
        # Generate using custom curves
        print("Generating image with custom Bézier curves...")
        image = pipeline.generate(
            prompt="stylized typography with flowing curves",
            bezier_curves=curves_file,  # Load from file
            width=512,
            height=512,
            num_steps=25,
            guidance=8.0,
            seed=123
        )
        
        # Save result
        output_path = "custom_curves_result.png"
        image.save(output_path)
        print(f"✅ Custom curves result saved to {output_path}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error with custom curves: {e}")
        return False


def main():
    """Run all examples."""
    print("🚀 BezierFluxPipeline Examples")
    print("=" * 50)
    print()
    
    # Note about model loading
    print("⚠️  Note: These examples require FLUX.1-Fill-dev model which is large (~12B parameters)")
    print("   Model initialization may take several minutes on first run.")
    print("   Ensure you have sufficient GPU memory (12GB+ recommended).")
    print()
    
    examples = [
        ("Basic Font Generation", example_basic_font_generation),
        ("Batch Character Generation", example_batch_character_generation), 
        ("Font Style Transfer", example_style_transfer),
        ("Custom Bézier Curves", example_custom_bezier_curves)
    ]
    
    results = []
    
    for name, example_func in examples:
        print(f"Running: {name}")
        try:
            success = example_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")
            results.append((name, False))
        
        print()
    
    # Summary
    print("📊 Example Results Summary")
    print("=" * 50)
    
    for name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {name}: {status}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {successful}/{total} examples completed successfully")
    
    if successful == total:
        print("🎉 All examples completed successfully!")
        print("\nGenerated files:")
        print("  - generated_character_*.png: Individual character generations")
        print("  - style_transfer_result.png: Style transfer result")
        print("  - custom_curves_result.png: Custom Bézier curves result")
        print("  - example_A_curves.json: Example Bézier curve data")
        print("  - custom_curves.json: Custom curve specification")
    else:
        print("⚠️  Some examples failed. Check error messages above.")
        print("   Common issues:")
        print("   - Insufficient GPU memory (need 12GB+ for FLUX.1-Fill-dev)")
        print("   - Missing model files or network issues")
        print("   - CUDA not available")


if __name__ == "__main__":
    main()