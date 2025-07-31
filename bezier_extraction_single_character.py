#!/usr/bin/env python3
"""
Script to process a single Chinese character directory for Bézier curve extraction.
Uses the same high-detail parameters as the main bezier_extraction.py
"""

import os
import sys
import time
from pathlib import Path
from bezier_extraction import BezierCurveExtractor

def process_single_character(input_path: str, output_dir: str = None, character_name: str = None):
    """
    Process a single image file or all images in a character directory to extract Bézier curves.

    Args:
        input_path: Path to either a single image file or a character directory containing images
        output_dir: Output directory (if None, uses character_name + "_bezier_output" or "single_image_output")
        character_name: Name of the character (if None, extracted from directory/file name)
    """
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"Error: Input path not found: {input_path}")
        return False

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # Determine if input is a file or directory
    is_single_file = os.path.isfile(input_path)
    is_directory = os.path.isdir(input_path)

    if not (is_single_file or is_directory):
        print(f"Error: Input path is neither a file nor a directory: {input_path}")
        return False

    # Handle single file case
    if is_single_file:
        # Check if it's an image file
        if not any(input_path.lower().endswith(ext) for ext in image_extensions):
            print(f"Error: File is not a supported image format: {input_path}")
            print(f"Supported formats: {', '.join(image_extensions)}")
            return False

        # Extract file info
        file_dir = os.path.dirname(input_path)
        file_name = os.path.basename(input_path)
        base_name = os.path.splitext(file_name)[0]

        if character_name is None:
            character_name = base_name

        if output_dir is None:
            output_dir = f"{base_name}_bezier_output"

        # Create list with single image
        image_files = [file_name]
        source_dir = file_dir
        processing_type = "Single Image"

    else:  # Directory case
        # Extract character name if not provided
        if character_name is None:
            character_name = os.path.basename(input_path.rstrip('/\\'))

        # Set default output directory if not provided
        if output_dir is None:
            output_dir = f"{character_name}_bezier_output"

        # Find all images in the character directory
        image_files = []
        for file in os.listdir(input_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)

        if not image_files:
            print(f"No image files found in directory: {input_path}")
            return False

        source_dir = input_path
        processing_type = "Character Directory"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"{processing_type} Bézier Curve Extraction")
    print("=" * 50)
    print(f"Target: {character_name}")
    print(f"Source: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Images to process: {len(image_files)}")
    print("=" * 50)

    # Initialize extractor with high-detail parameters
    extractor = BezierCurveExtractor()

    print("High-Detail Parameters:")
    print(f"  - Smoothing factor: {extractor.smoothing_factor} (minimal smoothing)")
    print(f"  - Max points: {extractor.max_points} (high accuracy)")
    print(f"  - Blur kernel: {extractor.blur_kernel_size}x{extractor.blur_kernel_size} (minimal blur)")
    print(f"  - Adaptive block size: {extractor.adaptive_block_size} (fine detail)")
    print(f"  - Adaptive C: {extractor.adaptive_c} (minimal threshold adjustment)")
    print(f"  - Min contour area: {extractor.min_contour_area} (captures small strokes)")
    print(f"  - Max segments: {extractor.max_segments} (detailed representation)")
    print(f"  - Curve resolution: {extractor.curve_resolution} (smooth curves)")
    print(f"  - Visualization alpha: {extractor.visualization_alpha} (visible overlay)")
    print("=" * 50)
    print()

    # Track statistics
    processed_images = 0
    failed_images = 0
    total_characters = 0
    total_bezier_segments = 0
    start_time = time.time()

    # Process each image
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(source_dir, image_file)

        # Create output file names
        base_name = os.path.splitext(image_file)[0]
        output_json = os.path.join(output_dir, f"{base_name}_bezier.json")
        output_viz = os.path.join(output_dir, f"{base_name}_visualization.jpg")

        try:
            # Progress indicator
            if len(image_files) > 1:
                progress = (i / len(image_files)) * 100
                print(f"[{i}/{len(image_files)}] ({progress:.1f}%) Processing: {image_file}")
            else:
                print(f"Processing: {image_file}")

            # Extract Bézier curves
            bezier_data = extractor.extract_character_bezier(image_path)

            # Save Bézier data
            extractor.save_bezier_data(bezier_data, output_json)

            # Create visualization
            extractor.visualize_bezier_curves(image_path, bezier_data, output_viz)

            # Update statistics
            chars_in_image = len(bezier_data['characters'])
            segments_in_image = sum(len(char['bezier_curves']) for char in bezier_data['characters'])
            total_characters += chars_in_image
            total_bezier_segments += segments_in_image
            processed_images += 1

            print(f"  -> JSON: {output_json}")
            print(f"  -> Visualization: {output_viz}")
            print(f"  -> Characters: {chars_in_image}, Bézier segments: {segments_in_image}")

            # Show sample data for single image processing
            if is_single_file and bezier_data['characters']:
                print(f"  -> Image dimensions: {bezier_data.get('image_dimensions', 'N/A')}")
                sample_char = bezier_data['characters'][0]
                print(f"  -> Largest character area: {sample_char['contour_area']:.1f}")
                print(f"  -> Bounding box: {sample_char['bounding_box']}")
                if sample_char['bezier_curves']:
                    sample_curve = sample_char['bezier_curves'][0]
                    print(f"  -> Sample control points: {len(sample_curve)} points")

        except Exception as e:
            failed_images += 1
            print(f"  -> ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

        print()  # Empty line for readability

    # Final statistics
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Target: {character_name}")
    print(f"Type: {processing_type}")
    print(f"Total images: {len(image_files)}")
    print(f"Successfully processed: {processed_images}")
    print(f"Failed: {failed_images}")
    print(f"Total characters extracted: {total_characters}")
    print(f"Total Bézier segments: {total_bezier_segments}")
    if processed_images > 0:
        print(f"Average characters per image: {total_characters / processed_images:.1f}")
    if total_characters > 0:
        print(f"Average segments per character: {total_bezier_segments / total_characters:.1f}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    if len(image_files) > 0:
        print(f"Average time per image: {elapsed_time / len(image_files):.2f} seconds")
    print("=" * 50)

    return processed_images > 0

def main():
    """
    Main function for command-line usage.
    """
    if len(sys.argv) < 2:
        print("Usage: python bezier_extraction_single_character.py <input_path> [output_directory]")
        print()
        print("Input can be either:")
        print("  - A single image file (e.g., image.jpg)")
        print("  - A character directory containing multiple images")
        print()
        print("Examples:")
        print("  # Process a single image")
        print("  python bezier_extraction_single_character.py chinese-calligraphy-dataset/chinese-calligraphy-dataset/佐/19527.jpg")
        print("  python bezier_extraction_single_character.py path/to/single_image.jpg custom_output")
        print()
        print("  # Process a character directory")
        print("  python bezier_extraction_single_character.py chinese-calligraphy-dataset/chinese-calligraphy-dataset/佐")
        print("  python bezier_extraction_single_character.py chinese-calligraphy-dataset/chinese-calligraphy-dataset/佐 custom_output")
        print()
        print("Available characters:")

        # Show available characters if dataset exists
        dataset_path = "chinese-calligraphy-dataset/chinese-calligraphy-dataset"
        if os.path.exists(dataset_path):
            characters = []
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    characters.append(item)

            if characters:
                characters.sort()
                print(f"  Found {len(characters)} character directories:")
                for i, char in enumerate(characters[:10]):  # Show first 10
                    print(f"    {char}")
                if len(characters) > 10:
                    print(f"    ... and {len(characters) - 10} more")

                print()
                print("Example paths:")
                print(f"  Directory: {dataset_path}/{characters[0]}")
                if len(characters) > 0:
                    # Try to find a sample image in the first character directory
                    sample_dir = os.path.join(dataset_path, characters[0])
                    if os.path.exists(sample_dir):
                        for file in os.listdir(sample_dir):
                            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                print(f"  Single image: {dataset_path}/{characters[0]}/{file}")
                                break

        return 1

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    success = process_single_character(input_path, output_dir)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())