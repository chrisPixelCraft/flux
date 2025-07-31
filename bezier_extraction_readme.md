# Calligraphy Bézier Curve Extractor

This project provides tools to extract character boundaries from calligraphy images and convert them into Bézier curves for further analysis and processing.

## Features

- **Automatic Character Detection**: Detects individual characters in calligraphy images
- **Boundary Extraction**: Extracts precise character boundaries using advanced image processing
- **Bézier Curve Conversion**: Converts boundaries to mathematical Bézier curves
- **Curve Segmentation**: Intelligently segments complex boundaries into multiple Bézier curves
- **Visualization**: Creates visual overlays showing extracted curves on original images
- **Batch Processing**: Processes entire datasets automatically
- **Multiple Output Formats**: Saves data in JSON or pickle format

## Installation

1. **Clone or download the project files**

2. **Install required dependencies**:
```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python (≥4.5.0)
- numpy (≥1.21.0)
- scipy (≥1.7.0)
- matplotlib (≥3.3.0)
- Pillow (≥8.0.0)

## Quick Start

### Test on a Single Image

```bash
python demo_single_image.py path/to/your/image.jpg
```

This will:
- Extract character boundaries from the image
- Convert them to Bézier curves
- Save the results to `demo_output/bezier_curves.json`
- Create a visualization at `demo_output/visualization.jpg`

### Process an Entire Dataset

```bash
python bezier_extraction.py
```

Or modify the script to point to your dataset:

```python
from bezier_extraction import process_calligraphy_dataset

# Process your dataset
dataset_dir = "path/to/your/calligraphy/dataset"
output_dir = "bezier_curves_output"
process_calligraphy_dataset(dataset_dir, output_dir)
```

## How It Works

### 1. Image Preprocessing
- Converts images to grayscale
- Applies Gaussian blur to reduce noise
- Uses adaptive thresholding for robust binarization
- Applies morphological operations to clean the image

### 2. Character Detection
- Finds contours using OpenCV
- Filters contours by area to remove noise
- Sorts characters by size

### 3. Boundary Smoothing
- Uses B-spline interpolation to smooth contours
- Reduces noise while preserving important features
- Resamples points for optimal processing

### 4. Bézier Curve Fitting
- Converts smooth boundaries to Bézier curves
- Uses least-squares fitting with Bernstein polynomials
- Segments long boundaries into multiple curves for better accuracy

### 5. Data Storage
- Saves control points for each Bézier curve
- Includes metadata (area, bounding box, etc.)
- Supports both JSON and pickle formats

## Output Format

The extracted data is saved in JSON format with the following structure:

```json
{
  "image_path": "path/to/image.jpg",
  "characters": [
    {
      "character_id": 0,
      "contour_area": 1250.5,
      "bounding_box": [x, y, width, height],
      "bezier_curves": [
        [
          [x1, y1],  // Control point 1
          [x2, y2],  // Control point 2
          [x3, y3],  // Control point 3
          [x4, y4]   // Control point 4
        ],
        // Additional curve segments...
      ],
      "original_contour_points": 156
    }
    // Additional characters...
  ]
}
```

## Customization

### Adjust Processing Parameters

```python
extractor = BezierCurveExtractor(
    smoothing_factor=0.1,    # Lower = more faithful to original, higher = smoother
    max_points=100           # Maximum points to sample from contours
)
```

### Modify Curve Fitting

```python
# Change the degree of Bézier curves (3 = cubic, 2 = quadratic)
control_points = extractor.fit_bezier_curve(x_points, y_points, degree=3)

# Adjust segmentation
bezier_segments = extractor.segment_contour_to_bezier(
    x_points, y_points,
    max_segments=10  # Maximum number of curve segments
)
```

### Filter Characters

```python
# Modify minimum area threshold in extract_character_contours
min_area = 200  # Increase to filter out smaller characters/noise
```

## Advanced Usage

### Working with Extracted Data

```python
import json
from bezier_extraction import BezierCurveExtractor

# Load saved data
with open('bezier_curves.json', 'r') as f:
    data = json.load(f)

# Access character data
for char in data['characters']:
    print(f"Character {char['character_id']}:")
    print(f"  Area: {char['contour_area']}")
    print(f"  Number of curve segments: {len(char['bezier_curves'])}")

    # Work with individual Bézier curves
    for i, curve in enumerate(char['bezier_curves']):
        control_points = np.array(curve)
        # Generate points on the curve
        extractor = BezierCurveExtractor()
        curve_points = extractor.evaluate_bezier_curve(control_points, 100)
```

### Custom Visualization

```python
import cv2
import numpy as np

# Create custom visualizations
def custom_visualization(image_path, bezier_data, output_path):
    img = cv2.imread(image_path)

    for char_data in bezier_data['characters']:
        for bezier_curve in char_data['bezier_curves']:
            control_points = np.array(bezier_curve)

            # Your custom drawing code here
            # e.g., different colors for different characters
            color = (0, 255, 0)  # Green
            cv2.polylines(img, [control_points.astype(int)], False, color, 2)

    cv2.imwrite(output_path, img)
```

## Troubleshooting

### Common Issues

1. **No characters detected**:
   - Check image quality and contrast
   - Adjust the `min_area` threshold
   - Try different preprocessing parameters

2. **Poor curve fitting**:
   - Increase `smoothing_factor` for smoother curves
   - Decrease `max_segments` for simpler representation
   - Adjust `max_points` for better sampling

3. **Memory issues with large datasets**:
   - Process images in batches
   - Reduce `max_points` parameter
   - Use pickle format for more efficient storage

### Performance Tips

- **For large datasets**: Process images in parallel using multiprocessing
- **For memory efficiency**: Process one image at a time instead of loading all data
- **For speed**: Reduce image resolution before processing if high precision isn't needed

## Applications

This tool is useful for:

- **Calligraphy analysis**: Study stroke patterns and character formation
- **Font generation**: Create digital fonts from handwritten samples
- **Style transfer**: Analyze and reproduce calligraphy styles
- **Character recognition**: Extract features for machine learning models
- **Digital preservation**: Create mathematical representations of historical texts
- **Educational tools**: Teach calligraphy through stroke analysis

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues or pull requests to improve the functionality.