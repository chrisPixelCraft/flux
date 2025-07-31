import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import pickle

class BezierCurveExtractor:
    def __init__(self,
                 smoothing_factor: float = 0.001,    # Minimal smoothing for maximum detail
                 max_points: int = 400,              # Maximum points for high accuracy
                 blur_kernel_size: int = 3,          # Minimal blur
                 adaptive_block_size: int = 7,       # Very small blocks for detail
                 adaptive_c: int = 1,                # Minimal threshold adjustment
                 min_contour_area: int = 25,         # Very low threshold for small strokes
                 max_segments: int = 50,             # Many segments for detailed representation
                 curve_resolution: int = 150,        # High resolution curves
                 visualization_alpha: float = 0.5):  # More visible overlay
        """
        Initialize the Bézier curve extractor with high-detail parameters.

        Args:
            smoothing_factor: Controls the smoothness of the spline fitting (lower = more detail)
            max_points: Maximum number of points to sample from contours
            blur_kernel_size: Size of Gaussian blur kernel for preprocessing
            adaptive_block_size: Block size for adaptive thresholding
            adaptive_c: Constant subtracted from mean in adaptive thresholding
            min_contour_area: Minimum area threshold for contour filtering
            max_segments: Maximum number of segments to create per contour
            curve_resolution: Number of points to evaluate for each Bézier curve
            visualization_alpha: Alpha blending factor for visualization overlay
        """
        self.smoothing_factor = smoothing_factor
        self.max_points = max_points
        self.blur_kernel_size = blur_kernel_size
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.min_contour_area = min_contour_area
        self.max_segments = max_segments
        self.curve_resolution = curve_resolution
        self.visualization_alpha = visualization_alpha

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess the calligraphy image for character extraction.

        Args:
            image_path: Path to the input image

        Returns:
            Preprocessed binary image
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)

        # Apply adaptive thresholding to handle varying lighting
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.adaptive_block_size, self.adaptive_c
        )

        # Apply morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        return cleaned

    def extract_character_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        Extract character contours from the binary image.

        Args:
            binary_image: Preprocessed binary image

        Returns:
            List of contours representing character boundaries
        """
        # Find contours
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area to remove noise
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area]

        # Sort contours by area (largest first)
        filtered_contours.sort(key=cv2.contourArea, reverse=True)

        return filtered_contours

    def smooth_contour(self, contour: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Smooth the contour using B-spline interpolation.

        Args:
            contour: Input contour points

        Returns:
            Tuple of (smoothed x coordinates, smoothed y coordinates)
        """
        # Extract x, y coordinates
        contour = contour.reshape(-1, 2)
        x, y = contour[:, 0], contour[:, 1]

        # Ensure we have enough points
        if len(x) < 4:
            return x, y

        # Resample to reduce number of points if necessary
        if len(x) > self.max_points:
            indices = np.linspace(0, len(x) - 1, self.max_points, dtype=int)
            x, y = x[indices], y[indices]

        # Fit a parametric spline
        try:
            tck, u = splprep([x, y], s=self.smoothing_factor * len(x), per=True)

            # Generate smooth curve
            u_new = np.linspace(0, 1, len(x))
            x_smooth, y_smooth = splev(u_new, tck)

            return np.array(x_smooth), np.array(y_smooth)
        except:
            # Fallback to original points if spline fitting fails
            return x, y

    def fit_bezier_curve(self, x_points: np.ndarray, y_points: np.ndarray,
                        degree: int = 3) -> np.ndarray:
        """
        Fit a Bézier curve to the given points.

        Args:
            x_points: X coordinates of the curve
            y_points: Y coordinates of the curve
            degree: Degree of the Bézier curve (default: cubic)

        Returns:
            Control points of the Bézier curve
        """
        n_points = len(x_points)
        n_control = degree + 1

        # Parameter values (0 to 1)
        t_values = np.linspace(0, 1, n_points)

        # Bernstein polynomial matrix
        def bernstein_poly(n, i, t):
            """Bernstein polynomial basis function"""
            from math import comb
            return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

        # Build the basis matrix
        B = np.zeros((n_points, n_control))
        for i in range(n_control):
            B[:, i] = [bernstein_poly(degree, i, t) for t in t_values]

        # Solve for control points using least squares
        try:
            control_x = np.linalg.lstsq(B, x_points, rcond=None)[0]
            control_y = np.linalg.lstsq(B, y_points, rcond=None)[0]

            # Combine into control points
            control_points = np.column_stack((control_x, control_y))
            return control_points
        except:
            # Fallback: use original points as control points
            n_fallback = min(n_control, len(x_points))
            indices = np.linspace(0, len(x_points) - 1, n_fallback, dtype=int)
            return np.column_stack((x_points[indices], y_points[indices]))

    def segment_contour_to_bezier(self, x_points: np.ndarray, y_points: np.ndarray) -> List[np.ndarray]:
        """
        Segment a long contour into multiple Bézier curves.

        Args:
            x_points: X coordinates of the contour
            y_points: Y coordinates of the contour

        Returns:
            List of control points for each Bézier segment
        """
        n_points = len(x_points)

        if n_points <= 10:  # Small contour, single Bézier curve
            return [self.fit_bezier_curve(x_points, y_points)]

        # Calculate segment size using the configured max_segments
        segment_size = max(10, n_points // self.max_segments)
        segments = []

        for i in range(0, n_points, segment_size):
            end_idx = min(i + segment_size + 3, n_points)  # Overlap for continuity
            if end_idx - i < 4:  # Skip very small segments
                continue

            segment_x = x_points[i:end_idx]
            segment_y = y_points[i:end_idx]

            control_points = self.fit_bezier_curve(segment_x, segment_y)
            segments.append(control_points)

        return segments

    def extract_character_bezier(self, image_path: str) -> Dict[str, Any]:
        """
        Extract Bézier curves for all characters in an image.

        Args:
            image_path: Path to the calligraphy image

        Returns:
            Dictionary containing Bézier curve data for each character
        """
        # Preprocess image
        binary_image = self.preprocess_image(image_path)

        # Extract contours
        contours = self.extract_character_contours(binary_image)

        result = {
            'image_path': image_path,
            'characters': []
        }

        for i, contour in enumerate(contours):
            # Smooth the contour
            x_smooth, y_smooth = self.smooth_contour(contour)

            # Convert to Bézier curves
            bezier_segments = self.segment_contour_to_bezier(x_smooth, y_smooth)

            character_data = {
                'character_id': i,
                'contour_area': float(cv2.contourArea(contour)),
                'bounding_box': cv2.boundingRect(contour),
                'bezier_curves': [segment.tolist() for segment in bezier_segments],
                'original_contour_points': len(contour)
            }

            result['characters'].append(character_data)

        return result

    def save_bezier_data(self, bezier_data: Dict[str, Any], output_path: str,
                        format: str = 'json') -> None:
        """
        Save Bézier curve data to file.

        Args:
            bezier_data: Bézier curve data dictionary
            output_path: Output file path
            format: Output format ('json' or 'pickle')
        """
        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(bezier_data, f, indent=2, ensure_ascii=False)
        elif format.lower() == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(bezier_data, f)
        else:
            raise ValueError("Format must be 'json' or 'pickle'")

    def visualize_bezier_curves(self, image_path: str, bezier_data: Dict[str, Any],
                               output_path: str = None) -> np.ndarray:
        """
        Visualize the extracted Bézier curves on the original image.

        Args:
            image_path: Path to the original image
            bezier_data: Bézier curve data
            output_path: Optional path to save the visualization

        Returns:
            Image with Bézier curves overlaid
        """
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Create overlay
        overlay = img.copy()

        # Draw Bézier curves for each character
        for char_data in bezier_data['characters']:
            color = (0, 255, 0)  # Green for Bézier curves

            for bezier_curve in char_data['bezier_curves']:
                control_points = np.array(bezier_curve)

                # Draw control polygon
                cv2.polylines(overlay, [control_points.astype(int)], False, (255, 0, 0), 1)

                # Draw control points
                for point in control_points:
                    cv2.circle(overlay, tuple(point.astype(int)), 3, (0, 0, 255), -1)

                # Generate and draw smooth Bézier curve
                curve_points = self.evaluate_bezier_curve(control_points, self.curve_resolution)
                cv2.polylines(overlay, [curve_points.astype(int)], False, color, 2)

        # Blend with original using configured alpha
        result = cv2.addWeighted(img, 1 - self.visualization_alpha, overlay, self.visualization_alpha, 0)

        if output_path:
            cv2.imwrite(output_path, result)

        return result

    def evaluate_bezier_curve(self, control_points: np.ndarray, num_points: int = 100) -> np.ndarray:
        """
        Evaluate points on a Bézier curve.

        Args:
            control_points: Control points of the Bézier curve
            num_points: Number of points to evaluate

        Returns:
            Points on the Bézier curve
        """
        degree = len(control_points) - 1
        t_values = np.linspace(0, 1, num_points)

        def bernstein_poly(n, i, t):
            from math import comb
            return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

        curve_points = np.zeros((num_points, 2))

        for i, t in enumerate(t_values):
            point = np.zeros(2)
            for j in range(degree + 1):
                basis = bernstein_poly(degree, j, t)
                point += basis * control_points[j]
            curve_points[i] = point

        return curve_points

def process_calligraphy_dataset(dataset_dir: str, output_dir: str):
    """
    Process an entire calligraphy dataset to extract Bézier curves.
    Maintains the same folder structure as the input dataset.

    Args:
        dataset_dir: Directory containing calligraphy images (e.g., "chinese-calligraphy-dataset")
        output_dir: Directory to save the Bézier curve data (will mirror the input structure)
    """
    import time
    from pathlib import Path

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize extractor with high-detail parameters
    extractor = BezierCurveExtractor()

    print("High-Detail Bézier Curve Extraction")
    print("=" * 50)
    print(f"Source: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Parameters:")
    print(f"  - Smoothing factor: {extractor.smoothing_factor}")
    print(f"  - Max points: {extractor.max_points}")
    print(f"  - Curve resolution: {extractor.curve_resolution}")
    print(f"  - Max segments: {extractor.max_segments}")
    print("=" * 50)

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # Track statistics
    total_images = 0
    processed_images = 0
    failed_images = 0
    total_characters = 0
    total_bezier_segments = 0
    start_time = time.time()

    # First, count total images for progress tracking
    print("Scanning dataset...")
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                total_images += 1

    print(f"Found {total_images} images to process")
    print()

    # Process all images in the dataset
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(root, file)

                # Calculate relative path from dataset root
                rel_path = os.path.relpath(image_path, dataset_dir)

                # Create corresponding output path
                output_name_json = os.path.splitext(rel_path)[0] + '_bezier.json'
                output_path_json = os.path.join(output_dir, output_name_json)

                # Create subdirectories if needed
                os.makedirs(os.path.dirname(output_path_json), exist_ok=True)

                try:
                    # Progress indicator
                    processed_images += 1
                    progress = (processed_images / total_images) * 100
                    print(f"[{processed_images}/{total_images}] ({progress:.1f}%) Processing: {rel_path}")

                    # Extract Bézier curves
                    bezier_data = extractor.extract_character_bezier(image_path)

                    # Save Bézier data
                    extractor.save_bezier_data(bezier_data, output_path_json)

                    # Update statistics
                    chars_in_image = len(bezier_data['characters'])
                    segments_in_image = sum(len(char['bezier_curves']) for char in bezier_data['characters'])
                    total_characters += chars_in_image
                    total_bezier_segments += segments_in_image

                    print(f"  -> JSON: {output_path_json}")
                    print(f"  -> Characters: {chars_in_image}, Bézier segments: {segments_in_image}")

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
    print(f"Total images: {total_images}")
    print(f"Successfully processed: {processed_images - failed_images}")
    print(f"Failed: {failed_images}")
    print(f"Total characters extracted: {total_characters}")
    print(f"Total Bézier segments: {total_bezier_segments}")
    print(f"Average characters per image: {total_characters / max(1, processed_images - failed_images):.1f}")
    print(f"Average segments per character: {total_bezier_segments / max(1, total_characters):.1f}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Average time per image: {elapsed_time / max(1, total_images):.2f} seconds")
    print("=" * 50)

if __name__ == "__main__":
    # Example usage
    dataset_dir = "chinese-calligraphy-dataset"  # Adjust this path
    output_dir = "bezier_curves_output_no_visualization"

    print("Processing calligraphy dataset...")
    process_calligraphy_dataset(dataset_dir, output_dir)
    print("Done!")

