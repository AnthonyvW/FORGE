"""
Sequential Image Stitching Tool

This module performs sequential stitching on images to create panoramas.
Images are stitched in order: 1+2, then result+3, then result+4, etc.

Usage:
    python sequentialImageStitch.py [folder_path] [axis] [options]
    
Arguments:
    folder_path: Path to directory containing images (default: current directory)
    axis: 'x' or 'y' - which coordinate varies in the images (default: 'y')
          - Use 'y' for images captured top-to-bottom (will be rotated -90°)
          - Use 'x' for images captured left-to-right (no rotation)

Options:
    --debug              Enable saving debug images (eval regions, matched comparisons)
    --keep-intermediates Keep intermediate stitching results after completion

The script expects:
- A directory containing images to stitch
- Images should be named with coordinates like "Y122 X019 Z033 F37571.tiff"
- The coordinate letter (X or Y) can appear anywhere in the filename
- Supported formats: TIFF, TIF, JPG, JPEG, PNG

Examples:
    python sequentialImageStitch.py /path/to/images y                    # Basic (final output only)
    python sequentialImageStitch.py /path/to/images x --debug            # With debug images
    python sequentialImageStitch.py . y --keep-intermediates             # Keep intermediates
    python sequentialImageStitch.py . y --debug --keep-intermediates     # Full output
"""

import os
import sys
import shutil
import re
import argparse
import time
from pathlib import Path
from typing import Tuple, Optional

# Third-party imports
import cv2 as cv
import numpy as np


def find_alignment(img1_path: Path, img2_path: Path, 
                  rotate_images: bool = False, pair_info: str = "", 
                  save_debug: bool = False, debug_dir: Optional[Path] = None) -> Optional[Tuple[int, int, float]]:
    """
    Find alignment between two images without creating the stitched output.
    
    Args:
        img1_path: Path to first image (left image)
        img2_path: Path to second image (right image)
        rotate_images: Whether to rotate images 90° CCW before analysis
        pair_info: String identifier for this pair (used in debug filenames)
        save_debug: Whether to save debug images
        debug_dir: Directory to save debug images (if save_debug is True)
    
    Returns:
        Tuple of (x_overlap, y_offset, score) if alignment found, None otherwise
    """
    try:
        img1 = cv.imread(str(img1_path))  # Left image
        img2 = cv.imread(str(img2_path))  # Right image
        
        if img1 is None:
            print(f"    ERROR: Failed to load img1 from {img1_path}")
            return False
        if img2 is None:
            print(f"    ERROR: Failed to load img2 from {img2_path}")
            return False
        
        # Rotate images if requested (only for initial images from Y-axis scans)
        if rotate_images:
            img1 = cv.rotate(img1, cv.ROTATE_90_COUNTERCLOCKWISE)
            img2 = cv.rotate(img2, cv.ROTATE_90_COUNTERCLOCKWISE)
        
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Calculate edge regions for evaluation
        max_expected_overlap = int(w1 * 0.95)  # 95% max overlap
        
        # For img1 (left image): only use right edge that could overlap
        img1_eval_width = min(max_expected_overlap, w1)
        img1_eval_start = w1 - img1_eval_width  # Right edge of img1
        img1_eval_region = img1[:, img1_eval_start:]
        
        # For img2 (right image): only use left edge that could overlap  
        img2_eval_width = min(max_expected_overlap, w2)
        img2_eval_start = 0  # Left edge of img2
        img2_eval_region = img2[:, img2_eval_start:img2_eval_start + img2_eval_width]
        
        # Save evaluation regions for debugging (only if save_debug is True)
        if save_debug and debug_dir:
            debug_dir.mkdir(exist_ok=True)
            
            # Use pair_info for filename prefix
            prefix = f"{pair_info}_" if pair_info else ""
            
            # Save img1 with rectangle showing eval region
            img1_debug = img1.copy()
            cv.rectangle(img1_debug, (img1_eval_start, 0), (w1-1, h1-1), (0, 255, 0), 3)
            cv.putText(img1_debug, f"IMG1: {img1_path.name}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(img1_debug, "RIGHT EDGE EVAL (green box)", (10, 60), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.imwrite(str(debug_dir / f"{prefix}img1_eval_region.jpg"), img1_debug)
            
            # Save img2 with rectangle showing eval region
            img2_debug = img2.copy()
            cv.rectangle(img2_debug, (0, 0), (img2_eval_width-1, h2-1), (0, 0, 255), 3)
            cv.putText(img2_debug, f"IMG2: {img2_path.name}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.putText(img2_debug, "LEFT EDGE EVAL (red box)", (10, 60), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.imwrite(str(debug_dir / f"{prefix}img2_eval_region.jpg"), img2_debug)
            
            # Save just the eval regions themselves
            cv.imwrite(str(debug_dir / f"{prefix}img1_eval_only.jpg"), img1_eval_region)
            cv.imwrite(str(debug_dir / f"{prefix}img2_eval_only.jpg"), img2_eval_region)
            
        # Convert evaluation regions to grayscale
        gray1_eval = cv.cvtColor(img1_eval_region, cv.COLOR_BGR2GRAY)
        gray2_eval = cv.cvtColor(img2_eval_region, cv.COLOR_BGR2GRAY)
        
        # Stage 1: Coarse horizontal search on evaluation regions
        min_eval_overlap = int(min(img1_eval_width, img2_eval_width) * 0.3)
        max_eval_overlap = int(min(img1_eval_width, img2_eval_width) * 0.95)
        
        # Use center strips for coarse search
        eval_height = min(gray1_eval.shape[0], gray2_eval.shape[0])
        y1_start = (gray1_eval.shape[0] - eval_height) // 2
        y2_start = (gray2_eval.shape[0] - eval_height) // 2
        gray1_strip = gray1_eval[y1_start:y1_start + eval_height, :]
        gray2_strip = gray2_eval[y2_start:y2_start + eval_height, :]
        
        best_coarse_overlap = min_eval_overlap
        best_coarse_score = -1
        
        print(f"    Stage 1: Coarse search overlap {min_eval_overlap} to {max_eval_overlap}")
        
        for overlap in range(min_eval_overlap, max_eval_overlap, 20):
            if overlap > min(img1_eval_width, img2_eval_width):
                continue
                
            # Extract overlapping regions from evaluation areas
            region1 = gray1_strip[:, -overlap:]  # Right part of img1's eval region
            region2 = gray2_strip[:, :overlap]   # Left part of img2's eval region
            
            if region1.shape != region2.shape:
                continue
            
            result = cv.matchTemplate(region1, region2, cv.TM_CCOEFF_NORMED)
            score = result[0, 0]
            
            if score > best_coarse_score:
                best_coarse_score = score
                best_coarse_overlap = overlap
        
        print(f"    Coarse result: eval_overlap {best_coarse_overlap}, score {best_coarse_score:.4f}")
        
        # Stage 2: Fine X,Y search around coarse result
        x_range = 30
        y_range = 15
        
        x_min = max(min_eval_overlap, best_coarse_overlap - x_range)
        x_max = min(max_eval_overlap, best_coarse_overlap + x_range)
        
        best_eval_x_overlap = best_coarse_overlap
        best_y_offset = 0
        best_xy_score = best_coarse_score
        
        print(f"    Stage 2: Fine X,Y search on eval regions")
        
        for x_overlap in range(x_min, x_max + 1, 3):
            if x_overlap > min(img1_eval_width, img2_eval_width):
                continue
                
            for y_offset in range(-y_range, y_range + 1, 2):
                # Calculate comparison regions with Y offset
                if y_offset >= 0:
                    available_height1 = gray1_eval.shape[0] - y_offset
                    available_height2 = gray2_eval.shape[0]
                    compare_height = min(available_height1, available_height2)
                    
                    if compare_height <= 50:
                        continue
                    
                    region1 = gray1_eval[y_offset:y_offset + compare_height, -x_overlap:]
                    region2 = gray2_eval[:compare_height, :x_overlap]
                else:
                    available_height1 = gray1_eval.shape[0]
                    available_height2 = gray2_eval.shape[0] + y_offset
                    compare_height = min(available_height1, available_height2)
                    
                    if compare_height <= 50:
                        continue
                    
                    region1 = gray1_eval[:compare_height, -x_overlap:]
                    region2 = gray2_eval[-y_offset:-y_offset + compare_height, :x_overlap]
                
                if region1.shape != region2.shape:
                    continue
                
                result = cv.matchTemplate(region1, region2, cv.TM_CCOEFF_NORMED)
                score = result[0, 0]
                
                if score > best_xy_score:
                    best_xy_score = score
                    best_eval_x_overlap = x_overlap
                    best_y_offset = y_offset
        
        # Convert evaluation region overlap back to full image coordinates
        actual_x_overlap = best_eval_x_overlap
        
        print(f"    Fine alignment: eval_x_overlap={best_eval_x_overlap}, actual_x_overlap={actual_x_overlap}, Y_offset={best_y_offset}, score={best_xy_score:.4f}")
        
        # Create visualization of matched regions FROM EVAL REGIONS
        if best_y_offset >= 0:
            compare_height = min(gray1_eval.shape[0] - best_y_offset, gray2_eval.shape[0])
            matched_region1 = img1_eval_region[best_y_offset:best_y_offset + compare_height, -actual_x_overlap:]
            matched_region2 = img2_eval_region[:compare_height, :actual_x_overlap]
        else:
            compare_height = min(gray1_eval.shape[0], gray2_eval.shape[0] + best_y_offset)
            matched_region1 = img1_eval_region[:compare_height, -actual_x_overlap:]
            matched_region2 = img2_eval_region[-best_y_offset:-best_y_offset + compare_height, :actual_x_overlap]
        
        # Save matched comparison (only if save_debug is True)
        if save_debug and debug_dir:
            # Create side-by-side comparison
            comparison = np.hstack([matched_region1, matched_region2])
            
            # Add text labels
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(comparison, "IMG1 RIGHT edge (overlap)", (10, 30), 
                      font, 0.8, (0, 255, 0), 2)
            cv.putText(comparison, "IMG2 LEFT edge (overlap)", (matched_region1.shape[1] + 10, 30), 
                      font, 0.8, (0, 0, 255), 2)
            cv.putText(comparison, f"Score: {best_xy_score:.4f}", (10, 60), 
                      font, 0.8, (255, 255, 255), 2)
            cv.putText(comparison, f"Overlap: {actual_x_overlap}px, Y-offset: {best_y_offset}px", (10, 90), 
                      font, 0.8, (255, 255, 255), 2)
            
            # Yellow dividing line
            cv.line(comparison, (matched_region1.shape[1], 0), 
                   (matched_region1.shape[1], comparison.shape[0]), (0, 255, 255), 2)
            
            # Use pair_info for filename prefix
            prefix = f"{pair_info}_" if pair_info else ""
            match_path = debug_dir / f"{prefix}matched_comparison.jpg"
            cv.imwrite(str(match_path), comparison)
        
        # Return alignment parameters if score is good enough
        if best_xy_score > 0.7:
            print(f"    ✓ Alignment found: overlap={actual_x_overlap}px, y_offset={best_y_offset}px, score={best_xy_score:.4f}")
            return (actual_x_overlap, best_y_offset, best_xy_score)
        else:
            print(f"    ✗ No acceptable alignment (score {best_xy_score:.4f} < 0.7)")
            return None
        
    except Exception as e:
        print(f"Alignment error: {e}")
        import traceback
        traceback.print_exc()
        return None


def sequential_stitch_images(images_dir: Path, output_dir: Path, axis: str = 'y', 
                             save_debug: bool = False, keep_intermediates: bool = False):
    """
    Sequentially stitch images by finding all offsets, then creating final image.
    Process: Find offsets for pairs 1+2, 2+3, 3+4, etc., then assemble all images.
    
    Args:
        images_dir: Path to directory containing images
        output_dir: Path to output directory
        axis: 'x' or 'y' - which coordinate varies in the images
        save_debug: Whether to save debug images
        keep_intermediates: Whether to keep intermediate stitching results
    """
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return
    
    # Get all image files
    image_files = []
    for ext in ['*.tiff', '*.tif', '*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(images_dir.glob(ext)))
        image_files.extend(list(images_dir.glob(ext.upper())))
    
    # Remove duplicates
    unique_images = []
    seen_paths = set()
    for img_path in image_files:
        normalized = str(img_path.resolve())
        if normalized not in seen_paths:
            seen_paths.add(normalized)
            unique_images.append(img_path)
    
    if not unique_images:
        print(f"No images found in directory")
        return
    
    print(f"Starting sequential stitching along {axis.upper()} axis")
    print(f"Found {len(unique_images)} unique images to stitch")
    
    # Extract coordinates and sort images
    images_with_coords = []
    coord_letter = axis.upper()
    
    for img_path in unique_images:
        try:
            filename = img_path.stem
            coord_match = re.search(rf'{coord_letter}(\d+)', filename, re.IGNORECASE)
            
            if coord_match:
                coord_pos = int(coord_match.group(1))
                images_with_coords.append((coord_pos, img_path))
                
                # Also find the other coordinate for display
                other_letter = 'Y' if axis == 'x' else 'X'
                other_match = re.search(rf'{other_letter}(\d+)', filename, re.IGNORECASE)
                
                if other_match:
                    other_pos = int(other_match.group(1))
                    print(f"  Found: {img_path.name} -> {coord_letter}={coord_pos}, {other_letter}={other_pos}")
                else:
                    print(f"  Found: {img_path.name} -> {coord_letter}={coord_pos}")
            else:
                print(f"Skipping {img_path.name}: No {coord_letter} coordinate found")
        except (IndexError, ValueError) as e:
            print(f"Skipping {img_path.name}: Could not parse coordinates - {e}")
            continue
    
    if len(images_with_coords) < 2:
        print(f"Need at least 2 images to stitch, found {len(images_with_coords)}")
        return
    
    # Sort by coordinate
    images_with_coords.sort(key=lambda x: x[0])
    sorted_images = [img_path for _, img_path in images_with_coords]
    
    print(f"\nImages sorted by {coord_letter} coordinate:")
    for i, (coord_pos, img_path) in enumerate(images_with_coords):
        print(f"  [{i}] {coord_letter}{coord_pos}: {img_path.name}")
    
    # Create working directory and debug directory
    working_dir = output_dir / "stitch_working"
    working_dir.mkdir(exist_ok=True)
    
    debug_dir = output_dir / "eval_regions" if save_debug else None
    if debug_dir:
        debug_dir.mkdir(exist_ok=True)
    
    print(f"\nProcessing {len(sorted_images)} images")
    
    try:
        # Phase 1: Find alignment offsets for all consecutive pairs
        print("\n" + "=" * 80)
        print("PHASE 1: Finding Alignment Offsets")
        print("=" * 80)
        
        pair_offsets = []  # List of (x_overlap, y_offset, score) tuples
        
        for i in range(len(sorted_images) - 1):
            img1_path = sorted_images[i]
            img2_path = sorted_images[i + 1]
            
            pair_info = f"pair{i+1}"
            
            print(f"\nPair {i+1}/{len(sorted_images)-1}: {img1_path.name} + {img2_path.name}")
            
            pair_start = time.time()
            
            # Find alignment (rotate if axis='y' to convert vertical scan to horizontal)
            alignment = find_alignment(img1_path, img2_path, 
                                     rotate_images=(axis == 'y'),  # Rotate for Y-axis scans
                                     pair_info=pair_info, 
                                     save_debug=save_debug,
                                     debug_dir=debug_dir)
            
            pair_elapsed = time.time() - pair_start
            
            if alignment:
                x_overlap, y_offset, score = alignment
                pair_offsets.append((x_overlap, y_offset, score))
                print(f"  Time: {pair_elapsed:.1f}s")
            else:
                print(f"  ✗ Alignment failed ({pair_elapsed:.1f}s)")
                print(f"\n⚠️  STOPPING: Failed to align pair {i+1}")
                print(f"Successfully aligned {i} out of {len(sorted_images)-1} pairs")
                
                # Create partial result if we have any successful alignments
                if pair_offsets and i > 0:
                    print("\n" + "=" * 80)
                    print(f"CREATING PARTIAL RESULT ({i+1} images)")
                    print("=" * 80)
                    partial_images = sorted_images[:i+1]
                    partial_offsets = pair_offsets
                    create_final_stitched_image(partial_images, partial_offsets, output_dir, axis, 
                                               f"partial_stitched_{i+1}_images.tiff")
                
                return
        
        print(f"\n✓ All {len(sorted_images)-1} pairs aligned successfully!")
        
        # Phase 2: Create final stitched image using all offsets
        print("\n" + "=" * 80)
        print("PHASE 2: Creating Final Stitched Image")
        print("=" * 80)
        
        create_final_stitched_image(sorted_images, pair_offsets, output_dir, axis, "final_stitched.tiff")
        
    except Exception as e:
        print(f"Error during sequential stitching: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Handle intermediate results based on keep_intermediates flag
        if keep_intermediates:
            print(f"\nIntermediate stitching results saved in: {working_dir}")
        else:
            # Clean up intermediate files
            if working_dir.exists():
                try:
                    shutil.rmtree(working_dir)
                    print(f"\nIntermediate stitching results cleaned up")
                except Exception as e:
                    print(f"Warning: Could not remove intermediate directory: {e}")
        
        # Handle debug images based on save_debug flag
        if not save_debug and debug_dir and debug_dir.exists():
            try:
                shutil.rmtree(debug_dir)
            except Exception as e:
                print(f"Warning: Could not remove debug directory: {e}")


def create_final_stitched_image(image_paths: list, offsets: list, output_dir: Path, 
                                axis: str, output_filename: str):
    """
    Create final stitched image from a list of images and their pairwise offsets.
    
    Args:
        image_paths: List of image paths in order
        offsets: List of (x_overlap, y_offset, score) tuples for consecutive pairs
        output_dir: Directory to save output
        axis: 'x' or 'y' - determines if rotation is needed
        output_filename: Name for the output file
    """
    print(f"\nAssembling {len(image_paths)} images...")
    
    # Load all images (rotate if axis is 'y')
    images = []
    for img_path in image_paths:
        img = cv.imread(str(img_path))
        if img is None:
            print(f"ERROR: Could not load {img_path}")
            return
        
        # Rotate if needed
        if axis == 'y':
            img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
        
        images.append(img)
        print(f"  Loaded: {img_path.name} -> {img.shape[1]}x{img.shape[0]}")
    
    # Calculate cumulative positions for each image
    # First image starts at x=0
    image_positions = [(0, 0)]  # (x_start, y_offset) for first image
    
    current_x = 0
    for i, (x_overlap, y_offset, score) in enumerate(offsets):
        # Each subsequent image is positioned based on the previous image and the overlap
        prev_img = images[i]
        current_x = current_x + prev_img.shape[1] - x_overlap
        image_positions.append((current_x, y_offset))
        print(f"  Image {i+1}: overlap={x_overlap}px, y_offset={y_offset}px -> x_pos={current_x}")
    
    # Calculate canvas size
    # Find total width (rightmost extent)
    total_width = 0
    for i, (x_pos, y_offset) in enumerate(image_positions):
        img = images[i]
        right_edge = x_pos + img.shape[1]
        total_width = max(total_width, right_edge)
    
    # Find total height (accounting for y offsets)
    min_y = 0
    max_y = 0
    for i, (x_pos, y_offset) in enumerate(image_positions):
        img = images[i]
        min_y = min(min_y, y_offset)
        max_y = max(max_y, y_offset + img.shape[0])
    
    total_height = max_y - min_y
    
    print(f"\nCanvas size: {total_width}x{total_height}")
    print(f"Y range: {min_y} to {max_y}")
    
    # Create canvas
    canvas = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    
    # Place each image on the canvas
    for i, (x_pos, y_offset) in enumerate(image_positions):
        img = images[i]
        h, w = img.shape[:2]
        
        # Adjust y position relative to min_y
        y_pos = y_offset - min_y
        
        print(f"  Placing image {i}: {image_paths[i].name} at ({x_pos}, {y_pos})")
        
        # Place image
        canvas[y_pos:y_pos + h, x_pos:x_pos + w] = img
    
    # Save result
    output_path = output_dir / output_filename
    cv.imwrite(str(output_path), canvas)
    
    print(f"\n🎉 Stitched image created!")
    print(f"Output: {output_path}")
    print(f"Size: {total_width}x{total_height}")
    print(f"Successfully combined {len(image_paths)} images")


def main():
    """Main function - runs sequential stitching on images in specified folder."""
    parser = argparse.ArgumentParser(
        description='Sequential Image Stitching Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sequentialImageStitch.py /path/to/images y              # Basic stitching
  python sequentialImageStitch.py /path/to/images x --debug      # With debug images
  python sequentialImageStitch.py . y --keep-intermediates       # Keep intermediate files
  python sequentialImageStitch.py . y --debug --keep-intermediates  # Full output
        """
    )
    
    parser.add_argument(
        'folder_path',
        nargs='?',
        default='.',
        help='Path to directory containing images (default: current directory)'
    )
    
    parser.add_argument(
        'axis',
        nargs='?',
        default='y',
        choices=['x', 'y', 'X', 'Y'],
        help='Axis along which images vary: "x" for left-to-right, "y" for top-to-bottom (default: y)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable saving debug images (eval regions, matched comparisons)'
    )
    
    parser.add_argument(
        '--keep-intermediates',
        action='store_true',
        help='Keep intermediate stitching results after completion'
    )
    
    try:
        args = parser.parse_args()
        
        images_dir = Path(args.folder_path).absolute()
        axis = args.axis.lower()
        save_debug = args.debug
        keep_intermediates = args.keep_intermediates
        
        # Check if directory exists
        if not images_dir.exists():
            print(f"Error: Directory '{images_dir}' does not exist!")
            return 1
        
        if not images_dir.is_dir():
            print(f"Error: '{images_dir}' is not a directory!")
            return 1
        
        # Check for images
        image_files = []
        for ext in ['*.tiff', '*.tif', '*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(images_dir.glob(ext)))
            image_files.extend(list(images_dir.glob(ext.upper())))
        
        if not image_files:
            print(f"Error: No images found in directory '{images_dir}'!")
            print("Supported formats: TIFF, TIF, JPG, JPEG, PNG")
            return 1
        
        print("=" * 80)
        print("SEQUENTIAL IMAGE STITCHING")
        print("=" * 80)
        print(f"Found {len(image_files)} images in directory")
        print(f"Stitching along {axis.upper()} axis")
        if save_debug or keep_intermediates:
            options = []
            if save_debug:
                options.append('debug images')
            if keep_intermediates:
                options.append('keep intermediates')
            print(f"Options: {', '.join(options)}")
        print()
        
        # Run sequential stitching
        sequential_stitch_images(images_dir, images_dir, axis, save_debug, keep_intermediates)
        
        print("\n" + "=" * 80)
        print("STITCHING PROCESS COMPLETED")
        print("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())