"""
Best Image Hierarchical Stitching Tool

This module performs hierarchical stitching on images to create high-quality panoramas.

Usage:
    python bestImageStitch.py [folder_path] [axis] [options]
    
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
    python bestImageStitch.py /path/to/images y                    # Basic (final output only)
    python bestImageStitch.py /path/to/images x --debug            # With debug images
    python bestImageStitch.py . y --keep-intermediates             # Keep intermediates
    python bestImageStitch.py . y --debug --keep-intermediates     # Full output
"""

import os
import sys
import shutil
import re
import argparse
import time
from pathlib import Path
from collections import deque

# Third-party imports
import cv2 as cv
from stitching import AffineStitcher, Stitcher
import numpy as np



def stitch_image_pair(img1_path: Path, img2_path: Path, output_path: Path, axis: str = 'y', pair_info: str = "", save_debug: bool = False) -> bool:
    """
    Optimized stitching that only evaluates relevant edge regions.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        output_path: Path to save stitched result
        axis: 'x' or 'y' - determines if rotation is needed ('y' requires rotation)
    """
    try:
        img1 = cv.imread(str(img1_path))  # Left/Top image (lower coordinate)
        img2 = cv.imread(str(img2_path))  # Right/Bottom image (higher coordinate)
        
        if img1 is None:
            print(f"    ERROR: Failed to load img1 from {img1_path}")
            return False
        if img2 is None:
            print(f"    ERROR: Failed to load img2 from {img2_path}")
            return False
        
        # Only rotate images if stitching along Y axis (top-to-bottom needs to become left-to-right)
        if axis == 'y':
            img1 = cv.rotate(img1, cv.ROTATE_90_COUNTERCLOCKWISE)
            img2 = cv.rotate(img2, cv.ROTATE_90_COUNTERCLOCKWISE)
        
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Calculate edge regions for evaluation
        max_expected_overlap = int(w1 * 0.95)  # 90% max overlap
        
        # For img1 (left image): only use right edge that could overlap
        img1_eval_width = min(max_expected_overlap, w1)
        img1_eval_start = w1 - img1_eval_width  # Right edge of img1
        img1_eval_region = img1[:, img1_eval_start:]
        
        # For img2 (right image): only use left edge that could overlap  
        img2_eval_width = min(max_expected_overlap, w2)
        img2_eval_start = 0  # Left edge of img2
        img2_eval_region = img2[:, img2_eval_start:img2_eval_start + img2_eval_width]
        
        # Save evaluation regions for debugging (only if save_debug is True)
        if save_debug:
            debug_dir = output_path.parent / "eval_regions"
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
        # Overlap is measured relative to the evaluation regions
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
        # The overlap in eval region corresponds to this overlap in full images:
        actual_x_overlap = best_eval_x_overlap
        
        print(f"    Fine alignment: eval_x_overlap={best_eval_x_overlap}, actual_x_overlap={actual_x_overlap}, Y_offset={best_y_offset}, score={best_xy_score:.4f}")
        
        # Create visualization of matched regions FROM EVAL REGIONS (where matching actually happened)
        # Extract the final matched regions for visualization
        if best_y_offset >= 0:
            compare_height = min(gray1_eval.shape[0] - best_y_offset, gray2_eval.shape[0])
            # From eval regions, not full images!
            matched_region1 = img1_eval_region[best_y_offset:best_y_offset + compare_height, -actual_x_overlap:]
            matched_region2 = img2_eval_region[:compare_height, :actual_x_overlap]
        else:
            compare_height = min(gray1_eval.shape[0], gray2_eval.shape[0] + best_y_offset)
            # From eval regions, not full images!
            matched_region1 = img1_eval_region[:compare_height, -actual_x_overlap:]
            matched_region2 = img2_eval_region[-best_y_offset:-best_y_offset + compare_height, :actual_x_overlap]
        
        # Save matched comparison (only if save_debug is True)
        if save_debug:
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
            debug_dir = output_path.parent / "eval_regions"
            debug_dir.mkdir(exist_ok=True)
            match_path = debug_dir / f"{prefix}matched_comparison.jpg"
            cv.imwrite(str(match_path), comparison)
        
        # Stage 3: Create output using full images
        if best_xy_score > 0.7:
            # Calculate canvas size
            if best_y_offset >= 0:
                canvas_height = max(h1, h2 + best_y_offset)
                img1_y_start = 0
                img2_y_start = best_y_offset
            else:
                canvas_height = max(h1 - best_y_offset, h2)
                img1_y_start = -best_y_offset
                img2_y_start = 0
            
            total_width = w1 + w2 - actual_x_overlap
            result = np.zeros((canvas_height, total_width, 3), dtype=np.uint8)
            
            # Place full img1 (left image - it should be on the left side)
            result[img1_y_start:img1_y_start + h1, :w1] = img1
            
            # Place full img2 (right image - offset to the right with overlap)
            img2_x_start = w1 - actual_x_overlap
            result[img2_y_start:img2_y_start + h2, img2_x_start:img2_x_start + w2] = img2
            
            cv.imwrite(str(output_path), result)
            
            # Enhanced Debug visualization (only if save_debug is True)
            if save_debug:
                # Calculate x_seam for debug visualization
                x_seam = w1 - actual_x_overlap
                img2_x_start = w1 - actual_x_overlap
                
                debug_img = result.copy()
                
                # Draw the overlap seam in bright green
                cv.line(debug_img, (x_seam, 0), (x_seam, canvas_height), (0, 255, 0), 3)
                
                # Draw img1 boundary in blue
                cv.rectangle(debug_img, (0, img1_y_start), (w1-1, img1_y_start + h1-1), (255, 0, 0), 2)
                
                # Draw img2 boundary in red
                cv.rectangle(debug_img, (img2_x_start, img2_y_start), 
                            (img2_x_start + w2-1, img2_y_start + h2-1), (0, 0, 255), 2)
                
                # Draw Y offset line if present
                if best_y_offset != 0:
                    y_line = max(img1_y_start, img2_y_start) + min(h1, h2) // 2
                    cv.line(debug_img, (0, y_line), (total_width, y_line), (255, 255, 0), 2)
                
                # Add comprehensive text information
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                y_text = 30
                line_height = 35
                
                # Background for text readability
                cv.rectangle(debug_img, (5, 5), (600, 350), (0, 0, 0), -1)
                cv.rectangle(debug_img, (5, 5), (600, 350), (255, 255, 255), 2)
                
                # Display comprehensive info
                texts = [
                    f"IMG1 (BLUE): {img1_path.name[:40]}",
                    f"  Size: {w1}x{h1}, Y_start: {img1_y_start}",
                    f"  Position: (0, {img1_y_start}) to ({w1}, {img1_y_start + h1})",
                    f"",
                    f"IMG2 (RED): {img2_path.name[:40]}",
                    f"  Size: {w2}x{h2}, Y_start: {img2_y_start}",
                    f"  Position: ({img2_x_start}, {img2_y_start}) to ({img2_x_start + w2}, {img2_y_start + h2})",
                    f"",
                    f"OVERLAP: {actual_x_overlap}px (GREEN line at x={x_seam})",
                    f"Y_OFFSET: {best_y_offset}px, Score: {best_xy_score:.4f}",
                    f"Canvas: {total_width}x{canvas_height}"
                ]
                
                for i, text in enumerate(texts):
                    cv.putText(debug_img, text, (10, y_text + i * line_height), 
                              font, font_scale, (255, 255, 255), thickness)
                
                debug_path = output_path.parent / f"optimized_debug_{output_path.name}"
                cv.imwrite(str(debug_path), debug_img)
            
            return True
        else:
            print("    No acceptable alignment found (score < 0.7)")
            return False
        
    except Exception as e:
        print(f"Optimized stitch error: {e}")
        return False


def hierarchical_stitch_best_images(best_images_dir: Path = None, output_dir: Path = None, axis: str = 'y', save_debug: bool = False, keep_intermediates: bool = False):
    """
    Hierarchically stitch images from a directory.
    
    Args:
        best_images_dir (Path): Path to the directory containing images. If None, uses current directory
        output_dir (Path): Path to the output directory. If None, uses current directory
        axis (str): Axis along which images vary - 'x' or 'y' (default: 'y')
    """
    if best_images_dir is None:
        best_images_dir = Path('.').absolute()
    
    if output_dir is None:
        output_dir = Path('.').absolute()
    
    if not best_images_dir.exists():
        print(f"Best images directory not found: {best_images_dir}")
        print("Please ensure the directory exists.")
        return
    
    # Get all image files in the directory (no longer filtering by strategy)
    best_images = []
    for ext in ['*.tiff', '*.tif', '*.jpg', '*.jpeg', '*.png']:
        best_images.extend(list(best_images_dir.glob(ext)))
        best_images.extend(list(best_images_dir.glob(ext.upper())))
    
    # Remove duplicates (can happen with case-insensitive filesystems or mixed case extensions)
    best_images_unique = []
    seen_paths = set()
    for img_path in best_images:
        # Normalize path to detect duplicates
        normalized = str(img_path.resolve())
        if normalized not in seen_paths:
            seen_paths.add(normalized)
            best_images_unique.append(img_path)
    
    best_images = best_images_unique
    
    if not best_images:
        print(f"No images found in directory")
        return
    
    print(f"Starting hierarchical stitching along {axis.upper()} axis")
    print(f"Found {len(best_images)} unique images to stitch")
    
    # Extract coordinates and sort images
    images_with_coords = []
    coord_letter = axis.upper()  # 'X' or 'Y'
    
    for img_path in best_images:
        try:
            # Use regex to find X and Y coordinates anywhere in the filename
            filename = img_path.stem  # Get filename without extension
            
            # Find the coordinate for the specified axis
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
    
    # Sort by the specified coordinate
    images_with_coords.sort(key=lambda x: x[0])
    
    print(f"\nImages sorted by {coord_letter} coordinate:")
    for coord_pos, img_path in images_with_coords:
        print(f"  {coord_letter}{coord_pos}: {img_path.name}")
    
    # Create working directory for intermediate results
    working_dir = output_dir / "stitch_working"
    working_dir.mkdir(exist_ok=True)
    
    try:
        # Start hierarchical stitching
        current_images = [img_path for _, img_path in images_with_coords]
        level = 0
        
        # Timing tracking for estimates
        pair_times = deque(maxlen=3)  # Keep last 3 pair times for averaging
        total_pairs_to_process = len(current_images) - 1  # N-1 pairs total
        pairs_completed = 0
        
        print(f"\nInitial image list ({len(current_images)} images):")
        for idx, img in enumerate(current_images):
            print(f"  [{idx}] {img.name}")
        
        # Check for duplicates in initial list
        if len(current_images) != len(set(str(p.resolve()) for p in current_images)):
            print("\n⚠️  WARNING: Duplicate images detected in initial list!")
            seen = {}
            for idx, img in enumerate(current_images):
                key = str(img.resolve())
                if key in seen:
                    print(f"    Duplicate at index {idx}: {img.name} (same as index {seen[key]})")
                else:
                    seen[key] = idx
        
        while len(current_images) > 1:
            level += 1
            print(f"\n--- Stitching Level {level} ---")
            print(f"Processing {len(current_images)} images")
            
            next_level_images = []
            pairs_processed = 0
            
            # Process pairs of neighboring images
            for i in range(0, len(current_images), 2):
                if i + 1 < len(current_images):
                    # Pair of images
                    img1 = current_images[i]
                    img2 = current_images[i + 1]
                    
                    # Validate we're not trying to stitch an image with itself
                    if img1 == img2:
                        print(f"    ERROR: Pair {pairs_processed + 1} has the same image twice!")
                        print(f"      This indicates a bug in the image list. Skipping this pair.")
                        continue
                    
                    # Create output filename for this pair
                    output_name = f"level{level}_pair{pairs_processed + 1}.tiff"
                    output_path = working_dir / output_name
                    
                    print(f"  Stitching Pair {pairs_processed + 1}: {img1.name}, {img2.name}")
                    
                    # Create pair info for debug filenames
                    pair_info = f"level{level}_pair{pairs_processed + 1}"
                    
                    # Time this pair
                    pair_start_time = time.time()
                    
                    if stitch_image_pair(img1, img2, output_path, axis, pair_info, save_debug):
                        pair_elapsed = time.time() - pair_start_time
                        pair_times.append(pair_elapsed)
                        pairs_completed += 1
                        
                        next_level_images.append(output_path)
                        
                        # Format success message with timing
                        success_msg = f"    Success - Added {output_path.name} to next level ({pair_elapsed:.1f}s)"
                        
                        # Add time estimate after first pair
                        if pairs_completed > 0:
                            avg_time = sum(pair_times) / len(pair_times)
                            remaining_pairs = total_pairs_to_process - pairs_completed
                            estimated_remaining = avg_time * remaining_pairs
                            
                            # Format time estimate
                            if estimated_remaining < 60:
                                time_str = f"{estimated_remaining:.0f}s"
                            elif estimated_remaining < 3600:
                                minutes = estimated_remaining / 60
                                time_str = f"{minutes:.1f}m"
                            else:
                                hours = estimated_remaining / 3600
                                time_str = f"{hours:.1f}h"
                            
                            success_msg += f" | Est. remaining: {time_str} ({remaining_pairs} pairs left)"
                        
                        print(success_msg)
                        pairs_processed += 1
                    else:
                        pair_elapsed = time.time() - pair_start_time
                        print(f"    Failed to stitch pair ({pair_elapsed:.1f}s)")
                        # If stitching fails, keep both images for next level separately
                        next_level_images.append(img1)
                        next_level_images.append(img2)
                        print(f"    Added both images separately to next level")
                        
                else:
                    # Odd image out, carry to next level
                    print(f"  Carrying forward (odd image): {current_images[i].name}")
                    next_level_images.append(current_images[i])
            
            current_images = next_level_images
            print(f"Level {level} completed. {len(current_images)} images remain:")
            for idx, img in enumerate(current_images):
                print(f"    [{idx}] {img.name}")
        
        # Move final result to main output directory
        if current_images:
            final_image = current_images[0]
            final_output = output_dir / "final_stitched.tiff"
            
            # Remove existing final output if it exists
            if final_output.exists():
                final_output.unlink()
                print(f"Removed existing final output: {final_output}")
            
            if final_image.parent == working_dir:
                # It's an intermediate result, move it
                final_image.rename(final_output)
            else:
                # It's an original image (only one image was provided), copy it
                shutil.copy2(final_image, final_output)
            
            print(f"\n🎉 Image stitching completed!")
            print(f"Final result: {final_output}")
        
    except Exception as e:
        print(f"Error during hierarchical stitching: {e}")
        
    finally:
        # Handle intermediate results based on keep_intermediates flag
        if keep_intermediates:
            print(f"Intermediate stitching results saved in: {working_dir}")
            print("Intermediate files have been preserved for inspection.")
        else:
            # Clean up intermediate files
            if working_dir.exists():
                try:
                    shutil.rmtree(working_dir)
                    print(f"Intermediate stitching results cleaned up")
                except Exception as e:
                    print(f"Warning: Could not remove intermediate directory: {e}")
        
        # Handle debug images based on save_debug flag
        if not save_debug:
            debug_dir = output_dir / "eval_regions"
            if debug_dir.exists():
                try:
                    shutil.rmtree(debug_dir)
                    print(f"Debug images cleaned up")
                except Exception as e:
                    print(f"Warning: Could not remove debug directory: {e}")


def process_best_image_stitching(best_images_dir: Path = None, output_dir: Path = None, axis: str = 'y', save_debug: bool = False, keep_intermediates: bool = False):
    """
    Complete pipeline: stitch images hierarchically.
    
    Args:
        best_images_dir (Path): Path to the directory containing images. If None, uses current directory
        output_dir (Path): Path to the output directory. If None, uses current directory
        axis (str): Axis along which images vary - 'x' or 'y' (default: 'y')
        save_debug (bool): Whether to save debug images (default: False)
        keep_intermediates (bool): Whether to keep intermediate stitching results (default: False)
    """
    if best_images_dir is None:
        best_images_dir = Path('.').absolute()
        
    if output_dir is None:
        output_dir = Path('.').absolute()
    
    print("=" * 80)
    print("HIERARCHICAL IMAGE STITCHING")
    print("=" * 80)
    
    try:
        # Check if directory exists
        if not best_images_dir.exists():
            print(f"Error: Directory not found: {best_images_dir}")
            print("Please ensure the directory exists.")
            return
        
        # Hierarchical stitching
        print("Starting hierarchical stitching...")
        hierarchical_stitch_best_images(best_images_dir, output_dir, axis, save_debug, keep_intermediates)
        
        print("\n" + "=" * 80)
        print("STITCHING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Print summary
        if best_images_dir.exists():
            best_files = []
            for ext in ['*.tiff', '*.tif', '*.jpg', '*.jpeg', '*.png']:
                best_files.extend(list(best_images_dir.glob(ext)))
                best_files.extend(list(best_images_dir.glob(ext.upper())))
            print(f"\nProcessed {len(best_files)} images:")
            for file in sorted(best_files):
                print(f"  - {file.name}")
        
        final_result = output_dir / "final_stitched.tiff"
        if final_result.exists():
            print(f"\nFinal stitched result: {final_result}")
        else:
            print(f"\nNo final result found. Check for errors above.")
        
    except Exception as e:
        print(f"Error during processing: {e}")


def main():
    """Main function - runs the hierarchical stitching on images in specified folder."""
    parser = argparse.ArgumentParser(
        description='Hierarchical Image Stitching Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bestImageStitch.py /path/to/images y              # Basic stitching (no debug/intermediates)
  python bestImageStitch.py /path/to/images x --debug      # With debug images
  python bestImageStitch.py . y --keep-intermediates       # Keep intermediate files
  python bestImageStitch.py . y --debug --keep-intermediates  # Full output
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
        help='Enable saving debug images (eval regions, matched comparisons, annotated outputs)'
    )
    
    parser.add_argument(
        '--keep-intermediates',
        action='store_true',
        help='Keep intermediate stitching results after completion'
    )
    
    try:
        args = parser.parse_args()
        
        # Convert to Path and normalize
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
        
        # Check if there are any images in the directory
        image_files = []
        for ext in ['*.tiff', '*.tif', '*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(images_dir.glob(ext)))
            image_files.extend(list(images_dir.glob(ext.upper())))
        
        # Remove duplicates
        image_files_unique = []
        seen = set()
        for img in image_files:
            normalized = str(img.resolve())
            if normalized not in seen:
                seen.add(normalized)
                image_files_unique.append(img)
        image_files = image_files_unique
        
        if not image_files:
            print(f"Error: No images found in directory '{images_dir}'!")
            print("Supported formats: TIFF, TIF, JPG, JPEG, PNG")
            return 1
        
        print(f"Found {len(image_files)} images in directory")
        print(f"Stitching along {axis.upper()} axis")
        if save_debug or keep_intermediates:
            print(f"Options: {'debug images' if save_debug else ''}{' ' if save_debug and keep_intermediates else ''}{'keep intermediates' if keep_intermediates else ''}")
        print()
        
        # Run the stitching pipeline
        process_best_image_stitching(images_dir, images_dir, axis, save_debug, keep_intermediates)
        
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