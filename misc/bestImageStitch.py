"""
Best Image Hierarchical Stitching Tool

This module performs hierarchical stitching on pre-selected best images to create high-quality panoramas.

Usage:
    python best_image_stitcher.py

The script expects:
- A 'best_images' directory containing the pre-selected best images
- Images should be named with coordinates like "X{x}Y{y}_best_single.tiff"
"""

import os
import shutil
from pathlib import Path

# Third-party imports
import cv2 as cv
from stitching import AffineStitcher, Stitcher
import numpy as np



def stitch_image_pair(img1_path: Path, img2_path: Path, output_path: Path) -> bool:
    """
    Optimized stitching that only evaluates relevant edge regions.
    """
    try:
        img1 = cv.imread(str(img1_path))  # Right image
        img2 = cv.imread(str(img2_path))  # Left image
        
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        print(f"Image 1: {h1}x{w1}, Image 2: {h2}x{w2}")
        
        # Calculate edge regions for evaluation
        original_width = 2592  # Original image width
        max_expected_overlap = int(original_width * 0.9)  # 90% max overlap
        
        # For img1 (right image): only use left edge that could overlap
        img1_eval_width = min(max_expected_overlap, w1)
        img1_eval_start = 0  # Left edge of img1
        img1_eval_region = img1[:, img1_eval_start:img1_eval_start + img1_eval_width]
        
        # For img2 (left image): only use right edge that could overlap  
        img2_eval_width = min(max_expected_overlap, w2)
        img2_eval_start = w2 - img2_eval_width  # Right edge of img2
        img2_eval_region = img2[:, img2_eval_start:]
        
        print(f"Evaluation regions - img1: {img1_eval_region.shape}, img2: {img2_eval_region.shape}")
        print(f"Speed improvement: evaluating {img1_eval_width + img2_eval_width} pixels instead of {w1 + w2}")
        
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
        
        print(f"Stage 1: Coarse search on eval regions - overlap {min_eval_overlap} to {max_eval_overlap}")
        
        for overlap in range(min_eval_overlap, max_eval_overlap, 20):
            if overlap > min(img1_eval_width, img2_eval_width):
                continue
                
            # Extract overlapping regions from evaluation areas
            region1 = gray1_strip[:, :overlap]  # Left part of img1's eval region
            region2 = gray2_strip[:, -overlap:]  # Right part of img2's eval region
            
            if region1.shape != region2.shape:
                continue
            
            result = cv.matchTemplate(region1, region2, cv.TM_CCOEFF_NORMED)
            score = result[0, 0]
            
            if score > best_coarse_score:
                best_coarse_score = score
                best_coarse_overlap = overlap
        
        print(f"Coarse result: eval_overlap {best_coarse_overlap}, score {best_coarse_score:.4f}")
        
        # Stage 2: Fine X,Y search around coarse result
        x_range = 30
        y_range = 15
        
        x_min = max(min_eval_overlap, best_coarse_overlap - x_range)
        x_max = min(max_eval_overlap, best_coarse_overlap + x_range)
        
        best_eval_x_overlap = best_coarse_overlap
        best_y_offset = 0
        best_xy_score = best_coarse_score
        
        print(f"Stage 2: Fine X,Y search on eval regions")
        
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
                    
                    region1 = gray1_eval[y_offset:y_offset + compare_height, :x_overlap]
                    region2 = gray2_eval[:compare_height, -x_overlap:]
                else:
                    available_height1 = gray1_eval.shape[0]
                    available_height2 = gray2_eval.shape[0] + y_offset
                    compare_height = min(available_height1, available_height2)
                    
                    if compare_height <= 50:
                        continue
                    
                    region1 = gray1_eval[:compare_height, :x_overlap]
                    region2 = gray2_eval[-y_offset:-y_offset + compare_height, -x_overlap:]
                
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
        
        print(f"Final alignment: eval_x_overlap={best_eval_x_overlap}, actual_x_overlap={actual_x_overlap}")
        print(f"Y_offset={best_y_offset}, score={best_xy_score:.4f}")
        
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
            
            total_width = w2 + w1 - actual_x_overlap
            result = np.zeros((canvas_height, total_width, 3), dtype=np.uint8)
            
            # Place full img2 (left image)
            result[img2_y_start:img2_y_start + h2, :w2] = img2
            
            # Place full img1 (right image) with calculated offsets
            img1_x_start = w2 - actual_x_overlap
            result[img1_y_start:img1_y_start + h1, img1_x_start:img1_x_start + w1] = img1
            
            cv.imwrite(str(output_path), result)
            
            print(f"Output: {canvas_height}x{total_width}")
            print(f"X overlap: {actual_x_overlap} pixels, Y offset: {best_y_offset} pixels")
            
            # Debug visualization
            debug_img = result.copy()
            x_seam = w2 - actual_x_overlap
            cv.line(debug_img, (x_seam, 0), (x_seam, canvas_height), (0, 255, 0), 2)
            
            if best_y_offset != 0:
                y_line = max(img1_y_start, img2_y_start) + min(h1, h2) // 2
                cv.line(debug_img, (0, y_line), (total_width, y_line), (0, 0, 255), 1)
            
            cv.putText(debug_img, f"X:{actual_x_overlap} Y:{best_y_offset}", 
                      (x_seam + 10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            debug_path = output_path.parent / f"optimized_debug_{output_path.name}"
            cv.imwrite(str(debug_path), debug_img)
            
            return True
        else:
            print("No acceptable alignment found")
            return False
        
    except Exception as e:
        print(f"Optimized stitch error: {e}")
        return False


def hierarchical_stitch_best_images(best_images_dir: Path = None, output_dir: Path = None, strategy: str = "best_single"):
    """
    Hierarchically stitch the best single images from a pre-existing best_images directory.
    
    Args:
        best_images_dir (Path): Path to the directory containing best images. If None, uses 'best_images'
        output_dir (Path): Path to the output directory. If None, uses current directory
        strategy (str): Strategy suffix for the images to stitch
    """
    if best_images_dir is None:
        best_images_dir = Path('best_images').absolute()
    
    if output_dir is None:
        output_dir = Path('.').absolute()
    
    if not best_images_dir.exists():
        print(f"Best images directory not found: {best_images_dir}")
        print("Please ensure you have a 'best_images' directory with pre-selected images.")
        return
    
    # Get images for this strategy
    best_images = list(best_images_dir.glob(f"*_{strategy}.tiff"))
    
    if not best_images:
        print(f"No best images found for strategy: {strategy}")
        return
    
    print(f"Starting hierarchical stitching of best single images")
    print(f"Found {len(best_images)} images to stitch")
    
    # Extract X coordinates and sort images
    images_with_coords = []
    for img_path in best_images:
        try:
            # Extract folder name from best image name
            base_name = img_path.stem  # Remove .tiff
            folder_name = base_name.replace(f"_{strategy}", "")  # Remove strategy suffix
            
            # Extract X coordinate
            x_pos = int(folder_name.split('Y')[0].split('X')[1])
            images_with_coords.append((x_pos, img_path))
            
        except (IndexError, ValueError) as e:
            print(f"Skipping {img_path.name}: Could not parse X coordinate - {e}")
            continue
    
    if len(images_with_coords) < 2:
        print(f"Need at least 2 images to stitch, found {len(images_with_coords)}")
        return
    
    # Sort by X coordinate
    images_with_coords.sort(key=lambda x: x[0])
    
    print("Best images sorted by X coordinate:")
    for x_pos, img_path in images_with_coords:
        print(f"  X{x_pos}: {img_path.name}")
    
    # Create working directory for intermediate results
    working_dir = output_dir / f"stitch_working_{strategy}"
    working_dir.mkdir(exist_ok=True)
    
    try:
        # Start hierarchical stitching
        current_images = [img_path for _, img_path in images_with_coords]
        level = 0
        
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
                    
                    # Create output filename for this pair
                    output_name = f"level{level}_pair{pairs_processed + 1}_{strategy}.tiff"
                    output_path = working_dir / output_name
                    
                    print(f"  Stitching pair {pairs_processed + 1}: {img1.name} + {img2.name}")
                    
                    if stitch_image_pair(img1, img2, output_path):
                        next_level_images.append(output_path)
                        pairs_processed += 1
                    else:
                        print(f"    Failed to stitch pair, skipping")
                        # If stitching fails, keep the first image for next level
                        next_level_images.append(img1)
                        
                else:
                    # Odd image out, carry to next level
                    print(f"  Carrying forward: {current_images[i].name}")
                    next_level_images.append(current_images[i])
            
            current_images = next_level_images
            print(f"Level {level} completed. {len(current_images)} images remain.")
        
        # Move final result to main output directory
        if current_images:
            final_image = current_images[0]
            final_output = output_dir / f"final_best_single_stitched.tiff"
            
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
            
            print(f"\n🎉 Best single image stitching completed!")
            print(f"Final result: {final_output}")
            print(f"This should be the sharpest result since it uses original unprocessed images!")
        
    except Exception as e:
        print(f"Error during hierarchical stitching: {e}")
        
    finally:
        # Keep working directory with intermediate results
        print(f"Intermediate stitching results saved in: {working_dir}")
        print("Intermediate files have been preserved for inspection.")


def process_best_image_stitching(best_images_dir: Path = None, output_dir: Path = None):
    """
    Complete pipeline: stitch pre-selected best images hierarchically.
    
    Args:
        best_images_dir (Path): Path to the directory containing best images. If None, uses 'best_images'
        output_dir (Path): Path to the output directory. If None, uses current directory
    """
    if best_images_dir is None:
        best_images_dir = Path('best_images').absolute()
        
    if output_dir is None:
        output_dir = Path('.').absolute()
    
    print("=" * 80)
    print("HIERARCHICAL STITCHING OF BEST IMAGES")
    print("=" * 80)
    
    try:
        # Check if best_images directory exists
        if not best_images_dir.exists():
            print(f"Error: Best images directory not found: {best_images_dir}")
            print("Please ensure you have a 'best_images' directory with your pre-selected images.")
            return
        
        # Hierarchical stitching of best single images
        print("Starting hierarchical stitching...")
        hierarchical_stitch_best_images(best_images_dir, output_dir, "best_single")
        
        print("\n" + "=" * 80)
        print("STITCHING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Print summary
        if best_images_dir.exists():
            best_files = list(best_images_dir.glob("*.tiff")) + list(best_images_dir.glob("*.jpg")) + list(best_images_dir.glob("*.png"))
            print(f"\nProcessed {len(best_files)} best images:")
            for file in sorted(best_files):
                print(f"  - {file.name}")
        
        final_result = output_dir / "final_best_single_stitched.tiff"
        if final_result.exists():
            print(f"\nFinal stitched result: {final_result}")
            print("This should be your sharpest panorama!")
        else:
            print(f"\nNo final result found. Check for errors above.")
        
    except Exception as e:
        print(f"Error during processing: {e}")


def main():
    """Main function - runs the hierarchical stitching on pre-selected best images."""
    try:
        # Check if best_images directory exists
        best_images_dir = Path('output/best_images').absolute()
        if not best_images_dir.exists():
            print(f"Error: Best images directory '{best_images_dir}' does not exist!")
            print("Please ensure you have a 'best_images' directory with your pre-selected images.")
            return
        
        # Check if there are any images in the best_images directory
        image_files = []
        for ext in ['*.tiff', '*.tif', '*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(best_images_dir.glob(ext)))
            image_files.extend(list(best_images_dir.glob(ext.upper())))
        
        if not image_files:
            print("Error: No images found in best_images directory!")
            print("Please ensure your best_images directory contains your pre-selected images.")
            return
        
        print(f"Found {len(image_files)} images in best_images directory:")
        for img_file in sorted(image_files):
            print(f"  - {img_file.name}")
        
        # Run the stitching pipeline
        process_best_image_stitching(best_images_dir)
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()