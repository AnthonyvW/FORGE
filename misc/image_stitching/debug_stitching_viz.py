"""
Debug Visualization Add-On for bestImageStitch.py

INSTALLATION:
1. Copy this file to the same directory as bestImageStitch.py
2. In bestImageStitch.py, add at the top (after imports):
   
   from debug_stitching_viz import create_debug_visualization
   
3. In create_final_stitched_image(), after saving the output, add:
   
   # Create debug visualization
   create_debug_visualization(canvas, image_paths, images, image_positions, 
                              offsets, output_dir, output_filename, axis, 
                              min_y, total_width, total_height, scale=1.0)

That's it! Now you'll get a _DEBUG.jpg file with annotations.

Alternatively, use this as a standalone post-processing script.
"""

import cv2 as cv
import numpy as np
from pathlib import Path


def create_debug_visualization(canvas, image_paths, images, image_positions, offsets,
                               output_dir, output_filename, axis, min_y, 
                               total_width, total_height, scale=1.0):
    """
    Create annotated debug visualization showing image boundaries and pair info
    
    Args:
        canvas: The stitched image (numpy array)
        image_paths: List of Path objects for source images
        images: List of loaded image arrays
        image_positions: List of (x_pos, y_offset) tuples
        offsets: List of (overlap, y_offset, score, confidence, flags) tuples
        output_dir: Directory to save debug image
        output_filename: Base filename (will add _DEBUG)
        axis: 'x' or 'y'
        min_y: Minimum Y offset
        total_width: Canvas width
        total_height: Canvas height
        scale: Optional scale factor (1.0 = full size)
    """
    
    print(f"\n📊 Creating debug visualization...")
    
    # Create debug canvas
    debug_canvas = canvas.copy()
    
    # Scale up for better label readability (4x larger)
    scale = 1.0
    debug_width = int(total_width * scale)
    debug_height = int(total_height * scale)
    debug_canvas = cv.resize(debug_canvas, (debug_width, debug_height))
    print(f"  Scaled to {debug_width}x{debug_height} (4x) for readability")
    
    # Font settings (adaptive to size)
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(1.0, debug_width / 3000))
    thickness = max(1, int(2 * font_scale))
    line_thickness = max(1, int(3 * font_scale))
    
    # Colors
    COLOR_BOUNDARY = (0, 255, 255)      # Yellow
    COLOR_OVERLAP = (255, 0, 255)       # Magenta  
    COLOR_HIGH = (0, 255, 0)            # Green
    COLOR_MEDIUM = (0, 165, 255)        # Orange
    COLOR_LOW = (0, 0, 255)             # Red
    COLOR_TEXT_BG = (0, 0, 0)           # Black
    COLOR_TEXT = (255, 255, 255)        # White
    
    # ========================================================================
    # DRAW IMAGE BOUNDARIES AND LABELS
    # ========================================================================
    
    for i, (x_pos, y_offset) in enumerate(image_positions):
        img = images[i]
        h, w = img.shape[:2]
        y_pos = y_offset - min_y
        
        # Scale coordinates
        x1 = int(x_pos * scale)
        x2 = int((x_pos + w) * scale)
        y1 = int(y_pos * scale)
        y2 = int((y_pos + h) * scale)
        
        # Draw boundary
        cv.rectangle(debug_canvas, (x1, y1), (x2, y2), COLOR_BOUNDARY, line_thickness)
        
        # Image number (top-left) with semi-transparent background
        img_label = f"#{i}"
        text_size = cv.getTextSize(img_label, font, font_scale, thickness)[0]
        text_x = x1 + 10
        text_y = y1 + text_size[1] + 10
        
        # Semi-transparent background
        overlay = debug_canvas.copy()
        cv.rectangle(overlay, 
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    COLOR_TEXT_BG, -1)
        cv.addWeighted(overlay, 0.6, debug_canvas, 0.4, 0, debug_canvas)
        
        # Text
        cv.putText(debug_canvas, img_label, (text_x, text_y),
                  font, font_scale, COLOR_TEXT, thickness)
        
        # Filename (bottom) with semi-transparent background
        filename = image_paths[i].stem
        # Extract coordinate if present
        import re
        coord_match = re.search(r'[YX](\d+)', filename, re.IGNORECASE)
        if coord_match:
            short_name = coord_match.group(0)  # e.g., "Y187"
        else:
            short_name = filename[:12]
        
        fn_scale = font_scale * 0.7
        fn_thickness = max(1, thickness - 1)
        fn_size = cv.getTextSize(short_name, font, fn_scale, fn_thickness)[0]
        fn_x = x1 + 10
        fn_y = y2 - 10
        
        # Semi-transparent background
        overlay = debug_canvas.copy()
        cv.rectangle(overlay,
                    (fn_x - 5, fn_y - fn_size[1] - 5),
                    (fn_x + fn_size[0] + 5, fn_y + 5),
                    COLOR_TEXT_BG, -1)
        cv.addWeighted(overlay, 0.6, debug_canvas, 0.4, 0, debug_canvas)
        
        cv.putText(debug_canvas, short_name, (fn_x, fn_y),
                  font, fn_scale, COLOR_TEXT, fn_thickness)
    
    # ========================================================================
    # DRAW OVERLAP REGIONS AND PAIR ANNOTATIONS
    # ========================================================================
    
    for i in range(len(images) - 1):
        x_overlap = offsets[i][0]
        y_offset_pair = offsets[i][1]
        score = offsets[i][2] if len(offsets[i]) > 2 else 0.0
        confidence = offsets[i][3] if len(offsets[i]) > 3 else "UNK"
        flags = offsets[i][4] if len(offsets[i]) > 4 else []
        
        if x_overlap <= 0:
            continue
        
        x1_pos, y1_offset = image_positions[i]
        x2_pos, y2_offset = image_positions[i + 1]
        
        overlap_start_x = x2_pos
        overlap_end_x = x1_pos + images[i].shape[1]
        
        if overlap_start_x >= overlap_end_x:
            continue
        
        # Scale coordinates
        overlap_x1 = int(overlap_start_x * scale)
        overlap_x2 = int(overlap_end_x * scale)
        
        # Y range for overlap
        overlap_y1 = int((max(y1_offset, y2_offset) - min_y) * scale)
        overlap_y2 = int((min(y1_offset + images[i].shape[0], 
                              y2_offset + images[i+1].shape[0]) - min_y) * scale)
        
        # Color by confidence
        if confidence == "HIGH":
            pair_color = COLOR_HIGH
        elif confidence == "MEDIUM":
            pair_color = COLOR_MEDIUM
        else:
            pair_color = COLOR_LOW
        
        # Draw vertical lines at overlap boundaries
        cv.line(debug_canvas, (overlap_x1, 0), (overlap_x1, debug_height), 
               COLOR_OVERLAP, line_thickness)
        cv.line(debug_canvas, (overlap_x2, 0), (overlap_x2, debug_height), 
               COLOR_OVERLAP, line_thickness)
        
        # Semi-transparent overlay
        overlay = debug_canvas.copy()
        cv.rectangle(overlay, (overlap_x1, overlap_y1), (overlap_x2, overlap_y2),
                    pair_color, -1)
        cv.addWeighted(overlay, 0.15, debug_canvas, 0.85, 0, debug_canvas)
        
        # Pair annotation box
        pair_center_x = (overlap_x1 + overlap_x2) // 2
        pair_center_y = (overlap_y1 + overlap_y2) // 2
        
        # Multi-line label
        pair_label = f"P{i+1}"
        overlap_label = f"{x_overlap}px"
        score_label = f"{score:.3f}"
        conf_short = confidence[0] if confidence else "?"  # H/M/L
        
        labels = [pair_label, overlap_label, f"y={y_offset_pair}", score_label, conf_short]
        
        # Calculate box size
        max_width = max(cv.getTextSize(l, font, font_scale * 0.7, thickness)[0][0] 
                       for l in labels)
        line_height = int(cv.getTextSize("A", font, font_scale * 0.7, thickness)[0][1] * 1.4)
        box_height = len(labels) * line_height + 15
        box_width = max_width + 15
        
        # Draw background box with transparency
        label_x1 = pair_center_x - box_width // 2
        label_y1 = pair_center_y - box_height // 2
        label_x2 = pair_center_x + box_width // 2
        label_y2 = pair_center_y + box_height // 2
        
        # Semi-transparent background
        overlay = debug_canvas.copy()
        cv.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2),
                    COLOR_TEXT_BG, -1)
        cv.addWeighted(overlay, 0.7, debug_canvas, 0.3, 0, debug_canvas)
        
        # Border (opaque)
        cv.rectangle(debug_canvas, (label_x1, label_y1), (label_x2, label_y2),
                    pair_color, line_thickness)
        
        # Draw text
        text_y = label_y1 + line_height
        for label in labels:
            text_size = cv.getTextSize(label, font, font_scale * 0.7, thickness)[0]
            text_x = pair_center_x - text_size[0] // 2
            cv.putText(debug_canvas, label, (text_x, text_y),
                      font, font_scale * 0.7, pair_color, thickness)
            text_y += line_height
        
        # Flags (if any) with semi-transparent background
        if flags:
            flag_text = ", ".join(str(f)[:20] for f in flags[:2])
            flag_scale = font_scale * 0.5
            flag_size = cv.getTextSize(flag_text, font, flag_scale, max(1, thickness-1))[0]
            flag_x = pair_center_x - flag_size[0] // 2
            flag_y = label_y2 + flag_size[1] + 8
            
            # Semi-transparent background
            overlay = debug_canvas.copy()
            cv.rectangle(overlay,
                        (flag_x - 3, flag_y - flag_size[1] - 3),
                        (flag_x + flag_size[0] + 3, flag_y + 3),
                        COLOR_TEXT_BG, -1)
            cv.addWeighted(overlay, 0.6, debug_canvas, 0.4, 0, debug_canvas)
            
            cv.putText(debug_canvas, flag_text, (flag_x, flag_y),
                      font, flag_scale, (150, 150, 150), max(1, thickness - 1))
    
    # ========================================================================
    # LEGEND
    # ========================================================================
    
    legend_x = 20
    legend_y = 40
    legend_spacing = int(30 * font_scale)
    
    # Legend background (semi-transparent)
    legend_height = 6 * legend_spacing + 20
    overlay = debug_canvas.copy()
    cv.rectangle(overlay, 
                (legend_x - 10, legend_y - 30),
                (legend_x + 200, legend_y + legend_height),
                (40, 40, 40), -1)
    cv.addWeighted(overlay, 0.7, debug_canvas, 0.3, 0, debug_canvas)
    
    cv.putText(debug_canvas, "LEGEND", (legend_x, legend_y),
              font, font_scale * 0.8, COLOR_TEXT, thickness)
    
    legend_items = [
        (COLOR_BOUNDARY, "Image Edge"),
        (COLOR_OVERLAP, "Overlap"),
        (COLOR_HIGH, "HIGH Conf"),
        (COLOR_MEDIUM, "MED Conf"),
        (COLOR_LOW, "LOW Conf"),
    ]
    
    for idx, (color, label) in enumerate(legend_items):
        y_pos = legend_y + (idx + 1) * legend_spacing
        
        cv.rectangle(debug_canvas, 
                    (legend_x, y_pos - 10), 
                    (legend_x + 18, y_pos + 2),
                    color, -1)
        
        cv.putText(debug_canvas, label, (legend_x + 25, y_pos),
                  font, font_scale * 0.6, COLOR_TEXT, max(1, thickness - 1))
    
    # ========================================================================
    # SAVE
    # ========================================================================
    
    # Determine output filename
    base_name = output_filename.replace('.tiff', '').replace('.tif', '').replace('.jpg', '')
    debug_filename = f"{base_name}_DEBUG.jpg"
    debug_path = output_dir / debug_filename
    
    cv.imwrite(str(debug_path), debug_canvas, [cv.IMWRITE_JPEG_QUALITY, 95])
    
    print(f"📊 Debug visualization: {debug_path}")
    print(f"   Legend: Yellow=boundaries, Magenta=overlaps, Green/Orange/Red=confidence")


def add_debug_to_existing_stitcher():
    """
    Instructions for integrating into bestImageStitch.py
    """
    instructions = """
    TO ADD DEBUG VISUALIZATION TO bestImageStitch.py:
    
    1. Copy this file (debug_stitching_viz.py) to same directory
    
    2. Add import at top of bestImageStitch.py:
       from debug_stitching_viz import create_debug_visualization
    
    3. In create_final_stitched_image(), find the line:
       cv.imwrite(str(output_path), canvas)
    
    4. Right after that line, add:
       
       # Generate debug visualization
       create_debug_visualization(
           canvas, image_paths, images, image_positions, offsets,
           output_dir, output_filename, axis, min_y, 
           total_width, total_height
       )
    
    5. Run normally:
       python bestImageStitch.py /path/to/images y --refine
    
    6. You'll get two outputs:
       - final_stitched.tiff (clean result)
       - final_stitched_DEBUG.jpg (annotated version)
    """
    print(instructions)


if __name__ == "__main__":
    add_debug_to_existing_stitcher()