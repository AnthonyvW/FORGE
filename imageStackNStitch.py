import os
from pathlib import Path
from stitching import AffineStitcher
import cv2 as cv
import glob

def process_folders():
    # Get the absolute path to the output directory
    output_dir = Path('output').absolute()
    
    # Create focus_stacked directory if it doesn't exist
    focus_stacked_dir = output_dir / "focus_stacked"
    focus_stacked_dir.mkdir(exist_ok=True)
    
    # Check if output directory exists
    if not output_dir.exists():
        print(f"Error: Directory '{output_dir}' does not exist")
        return
    
    # First, stack all images in each folder
    for folder in output_dir.iterdir():
        if folder.is_dir() and folder.name != "focus_stacked":  # Skip the focus_stacked directory
            folder_name = folder.name
            output_file = focus_stacked_dir / f"{folder_name}.png"
            
            # Use the folder path instead of input/pla
            input_path = folder / "*"  # Add wildcard to include all files
            
            # Construct the command with quotes around paths
            command = (
                f".\\focus-stack\\focus-stack.exe "
                f"\"{input_path}\" "  # Add quotes around input path
                f"--output=\"{output_file}\" "  # Add quotes around output path
                f"--consistency=0 "
                f"--no-whitebalance "
                f"--no-contrast"
            )
            
            print(f"Processing folder: {folder_name}")
            print(f"Running command: {command}")
            
            # Execute the command
            os.system(command)
            
            print(f"Completed stacking {folder_name}\n")

    # Now stitch all the stacked images together
    print("Starting stitching process...")
    
    # Get all PNG files in the focus_stacked directory
    stacked_images = list(focus_stacked_dir.glob("*.png"))
    if not stacked_images:
        print("No stacked images found to stitch!")
        return

    # Convert Path objects to strings
    stacked_images = [str(img) for img in stacked_images]
    
    # Configure the stitcher
    settings = {
        "crop": False,
        "detector": "sift",
        "confidence_threshold": 0.2
    }    

    try:
        # Create and run the stitcher
        stitcher = AffineStitcher(**settings)
        panorama = stitcher.stitch(stacked_images)
        
        # Save the final stitched image in the output directory
        output_path = output_dir / "final_stitched.png"
        cv.imwrite(str(output_path), panorama)
        print(f"Successfully created stitched image: {output_path}")
        
    except Exception as e:
        print(f"Error during stitching: {e}")

if __name__ == "__main__":
    process_folders()