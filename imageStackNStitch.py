import os
from pathlib import Path
from stitching import AffineStitcher
import cv2 as cv
import glob

def focus_stack_images(output_dir: Path) -> Path:
    """
    Perform focus stacking on images in subdirectories of the output directory.
    
    Args:
        output_dir (Path): Path to the main output directory
        
    Returns:
        Path: Path to the directory containing focus stacked images
    """
    # Create focus_stacked directory if it doesn't exist
    focus_stacked_dir = output_dir / "focus_stacked"
    focus_stacked_dir.mkdir(exist_ok=True)
    
    # Check if output directory exists
    if not output_dir.exists():
        raise FileNotFoundError(f"Directory '{output_dir}' does not exist")
    
    # Stack all images in each folder
    for folder in output_dir.iterdir():
        if folder.is_dir() and folder.name != "focus_stacked":
            folder_name = folder.name
            output_file = focus_stacked_dir / f"{folder_name}.png"
            
            # Use the folder path with wildcard to include all files
            input_path = folder / "*"
            
            # Construct the command with quotes around paths
            command = (
                f".\\focus-stack\\focus-stack.exe "
                f"\"{input_path}\" "
                f"--output=\"{output_file}\" "
                f"--consistency=0 "
                f"--no-whitebalance "
                f"--no-contrast"
            )
            
            print(f"Processing folder: {folder_name}")
            print(f"Running command: {command}")
            
            # Execute the command
            os.system(command)
            print(f"Completed stacking {folder_name}\n")
    
    return focus_stacked_dir

def stitch_images(focus_stacked_dir: Path, output_dir: Path) -> None:
    """
    Stitch together focus stacked images into a panorama.
    
    Args:
        focus_stacked_dir (Path): Directory containing the focus stacked images
        output_dir (Path): Directory where the final stitched image will be saved
    """
    print("Starting stitching process...")
    
    # Get all PNG files in the focus_stacked directory
    stacked_images = list(focus_stacked_dir.glob("*.png"))
    if not stacked_images:
        raise ValueError("No stacked images found to stitch!")

    # Convert Path objects to strings
    stacked_images = [str(img) for img in stacked_images]
    
    # Configure the stitcher
    settings = {
        "crop": False,
        "detector": "sift",
        "confidence_threshold": 0.2, 
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
        raise RuntimeError(f"Error during stitching: {e}")

def stitch_by_y_position(focus_stacked_dir: Path, output_dir: Path) -> None:
    """
    Stitch together focus stacked images that share the same y-position.
    Images should be named in the format: X{x_position}Y{y_position}.png
    
    Args:
        focus_stacked_dir (Path): Directory containing the focus stacked images
        output_dir (Path): Directory where the y-stitched images will be saved
    """
    # Create output directory for y-stitched images
    y_stitched_dir = output_dir / "focus_stacked_y_stitched"
    y_stitched_dir.mkdir(exist_ok=True)
    
    # Get all PNG files and organize them by y-position
    images_by_y = {}
    for image_path in focus_stacked_dir.glob("*.png"):
        # Extract x and y positions from filename
        try:
            # Find the Y position in the filename
            filename = image_path.stem  # Get filename without extension
            y_pos = int(filename.split('Y')[1])  # Get number after Y
            
            # Add image to appropriate y-position group
            if y_pos not in images_by_y:
                images_by_y[y_pos] = []
            images_by_y[y_pos].append(str(image_path))
            
        except (IndexError, ValueError) as e:
            print(f"Skipping {image_path}: Could not parse coordinates - {e}")
            continue
    
    # Configure the stitcher
    settings = {
        "crop": False,
        "detector": "sift",
        "confidence_threshold": 0.2,
    }
    
    # Process each y-position group
    for y_pos, image_paths in images_by_y.items():
        if len(image_paths) < 2:
            print(f"Skipping y-position {y_pos}: Only {len(image_paths)} image(s) found")
            continue
            
        try:
            # Sort images by x-position to ensure correct order
            image_paths.sort(key=lambda x: int(Path(x).stem.split('Y')[0].split('X')[1]))
            
            # Create and run the stitcher
            stitcher = AffineStitcher(**settings)
            panorama = stitcher.stitch(image_paths)
            
            # Save the stitched image for this y-position
            output_path = y_stitched_dir / f"y_position_{y_pos}.png"
            cv.imwrite(str(output_path), panorama)
            print(f"Successfully created y-stitched image for y-position {y_pos}")
            
        except Exception as e:
            print(f"Error stitching y-position {y_pos}: {e}")
            continue

def stitch_n_y_images(output_dir: Path, n: int) -> None:
    """
    Stitch together the first N y-stitched images.
    
    Args:
        output_dir (Path): Directory containing the y-stitched images
        n (int): Number of y-stitched images to combine
    """
    y_stitched_dir = output_dir / "focus_stacked_y_stitched"
    if not y_stitched_dir.exists():
        raise FileNotFoundError(f"Y-stitched directory not found: {y_stitched_dir}")
    
    # Get all PNG files and sort them by y-position number
    y_stitched_images = list(y_stitched_dir.glob("*.png"))
    y_stitched_images.sort(key=lambda x: int(x.stem.split('_')[-1]))
    
    if len(y_stitched_images) < n:
        raise ValueError(f"Requested {n} images but only found {len(y_stitched_images)}")
    
    # Take only the first N images
    images_to_stitch = y_stitched_images[:n]
    
    print("\nPreparing to stitch the following images:")
    for img in images_to_stitch:
        print(f"- {img.name}")
    
    # Convert Path objects to strings
    images_to_stitch = [str(img) for img in images_to_stitch]
    
    # Configure the stitcher
    settings = {
        "crop": False,
        "detector": "sift",
        "confidence_threshold": 0.2,
    }
    
    try:
        # Create and run the stitcher
        stitcher = AffineStitcher(**settings)
        panorama = stitcher.stitch(images_to_stitch)
        
        # Save the final stitched image
        output_path = output_dir / f"final_stitched_{n}_rows.png"
        cv.imwrite(str(output_path), panorama)
        print(f"Successfully created stitched image from {n} rows: {output_path}")
        
    except Exception as e:
        raise RuntimeError(f"Error during stitching N rows: {e}")

def process_folders(n_rows: int = None):
    """
    Main function to process folders by focus stacking and stitching images.
    
    Args:
        n_rows (int, optional): Number of y-stitched rows to combine in final image.
            If None, all rows will be combined.
    """
    # Get the absolute path to the output directory
    output_dir = Path('output').absolute()
    focus_stacked_dir = Path('output/focus_stacked').absolute()
    
    try:
        # First perform focus stacking
        #focus_stacked_dir = focus_stack_images(output_dir)
        
        # Then stitch images with the same y-position
        #stitch_by_y_position(focus_stacked_dir, output_dir)
        
        if n_rows is not None:
            # Stitch together first N rows
            stitch_n_y_images(output_dir, n_rows)
        else:
            # Stitch all focus stacked images together
            stitch_images(focus_stacked_dir, output_dir)
        
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    process_folders(2)