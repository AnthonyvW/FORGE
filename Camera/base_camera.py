from abc import ABC, abstractmethod
import pygame
import time
from pathlib import Path
from PIL import Image
import tkinter as tk
from tkinter import filedialog

class BaseCamera(ABC):
    """Abstract base class defining the camera interface."""
    
    def __init__(self, frame_width: int, frame_height: int):
        self.width = frame_width
        self.height = frame_height
        self.frame = None
        self.scale = 1
        self.capture_path = "./output/"
        self.capture_name = "sample"
        self.capture_index = 1
        self.printer_position = (0, 0, 0)
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Create fallback surface
        self.fallback_surface = pygame.Surface((frame_width, frame_height))
        self.fallback_surface.fill((0, 0, 0))  # Black background

        self.initialize()
        self.resize(frame_width, frame_height)
    
    @abstractmethod
    def initialize(self):
        """Initialize camera hardware and settings."""
        pass
    
    @abstractmethod
    def resize(self, frame_width: int, frame_height: int):
        """Resize camera frame dimensions."""
        pass
    
    @abstractmethod
    def update(self):
        """Update camera frame."""
        pass
    
    def capture_image(self):
        """Capture a still image."""
        self.is_taking_image = True
        self.camera.Snap(0)

    def save_image(self, is_automated: bool, folder: str = "", filename: str = ""):
        """
        Save captured image to disk.
        
        Args:
            folder (str): Optional subfolder within capture_path to save the image
            filename (str): Optional custom filename (without extension)
                          If not provided, uses capture_name + index + position
        """
        while self.is_taking_image:
            time.sleep(0.01)

        try:
            # Build the complete save path
            save_path = Path(self.capture_path).joinpath(self.capture_name) if is_automated else Path(self.capture_path)
            if folder:
                save_path = save_path / folder
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Build the filename
            if filename:
                # Use provided filename
                final_filename = filename
            else:
                # Use default naming pattern with capture_name, index and position
                position_suffix = f"PX{self.printer_position[0]}Y{self.printer_position[1]}Z{self.printer_position[2]}"
                final_filename = f"{self.capture_name}{self.capture_index}{position_suffix}"
                self.capture_index += 1
            
            # Add file extension and combine with path
            full_path = save_path / f"{final_filename}.{self.settings.fformat}"
            
            print(f"Saving Image: {full_path}")
            img = Image.fromarray(self.last_image)
            img.save(str(full_path))
            
        except Exception as e:
            print(f"Error saving image: {e}")
    
    def get_frame(self):
        """Get the current frame scaled appropriately."""
        if self.frame is None:
            return self.fallback_surface
            
        try:
            return pygame.transform.scale_by(self.frame, self.scale)
        except Exception as e:
            print(f"Error scaling frame: {e}")
            return self.fallback_surface
    
    def set_capture_path(self, path: str):
        """Set path for saving captured images."""
        self.capture_path = path

    def set_capture_name(self, name: str):
        """Set path for saving captured images."""
        self.capture_name = name

    def select_capture_path(self):
        """Open a folder selection dialog to set the capture path."""
        root = tk.Tk()
        root.withdraw()  # Hide the main Tk window
        selected_folder = filedialog.askdirectory(title="Select Capture Folder")
        root.destroy()

        if selected_folder:  # User didn't cancel
            self.set_capture_path(selected_folder)
            print(f"Capture path set to: {self.capture_path}")
        return self.capture_path