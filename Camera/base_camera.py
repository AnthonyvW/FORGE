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
        # Public-ish, common state
        self.name = ""
        self.is_taking_image = False
        self.last_image = None
        # Safe default for save_image() until a subclass loads real settings
        self.settings = type("DefaultSettings", (), {"fformat": "png"})()

        # Dimensions of the UI frame we render into
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Camera-native dimensions (subclasses may set real values during initialize())
        self.width = frame_width
        self.height = frame_height

        # Current live frame (pygame Surface) and scale
        self.frame = None
        self.scale = 1

        # Capture path/name/index and optional printer position
        self.capture_path = "./output/"
        self.capture_name = "sample"
        self.capture_index = 1
        self.printer_position = (0, 0, 0)

        # Create fallback surface
        self.fallback_surface = pygame.Surface((frame_width, frame_height))
        self.fallback_surface.fill((0, 0, 0))  # Black background

        # Allow subclasses to do pre-initialize work (e.g., load SDKs) before initialize()
        self.pre_initialize()
        self.initialize()
        # Run the default resize scaffold once so scale/fallback are consistent
        self.resize(frame_width, frame_height)

    # ----- Lifecycle hooks -----
    def pre_initialize(self):
        """Optional hook to run before initialize(); subclasses may override."""
        pass

    @abstractmethod
    def initialize(self):
        """Initialize camera hardware and settings."""
        pass

    # Provide a default resize scaffold that most cameras can reuse.
    # Subclasses can override if they have special behavior.
    def resize(self, frame_width: int, frame_height: int):
        """Resize the UI frame dimensions and update scaling/fallback surface."""
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Update fallback surface size
        self.fallback_surface = pygame.Surface((frame_width, frame_height))
        self.fallback_surface.fill((0, 0, 0))

        # Recompute scale if we have a live frame
        if self.frame is not None:
            try:
                fw, fh = self.frame.get_width(), self.frame.get_height()
                if fw > 0 and fh > 0:
                    self.scale = min(frame_width / fw, frame_height / fh)
            except Exception:
                # Keep previous scale on error
                pass

    @abstractmethod
    def update(self):
        """Update camera frame."""
        pass

    @abstractmethod
    def capture_image(self):
        """Capture a still image (subclass must implement)."""
        pass

    def get_last_image(self):
        """Get the last captured image, waiting if a capture is in progress."""
        while self.is_taking_image:
            time.sleep(0.01)
        return self.last_image

    def save_image(self, is_automated: bool, folder: str = "", filename: str = ""):
        """
        Save captured image to disk.

        Args:
            is_automated (bool): Whether to nest images under capture_name/
            folder (str): Optional subfolder within capture_path to save the image
            filename (str): Optional custom filename (without extension)
        """
        # Wait for any in-flight capture first
        while self.is_taking_image:
            time.sleep(0.01)

        img_data = self.last_image
        if img_data is None:
            print("No image to save (last_image is None).")
            return

        try:
            import numpy as np
            from PIL import Image

            # Accept either NumPy array or pygame.Surface
            if hasattr(img_data, "get_view") and hasattr(img_data, "get_size"):
                # Pygame Surface
                w, h = img_data.get_size()
                raw = pygame.image.tostring(img_data, "RGB")
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            else:
                arr = np.array(img_data)  # makes a copy if needed

            # Validate shape/dtype
            if arr.ndim == 2:
                # grayscale -> expand to 3 channels
                arr = np.stack([arr]*3, axis=-1)
            if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                raise ValueError(f"Unsupported image shape: {arr.shape}")

            if arr.dtype != np.uint8:
                # Try to scale/clip to uint8
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            if arr.shape[2] == 4:
                mode = "RGBA"
            else:
                mode = "RGB"

            # Build save path
            save_path = Path(self.capture_path).joinpath(self.capture_name) if is_automated else Path(self.capture_path)
            if folder:
                save_path = save_path / folder
            save_path.mkdir(parents=True, exist_ok=True)

            # Build filename
            if filename:
                final_filename = filename
            else:
                x, y, z = getattr(self, "printer_position", (0, 0, 0))
                position_suffix = f"PX{x}Y{y}Z{z}"
                final_filename = f"{self.capture_name}{self.capture_index}{position_suffix}"
                self.capture_index += 1

            fformat = getattr(self.settings, "fformat", "png") or "png"
            full_path = save_path / f"{final_filename}.{fformat}"

            print(f"Saving Image: {full_path}")
            Image.fromarray(arr, mode=mode).save(str(full_path))

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
        """Set capture name prefix used for automated saves."""
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

    # Provide a default close() that gracefully shuts down common SDKs
    def close(self):
        """Clean up camera resources if possible."""
        cam = getattr(self, "camera", None)
        if cam is not None:
            try:
                close_fn = getattr(cam, "Close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass
            finally:
                self.camera = None
