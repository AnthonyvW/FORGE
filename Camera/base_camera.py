from abc import ABC, abstractmethod
import time
from pathlib import Path
from PIL import Image
import tkinter as tk
import numpy as np
from tkinter import filedialog

from .camera_settings import (
    CameraSettings,
    CameraSettingsManager,
    ACTIVE_FILENAME,
    DEFAULT_FILENAME
)


class BaseCamera(ABC):
    """Abstract base class defining the camera interface."""

    # Subclasses may override to control config subfolder name
    CONFIG_SUBDIR: str | None = None

    def __init__(self):
        # Public-ish, common state
        self.name = ""
        self.is_taking_image = False
        self.last_image: np.ndarray | None = None   # (H, W, 3) RGB uint8
        self.last_stream_array: np.ndarray | None = None  # (H, W, 3) RGB uint8

        self.last_image_ts: float = 0.0
        self.last_stream_ts: float = 0.0

        self.initialized = False
        # Safe default for save_image() until a subclass loads real settings
        self.settings = CameraSettings()
        self._scope = self.get_impl_key()
        CameraSettingsManager.scope_dir(self._scope)

        # Camera-native dimensions (subclasses may set real values during initialize())
        self.width = 1280
        self.height = 720

        # Capture path/name/index and optional printer position
        self.capture_path = "./output/"
        self.capture_name = "sample"
        self.capture_index = 1
        self.printer_position = (0, 0, 0)

        # Config roots
        self._scope = self.get_impl_key()
        CameraSettingsManager.scope_dir(self._scope)
        self.impl_config_dir = self.get_config_dir()  # e.g., config/amscope
        self.impl_config_dir.mkdir(parents=True, exist_ok=True)

        # Allow subclasses to do pre-initialize work (e.g., load SDKs) before initialize()
        self.pre_initialize()
        self.initialized = self.initialize()

    # ----- Lifecycle hooks -----
    def pre_initialize(self):
        """Optional hook to run before initialize(); subclasses may override."""
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize camera hardware and settings."""
        pass

    @abstractmethod
    def update(self):
        """Update camera frame."""
        pass

    @abstractmethod
    def capture_image(self):
        """Capture a still image (subclass must implement)."""
        pass

    # -------------------------------
    # Config & settings convenience
    # -------------------------------
    def get_impl_key(self) -> str:
        """
        Returns the implementation key used for config subfolder naming.
        Default: lowercased class name with trailing 'camera' removed (e.g., AmscopeCamera -> 'amscope').
        Subclasses can override by setting CONFIG_SUBDIR.
        """
        if isinstance(self.CONFIG_SUBDIR, str) and self.CONFIG_SUBDIR.strip():
            return self.CONFIG_SUBDIR.strip()
        cls = self.__class__.__name__
        return (cls[:-6] if cls.lower().endswith("camera") else cls).lower()

    def get_config_dir(self) -> Path:
        return CameraSettingsManager.scope_dir(self._scope)

    def load_and_apply_settings(self, filename: str = ACTIVE_FILENAME):
        """
        Load settings from YAML and apply to the live camera.
        If the active file is missing, this falls back to default_settings.yaml, else built-in defaults.
        """
        loaded = CameraSettingsManager.load(self._scope)
        self.settings = loaded
        self.apply_settings(self.settings)

    def apply_settings(self, settings):
        """
        Apply settings to the hardware. By default this calls a subclass hook named _apply_settings
        if present. Subclasses should implement _apply_settings(settings: CameraSettings).
        """
        hook = getattr(self, "_apply_settings", None)
        if callable(hook):
            hook(settings)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement _apply_settings(settings) or override apply_settings()."
            )

    def save_settings(self, filename: str = ACTIVE_FILENAME):
        """
        Persist current settings to YAML in the per-implementation folder.
        Automatically creates a timestamped backup of the previous version and keeps the 5 most recent.
        """
        CameraSettingsManager.save(self._scope, self.settings)

    def set_settings(self, settings, persist: bool = False, filename: str = ACTIVE_FILENAME):
        """
        Replace the entire settings object, apply immediately, optionally persist to disk.
        """
        self.settings = settings
        self.apply_settings(self.settings)
        if persist:
            self.save_settings(filename=filename)

    def update_settings(self, persist: bool = False, filename: str = ACTIVE_FILENAME, **updates):
        """
        Update one or more attributes on the current settings, apply immediately, and
        optionally persist to disk. Example:

            camera.update_settings(temp=6500, tint=900, linear=1, persist=True)
        """
        # If settings hasn't been loaded yet, attempt to load from disk first.
        if not hasattr(self.settings, "__dict__"):
            self.load_and_apply_settings(filename=filename)

        # Apply updates (only for existing attributes to avoid silent typos)
        for k, v in updates.items():
            if hasattr(self.settings, k):
                setattr(self.settings, k, v)
            else:
                raise AttributeError(f"Unknown camera setting '{k}'")

        # Push to hardware and optionally persist
        self.apply_settings(self.settings)
        if persist:
            self.save_settings(filename=filename)

    # ----- Defaults helpers -----
    def get_default_config_path(self) -> Path:
        return self.get_config_path(DEFAULT_FILENAME)

    def write_default_settings(self, settings: CameraSettings | None = None) -> Path:
        """
        Write default_settings.yaml in this camera's config directory.
        If 'settings' is None, writes built-in defaults.
        """
        return CameraSettingsManager.write_defaults(self._scope, settings)

    def load_default_settings(self):
        """
        Load defaults from default_settings.yaml (or built-in defaults if file doesn't exist),
        apply to hardware, but do NOT persist to the active file.
        """
        defaults = CameraSettingsManager.load_defaults(self._scope)
        self.set_settings(defaults, persist=False)
        return defaults

    def restore_default_settings(self, persist: bool = True):
        """
        Restore defaults into the active settings file (backup the current one), apply, and optionally persist.
        Useful for a "Restore Defaults" button in the UI.
        """
        restored = CameraSettingsManager.restore_defaults_into_active(self._scope)
        self.set_settings(restored, persist=False)
        if persist:
            self.save_settings()
        return restored

    # -------------------------------
    # Image helpers
    # -------------------------------
    def get_last_image(self):
        """Get the last captured image, waiting if a capture is in progress."""
        while self.is_taking_image:
            time.sleep(0.01)
        return self.last_image

    def get_last_stream_array(self) -> np.ndarray | None:
        """Return latest live-stream RGB frame as (H, W, 3) uint8, or None."""
        return self.last_stream_array

    def get_last_frame(self, prefer: str = "latest", wait_for_still: bool = True):
        """
        Return the latest RGB frame (H, W, 3) uint8 from either a still or the stream.

        prefer:
        - "latest" (default): whichever arrived most recently (compares timestamps)
        - "still"           : still if present, else stream
        - "stream"          : stream if present, else still

        wait_for_still:
        - If True, block briefly if a still capture is currently in progress.
        """
        if wait_for_still and self.is_taking_image:
            while self.is_taking_image:
                time.sleep(0.01)

        # Fast paths for legacy behavior
        if prefer == "still":
            return self.last_image if self.last_image is not None else self.last_stream_array
        if prefer == "stream":
            return self.last_stream_array if self.last_stream_array is not None else self.last_image

        # "latest" behavior: pick the freshest we’ve seen
        li, ls = self.last_image, self.last_stream_array
        ti, ts = self.last_image_ts, self.last_stream_ts

        if li is None and ls is None:
            return None
        if li is None:
            return ls
        if ls is None:
            return li
        return li if ti >= ts else ls

    def save_image(self, is_automated: bool, folder: str = "", filename: str = ""):
        while self.is_taking_image:
            time.sleep(0.01)

        arr = self.last_image
        if arr is None:
            print("No image to save (last_image is None).")
            return

        try:
            arr = np.asarray(arr)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                raise ValueError(f"Unsupported image shape: {arr.shape}")
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            mode = "RGBA" if arr.shape[2] == 4 else "RGB"

            save_path = Path(self.capture_path)
            if is_automated:
                save_path = save_path / self.capture_name
            if folder:
                save_path = save_path / folder
            save_path.mkdir(parents=True, exist_ok=True)

            if filename:
                final_filename = filename
            else:
                x, y, z = getattr(self, "printer_position", (0, 0, 0))
                final_filename = f"{self.capture_name}{self.capture_index}PX{x}Y{y}Z{z}"
                self.capture_index += 1

            fformat = self.settings.fformat
            full_path = save_path / f"{final_filename}.{fformat}"
            print(f"Saving Image: {full_path}")
            Image.fromarray(arr, mode=mode).save(str(full_path))
        except Exception as e:
            print(f"Error saving image: {e}")

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
