"""
Base camera class that defines the interface for camera operations.
All specific camera implementations should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CameraResolution:
    """Represents a camera resolution"""
    width: int
    height: int
    
    def __str__(self):
        return f"{self.width}*{self.height}"


@dataclass
class CameraInfo:
    """Basic camera information"""
    id: str
    displayname: str
    model: Any  # Model-specific information


class BaseCamera(ABC):
    """
    Abstract base class for camera operations.
    Defines the interface that all camera implementations must follow.
    
    Each camera implementation should handle its own SDK loading in the
    ensure_sdk_loaded() method. This is typically called once before any
    camera operations.
    """
    
    # Class-level flag to track if SDK has been loaded
    _sdk_loaded = False
    
    def __init__(self):
        self._is_open = False
        self._callback = None
        self._callback_context = None
        
    @property
    def is_open(self) -> bool:
        """Check if camera is currently open"""
        return self._is_open
    
    @classmethod
    @abstractmethod
    def ensure_sdk_loaded(cls, sdk_path: Optional[Path] = None) -> bool:
        """
        Ensure the camera SDK is loaded and ready to use.
        
        This method should be called before any camera operations.
        Implementations should handle:
        - Loading vendor SDK libraries
        - Platform-specific initialization
        - Setting up library search paths
        - Extracting SDK files if needed
        
        Args:
            sdk_path: Optional path to SDK location. If None, use default location.
            
        Returns:
            True if SDK is loaded successfully, False otherwise
            
        Note:
            This is a class method so it can be called before instantiating cameras.
            Most implementations should track SDK load state to avoid reloading.
        """
        pass
    
    @classmethod
    def is_sdk_loaded(cls) -> bool:
        """
        Check if SDK has been loaded.
        
        Returns:
            True if SDK is loaded, False otherwise
        """
        return cls._sdk_loaded
    
    @abstractmethod
    def open(self, camera_id: str) -> bool:
        """
        Open camera connection
        
        Args:
            camera_id: Identifier for the camera to open
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close camera connection and cleanup resources"""
        pass
    
    @abstractmethod
    def start_capture(self, callback: Callable, context: Any) -> bool:
        """
        Start capturing frames
        
        Args:
            callback: Function to call when events occur
            context: Context object to pass to callback
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def stop_capture(self):
        """Stop capturing frames"""
        pass
    
    @abstractmethod
    def pull_image(self, buffer: bytes, bits_per_pixel: int = 24) -> bool:
        """
        Pull the latest image into provided buffer
        
        Args:
            buffer: Pre-allocated buffer to receive image data
            bits_per_pixel: Bits per pixel (typically 24 for RGB)
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def snap_image(self, resolution_index: int = 0) -> bool:
        """
        Capture a still image at specified resolution
        
        Args:
            resolution_index: Index of resolution to use
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_resolutions(self) -> list[CameraResolution]:
        """
        Get available camera resolutions
        
        Returns:
            List of available resolutions
        """
        pass
    
    @abstractmethod
    def get_current_resolution(self) -> Tuple[int, int, int]:
        """
        Get current resolution
        
        Returns:
            Tuple of (resolution_index, width, height)
        """
        pass
    
    @abstractmethod
    def set_resolution(self, resolution_index: int) -> bool:
        """
        Set camera resolution
        
        Args:
            resolution_index: Index of resolution to use
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_exposure_range(self) -> Tuple[int, int, int]:
        """
        Get exposure time range
        
        Returns:
            Tuple of (min, max, default) values
        """
        pass
    
    @abstractmethod
    def get_exposure_time(self) -> int:
        """
        Get current exposure time
        
        Returns:
            Current exposure time in microseconds
        """
        pass
    
    @abstractmethod
    def set_exposure_time(self, time_us: int) -> bool:
        """
        Set exposure time
        
        Args:
            time_us: Exposure time in microseconds
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_gain_range(self) -> Tuple[int, int, int]:
        """
        Get gain range
        
        Returns:
            Tuple of (min, max, default) values in percent
        """
        pass
    
    @abstractmethod
    def get_gain(self) -> int:
        """
        Get current gain
        
        Returns:
            Current gain in percent
        """
        pass
    
    @abstractmethod
    def set_gain(self, gain_percent: int) -> bool:
        """
        Set gain
        
        Args:
            gain_percent: Gain in percent
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_auto_exposure(self) -> bool:
        """
        Get auto exposure state
        
        Returns:
            True if auto exposure is enabled, False otherwise
        """
        pass
    
    @abstractmethod
    def set_auto_exposure(self, enabled: bool) -> bool:
        """
        Set auto exposure state
        
        Args:
            enabled: True to enable, False to disable
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def supports_white_balance(self) -> bool:
        """
        Check if camera supports white balance
        
        Returns:
            True if white balance is supported, False otherwise
        """
        pass
    
    @abstractmethod
    def get_white_balance_range(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Get white balance range
        
        Returns:
            Tuple of ((temp_min, temp_max), (tint_min, tint_max))
        """
        pass
    
    @abstractmethod
    def get_white_balance(self) -> Tuple[int, int]:
        """
        Get current white balance
        
        Returns:
            Tuple of (temperature, tint)
        """
        pass
    
    @abstractmethod
    def set_white_balance(self, temperature: int, tint: int) -> bool:
        """
        Set white balance
        
        Args:
            temperature: Color temperature value
            tint: Tint value
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def auto_white_balance(self) -> bool:
        """
        Perform one-time auto white balance
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_frame_rate(self) -> Tuple[int, int, int]:
        """
        Get current frame rate information
        
        Returns:
            Tuple of (frames_in_period, time_period_ms, total_frames)
        """
        pass
    
    @staticmethod
    @abstractmethod
    def enumerate_cameras() -> list[CameraInfo]:
        """
        Enumerate available cameras
        
        Returns:
            List of available camera information
        """
        pass
    
    @abstractmethod
    def supports_still_capture(self) -> bool:
        """
        Check if camera supports separate still image capture
        
        Returns:
            True if supported, False otherwise
        """
        pass
    
    @abstractmethod
    def get_still_resolutions(self) -> list[CameraResolution]:
        """
        Get available still image resolutions
        
        Returns:
            List of available still resolutions
        """
        pass
