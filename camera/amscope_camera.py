"""
Amscope camera implementation using the amcam SDK.
"""

from typing import Tuple, Callable, Any, Optional, TYPE_CHECKING
from pathlib import Path
from camera.base_camera import BaseCamera, CameraResolution, CameraInfo
from logger import get_logger

# Module-level reference to the loaded SDK
_amcam = None

# Type hints for IDE support (won't execute at runtime when checking types)
if TYPE_CHECKING:
    import amcam  # This is just for type hints, won't actually import


class AmscopeCamera(BaseCamera):
    """
    Amscope camera implementation using the amcam SDK.
    Wraps the amcam library to conform to the BaseCamera interface.
    
    The SDK must be loaded before using this class:
        AmscopeCamera.ensure_sdk_loaded()
    
    Or it will be loaded automatically on first use.
    """
    
    # Class-level flag to track SDK loading
    _sdk_loaded = False
    
    def __init__(self):
        super().__init__()
        
        # Ensure SDK is loaded before instantiating
        if not AmscopeCamera._sdk_loaded:
            AmscopeCamera.ensure_sdk_loaded()
        
        self._hcam: Optional[Any] = None  # Will be amcam.Amcam after SDK loads
        self._camera_info: Optional[CameraInfo] = None
    
    @classmethod
    def ensure_sdk_loaded(cls, sdk_path: Optional[Path] = None) -> bool:
        """
        Ensure the Amscope SDK is loaded and ready to use.
        
        Args:
            sdk_path: Optional path to SDK base directory.
                     If None, auto-detects from project structure.
        
        Returns:
            True if SDK loaded successfully, False otherwise
        """
        global _amcam
        
        if cls._sdk_loaded and _amcam is not None:
            return True
        
        logger = get_logger()
        
        try:
            from camera.sdk_loaders.amscope_sdk_loader import AmscopeSdkLoader
            
            loader = AmscopeSdkLoader(sdk_path)
            _amcam = loader.load()
            
            cls._sdk_loaded = True
            logger.info("Amscope SDK loaded successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load Amscope SDK: {e}")
            logger.info("Attempting fallback to direct import...")
            
            try:
                # Fallback to direct import if loader fails
                import amcam as amcam_module
                _amcam = amcam_module
                cls._sdk_loaded = True
                logger.info("Amscope SDK loaded via direct import")
                return True
            except ImportError as ie:
                logger.error(f"Direct import also failed: {ie}")
                return False
    
    @staticmethod
    def _get_sdk():
        """Get the loaded SDK module"""
        global _amcam
        if _amcam is None:
            raise RuntimeError(
                "Amscope SDK not loaded. Call AmscopeCamera.ensure_sdk_loaded() first."
            )
        return _amcam
    
    # Class-level event constant accessors
    @classmethod
    def get_event_constants(cls):
        """
        Get event constants as a namespace object.
        Useful for accessing events without a camera instance.
        
        Returns:
            SimpleNamespace with event constants
        """
        from types import SimpleNamespace
        amcam = cls._get_sdk_static()
        return SimpleNamespace(
            IMAGE=amcam.AMCAM_EVENT_IMAGE,
            EXPOSURE=amcam.AMCAM_EVENT_EXPOSURE,
            TEMPTINT=amcam.AMCAM_EVENT_TEMPTINT,
            STILLIMAGE=amcam.AMCAM_EVENT_STILLIMAGE,
            ERROR=amcam.AMCAM_EVENT_ERROR,
            DISCONNECTED=amcam.AMCAM_EVENT_DISCONNECTED
        )
    
    # Event type constants - these are properties since SDK loads dynamically
    @property
    def EVENT_IMAGE(self):
        return self._get_sdk().AMCAM_EVENT_IMAGE
    
    @property
    def EVENT_EXPOSURE(self):
        return self._get_sdk().AMCAM_EVENT_EXPOSURE
    
    @property
    def EVENT_TEMPTINT(self):
        return self._get_sdk().AMCAM_EVENT_TEMPTINT
    
    @property
    def EVENT_STILLIMAGE(self):
        return self._get_sdk().AMCAM_EVENT_STILLIMAGE
    
    @property
    def EVENT_ERROR(self):
        return self._get_sdk().AMCAM_EVENT_ERROR
    
    @property
    def EVENT_DISCONNECTED(self):
        return self._get_sdk().AMCAM_EVENT_DISCONNECTED
        
    @property
    def handle(self) -> Optional[Any]:
        """Get the underlying amcam handle"""
        return self._hcam
    
    def open(self, camera_id: str) -> bool:
        """Open connection to Amscope camera"""
        amcam = self._get_sdk()
        try:
            self._hcam = amcam.Amcam.Open(camera_id)
            if self._hcam:
                self._is_open = True
                # Set RGB byte order for Qt compatibility
                self._hcam.put_Option(amcam.AMCAM_OPTION_BYTEORDER, 0)
                return True
            return False
        except self._get_sdk().HRESULTException:
            return False
    
    def close(self):
        """Close camera connection"""
        if self._hcam:
            self._hcam.Close()
            self._hcam = None
        self._is_open = False
        self._callback = None
        self._callback_context = None
        self._camera_info = None
    
    def start_capture(self, callback: Callable, context: Any) -> bool:
        """Start capturing frames with callback"""
        if not self._hcam:
            return False
        
        amcam = self._get_sdk()
        try:
            self._callback = callback
            self._callback_context = context
            self._hcam.StartPullModeWithCallback(self._event_callback_wrapper, self)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def stop_capture(self):
        """Stop capturing frames"""
        if self._hcam:
            amcam = self._get_sdk()
            try:
                self._hcam.Stop()
            except self._get_sdk().HRESULTException:
                pass
    
    def pull_image(self, buffer: bytes, bits_per_pixel: int = 24) -> bool:
        """Pull the latest image into buffer"""
        if not self._hcam:
            return False
        
        amcam = self._get_sdk()
        try:
            self._hcam.PullImageV4(buffer, 0, bits_per_pixel, 0, None)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def snap_image(self, resolution_index: int = 0) -> bool:
        """Capture a still image"""
        if not self._hcam:
            return False
        
        amcam = self._get_sdk()
        try:
            self._hcam.Snap(resolution_index)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def pull_still_image(self, buffer: bytes, bits_per_pixel: int = 24) -> Tuple[bool, int, int]:
        """
        Pull a still image into buffer
        
        Args:
            buffer: Buffer to receive image data (should be large enough)
            bits_per_pixel: Bits per pixel (typically 24)
            
        Returns:
            Tuple of (success, width, height)
        """
        if not self._hcam:
            return False, 0, 0
        
        amcam = self._get_sdk()
        info = amcam.AmcamFrameInfoV3()
        try:
            # First peek to get dimensions
            self._hcam.PullImageV3(None, 1, bits_per_pixel, 0, info)
            if info.width > 0 and info.height > 0:
                # Then pull the actual image
                self._hcam.PullImageV3(buffer, 1, bits_per_pixel, 0, info)
                return True, info.width, info.height
            return False, 0, 0
        except self._get_sdk().HRESULTException:
            return False, 0, 0
    
    def get_resolutions(self) -> list[CameraResolution]:
        """Get available preview resolutions"""
        if not self._camera_info or not self._camera_info.model:
            return []
        
        resolutions = []
        for i in range(self._camera_info.model.preview):
            res = self._camera_info.model.res[i]
            resolutions.append(CameraResolution(res.width, res.height))
        return resolutions
    
    def get_current_resolution(self) -> Tuple[int, int, int]:
        """Get current resolution index, width, and height"""
        if not self._hcam or not self._camera_info:
            return 0, 0, 0
        
        res_index = self._hcam.get_eSize()
        res = self._camera_info.model.res[res_index]
        return res_index, res.width, res.height
    
    def set_resolution(self, resolution_index: int) -> bool:
        """Set camera resolution"""
        if not self._hcam:
            return False
        
        amcam = self._get_sdk()
        try:
            self._hcam.put_eSize(resolution_index)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def get_exposure_range(self) -> Tuple[int, int, int]:
        """Get exposure time range (min, max, default) in microseconds"""
        if not self._hcam:
            return 0, 0, 0
        
        amcam = self._get_sdk()
        try:
            return self._hcam.get_ExpTimeRange()
        except self._get_sdk().HRESULTException:
            return 0, 0, 0
    
    def get_exposure_time(self) -> int:
        """Get current exposure time in microseconds"""
        amcam = self._get_sdk()
        if not self._hcam:
            return 0
        
        try:
            return self._hcam.get_ExpoTime()
        except self._get_sdk().HRESULTException:
            return 0
    
    def set_exposure_time(self, time_us: int) -> bool:
        """Set exposure time in microseconds"""
        amcam = self._get_sdk()
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_ExpoTime(time_us)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def get_gain_range(self) -> Tuple[int, int, int]:
        """Get gain range (min, max, default) in percent"""
        amcam = self._get_sdk()
        if not self._hcam:
            return 0, 0, 0
        
        try:
            return self._hcam.get_ExpoAGainRange()
        except self._get_sdk().HRESULTException:
            return 0, 0, 0
    
    def get_gain(self) -> int:
        """Get current gain in percent"""
        amcam = self._get_sdk()
        if not self._hcam:
            return 0
        
        try:
            return self._hcam.get_ExpoAGain()
        except self._get_sdk().HRESULTException:
            return 0
    
    def set_gain(self, gain_percent: int) -> bool:
        """Set gain in percent"""
        amcam = self._get_sdk()
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_ExpoAGain(gain_percent)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def get_auto_exposure(self) -> bool:
        """Get auto exposure state"""
        amcam = self._get_sdk()
        if not self._hcam:
            return False
        
        try:
            return self._hcam.get_AutoExpoEnable() == 1
        except self._get_sdk().HRESULTException:
            return False
    
    def set_auto_exposure(self, enabled: bool) -> bool:
        """Set auto exposure state"""
        amcam = self._get_sdk()
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_AutoExpoEnable(1 if enabled else 0)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def supports_white_balance(self) -> bool:
        """Check if camera supports white balance (not monochrome)"""
        if not self._camera_info or not self._camera_info.model:
            return False
        
        amcam = self._get_sdk()
        return (self._camera_info.model.flag & amcam.AMCAM_FLAG_MONO) == 0
    
    def get_white_balance_range(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get white balance range ((temp_min, temp_max), (tint_min, tint_max))"""
        amcam = self._get_sdk()
        return ((amcam.AMCAM_TEMP_MIN, amcam.AMCAM_TEMP_MAX),
                (amcam.AMCAM_TINT_MIN, amcam.AMCAM_TINT_MAX))
    
    def get_white_balance(self) -> Tuple[int, int]:
        """Get current white balance (temperature, tint)"""
        amcam = self._get_sdk()
        if not self._hcam:
            return amcam.AMCAM_TEMP_DEF, amcam.AMCAM_TINT_DEF
        
        try:
            return self._hcam.get_TempTint()
        except self._get_sdk().HRESULTException:
            return amcam.AMCAM_TEMP_DEF, amcam.AMCAM_TINT_DEF
    
    def set_white_balance(self, temperature: int, tint: int) -> bool:
        """Set white balance"""
        amcam = self._get_sdk()
        if not self._hcam:
            return False
        
        try:
            self._hcam.put_TempTint(temperature, tint)
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def auto_white_balance(self) -> bool:
        """Perform one-time auto white balance"""
        amcam = self._get_sdk()
        if not self._hcam:
            return False
        
        try:
            self._hcam.AwbOnce()
            return True
        except self._get_sdk().HRESULTException:
            return False
    
    def get_frame_rate(self) -> Tuple[int, int, int]:
        """Get frame rate info (frames_in_period, time_period_ms, total_frames)"""
        amcam = self._get_sdk()
        if not self._hcam:
            return 0, 0, 0
        
        try:
            return self._hcam.get_FrameRate()
        except self._get_sdk().HRESULTException:
            return 0, 0, 0
    
    @classmethod
    def enumerate_cameras(cls) -> list[CameraInfo]:
        """Enumerate available Amscope cameras"""
        # Ensure SDK is loaded
        if not cls._sdk_loaded:
            cls.ensure_sdk_loaded()
        
        amcam = cls._get_sdk_static()
        cameras = []
        arr = amcam.Amcam.EnumV2()
        for cam in arr:
            info = CameraInfo(
                id=cam.id,
                displayname=cam.displayname,
                model=cam.model
            )
            cameras.append(info)
        return cameras
    
    @staticmethod
    def _get_sdk_static():
        """Static method to get SDK (for use in classmethods)"""
        global _amcam
        if _amcam is None:
            raise RuntimeError(
                "Amscope SDK not loaded. Call AmscopeCamera.ensure_sdk_loaded() first."
            )
        return _amcam
    
    def set_camera_info(self, info: CameraInfo):
        """Set camera information (needed before opening)"""
        self._camera_info = info
    
    def supports_still_capture(self) -> bool:
        """Check if camera supports separate still image capture"""
        if not self._camera_info or not self._camera_info.model:
            return False
        
        return self._camera_info.model.still > 0
    
    def get_still_resolutions(self) -> list[CameraResolution]:
        """Get available still image resolutions"""
        if not self._camera_info or not self._camera_info.model:
            return []
        
        resolutions = []
        for i in range(self._camera_info.model.still):
            res = self._camera_info.model.res[i]
            resolutions.append(CameraResolution(res.width, res.height))
        return resolutions
    
    @staticmethod
    def calculate_buffer_size(width: int, height: int, bits_per_pixel: int = 24) -> int:
        """
        Calculate required buffer size for image data
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            bits_per_pixel: Bits per pixel (typically 24 for RGB)
            
        Returns:
            Buffer size in bytes
        """
        amcam = AmscopeCamera._get_sdk_static()
        return amcam.TDIBWIDTHBYTES(width * bits_per_pixel) * height
    
    @staticmethod
    def calculate_stride(width: int, bits_per_pixel: int = 24) -> int:
        """
        Calculate image stride (bytes per row)
        
        Args:
            width: Image width in pixels
            bits_per_pixel: Bits per pixel (typically 24 for RGB)
            
        Returns:
            Stride in bytes
        """
        amcam = AmscopeCamera._get_sdk_static()
        return amcam.TDIBWIDTHBYTES(width * bits_per_pixel)
    
    @classmethod
    def enable_gige(cls, callback: Optional[Callable] = None, context: Any = None):
        """
        Enable GigE camera support
        
        Args:
            callback: Optional callback for GigE events
            context: Optional context for callback
        """
        # Ensure SDK is loaded
        if not cls._sdk_loaded:
            cls.ensure_sdk_loaded()
        
        amcam = cls._get_sdk_static()
        amcam.Amcam.GigeEnable(callback, context)
    
    def _event_callback_wrapper(self, event: int, context: Any):
        """
        Internal wrapper for camera events.
        Translates amcam events to the callback registered with start_capture.
        """
        if self._callback and self._callback_context:
            # Call the registered callback with the event
            self._callback(event, self._callback_context)
