"""
Application context for managing shared resources and state.
Provides a singleton pattern for accessing camera and other shared resources.
"""

from typing import Optional
from camera.base_camera import BaseCamera
from camera.amscope_camera import AmscopeCamera
from logger import get_logger


class AppContext:
    """
    Singleton application context managing shared resources.
    """
    _instance: Optional['AppContext'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._camera: Optional[BaseCamera] = None
        self._camera_initialized = False
        self._initialized = True
    
    @property
    def camera(self) -> Optional[BaseCamera]:
        """Get the camera instance, initializing if needed"""
        if not self._camera_initialized:
            self._initialize_camera()
        return self._camera
    
    def _initialize_camera(self):
        """Initialize the camera subsystem"""
        if self._camera_initialized:
            return
            
        logger = get_logger()
        try:
            # Load SDK
            AmscopeCamera.ensure_sdk_loaded()
            
            # Enable GigE support
            AmscopeCamera.enable_gige(None, None)
            
            # Create camera instance
            self._camera = AmscopeCamera()
            self._camera_initialized = True
            
            logger.info("Camera subsystem initialized")
        except Exception as e:
            logger.error(f"Failed to initialize camera subsystem: {e}")
            self._camera = None
            self._camera_initialized = True
    
    def cleanup(self):
        """Cleanup resources"""
        if self._camera and self._camera.is_open:
            self._camera.close()
        self._camera = None
        self._camera_initialized = False


# Global instance accessor
def get_app_context() -> AppContext:
    """Get the global application context"""
    return AppContext()
