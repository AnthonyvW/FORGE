"""
Application context for managing shared resources and state.
Provides a singleton pattern for accessing camera and other shared resources.
"""

from typing import Optional, TYPE_CHECKING
from camera.base_camera import BaseCamera
from camera.amscope_camera import AmscopeCamera
from logger import get_logger
from forgeConfig import ForgeSettingsManager, ForgeSettings

if TYPE_CHECKING:
    from UI.settings.settings_main import SettingsDialog
    from UI.widgets.toast_widget import ToastManager


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
        self._settings_dialog: Optional['SettingsDialog'] = None
        self._settings_manager: Optional[ForgeSettingsManager] = None
        self._settings: Optional[ForgeSettings] = None
        self._toast_manager: Optional['ToastManager'] = None
        self._main_window = None
        self._initialized = True
        
        # Load settings
        self._load_settings()
    
    @property
    def camera(self) -> Optional[BaseCamera]:
        """Get the camera instance, initializing if needed"""
        if not self._camera_initialized:
            self._initialize_camera()
        return self._camera
    
    @property
    def settings(self) -> Optional[ForgeSettings]:
        """Get the Forge settings"""
        return self._settings
    
    @property
    def settings_dialog(self) -> Optional['SettingsDialog']:
        """Get the settings dialog instance"""
        return self._settings_dialog
    
    @property
    def toast(self) -> Optional['ToastManager']:
        """Get the toast manager instance"""
        return self._toast_manager
    
    def register_main_window(self, window):
        """Register the main window instance"""
        self._main_window = window
        # Initialize toast manager when main window is registered
        if self._toast_manager is None:
            from UI.widgets.toast_widget import ToastManager
            self._toast_manager = ToastManager(window)
    
    def register_settings_dialog(self, dialog: 'SettingsDialog'):
        """Register the settings dialog instance"""
        self._settings_dialog = dialog
    
    def open_settings(self, category: str):
        """
        Open settings dialog to a specific category.
        
        Args:
            category: Name of the settings category to open to
        """
        if self._settings_dialog:
            self._settings_dialog.open_to(category)
            self._settings_dialog.show()
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()
    
    def _load_settings(self):
        """Load Forge application settings"""
        logger = get_logger()
        try:
            self._settings_manager = ForgeSettingsManager()
            self._settings = self._settings_manager.load()
            logger.info(f"Forge settings loaded - version: {self._settings.version}")
        except Exception as e:
            logger.error(f"Failed to load Forge settings: {e}")
            # Create default settings if loading fails
            self._settings = ForgeSettings()
            logger.warning("Using default Forge settings")
    
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
        self._settings_dialog = None
        self._settings_manager = None
        self._settings = None
        self._toast_manager = None
        self._main_window = None


# Global instance accessor
def get_app_context() -> AppContext:
    """Get the global application context"""
    return AppContext()

def open_settings(category: str):
    AppContext().open_settings(category)