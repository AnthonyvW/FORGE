import sys
import time
from pathlib import Path
import pygame
import numpy as np
from PIL import Image
import camera.amcam.amcam as amcam
from .base import BaseCamera
from .settings import CameraSettings, CameraSettingsManager
from image_processing.analyzers import ImageAnalyzer

class AmscopeCamera(BaseCamera):
    def __init__(self, frame_width: int, frame_height: int):
        super().__init__(frame_width, frame_height)
        self.camera = None
        self.frame = None
        self.name = ""
        self.runtime = 0
        self.is_taking_image = False
        self.buffer = None
        self.last_image = None
        self.settings = None
        self.scale = 1.0
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Create fallback surface
        self.fallback_surface = pygame.Surface((frame_width, frame_height))
        self.fallback_surface.fill((0, 0, 0))  # Black background
        
        # Initialize the camera right away
        self.initialize()

    def initialize(self):
        """Initialize the Amscope camera."""
        try:
            available_cameras = amcam.Amcam.EnumV2()
            if not available_cameras:
                raise Exception("Failed to Find Amscope Camera")
            
            self.name = available_cameras[0].displayname
            self.camera = amcam.Amcam.Open(available_cameras[0].id)
            
            if not self.camera:
                raise Exception("Failed to open Amscope Camera")
                
            self.width, self.height = self.camera.get_Size()
            self.buffer = bytes((self.width * 24 + 31) // 32 * 4 * self.height)
            
            if sys.platform == 'win32':
                self.camera.put_Option(amcam.AMCAM_OPTION_BYTEORDER, 0)
                
            # Start the stream immediately after initialization
            self.start_stream()
            return True
            
        except amcam.HRESULTException as e:
            print(f"Error initializing camera: {e}")
            self.camera = None
            return False
        except Exception as e:
            print(f"Unexpected error initializing camera: {e}")
            self.camera = None
            return False

    def start_stream(self):
        """Start the camera stream with configured settings."""
        if self.camera is None:
            print("Cannot start stream - camera not initialized")
            return
            
        try:
            # Load and apply settings first
            self.settings = CameraSettingsManager.load_settings("amscope_camera_configuration.yaml")
            self._apply_settings(self.settings)
            
            # Start the pull mode BEFORE trying to stream
            self.camera.StartPullModeWithCallback(self._camera_callback, self)
            self.resize(self.width, self.height)
            
        except amcam.HRESULTException as e:
            print(f"Error starting stream: {e}")
        except Exception as e:
            print(f"Unexpected error starting stream: {e}")

    def _apply_settings(self, settings: CameraSettings):
        """Apply camera settings."""
        try:
            self.camera.put_AutoExpoEnable(settings.auto_expo)
            self.camera.put_AutoExpoTarget(settings.exposure)
            self.camera.put_TempTint(settings.temp, settings.tint)
            self.camera.put_LevelRange(settings.levelrange_low, settings.levelrange_high)
            self.camera.put_Contrast(settings.contrast)
            self.camera.put_Hue(settings.hue)
            self.camera.put_Saturation(settings.saturation)
            self.camera.put_Brightness(settings.brightness)
            self.camera.put_Gamma(settings.gamma)
            self.camera.put_Option(amcam.AMCAM_OPTION_SHARPENING, settings.sharpening)
            self.camera.put_Option(amcam.AMCAM_OPTION_LINEAR, settings.linear)
            
            curve_options = {'Off': 0, 'Polynomial': 1, 'Logarithmic': 2}
            self.camera.put_Option(amcam.AMCAM_OPTION_CURVE, curve_options.get(settings.curve, 1))
            
        except amcam.HRESULTException as e:
            print(f"Error applying settings: {e}")

    @staticmethod
    def _camera_callback(event, _self):
        """Handle camera events."""
        if event == amcam.AMCAM_EVENT_STILLIMAGE:
            _self._process_frame()
        elif event == amcam.AMCAM_EVENT_IMAGE:
            # Call stream directly from the callback when an image is ready
            try:
                _self.camera.PullImageV2(_self.buffer, 24, None)
                _self.frame = pygame.image.frombuffer(_self.buffer, [_self.width, _self.height], 'RGB')
            except amcam.HRESULTException as e:
                print(f"Error in callback stream: {e}")
        elif event == amcam.AMCAM_EVENT_EXPO_START:
            print("Exposure start event detected")

    def stream(self):
        """This method is now mainly used for error handling and initialization"""
        if self.camera is None:
            print("Camera not initialized. Attempting to initialize...")
            if not self.initialize():
                return
            
        # Ensure buffer is initialized
        if self.buffer is None:
            self.width, self.height = self.camera.get_Size()
            self.buffer = bytes((self.width * 24 + 31) // 32 * 4 * self.height)

    def capture_image(self):
        """Capture a still image."""
        self.is_taking_image = True
        self.camera.Snap(0)

    def _process_frame(self):
        """Process captured frame."""
        self.is_taking_image = True
        try:
            cam_width, cam_height = self.camera.get_StillResolution(0)
            buffer_size = cam_width * cam_height * 3
            buf = bytes(buffer_size)
            self.camera.PullStillImageV2(buf, 24, None)
            
            decoded = np.frombuffer(buf, np.uint8)
            self.last_image = decoded.reshape((cam_height, cam_width, 3))
            
        except amcam.HRESULTException as e:
            print(f"Error processing frame: {e}")
        finally:
            self.is_taking_image = False

    def get_last_image(self) -> np.ndarray:
        """Get the last captured image, waiting if an image capture is in progress."""
        while self.is_taking_image:
            time.sleep(0.01)
        return self.last_image

    def resize(self, frame_width: int, frame_height: int):
        """Resize camera frame dimensions."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Update fallback surface size
        self.fallback_surface = pygame.Surface((frame_width, frame_height))
        self.fallback_surface.fill((0, 0, 0))
        
        if self.frame is not None:
            self.scale = min(frame_width/self.frame.get_width(), 
                           frame_height/self.frame.get_height())

    def update(self):
        """Update camera frame."""
        if not self.is_taking_image and self.camera is not None:
            self.stream()

    def close(self):
        """Clean up camera resources."""
        if self.camera:
            self.camera.Close()
            self.camera = None