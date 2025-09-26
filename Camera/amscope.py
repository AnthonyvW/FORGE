import time
from pathlib import Path
import pygame
import numpy as np
from PIL import Image

import os
import sys
import platform
import importlib.util
import ctypes
import zipfile

from .base_camera import BaseCamera
from .camera_settings import CameraSettings, CameraSettingsManager
from image_processing.analyzers import ImageAnalyzer

class AmscopeCamera(BaseCamera):
    # Optional explicit subdir override; otherwise BaseCamera will derive 'amscope'
    CONFIG_SUBDIR = "amscope"

    def __init__(self, frame_width: int, frame_height: int):
        # Minimal vendor state; BaseCamera handles common fields
        self.amcam = None
        self._callback_ref = None  # must keep a reference to avoid garbage collection
        self.buffer = None
        self.camera = None
        self.frame = None
        super().__init__(frame_width, frame_height)

    # Load vendor SDK before initialize()
    def pre_initialize(self):
        self._load_amcam()

    def _ensure_sdk(self, project_root: Path) -> tuple[Path, Path]:
        """
        Ensure the AmScope SDK is available under:
            project_root / "3rd_party_imports" / "official_amscope"
        If not, extract the first amcamsdk*.zip in 3rd_party_imports.
        Returns (sdk_root_dir, sdk_py_path).
        """
        sdk_dir = project_root / "3rd_party_imports"
        official_dir = sdk_dir / "official_amscope"
        sdk_py = official_dir / "python" / "amcam.py"

        # Already extracted?
        if sdk_py.is_file():
            return official_dir, sdk_py

        # Look for a zip starting with "amcamsdk"
        for f in sdk_dir.iterdir():
            if f.is_file() and f.name.lower().startswith("amcamsdk") and f.suffix.lower() == ".zip":
                with zipfile.ZipFile(f, "r") as zf:
                    zf.extractall(official_dir)
                break
        else:
            raise RuntimeError(f"No AmScope SDK found in {sdk_dir}")

        # Handle case with nested folder
        if not sdk_py.is_file():
            subdirs = [d for d in official_dir.iterdir() if d.is_dir()]
            if len(subdirs) == 1 and (subdirs[0] / "python" / "amcam.py").is_file():
                tmp = subdirs[0]
                for item in tmp.iterdir():
                    shutil.move(str(item), official_dir)
                tmp.rmdir()

        if not sdk_py.is_file():
            raise RuntimeError("Extracted SDK does not contain python/amcam.py")

        return official_dir, sdk_py

    def _load_amcam(self):
        project_root = Path(__file__).resolve().parent.parent

        sdk_root, sdk_py = self._ensure_sdk(project_root)

        # Determine platform and architecture
        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == 'windows':
            dll_dir = os.path.join(sdk_root, 'win', 'x64')
        elif system == 'linux':
            arch_map = {
                'x86_64': 'x64',
                'amd64': 'x64',
                'i386': 'x86',
                'i686': 'x86',
                'arm64': 'arm64',
                'aarch64': 'arm64',
                'armv7l': 'armhf',
                'armv6l': 'armel'
            }
            subarch = arch_map.get(machine)
            if not subarch:
                raise RuntimeError(f"Unsupported Linux architecture: {machine}")
            dll_dir = os.path.join(sdk_root, 'linux', subarch)
        elif system == 'darwin':
            dll_dir = os.path.join(sdk_root, 'mac')
        else:
            raise RuntimeError(f"Unsupported operating system: {system}")

        # Update PATH or add_dll_directory for shared library resolution
        if system == 'windows':
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(dll_dir)
            else:
                os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')
        else:
            os.environ['LD_LIBRARY_PATH'] = dll_dir + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

        # Dynamically import amcam.py and override __file__ so its LoadLibrary logic works
        spec = importlib.util.spec_from_file_location("amcam", sdk_py)
        amcam_module = importlib.util.module_from_spec(spec)
        amcam_module.__file__ = os.path.join(dll_dir, 'amcam.py')  # Trick __file__ logic
        sys.modules["amcam"] = amcam_module
        spec.loader.exec_module(amcam_module)

        self.amcam = amcam_module

    def initialize(self):
        """Initialize the Amscope camera."""
        try:
            available_cameras = self.amcam.Amcam.EnumV2()
            if not available_cameras:
                raise Exception("Failed to Find Amscope Camera")

            self.name = available_cameras[0].displayname
            self.camera = self.amcam.Amcam.Open(available_cameras[0].id)

            if not self.camera:
                raise Exception("Failed to open Amscope Camera")

            self.width, self.height = self.camera.get_Size()
            self.buffer = bytes((self.width * 24 + 31) // 32 * 4 * self.height)

            if sys.platform == 'win32':
                self.camera.put_Option(self.amcam.AMCAM_OPTION_BYTEORDER, 0)

            # Start the stream immediately after initialization
            self.start_stream()
            return True

        except self.amcam.HRESULTException as e:
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
            # Load and apply settings from config/amscope/settings.yaml (via BaseCamera helpers)
            self.load_and_apply_settings(filename="settings.yaml")

            # Start the pull mode BEFORE trying to stream
            self.camera.StartPullModeWithCallback(self._camera_callback, self)
            # Recompute scale now that width/height are known
            self.resize(self.frame_width, self.frame_height)

        except self.amcam.HRESULTException as e:
            print(f"Error starting stream: {e}")
        except Exception as e:
            print(f"Unexpected error starting stream: {e}")

    def _apply_settings(self, settings: CameraSettings):
        """Apply camera settings to the hardware."""
        
        # if camera is not initialized, don't apply settings
        if not self.initialized:
            return

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
            self.camera.put_Option(self.amcam.AMCAM_OPTION_SHARPENING, settings.sharpening)
            self.camera.put_Option(self.amcam.AMCAM_OPTION_LINEAR, settings.linear)

            curve_options = {'Off': 0, 'Polynomial': 1, 'Logarithmic': 2}
            self.camera.put_Option(self.amcam.AMCAM_OPTION_CURVE, curve_options.get(settings.curve, 1))

        except self.amcam.HRESULTException as e:
            print(f"Error applying settings: {e}")

    @staticmethod
    def _camera_callback(event, _self):
        """Handle camera events."""
        if event == _self.amcam.AMCAM_EVENT_IMAGE:
            try:
                _self.camera.PullImageV2(_self.buffer, 24, None)
                frame = pygame.image.frombuffer(_self.buffer, [_self.width, _self.height], "RGB")
                _self._set_frame(frame)
            except _self.amcam.HRESULTException as e:
                print(f"Error in callback stream: {e}")
        elif event == _self.amcam.AMCAM_EVENT_STILLIMAGE:
            _self._process_frame()  # will set last_image, see below
        elif event == _self.amcam.AMCAM_EVENT_EXPO_START:
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
        self.is_taking_image = True
        try:
            w, h = self.camera.get_StillResolution(0)
            buf = bytes(w * h * 3)
            self.camera.PullStillImageV2(buf, 24, None)
            arr = np.frombuffer(buf, np.uint8).reshape((h, w, 3))
            self.last_image = arr

            # Optional: briefly show the still in the preview, scaled correctly
            still_surface = pygame.image.frombuffer(buf, (w, h), "RGB")
            self._set_frame(still_surface)  # <- scale computed for still size
        except self.amcam.HRESULTException as e:
            print(f"Error processing frame: {e}")
        finally:
            self.is_taking_image = False

    def update(self):
        """Update camera frame."""
        if not self.is_taking_image and self.camera is not None:
            self.stream()
