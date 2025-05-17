from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class CameraSettings:
    """Data class for camera settings."""
    auto_expo: bool = False  # Changed from auto_exposure to match config
    exposure: int = 120
    temp: int = 11616
    tint: int = 925
    levelrange_low: Tuple[int, int, int, int] = (0, 0, 0, 0)  # Changed from level_range_low
    levelrange_high: Tuple[int, int, int, int] = (255, 255, 255, 255)  # Changed from level_range_high
    contrast: int = 0
    hue: int = 0
    saturation: int = 126
    brightness: int = -64
    gamma: int = 100
    wbgain: Tuple[int, int, int] = (0, 0, 0)  # Changed from wb_gain
    sharpening: int = 500
    linear: int = 0
    curve: str = 'Polynomial'
    fformat: str = 'png'  # Changed from image_file_format

class CameraSettingsManager:
    """Manager class for handling camera settings."""
    
    @staticmethod
    def load_settings(config_path: str) -> CameraSettings:
        """Load camera settings from YAML file."""
        import yaml
        
        try:
            with open(config_path, "r") as stream:
                settings_dict = yaml.safe_load(stream)
                return CameraSettings(**settings_dict)
        except Exception as e:
            print(f'Error loading camera settings: {e}')
            return CameraSettings()

    @staticmethod
    def save_settings(settings: CameraSettings, config_path: str):
        """Save camera settings to YAML file."""
        import yaml
        
        try:
            with open(config_path, "w") as stream:
                yaml.dump(settings.__dict__, stream)
        except Exception as e:
            print(f'Error saving camera settings: {e}')