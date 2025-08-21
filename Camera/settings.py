from dataclasses import dataclass
from typing import Tuple


#   (From the API...)
#   .-[ DEFAULT VALUES FOR THE IMAGE ]--------------------------------.
#   | Parameter                | Range         | Default              |
#   |-----------------------------------------------------------------|
#   | Auto Exposure Target     | 16~235        | 120                  |
#   | Temp                     | 2000~15000    | 6503                 |
#   | Tint                     | 200~2500      | 1000                 |
#   | LevelRange               | 0~255         | Low = 0, High = 255  |
#   | Contrast                 | -100~100      | 0                    |
#   | Hue                      | -180~180      | 0                    |
#   | Saturation               | 0~255         | 128                  |
#   | Brightness               | -64~64        | 0                    |
#   | Gamma                    | 20~180        | 100                  |
#   | WBGain                   | -127~127      | 0                    |
#   | Sharpening               | 0~500         | 0                    |
#   | Linear Tone Mapping      | 1/0           | 1                    |
#   | Curved Tone Mapping      | Log/Pol/Off   | 2 (Logarithmic)      |
#   '-----------------------------------------------------------------'

@dataclass
class CameraSettings:
    """Data class for camera settings, including value ranges."""
    # Values
    auto_expo: bool = False
    
    exposure: int = 120                      # Auto Exposure Target
    temp: int = 11616                        # White balance temperature
    tint: int = 925                          # White balance tint
    contrast: int = 0
    hue: int = 0
    saturation: int = 126
    brightness: int = -64
    gamma: int = 100
    sharpening: int = 500

    levelrange_low: Tuple[int, int, int, int] = (0, 0, 0, 0)
    levelrange_high: Tuple[int, int, int, int] = (255, 255, 255, 255)
    wbgain: Tuple[int, int, int] = (0, 0, 0)  # (R, G, B)
    linear: int = 0                           # 0/1
    curve: str = 'Polynomial'
    fformat: str = 'png'

    # Ranges (from table above)
    exposure_min: int = 16
    exposure_max: int = 235

    temp_min: int = 2000
    temp_max: int = 15000

    tint_min: int = 200
    tint_max: int = 2500

    # Applies per-channel for levelrange_low/high
    levelrange_min: int = 0
    levelrange_max: int = 255

    contrast_min: int = -100
    contrast_max: int = 100

    hue_min: int = -180
    hue_max: int = 180

    saturation_min: int = 0
    saturation_max: int = 255

    brightness_min: int = -64
    brightness_max: int = 64

    gamma_min: int = 20
    gamma_max: int = 180

    # Applies per-channel for wbgain (R,G,B)
    wbgain_min: int = -127
    wbgain_max: int = 127

    sharpening_min: int = 0
    sharpening_max: int = 500

    linear_min: int = 0
    linear_max: int = 1


class CameraSettingsManager:
    """Manager class for handling camera settings."""

    @staticmethod
    def load_settings(config_path: str) -> CameraSettings:
        """Load camera settings from YAML file."""
        import yaml
        try:
            with open(config_path, "r") as stream:
                data = yaml.safe_load(stream) or {}
                return CameraSettings(**data)
        except Exception as e:
            print(f'Error loading camera settings: {e}')
            return CameraSettings()

    @staticmethod
    def save_settings(settings: CameraSettings, config_path: str):
        """Save camera settings to YAML file, preserving key order."""
        import yaml
        try:
            with open(config_path, "w") as stream:
                yaml.safe_dump(
                    settings.__dict__,
                    stream,
                    sort_keys=False
                )
        except Exception as e:
            print(f'Error saving camera settings: {e}')
