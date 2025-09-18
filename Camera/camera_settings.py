from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from generic_config import ConfigManager, DEFAULT_FILENAME, ACTIVE_FILENAME


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

    # Ranges (API docs)
    exposure_min: int = 16
    exposure_max: int = 220

    temp_min: int = 2000
    temp_max: int = 15000

    tint_min: int = 200
    tint_max: int = 2500

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

    wbgain_min: int = -127
    wbgain_max: int = 127

    sharpening_min: int = 0
    sharpening_max: int = 500

    linear_min: int = 0
    linear_max: int = 1


# A pre-bound manager that knows how to load/save CameraSettings.
# You can instantiate this wherever you need camera configs.
def make_camera_settings_manager(
    *,
    default_filename: str = "default_settings.yaml",
    backup_dirname: str = "backups",
    backup_keep: int = 5,
) -> ConfigManager[CameraSettings]:
    return ConfigManager[CameraSettings](
        CameraSettings,
        default_filename=default_filename,
        backup_dirname=backup_dirname,
        backup_keep=backup_keep,
    )

CameraSettingsManager = make_camera_settings_manager(
    default_filename=DEFAULT_FILENAME,
    backup_dirname="backups",
    backup_keep=5,
)