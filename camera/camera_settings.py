from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Union
from pathlib import Path

from generic_config import ConfigManager, ConfigValidationError

from app_context import get_app_context

# -------------------------
# Enums for type safety
# -------------------------
class CurveType(str, Enum):
    """Tone mapping curve types."""
    LOGARITHMIC = 'Logarithmic'
    POLYNOMIAL = 'Polynomial'
    OFF = 'Off'


class FileFormat(str, Enum):
    """Supported image file formats."""
    PNG = 'png'
    TIFF = 'tiff'
    JPEG = 'jpeg'
    BMP = 'bmp'


# -------------------------
# Type-safe tuples
# -------------------------
class RGBALevel(NamedTuple):
    """RGBA level range values (0-255 each)."""
    r: int
    g: int
    b: int
    a: int
    
    def validate(self) -> None:
        """Ensure all values are in valid range."""
        for name, value in [('r', self.r), ('g', self.g), ('b', self.b), ('a', self.a)]:
            if not (0 <= value <= 255):
                raise ValueError(f"RGBALevel.{name} must be in range [0, 255], got {value}")


class RGBGain(NamedTuple):
    """RGB white balance gain values (-127 to 127 each)."""
    r: int
    g: int
    b: int
    
    def validate(self) -> None:
        """Ensure all values are in valid range."""
        for name, value in [('r', self.r), ('g', self.g), ('b', self.b)]:
            if not (-127 <= value <= 127):
                raise ValueError(f"RGBGain.{name} must be in range [-127, 127], got {value}")


# -------------------------
# Settings dataclass
# -------------------------
#   From the API documentation:
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
    """Camera image processing settings with validation."""
    
    # Version tracking (defaults to None, will be set to Forge version if missing)
    version: str = get_app_context().settings.version
    
    # Image processing parameters
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
    
    # Complex parameters (now type-safe)
    levelrange_low: RGBALevel = RGBALevel(0, 0, 0, 0)
    levelrange_high: RGBALevel = RGBALevel(255, 255, 255, 255)
    wbgain: RGBGain = RGBGain(0, 0, 0)
    
    # Tone mapping and format
    linear: int = 0                           # 0/1
    curve: CurveType = CurveType.POLYNOMIAL
    fformat: FileFormat = FileFormat.PNG
    
    @classmethod
    def get_ranges(cls) -> dict:
        """
        Return validation ranges for all numeric parameters.
        
        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        return {
            'exposure': (16, 220),
            'temp': (2000, 15000),
            'tint': (200, 2500),
            'levelrange': (0, 255),
            'contrast': (-100, 100),
            'hue': (-180, 180),
            'saturation': (0, 255),
            'brightness': (-64, 64),
            'gamma': (20, 180),
            'wbgain': (-127, 127),
            'sharpening': (0, 500),
            'linear': (0, 1),
        }
    
    def validate(self) -> None:
        """
        Validate all settings are within acceptable ranges.
        
        Raises:
            ValueError: If any parameter is outside its valid range
        """
        ranges = self.get_ranges()
        
        # Validate simple numeric parameters
        for param, (min_val, max_val) in ranges.items():
            if param in ('levelrange', 'wbgain'):
                continue  # Handled separately
            
            value = getattr(self, param)
            if not isinstance(value, bool) and not (min_val <= value <= max_val):
                raise ValueError(
                    f"{param} = {value} is outside valid range [{min_val}, {max_val}]"
                )
        
        # Validate complex types (they have their own validate methods)
        try:
            self.levelrange_low.validate()
        except ValueError as e:
            raise ValueError(f"levelrange_low invalid: {e}") from e
        
        try:
            self.levelrange_high.validate()
        except ValueError as e:
            raise ValueError(f"levelrange_high invalid: {e}") from e
        
        try:
            self.wbgain.validate()
        except ValueError as e:
            raise ValueError(f"wbgain invalid: {e}") from e
        
        # Validate enum types
        if not isinstance(self.curve, CurveType):
            raise ValueError(f"curve must be a CurveType enum, got {type(self.curve)}")
        
        if not isinstance(self.fformat, FileFormat):
            raise ValueError(f"fformat must be a FileFormat enum, got {type(self.fformat)}")
    
    def __post_init__(self) -> None:
        """
        Post-initialization hook to ensure enums are converted from strings.
        
        This allows YAML deserialization to work correctly by converting
        string values back to enum instances.
        """
        # Convert string values to enums if needed
        if isinstance(self.curve, str):
            self.curve = CurveType(self.curve)
        if isinstance(self.fformat, str):
            self.fformat = FileFormat(self.fformat)
        
        # Convert tuples/lists to NamedTuples if needed
        if isinstance(self.levelrange_low, (tuple, list)):
            self.levelrange_low = RGBALevel(*self.levelrange_low)
        if isinstance(self.levelrange_high, (tuple, list)):
            self.levelrange_high = RGBALevel(*self.levelrange_high)
        if isinstance(self.wbgain, (tuple, list)):
            self.wbgain = RGBGain(*self.wbgain)


# -------------------------
# Specialized manager
# -------------------------
class CameraSettingsManager(ConfigManager[CameraSettings]):
    """
    Specialized configuration manager for a single camera model.
    
    Each camera model should have its own manager instance.
    This ensures settings don't bleed between incompatible models.
    
    Directory structure:
        config/cameras/MU500/
            settings.yaml
            default_settings.yaml
            backups/
    
    Example usage:
        >>> # Create manager for MU500
        >>> mu500_mgr = CameraSettingsManager(model="MU500")
        >>> settings = mu500_mgr.load()
        >>> settings.exposure = 150
        >>> mu500_mgr.save(settings)
        >>> 
        >>> # Create separate manager for MU3000 (different settings!)
        >>> mu3000_mgr = CameraSettingsManager(model="MU3000")
        >>> settings = mu3000_mgr.load()  # Won't interfere with MU500
    """
    
    def __init__(
        self,
        *,
        model: str,
        base_dir: Union[str, Path] = "./config/cameras",
        default_filename: str = "default_settings.yaml",
        backup_dirname: str = "backups",
        backup_keep: int = 5,
    ) -> None:
        # Set root_dir to the model-specific directory
        model_dir = Path(base_dir) / model
        
        super().__init__(
            CameraSettings,
            root_dir=model_dir,
            default_filename=default_filename,
            backup_dirname=backup_dirname,
            backup_keep=backup_keep,
        )
        
        self.model = model
        self._logger.info(f"Initialized CameraSettingsManager for model '{model}' at {model_dir}")
