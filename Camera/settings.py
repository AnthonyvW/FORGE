from dataclasses import dataclass
from typing import Tuple, Any, Dict

import os
import time
import shutil
from pathlib import Path


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

DEFAULT_FILENAME = "default_settings.yaml"
ACTIVE_FILENAME = "settings.yaml"
BACKUP_DIRNAME = "backups"
BACKUP_KEEP = 5  # keep most recent N backups


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
    exposure_max: int = 220

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
    """Manager class for handling camera settings, defaults, and backups."""

    # -------------------------
    # YAML (de)serialization
    # -------------------------
    @staticmethod
    def _to_dict(settings: CameraSettings) -> Dict[str, Any]:
        # Using __dict__ is fine here; we keep the dataclass flat
        return dict(settings.__dict__)

    @staticmethod
    def _from_dict(data: Dict[str, Any]) -> CameraSettings:
        # Allow partial dicts; unknown keys are ignored by dataclass constructor via ** after filtering
        allowed = set(CameraSettings().__dict__.keys())
        filtered = {k: v for k, v in (data or {}).items() if k in allowed}
        return CameraSettings(**filtered)

    # -------------------------
    # Paths & directories
    # -------------------------
    @staticmethod
    def get_backup_dir(config_path: str) -> Path:
        cfg_path = Path(config_path)
        return cfg_path.parent / BACKUP_DIRNAME

    @staticmethod
    def get_default_path(config_path_or_dir: str) -> Path:
        p = Path(config_path_or_dir)
        if p.is_dir():
            return p / DEFAULT_FILENAME
        return p.parent / DEFAULT_FILENAME

    # -------------------------
    # Load / Save primitives
    # -------------------------
    @staticmethod
    def load_settings(config_path: str) -> CameraSettings:
        """Load camera settings from YAML file. Falls back to defaults if active file missing."""
        import yaml
        cfg = Path(config_path)
        try:
            if not cfg.exists():
                # try default_settings.yaml
                default_path = CameraSettingsManager.get_default_path(config_path)
                if default_path.exists():
                    with open(default_path, "r") as stream:
                        data = yaml.safe_load(stream) or {}
                        return CameraSettingsManager._from_dict(data)
                # final fallback: built-in defaults
                return CameraSettings()
            with open(cfg, "r") as stream:
                data = yaml.safe_load(stream) or {}
                return CameraSettingsManager._from_dict(data)
        except Exception as e:
            print(f'Error loading camera settings: {e}')
            return CameraSettings()

    @staticmethod
    def _backup_current_if_exists(config_path: str) -> None:
        """If the active settings file exists, copy it into backups/ with a timestamp, then prune."""
        src = Path(config_path)
        if not src.exists():
            return

        backup_dir = CameraSettingsManager.get_backup_dir(config_path)
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        dst = backup_dir / f"{src.stem}.{ts}{src.suffix}"
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"Warning: failed to create settings backup: {e}")
            return

        # prune old backups (keep newest BACKUP_KEEP)
        try:
            backups = sorted(backup_dir.glob(f"{src.stem}.*{src.suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in backups[BACKUP_KEEP:]:
                try:
                    old.unlink(missing_ok=True)  # py3.8+: omit if not available; fallback:
                except TypeError:
                    try:
                        old.unlink()
                    except Exception:
                        pass
        except Exception as e:
            print(f"Warning: failed to prune backups: {e}")

    @staticmethod
    def save_settings(settings: CameraSettings, config_path: str):
        """Save camera settings to YAML; creates a timestamped backup of the previous version and keeps up to 5."""
        import yaml
        cfg = Path(config_path)
        cfg.parent.mkdir(parents=True, exist_ok=True)

        # backup existing file first
        CameraSettingsManager._backup_current_if_exists(str(cfg))

        # write new file
        try:
            with open(cfg, "w") as stream:
                yaml.safe_dump(CameraSettingsManager._to_dict(settings), stream, sort_keys=False)
        except Exception as e:
            print(f'Error saving camera settings: {e}')

    # -------------------------
    # Defaults workflow
    # -------------------------
    @staticmethod
    def write_default_file(config_dir: str, settings: CameraSettings | None = None) -> Path:
        """Create or overwrite default_settings.yaml in the given directory."""
        import yaml
        cdir = Path(config_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        default_path = cdir / DEFAULT_FILENAME
        payload = CameraSettingsManager._to_dict(settings or CameraSettings())
        try:
            with open(default_path, "w") as stream:
                yaml.safe_dump(payload, stream, sort_keys=False)
        except Exception as e:
            print(f"Error writing default settings: {e}")
        return default_path

    @staticmethod
    def load_defaults(config_dir_or_path: str) -> CameraSettings:
        """Load settings from default_settings.yaml or use built-in defaults if missing."""
        import yaml
        default_path = CameraSettingsManager.get_default_path(config_dir_or_path)
        if not default_path.exists():
            return CameraSettings()
        try:
            with open(default_path, "r") as stream:
                data = yaml.safe_load(stream) or {}
                return CameraSettingsManager._from_dict(data)
        except Exception as e:
            print(f"Error loading default settings: {e}")
            return CameraSettings()

    @staticmethod
    def restore_defaults_to_active(config_path: str) -> CameraSettings:
        """
        Overwrite the active settings file with default_settings.yaml (if present),
        returning the restored settings (or built-in defaults if no default file exists).
        """
        defaults = CameraSettingsManager.load_defaults(config_path)
        # backup current active before overwrite
        CameraSettingsManager._backup_current_if_exists(config_path)
        CameraSettingsManager.save_settings(defaults, config_path)
        return defaults

    # -------------------------
    # Backup discovery helpers
    # -------------------------
    @staticmethod
    def list_backups(config_path: str) -> list[Path]:
        """Return backups for the given config file, newest first."""
        backup_dir = CameraSettingsManager.get_backup_dir(config_path)
        if not backup_dir.exists():
            return []
        try:
            backups = sorted(
                backup_dir.glob(f"{Path(config_path).stem}.*{Path(config_path).suffix}"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            return backups
        except Exception:
            return []

    @staticmethod
    def restore_from_backup(config_path: str, backup_path: str | Path) -> CameraSettings:
        """Restore a specific backup file to be the active settings and return it."""
        bp = Path(backup_path)
        if not bp.exists():
            raise FileNotFoundError(f"Backup not found: {bp}")
        CameraSettingsManager._backup_current_if_exists(config_path)
        shutil.copy2(bp, Path(config_path))
        return CameraSettingsManager.load_settings(config_path)
