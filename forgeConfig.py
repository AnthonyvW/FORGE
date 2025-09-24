from __future__ import annotations

from dataclasses import dataclass

from generic_config import ConfigManager, DEFAULT_FILENAME, ACTIVE_FILENAME

@dataclass
class ForgeSettings():
    serial_port: str = "COM9"
    windowWidth: int = 1440
    windowHeight: int = 810
    version: str = "1.1"

def make_forge_settings_manager(
    *,
    root_dir: str = "./config/forge",
    default_filename: str = "default_settings.yaml",
    backup_dirname: str = "backups",
    backup_keep: int = 5,
) -> ConfigManager[ForgeSettings]:
    return ConfigManager[ForgeSettings](
        ForgeSettings,
        root_dir=root_dir,
        default_filename=default_filename,
        backup_dirname=backup_dirname,
        backup_keep=backup_keep,
    )

ForgeSettingsManager = make_forge_settings_manager(
    root_dir="./config/forge",
    default_filename=DEFAULT_FILENAME,
    backup_dirname="backups",
    backup_keep=5,
)