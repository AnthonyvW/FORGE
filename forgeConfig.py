from __future__ import annotations

from dataclasses import dataclass
from typing import Union
from pathlib import Path

from generic_config import ConfigManager


@dataclass
class ForgeSettings:
    """Forge application settings."""
    version: str = "1.2"
    
    def validate(self) -> None:
        """
        Validate Forge settings.
        
        Raises:
            ValueError: If any setting is invalid
        """
        # Add validation logic as needed
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty string")


class ForgeSettingsManager(ConfigManager[ForgeSettings]):
    """
    Configuration manager for Forge application settings.
    
    Directory structure:
        config/forge/
            settings.yaml
            default_settings.yaml
            backups/
    
    Example usage:
        >>> forge_mgr = ForgeSettingsManager()
        >>> settings = forge_mgr.load()
        >>> settings.version = "1.2"
        >>> forge_mgr.save(settings)
    """
    
    def __init__(
        self,
        *,
        root_dir: Union[str, Path] = "./config/forge",
        default_filename: str = "default_settings.yaml",
        backup_dirname: str = "backups",
        backup_keep: int = 5,
    ) -> None:
        super().__init__(
            ForgeSettings,
            root_dir=root_dir,
            default_filename=default_filename,
            backup_dirname=backup_dirname,
            backup_keep=backup_keep,
        )