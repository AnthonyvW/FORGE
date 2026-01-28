# config_manager.py
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Generic, Iterator, List, Type, TypeVar, Union
import shutil
import time

from logger import get_logger

# File/dir names are generic—usable for ANY config
ACTIVE_FILENAME = "settings.yaml"
DEFAULT_FILENAME = "default_settings.yaml"
BACKUP_DIRNAME = "backups"
BACKUP_KEEP = 5  # keep most recent N backups

S = TypeVar("S")  # Config schema type (must be a dataclass)


class ConfigValidationError(Exception):
    """Raised when settings validation fails."""
    pass


class ConfigManager(Generic[S]):
    """
    Generic YAML-backed config manager for ANY dataclass-based settings.
    Manages a single configuration directory with active settings, defaults, and backups.
    
    Directory structure:
        root_dir/
            settings.yaml          # Active settings
            default_settings.yaml  # Factory defaults
            backups/               # Timestamped backups
                settings.20250128-143052.yaml
                settings.20250128-120301.yaml
    
    Example:
        >>> @dataclass
        ... class MySettings:
        ...     value: int = 10
        ...     def validate(self):
        ...         if self.value < 0:
        ...             raise ValueError("value must be non-negative")
        >>> 
        >>> manager = ConfigManager[MySettings](
        ...     MySettings, 
        ...     root_dir="./config/my_component"
        ... )
        >>> settings = manager.load()
        >>> settings.value = 20
        >>> manager.save(settings)
    """

    def __init__(
        self,
        schema_cls: Type[S],
        *,
        root_dir: Union[str, Path] = "./config",
        default_filename: str = DEFAULT_FILENAME,
        backup_dirname: str = BACKUP_DIRNAME,
        backup_keep: int = BACKUP_KEEP,
    ) -> None:
        """
        Initialize the config manager.
        
        Args:
            schema_cls: Dataclass type defining the settings schema
            root_dir: Directory for config files (settings, defaults, backups)
            default_filename: Name for the defaults file
            backup_dirname: Name for the backups subdirectory
            backup_keep: Number of backup files to retain (oldest are deleted)
        
        Raises:
            TypeError: If schema_cls is not a dataclass
        """
        if not is_dataclass(schema_cls):
            logger = get_logger()
            logger.error(f"Attempted to create ConfigManager with non-dataclass type: {schema_cls}")
            raise TypeError(f"schema_cls must be a dataclass type, got {type(schema_cls).__name__}")
        
        self.schema_cls = schema_cls
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.default_filename = default_filename
        self.backup_dirname = backup_dirname
        self.backup_keep = backup_keep
        self._logger = get_logger()
        
        self._logger.debug(f"Initialized ConfigManager for {schema_cls.__name__} at {self.root_dir}")

    # -------------------------
    # YAML (de)serialization
    # -------------------------
    def _to_dict(self, settings: S) -> Dict[str, Any]:
        """Convert settings dataclass to dictionary."""
        return asdict(settings)

    def _from_dict(self, data: Union[Dict[str, Any], None]) -> S:
        """
        Create settings instance from dictionary.
        Only includes fields that are defined in the schema.
        """
        data = data or {}
        allowed = {f.name for f in fields(self.schema_cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return self.schema_cls(**filtered)  # type: ignore

    def _validate(self, settings: S, context: str = "") -> None:
        """
        Validate settings if a validate() method exists.
        
        Args:
            settings: Settings instance to validate
            context: Additional context for error messages
        
        Raises:
            ConfigValidationError: If validation fails
        """
        if hasattr(settings, 'validate') and callable(settings.validate):
            try:
                settings.validate()
                self._logger.debug(f"Validation passed{' for ' + context if context else ''}")
            except Exception as e:
                error_msg = f"Settings validation failed{' for ' + context if context else ''}: {e}"
                self._logger.error(error_msg)
                raise ConfigValidationError(error_msg) from e

    # -------------------------
    # Path helpers
    # -------------------------
    def active_path(self) -> Path:
        """Get the path to the active settings file."""
        return self.root_dir / ACTIVE_FILENAME

    def default_path(self) -> Path:
        """Get the path to the default settings file."""
        return self.root_dir / self.default_filename

    def backup_dir(self) -> Path:
        """Get the backup directory, creating it if needed."""
        bd = self.root_dir / self.backup_dirname
        bd.mkdir(parents=True, exist_ok=True)
        return bd

    # -------------------------
    # Backup management
    # -------------------------
    def _backup_if_exists(self) -> None:
        """
        Create a timestamped backup of the active settings file if it exists.
        Also prunes old backups to maintain backup_keep limit.
        """
        src = self.active_path()
        if not src.exists():
            return
        
        ts = time.strftime("%Y%m%d-%H%M%S")
        dst = self.backup_dir() / f"{src.stem}.{ts}{src.suffix}"
        
        try:
            shutil.copy2(src, dst)
            self._logger.info(f"Created backup: {dst.name}")
        except Exception as e:
            self._logger.error(f"Failed to create settings backup: {e}")
            raise IOError("Failed to create backup") from e
        
        # Prune old backups
        try:
            backups: List[Path] = sorted(
                self.backup_dir().glob(f"{src.stem}.*{src.suffix}"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            
            if len(backups) > self.backup_keep:
                for old in backups[self.backup_keep:]:
                    try:
                        self._logger.debug(f"Pruning old backup: {old.name}")
                        old.unlink(missing_ok=True)
                    except Exception as e:
                        self._logger.warning(f"Failed to delete backup {old.name}: {e}")
                
                self._logger.info(f"Pruned {len(backups) - self.backup_keep} old backup(s)")
        except Exception as e:
            self._logger.warning(f"Failed to prune old backups: {e}")

    # -------------------------
    # Public API
    # -------------------------
    def load(self) -> S:
        """
        Load settings from the active settings file.
        
        Attempts to load in order:
        1. Active settings file
        2. Default settings file
        3. Fresh instance from schema
        
        Returns:
            Settings instance (validated if validate() method exists)
        
        Raises:
            ConfigValidationError: If loaded settings fail validation
        """
        import yaml
        
        p = self.active_path()
        
        # Try loading active settings
        if p.exists():
            try:
                with open(p, "r") as f:
                    data = yaml.safe_load(f) or {}
                settings = self._from_dict(data)
                self._validate(settings, "active settings")
                self._logger.info(f"Loaded active settings from {p.name}")
                return settings
            except ConfigValidationError:
                raise
            except Exception as e:
                self._logger.error(f"Failed to load settings from {p}: {e}")
                raise IOError("Failed to load active settings") from e
        
        # Fallback to defaults
        dp = self.default_path()
        if dp.exists():
            try:
                with open(dp, "r") as f:
                    data = yaml.safe_load(f) or {}
                settings = self._from_dict(data)
                self._validate(settings, "default settings")
                self._logger.info(f"Loaded default settings from {dp.name}")
                return settings
            except ConfigValidationError:
                raise
            except Exception as e:
                self._logger.error(f"Failed to load default settings from {dp}: {e}")
                raise IOError("Failed to load default settings") from e
        
        # Last resort: create fresh instance
        self._logger.info("No existing settings found, using fresh instance")
        settings = self.schema_cls()
        self._validate(settings, "fresh instance")
        return settings

    def load_from_file(self, path: Union[str, Path]) -> S:
        """
        Load settings from an arbitrary file path.
        
        This is useful for loading user-provided or downloaded configuration files.
        
        Args:
            path: Path to the settings file
        
        Returns:
            Settings instance (validated if validate() method exists)
        
        Raises:
            ConfigValidationError: If loaded settings fail validation
            IOError: If file cannot be read
        """
        import yaml
        
        p = Path(path)
        try:
            with open(p, "r") as f:
                data = yaml.safe_load(f) or {}
            settings = self._from_dict(data)
            self._validate(settings, f"file {p.name}")
            self._logger.info(f"Loaded settings from file: {p}")
            return settings
        except ConfigValidationError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to load settings from {p}: {e}")
            raise IOError(f"Failed to load settings from {path}") from e

    def save(self, settings: S) -> None:
        """
        Save settings to the active settings file.
        
        Creates a backup of existing settings before saving.
        
        Args:
            settings: Settings instance to save
        
        Raises:
            ConfigValidationError: If settings fail validation
            IOError: If file cannot be written
        """
        import yaml
        
        # Validate before saving
        self._validate(settings, "before save")
        
        # Backup existing file
        self._backup_if_exists()
        
        # Save new settings
        p = self.active_path()
        try:
            with open(p, "w") as f:
                yaml.safe_dump(self._to_dict(settings), f, sort_keys=False)
            self._logger.info(f"Saved settings to {p.name}")
        except Exception as e:
            self._logger.error(f"Failed to save settings to {p}: {e}")
            raise IOError("Failed to save settings") from e

    def write_defaults(self, settings: Union[S, None] = None) -> Path:
        """
        Write default settings file.
        
        Args:
            settings: Settings to write as defaults. If None, uses fresh schema instance.
        
        Returns:
            Path to the written defaults file
        
        Raises:
            ConfigValidationError: If settings fail validation
            IOError: If file cannot be written
        """
        import yaml
        
        settings_to_save = settings or self.schema_cls()
        self._validate(settings_to_save, "defaults")
        
        payload = self._to_dict(settings_to_save)
        dp = self.default_path()
        
        try:
            with open(dp, "w") as f:
                yaml.safe_dump(payload, f, sort_keys=False)
            self._logger.info(f"Wrote default settings to {dp.name}")
            return dp
        except Exception as e:
            self._logger.error(f"Failed to write default settings to {dp}: {e}")
            raise IOError("Failed to write default settings") from e

    def restore_defaults(self) -> S:
        """
        Restore default settings as the active settings.
        
        Creates a backup of current active settings before restoring.
        
        Returns:
            The restored default settings
        
        Raises:
            ConfigValidationError: If default settings fail validation
            IOError: If restore operation fails
        """
        defaults = self.load_defaults()
        self._backup_if_exists()
        self.save(defaults)
        self._logger.info("Restored defaults as active settings")
        return defaults

    def load_defaults(self) -> S:
        """
        Load default settings.
        
        Returns:
            Default settings instance
        
        Raises:
            ConfigValidationError: If default settings fail validation
            IOError: If defaults file cannot be read
        """
        import yaml
        
        dp = self.default_path()
        if not dp.exists():
            self._logger.debug("No defaults file, using fresh instance")
            settings = self.schema_cls()
            self._validate(settings, "fresh defaults")
            return settings
        
        try:
            with open(dp, "r") as f:
                data = yaml.safe_load(f) or {}
            settings = self._from_dict(data)
            self._validate(settings, "defaults")
            self._logger.info(f"Loaded default settings from {dp.name}")
            return settings
        except ConfigValidationError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to load default settings from {dp}: {e}")
            raise IOError("Failed to load defaults") from e

    def list_backups(self) -> List[Path]:
        """
        List all backup files, most recent first.
        
        Returns:
            List of backup file paths, sorted by modification time (newest first)
        """
        bd = self.backup_dir()
        try:
            backups = sorted(
                bd.glob(f"{ACTIVE_FILENAME.split('.')[0]}.*.yaml"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            self._logger.debug(f"Found {len(backups)} backup(s)")
            return backups
        except Exception as e:
            self._logger.warning(f"Failed to list backups: {e}")
            return []

    @contextmanager
    def edit(self) -> Iterator[S]:
        """
        Context manager for transactional settings editing.
        
        Loads settings, yields for editing, and automatically saves
        on successful exit. If an exception occurs, changes are discarded.
        
        Yields:
            Settings instance for editing
        
        Raises:
            ConfigValidationError: If edited settings fail validation
            IOError: If load or save operations fail
        
        Example:
            >>> with manager.edit() as settings:
            ...     settings.value = 150
            # Auto-saves on successful exit
        """
        self._logger.debug("Starting edit transaction")
        settings = self.load()
        try:
            yield settings
        except Exception as e:
            self._logger.error(f"Edit transaction failed: {e}")
            raise
        else:
            self.save(settings)
            self._logger.info("Edit transaction completed")
