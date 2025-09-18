# config_manager.py
from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Generic, List, Type, TypeVar
import shutil
import time

# File/dir names are generic—usable for ANY config
ACTIVE_FILENAME = "settings.yaml"
DEFAULT_FILENAME = "default_settings.yaml"
BACKUP_DIRNAME = "backups"
BACKUP_KEEP = 5  # keep most recent N backups

S = TypeVar("S")  # Config schema type (must be a dataclass)


class ConfigManager(Generic[S]):
    """
    Generic YAML-backed config manager for ANY dataclass-based settings.
    Handles: load/save, defaults file, timestamped backups (+ pruning), restore.
    """

    def __init__(
        self,
        schema_cls: Type[S],
        *,
        default_filename: str = DEFAULT_FILENAME,
        backup_dirname: str = BACKUP_DIRNAME,
        backup_keep: int = BACKUP_KEEP,
    ) -> None:
        if not is_dataclass(schema_cls):
            raise TypeError("schema_cls must be a dataclass type")
        self.schema_cls = schema_cls
        self.default_filename = default_filename
        self.backup_dirname = backup_dirname
        self.backup_keep = backup_keep

    # -------------------------
    # YAML (de)serialization
    # -------------------------
    def _to_dict(self, settings: S) -> Dict[str, Any]:
        return asdict(settings)

    def _from_dict(self, data: Dict[str, Any] | None) -> S:
        data = data or {}
        allowed = {f.name for f in fields(self.schema_cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}
        return self.schema_cls(**filtered)  # type: ignore[arg-type]

    # -------------------------
    # Paths & directories
    # -------------------------
    def get_backup_dir(self, config_path: str | Path) -> Path:
        cfg_path = Path(config_path)
        return cfg_path.parent / self.backup_dirname

    def get_default_path(self, config_path_or_dir: str | Path) -> Path:
        p = Path(config_path_or_dir)
        return (p / self.default_filename) if p.is_dir() else (p.parent / self.default_filename)

    # -------------------------
    # Load / Save primitives
    # -------------------------
    def load_settings(self, config_path: str | Path) -> S:
        """Load settings from YAML. Falls back to default file, then built-in defaults."""
        import yaml
        cfg = Path(config_path)
        try:
            if not cfg.exists():
                default_path = self.get_default_path(cfg)
                if default_path.exists():
                    with open(default_path, "r") as stream:
                        data = yaml.safe_load(stream) or {}
                        return self._from_dict(data)
                return self.schema_cls()  # built-in defaults
            with open(cfg, "r") as stream:
                data = yaml.safe_load(stream) or {}
                return self._from_dict(data)
        except Exception as e:
            print(f"Error loading settings: {e}")
            return self.schema_cls()

    def _backup_current_if_exists(self, config_path: str | Path) -> None:
        """If the active file exists, copy to backups/ with a timestamp, then prune."""
        src = Path(config_path)
        if not src.exists():
            return

        backup_dir = self.get_backup_dir(src)
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        dst = backup_dir / f"{src.stem}.{ts}{src.suffix}"
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"Warning: failed to create settings backup: {e}")
            return

        # prune old backups (keep newest backup_keep)
        try:
            backups: List[Path] = sorted(
                backup_dir.glob(f"{src.stem}.*{src.suffix}"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old in backups[self.backup_keep :]:
                try:
                    old.unlink(missing_ok=True)  # py3.8+: if not available, fallback below
                except TypeError:
                    try:
                        old.unlink()
                    except Exception:
                        pass
        except Exception as e:
            print(f"Warning: failed to prune backups: {e}")

    def save_settings(self, settings: S, config_path: str | Path) -> None:
        """Save settings to YAML; creates a timestamped backup of previous file."""
        import yaml
        cfg = Path(config_path)
        cfg.parent.mkdir(parents=True, exist_ok=True)

        self._backup_current_if_exists(cfg)

        try:
            with open(cfg, "w") as stream:
                yaml.safe_dump(self._to_dict(settings), stream, sort_keys=False)
        except Exception as e:
            print(f"Error saving settings: {e}")

    # -------------------------
    # Defaults workflow
    # -------------------------
    def write_default_file(self, config_dir: str | Path, settings: S | None = None) -> Path:
        """Create/overwrite the default YAML file in the given directory."""
        import yaml
        cdir = Path(config_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        default_path = cdir / self.default_filename
        payload = self._to_dict(settings or self.schema_cls())
        try:
            with open(default_path, "w") as stream:
                yaml.safe_dump(payload, stream, sort_keys=False)
        except Exception as e:
            print(f"Error writing default settings: {e}")
        return default_path

    def load_defaults(self, config_dir_or_path: str | Path) -> S:
        """Load settings from default file or return built-in defaults if missing."""
        import yaml
        default_path = self.get_default_path(config_dir_or_path)
        if not default_path.exists():
            return self.schema_cls()
        try:
            with open(default_path, "r") as stream:
                data = yaml.safe_load(stream) or {}
                return self._from_dict(data)
        except Exception as e:
            print(f"Error loading default settings: {e}")
            return self.schema_cls()

    def restore_defaults_to_active(self, config_path: str | Path) -> S:
        """
        Overwrite the active settings file with the defaults file (if present),
        returning the restored settings (or built-in defaults if no default file exists).
        """
        defaults = self.load_defaults(config_path)
        self._backup_current_if_exists(config_path)
        self.save_settings(defaults, config_path)
        return defaults

    # -------------------------
    # Backup discovery helpers
    # -------------------------
    def list_backups(self, config_path: str | Path) -> list[Path]:
        """Return backups for the given config file, newest first."""
        backup_dir = self.get_backup_dir(config_path)
        if not backup_dir.exists():
            return []
        try:
            backups = sorted(
                backup_dir.glob(f"{Path(config_path).stem}.*{Path(config_path).suffix}"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            return backups
        except Exception:
            return []

    def restore_from_backup(self, config_path: str | Path, backup_path: str | Path) -> S:
        """Restore a specific backup file to be the active settings and return it."""
        bp = Path(backup_path)
        if not bp.exists():
            raise FileNotFoundError(f"Backup not found: {bp}")
        self._backup_current_if_exists(config_path)
        shutil.copy2(bp, Path(config_path))
        return self.load_settings(config_path)
