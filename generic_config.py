# config_manager.py
from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Generic, List, Type, TypeVar, Callable
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
        root_dir: str | Path = "./config",
        scope_namer: Callable[[str], str] | None = None,
        default_filename: str = DEFAULT_FILENAME,
        backup_dirname: str = BACKUP_DIRNAME,
        backup_keep: int = BACKUP_KEEP,
    ) -> None:
        if not is_dataclass(schema_cls):
            raise TypeError("schema_cls must be a dataclass type")
        self.schema_cls = schema_cls
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.scope_namer = scope_namer or (lambda s: s)
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
        return self.schema_cls(**{k: v for k, v in data.items() if k in allowed})  # type: ignore

    def scope_dir(self, scope: str) -> Path:
        d = self.root_dir / self.scope_namer(scope)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def active_path(self, scope: str) -> Path:
        return self.scope_dir(scope) / ACTIVE_FILENAME

    def default_path(self, scope: str) -> Path:
        return self.scope_dir(scope) / self.default_filename

    def backup_dir(self, scope: str) -> Path:
        bd = self.scope_dir(scope) / self.backup_dirname
        bd.mkdir(parents=True, exist_ok=True)
        return bd

    def _backup_if_exists(self, scope: str) -> None:
        src = self.active_path(scope)
        if not src.exists():
            return
        ts = time.strftime("%Y%m%d-%H%M%S")
        dst = self.backup_dir(scope) / f"{src.stem}.{ts}{src.suffix}"
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"Warning: failed to create settings backup: {e}")
            return
        try:
            backups: List[Path] = sorted(
                self.backup_dir(scope).glob(f"{src.stem}.*{src.suffix}"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old in backups[self.backup_keep:]:
                try:
                    old.unlink(missing_ok=True)
                except TypeError:
                    old.unlink()
        except Exception as e:
            print(f"Warning: failed to prune backups: {e}")

    # -------- public scope-first API
    def load(self, scope: str) -> S:
        import yaml
        p = self.active_path(scope)
        if p.exists():
            try:
                with open(p, "r") as f:
                    return self._from_dict(yaml.safe_load(f) or {})
            except Exception as e:
                print(f"Error loading settings: {e}")
        # fallbacks
        dp = self.default_path(scope)
        if dp.exists():
            try:
                with open(dp, "r") as f:
                    return self._from_dict(yaml.safe_load(f) or {})
            except Exception as e:
                print(f"Error loading default settings: {e}")
        return self.schema_cls()

    def load_from_file(self, path: str | Path):
        import yaml
        p = Path(path)
        with open(p, "r") as f:
            data = yaml.safe_load(f) or {}
        return self._from_dict(data)

    def save(self, scope: str, settings: S) -> None:
        import yaml
        self._backup_if_exists(scope)
        try:
            with open(self.active_path(scope), "w") as f:
                yaml.safe_dump(self._to_dict(settings), f, sort_keys=False)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def write_defaults(self, scope: str, settings: S | None = None) -> Path:
        import yaml
        payload = self._to_dict(settings or self.schema_cls())
        dp = self.default_path(scope)
        try:
            with open(dp, "w") as f:
                yaml.safe_dump(payload, f, sort_keys=False)
        except Exception as e:
            print(f"Error writing default settings: {e}")
        return dp

    def restore_defaults_into_active(self, scope: str) -> S:
        defaults = self.load_defaults(scope)
        self._backup_if_exists(scope)
        self.save(scope, defaults)
        return defaults

    def load_defaults(self, scope: str) -> S:
        import yaml
        dp = self.default_path(scope)
        if not dp.exists():
            return self.schema_cls()
        try:
            with open(dp, "r") as f:
                return self._from_dict(yaml.safe_load(f) or {})
        except Exception as e:
            print(f"Error loading default settings: {e}")
            return self.schema_cls()

    def list_backups(self, scope: str) -> list[Path]:
        bd = self.backup_dir(scope)
        try:
            return sorted(bd.glob(f"{ACTIVE_FILENAME.split('.')[0]}.*.yaml"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            return []