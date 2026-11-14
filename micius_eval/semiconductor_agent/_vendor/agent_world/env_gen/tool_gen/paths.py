"""Deployment-specific defaults shared by ToolGen command line entry points."""

from __future__ import annotations

import json
import os
from pathlib import Path


LOCAL_PATHS_CONFIG = Path(__file__).resolve().parents[2] / 'config/toolgen_paths.local.json'


def _settings() -> dict:
    if not LOCAL_PATHS_CONFIG.is_file():
        return {}
    return json.loads(LOCAL_PATHS_CONFIG.read_text(encoding='utf-8'))


def configured_delivery_root() -> Path | None:
    value = os.environ.get('TOOLGEN_RESULTS_ROOT')
    if not value:
        value = _settings().get('delivery_root')
    if not value:
        return None
    if not isinstance(value, str) or not Path(value).expanduser().is_absolute():
        raise ValueError('ToolGen delivery_root must be an absolute path')
    return Path(value).expanduser()


def canonical_delivery_path(path: Path) -> Path:
    """Resolve an explicitly configured migration alias before loading a package."""
    target = configured_delivery_root()
    if target is None:
        return path
    absolute = Path(os.path.abspath(path.expanduser()))
    for source in _settings().get('source_delivery_roots', []):
        root = Path(source).expanduser()
        if not root.is_absolute():
            raise ValueError('ToolGen source_delivery_roots must be absolute paths')
        try:
            relative = absolute.relative_to(root)
        except ValueError:
            continue
        if len(relative.parts) >= 3 and relative.parts[0] == 'environments':
            migrated = target / relative
            if not migrated.is_file():
                raise FileNotFoundError(f'Migrated ToolGen file is missing: {migrated}')
            return migrated
    return path
