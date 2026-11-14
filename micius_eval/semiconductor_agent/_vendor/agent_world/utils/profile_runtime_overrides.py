"""Apply explicitly configured native executables to one software profile."""

import json
import os
from pathlib import Path


def profile_runtime_environment(software_root, environment):
    result = dict(environment)
    manifest_path = result.get("AGENT_WORLD_PROFILE_RUNTIME_OVERRIDES")
    if not manifest_path:
        return result
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("profiles"), dict):
        raise ValueError("Profile runtime overrides must contain a profiles object")
    overrides = manifest["profiles"].get(str(Path(software_root).resolve()), {})
    if not isinstance(overrides, dict):
        raise ValueError("Profile runtime overrides must be an environment object")
    for name, value in overrides.items():
        if name != "NGSPICE_BIN" or not isinstance(value, str) or not value:
            raise ValueError("Unsupported profile runtime override")
        executable = Path(value)
        if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("Profile runtime executable must be an existing absolute path and executable")
        result[name] = str(executable)
    return result
