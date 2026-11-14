"""Unpack (if needed) and relocate the Semiconductor Agent tool runtimes.

The software profiles are uv virtual environments whose interpreter links and
``pyvenv.cfg`` refer to the absolute prefix they were built under. This
script rewrites those references to the local dataset directory, so the data
can live anywhere. It is idempotent.

    python -m micius_eval.semiconductor_agent.setup_data --data-root $MICIUS_DATA/semiconductor_agent
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from .data import SOURCE_PREFIX, default_data_root

TEXT_SUFFIXES = {".cfg", ".pth", ".sh", ".txt", ".json", ".toml"}


def extract_archives(data_root: Path) -> None:
    """Extract ``archives/<name>.tar.zst.part-NNN`` streams (or whole ``.tar*`` files) once."""
    folder = data_root / "archives"
    if not folder.is_dir():
        return
    streams: dict[str, list[Path]] = {}
    for path in sorted(folder.iterdir()):
        if ".tar" in path.name and not path.name.endswith(".extracted"):
            streams.setdefault(path.name.split(".part-")[0], []).append(path)
    for name, parts in streams.items():
        done = folder / (name + ".extracted")
        if done.exists():
            continue
        print(f"extracting {name} ({len(parts)} part(s)) ...", flush=True)
        if name.endswith(".zst"):
            cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
            subprocess.run(["tar", "--use-compress-program=unzstd", "-xf", "-", "-C", str(data_root)],
                           stdin=cat.stdout, check=True)
            if cat.wait():
                raise RuntimeError(f"failed to read {name}")
        else:
            with tarfile.open(parts[0]) as tar:
                tar.extractall(data_root, filter="tar")
        done.touch()


def relocate(data_root: Path, old_prefix: str) -> dict[str, int]:
    new_prefix = str((data_root / "source_tree").resolve())
    stats = {"symlinks": 0, "text_files": 0}
    for dirpath, dirnames, filenames in os.walk(data_root / "source_tree"):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                target = os.readlink(path)
                if target.startswith(old_prefix):
                    path.unlink()
                    os.symlink(new_prefix + target[len(old_prefix):], path)
                    stats["symlinks"] += 1
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            in_bin = Path(dirpath).name in ("bin", "Scripts")
            if not (in_bin or path.suffix in TEXT_SUFFIXES or name == "pyvenv.cfg"):
                continue
            try:
                if path.stat().st_size > 2_000_000:
                    continue
                raw = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:4096] or old_prefix.encode() not in raw:
                continue
            data = raw.replace(old_prefix.encode(), new_prefix.encode())
            mode = path.stat().st_mode
            if path.stat().st_nlink > 1:  # break hardlinks created by deduplication
                tmp = path.with_name(path.name + ".relocate-tmp")
                tmp.write_bytes(data)
                os.chmod(tmp, mode)
                os.replace(tmp, path)
            else:
                path.write_bytes(data)
            stats["text_files"] += 1
    return stats


def check_interpreters(data_root: Path, limit: int = 3) -> list[str]:
    problems = []
    for cfg in sorted((data_root / "source_tree").rglob("pyvenv.cfg")):
        python = next((p for p in (cfg.parent / "bin").glob("python3*") if p.name.count(".") <= 1), None)
        if python is None:
            continue
        result = subprocess.run([str(python), "-c", "import sys; print(sys.version)"], capture_output=True, text=True)
        if result.returncode:
            problems.append(f"{cfg.parent}: {result.stderr.strip()[:200]}")
        limit -= 1
        if limit <= 0:
            break
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=None)
    a = ap.parse_args(argv)
    data_root = (a.data_root or default_data_root()).resolve()
    if not data_root.is_dir():
        sys.exit(f"{data_root} does not exist; download the data first (scripts/download_data.sh).")
    extract_archives(data_root)
    marker = data_root / ".relocated.json"
    old_prefix = SOURCE_PREFIX
    if marker.is_file():
        old_prefix = json.loads(marker.read_text())["prefix"]
    new_prefix = str((data_root / "source_tree").resolve())
    stats = {"symlinks": 0, "text_files": 0} if old_prefix == new_prefix else relocate(data_root, old_prefix)
    marker.write_text(json.dumps({"root": str(data_root), "prefix": new_prefix, **stats}, indent=2) + "\n")
    problems = check_interpreters(data_root)
    print(f"relocated {stats['symlinks']} links and {stats['text_files']} files to {new_prefix}")
    for p in problems:
        print("WARNING interpreter check failed:", p)
    print("Semiconductor Agent data is ready." if not problems else
          "Some interpreters failed; see docs/semiconductor_agent.md#troubleshooting (or use the Docker image).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
