#!/usr/bin/env python3
"""[Maintainers] Assemble the release directory for Semiconductor Agent.

Collects, for every evaluation task: its environment (task file, bundle,
initial state), its ToolGen delivery package, the pinned software profile and
interpreter it runs in, and its frozen grading rubric. Paths under the
original build prefix are mirrored into ``<output>/source_tree`` so that
``micius_eval.semiconductor_agent.setup_data`` can relocate them on the
user's machine.

    python tools/release/build_semiconductor_agent_data.py \
        --input-root   <taskgen_inputs_eval200/python_profile> \
        --source-tree  <real directory behind /data/agent-world/.../source_tree> \
        --rubric-index rubric_index.json \
        --output       release/semiconductor_agent  [--envs a b c] [--dedupe]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PREFIX = "/data/agent-world/toolgen-20261010/source_tree"
# Earlier batches first: same order used to build the original frozen maps.
RUBRIC_PREFERENCE = [
    "qwen_comparison_20261026_rubric_v3", "qwen_comparison_20261026_rubric_v3_key2",
    "qwen_comparison_20261026_rubric_v4_fixed", "qwen_comparison_20261027_split",
    "qwen38_27b_mix_20261027_rubric_fixed", "shared_frozen_rubrics_20261027",
]
SKIP_NAMES = {"uv-cache", ".uv-cache", "conda-cache", "runtime-cache", "mpl-cache", "__pycache__"}
warnings: list[str] = []
_locks: dict[str, threading.RLock] = {}
_master = threading.Lock()


def lock_for(path: Path) -> threading.RLock:
    with _master:
        return _locks.setdefault(str(path), threading.RLock())


def real(path: str, source_tree: Path) -> Path:
    if not path.startswith(PREFIX):
        raise ValueError(f"path outside build prefix: {path}")
    return source_tree / path[len(PREFIX):].lstrip("/")


def mirror(path: str, out: Path) -> Path:
    return out / "source_tree" / path[len(PREFIX):].lstrip("/")


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    def ignore(directory, names):
        return {n for n in names if n in SKIP_NAMES}
    def copy_file(s, d, *, follow_symlinks=True):
        try:
            return shutil.copy2(s, d, follow_symlinks=follow_symlinks)
        except (PermissionError, OSError) as exc:
            warnings.append(f"skip {s}: {exc}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_name(dst.name + ".partial")  # never leave a half-copied tree behind
    if partial.exists():
        shutil.rmtree(partial)
    try:
        shutil.copytree(src, partial, symlinks=True, ignore=ignore, copy_function=copy_file,
                        ignore_dangling_symlinks=True, dirs_exist_ok=False)
    except shutil.Error as exc:
        warnings.extend(f"copy error: {e}" for e in exc.args[0][:20])
    os.rename(partial, dst)


def runtime_root(path: str) -> str:
    """Widen a path inside an interpreter / host runtime to that runtime's root."""
    parts = path[len(PREFIX):].strip("/").split("/")
    for k, part in enumerate(parts):
        if part == "interpreters" and k + 1 < len(parts):
            return PREFIX + "/" + "/".join(parts[: k + 2])
        if part == "host_runtime":
            return PREFIX + "/" + "/".join(parts[: k + 1])
    return path


def copy_prefixed(path: str, source_tree: Path, out: Path) -> None:
    """Copy a prefixed path (file or directory), recreating symlinked ancestors."""
    path = runtime_root(path)
    rel = path[len(PREFIX):].strip("/").split("/")
    for k in range(1, len(rel)):
        ancestor = PREFIX + "/" + "/".join(rel[:k])
        if real(ancestor, source_tree).is_symlink():
            copy_prefixed(ancestor, source_tree, out)
            target = os.readlink(real(ancestor, source_tree))
            resolved = target if os.path.isabs(target) else os.path.normpath(os.path.join(os.path.dirname(ancestor), target))
            if resolved.startswith(PREFIX):
                copy_prefixed(resolved + "/" + "/".join(rel[k:]), source_tree, out)
            return
    src, dst = real(path, source_tree), mirror(path, out)
    with lock_for(dst):
        _copy_prefixed_locked(path, src, dst, source_tree, out)


def _copy_prefixed_locked(path: str, src: Path, dst: Path, source_tree: Path, out: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if src.is_symlink():
        target = os.readlink(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, dst)
        absolute = target if os.path.isabs(target) else os.path.normpath(os.path.join(os.path.dirname(path), target))
        if absolute.startswith(PREFIX):
            copy_prefixed(absolute, source_tree, out)
        return
    if src.is_dir():
        copy_tree(src, dst)
    elif src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        warnings.append(f"missing {src}")


def external_links(root: Path) -> set[str]:
    """Absolute symlink targets and venv homes under the build prefix."""
    found: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            if p.is_symlink():
                target = os.readlink(p)
                if target.startswith(PREFIX):
                    found.add(target)
        if "pyvenv.cfg" in filenames:
            for line in (Path(dirpath) / "pyvenv.cfg").read_text(errors="replace").splitlines():
                key, _, value = line.partition("=")
                if key.strip() == "home" and value.strip().startswith(PREFIX):
                    home = value.strip()
                    # copy the whole interpreter / runtime root, not just bin/
                    found.add(str(Path(home).parent) if home.endswith("/bin") else home)
    return found


def dedupe(root: Path) -> None:
    by_key: dict[tuple[int, str], Path] = {}
    saved = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_symlink() or not p.is_file():
                continue
            st = p.stat()
            if st.st_size < 4096:
                continue
            h = hashlib.sha256()
            with p.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    h.update(chunk)
            key = (st.st_size, h.hexdigest())
            first = by_key.setdefault(key, p)
            if first != p and first.stat().st_ino != st.st_ino and (first.stat().st_mode == st.st_mode):
                tmp = p.with_name(p.name + ".dedupe-tmp")
                os.link(first, tmp)
                os.replace(tmp, p)
                saved += st.st_size
    print(f"dedupe: {saved / 2**30:.1f} GiB shared via hardlinks", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--source-tree", type=Path, required=True)
    ap.add_argument("--rubric-index", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--envs", nargs="*")
    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--jobs", type=int, default=8, help="environments copied in parallel")
    a = ap.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((a.input_root / "validation_id_mapping.json").read_text())
    rubrics = json.loads(a.rubric_index.read_text())
    selected = [row for row in mapping if not a.envs or row["environment_id"] in a.envs]
    done = [0]

    def build_one(row: dict) -> dict:
        env = row["environment_id"]
        src_env = a.input_root / "environments" / env
        dst_env = out / "environments" / env
        if not (dst_env / "initial").exists():
            (dst_env / "intermediate").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_env / "tasks.json", dst_env / "tasks.json")
            shutil.copy2(src_env / "intermediate/step_5_bundle.json", dst_env / "intermediate/step_5_bundle.json")
            copy_tree(src_env / "initial", dst_env / "initial")
        bundle = json.loads((src_env / "intermediate/step_5_bundle.json").read_text())
        binding_path = bundle["runtime"]["binding_path"]
        binding = json.loads(real(binding_path, a.source_tree).read_text())
        delivery_root = binding_path.split("/delivery/")[0] + "/delivery"
        package = f"{delivery_root}/{binding['package_path']}"
        for sub in ("binding.json", "tools", "environment", "runtime", "software"):
            copy_prefixed(f"{package}/{sub}", a.source_tree, out)
        profile = binding.get("software_profile_path")
        if profile:
            copy_prefixed(f"{delivery_root}/{profile}", a.source_tree, out)
            pending = external_links(mirror(f"{delivery_root}/{profile}", out))
            seen: set[str] = set()
            while pending - seen:
                item = (pending - seen).pop()
                seen.add(item)
                copy_prefixed(item, a.source_tree, out)
                target = mirror(item, out)
                if target.is_dir() and not target.is_symlink():
                    pending |= external_links(target)
        # frozen rubric
        candidates = rubrics.get(row["task_id"], [])
        rank = lambda p: next((i for i, b in enumerate(RUBRIC_PREFERENCE) if f"/{b}/" in p), len(RUBRIC_PREFERENCE))
        if candidates:
            verifier = Path(sorted(candidates, key=rank)[0])
            dst = out / "rubrics" / row["task_id"]
            if not dst.exists():
                shutil.copytree(verifier, dst / "verifier")
                # frozen hashes + generation provenance (task id / task sha) checked by the verifier
                shutil.copy2(verifier.parent / "frozen_manifest.json", dst / "frozen_manifest.json")
                (dst / "generation").mkdir()
                provenance = json.loads((verifier.parent / "generation/result.json").read_text())
                provenance["verifier_dir"] = "verifier"  # drop the internal absolute path
                (dst / "generation/result.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
        else:
            warnings.append(f"no rubric for {row['task_id']}")
        done[0] += 1
        print(f"[{done[0]}/{len(selected)}] {env}", flush=True)
        return {"case_id": row["id"], "environment_id": env, "task_id": row["task_id"],
                "package_id": binding["package_id"], "software_profile": binding.get("software_profile")}

    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        rows = list(pool.map(build_one, selected))  # keeps the original task order
    with (out / "tasks.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "manifest.json").write_text(json.dumps({
        "benchmark": "Semiconductor Agent", "tasks": len(rows), "source_prefix": PREFIX,
        "runtime_backend": "python_profile"}, indent=2) + "\n")
    if a.dedupe:
        dedupe(out / "source_tree")
    (out / "build_warnings.txt").write_text("\n".join(warnings) + "\n")
    print(f"{len(rows)} tasks, {len(warnings)} warnings -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
