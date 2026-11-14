"""Read-only native file tools (Read / Grep / Glob) confined to the task workspace.

The model sees the workspace as ``/workspace``. Results follow the shape the
agent saw during data collection: a JSON object with ``output`` and a
``note`` status block. Anything outside the workspace, symlinks that escape it
and well-known secret files are refused.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from pathlib import Path
from typing import Any

VIRTUAL_ROOT = "/workspace"
READ_DEFAULT_MAX_CHARS = 80_000
READ_MAX_CHARS = 128_000
_SECRET = re.compile(r"(^|/)(\.env(\..*)?|id_rsa|id_ed25519|.*\.pem|credentials(\.json)?)$")
_SKIP_DIRS = {".git", ".hg", ".svn"}


class ToolInputError(ValueError):
    pass


def _brace_expand(pattern: str) -> list[str]:
    match = re.search(r"\{([^{}]*)\}", pattern)
    if not match:
        return [pattern]
    head, tail = pattern[: match.start()], pattern[match.end():]
    return [p for option in match.group(1).split(",") for p in _brace_expand(head + option + tail)]


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """Translate a ripgrep-style glob (``**`` aware) to a regex on posix paths."""
    out, i = "", 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif c == "*":
            out += "[^/]*"
            i += 1
        elif c == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(c)
            i += 1
    return re.compile(out + r"\Z")


class WorkspaceFiles:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    # -- path handling -------------------------------------------------
    def resolve(self, value: str | None) -> Path:
        if not value:
            return self.root
        value = str(value)
        if value == VIRTUAL_ROOT or value.startswith(VIRTUAL_ROOT + "/"):
            value = value[len(VIRTUAL_ROOT):].lstrip("/")
        elif value.startswith("/"):
            raise ToolInputError(f"Path is outside the working directory: {value}")
        path = (self.root / value).resolve()
        if path != self.root and not path.is_relative_to(self.root):
            raise ToolInputError(f"Path is outside the working directory: {value}")
        if _SECRET.search(path.as_posix()):
            raise ToolInputError(f"Refusing to access a sensitive file: {value}")
        return path

    def display(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() if path != self.root else "."

    def _walk(self, base: Path):
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            for name in sorted(filenames):
                path = Path(dirpath) / name
                if path.is_symlink() and not path.resolve().is_relative_to(self.root):
                    continue
                if _SECRET.search(path.as_posix()):
                    continue
                yield path

    # -- directory listing for the system prompt ----------------------
    def listing(self, depth: int = 2, limit: int = 60) -> str:
        lines: list[str] = []

        def entries(directory: Path):
            items = [p for p in directory.iterdir() if p.name not in _SKIP_DIRS]
            return sorted(items, key=lambda p: (not p.is_dir(), p.name))

        def visit(directory: Path, prefix: str, level: int):
            items = entries(directory)
            shown = items[:limit]
            for k, item in enumerate(shown):
                last = k == len(shown) - 1 and len(items) <= limit
                branch = "└── " if last else "├── "
                lines.append(prefix + branch + item.name + ("/" if item.is_dir() else ""))
                if item.is_dir() and level < depth and not item.name.startswith("."):
                    visit(item, prefix + ("    " if last else "│   "), level + 1)
            if len(items) > limit:
                lines.append(prefix + f"└── ... ({len(items) - limit} more)")

        visit(self.root, "", 1)
        return "\n".join(lines) + "\n"

    # -- tools ---------------------------------------------------------
    def read(self, path: str, line_offset: int = 1, n_lines: int | None = None,
             max_chars: int | None = None, column_offset: int = 0, **_: Any) -> dict[str, Any]:
        target = self.resolve(path)
        if not target.is_file():
            raise ToolInputError(f"File not found: {path}" if not target.exists() else f"Not a file: {path}")
        budget = min(int(max_chars or READ_DEFAULT_MAX_CHARS), READ_MAX_CHARS)
        raw = target.read_bytes()
        if b"\x00" in raw[:8192]:
            raise ToolInputError(f"Binary file cannot be read as text: {path}")
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)
        start = max(int(line_offset or 1), 1)
        end = total if n_lines is None else min(total, start - 1 + max(int(n_lines), 0))
        out: list[str] = []
        used = 0
        next_args = None
        number = start
        for number in range(start, end + 1):
            content = lines[number - 1]
            if number == start and column_offset:
                content = content[int(column_offset):]
            row = f"{number}\t{content}"
            if used + len(row) + 1 > budget:
                if not out:  # a single over-long line: return a fragment
                    room = max(budget - len(str(number)) - 2, 1)
                    out.append(f"{number}\t{content[:room]}")
                    next_args = {"path": path, "line_offset": number,
                                 "column_offset": int(column_offset or 0) + room}
                else:
                    next_args = {"path": path, "line_offset": number}
                break
            out.append(row)
            used += len(row) + 1
        read_count = len(out)
        if next_args:
            status = (f"{read_count} lines read from file starting from line {start}. Total lines in file: {total}. "
                      f"Result incomplete (effective max_chars: {budget}). Next Read: {json.dumps(next_args)}")
        else:
            tail = "End of file reached." if end >= total else "Requested range complete."
            status = (f"{read_count} lines read from file starting from line {start}. Total lines in file: {total}. "
                      f"Requested range complete. Effective max_chars: {budget}. {tail}")
        return {"output": "\n".join(out), "note": f"<system>{status}</system>"}

    def glob(self, pattern: str, path: str | None = None, head_limit: int = 100, offset: int = 0,
             include_dirs: bool = False, **_: Any) -> dict[str, Any]:
        base = self.resolve(path)
        if not base.is_dir():
            raise ToolInputError(f"Not a directory: {path}")
        patterns = [p if "/" in p else "**/" + p for p in _brace_expand(pattern)]
        regexes = [_glob_regex(p) for p in patterns]
        matches = [p for p in self._walk(base) if any(r.match(p.relative_to(base).as_posix()) for r in regexes)]
        matches.sort(key=lambda p: (-p.stat().st_mtime, p.as_posix()))
        page = matches[offset:] if not head_limit else matches[offset: offset + head_limit]
        output = "\n".join(self.display(p) for p in page)
        more = offset + len(page) < len(matches)
        note = f"Found {len(matches)} files" + (f"; showing {len(page)} from offset {offset}. Next offset: {offset + len(page)}" if more else "")
        return {"output": output or "No files found", "note": f"<system>{note}.</system>"}

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None, output_mode: str = "files_with_matches",
             head_limit: int = 100, offset: int = 0, multiline: bool = False, **kw: Any) -> dict[str, Any]:
        flags = re.IGNORECASE if kw.get("-i") else 0
        if multiline:
            flags |= re.DOTALL | re.MULTILINE
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            raise ToolInputError(f"Invalid regex: {exc}") from exc
        base = self.resolve(path)
        files = [base] if base.is_file() else list(self._walk(base))
        if glob:
            globs = [_glob_regex(g if g.startswith("**/") or "/" in g else "**/" + g) for g in _brace_expand(glob)]
            files = [f for f in files if any(r.match(f.as_posix().lstrip("/")) for r in globs)]
        before = int(kw.get("-B") or kw.get("-C") or 0)
        after = int(kw.get("-A") or kw.get("-C") or 0)
        number = kw.get("-n", True)
        results: list[str] = []
        counts: list[tuple[str, int]] = []
        for f in files:
            try:
                raw = f.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:8192]:
                continue
            text = raw.decode("utf-8", errors="replace")
            if multiline:
                hits = list(regex.finditer(text))
                if not hits:
                    continue
                counts.append((self.display(f), len(hits)))
                if output_mode == "content":
                    for h in hits:
                        line_no = text.count("\n", 0, h.start()) + 1
                        results.append(f"{self.display(f)}:{line_no}:{h.group(0)}")
                continue
            lines = text.splitlines()
            hit_lines = [i for i, line in enumerate(lines) if regex.search(line)]
            if not hit_lines:
                continue
            counts.append((self.display(f), len(hit_lines)))
            if output_mode == "content":
                shown: set[int] = set()
                for i in hit_lines:
                    for j in range(max(0, i - before), min(len(lines), i + after + 1)):
                        if j in shown:
                            continue
                        shown.add(j)
                        sep = ":" if j == i else "-"
                        prefix = f"{self.display(f)}{sep}{j + 1}{sep}" if number else f"{self.display(f)}{sep}"
                        results.append(prefix + lines[j])
        if output_mode == "count":
            results = [f"{name}:{n}" for name, n in counts]
        elif output_mode != "content":
            results = [name for name, _ in counts]
        page = results[offset:] if not head_limit else results[offset: offset + head_limit]
        more = offset + len(page) < len(results)
        note = f"{len(results)} result lines" + (f"; showing {len(page)} from offset {offset}. Next offset: {offset + len(page)}" if more else "")
        return {"output": "\n".join(page) or "No matches found", "note": f"<system>{note}.</system>"}

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Run one native tool. Returns ``(payload, is_error)``."""
        handler = {"Read": self.read, "Glob": self.glob, "Grep": self.grep}.get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}, True
        try:
            return handler(**arguments), False
        except (ToolInputError, TypeError, OSError, ValueError) as exc:
            return {"error": str(exc)}, True
