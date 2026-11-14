"""Dataset location, vendored runtime import path and task loading."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VENDOR = Path(__file__).resolve().parent / "_vendor" / "agent_world"
# Absolute prefix under which the tool runtimes were built. ``setup_data``
# rewrites it to the local dataset location; bundles are mapped at load time.
SOURCE_PREFIX = "/data/agent-world/toolgen-20261010/source_tree"


def enable_vendor() -> None:
    """Make the vendored Agent-World harness importable (harness, task_gen, ...)."""
    path = str(VENDOR)
    if path not in sys.path:
        sys.path.insert(0, path)
    # Tool workers are spawned as subprocesses that import ``harness`` too.
    current = os.environ.get("PYTHONPATH", "")
    if path not in current.split(os.pathsep):
        os.environ["PYTHONPATH"] = path + (os.pathsep + current if current else "")


def default_data_root() -> Path:
    root = os.environ.get("MICIUS_DATA")
    return Path(root).expanduser() / "semiconductor_agent" if root else Path("data/semiconductor_agent").resolve()


@dataclass(frozen=True)
class AgentCase:
    case_id: str
    case_name: str
    task: dict[str, Any]
    environment: dict[str, Any]
    initial_state: Path
    runtime: dict[str, Any]
    rubric_dir: Path | None
    source_run: Path
    reference_state: Path | None = None
    reference_calls: list[dict[str, Any]] = field(default_factory=list)


def _relocate(value: str, data_root: Path) -> str:
    if value.startswith(SOURCE_PREFIX):
        return str(data_root / "source_tree") + value[len(SOURCE_PREFIX):]
    return value


def check_ready(data_root: Path) -> None:
    if not (data_root / "tasks.jsonl").is_file():
        raise SystemExit(
            f"Semiconductor Agent data not found under {data_root}.\n"
            "Run `bash scripts/download_data.sh semiconductor_agent` (or set MICIUS_DATA)."
        )
    marker = data_root / ".relocated.json"
    if not marker.is_file() or json.loads(marker.read_text()).get("root") != str(data_root.resolve()):
        raise SystemExit(
            f"Tool runtimes under {data_root} are not relocated to this path yet.\n"
            f"Run `python -m micius_eval.semiconductor_agent.setup_data --data-root {data_root}`."
        )


def load_cases(data_root: Path, selected: set[str] | None = None) -> list[AgentCase]:
    data_root = data_root.resolve()
    cases: list[AgentCase] = []
    for line in (data_root / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if selected and row["case_id"] not in selected and row["task_id"] not in selected \
                and row["environment_id"] not in selected:
            continue
        source_run = data_root / "environments" / row["environment_id"]
        bundle = json.loads((source_run / "intermediate/step_5_bundle.json").read_text(encoding="utf-8"))
        tasks = json.loads((source_run / "tasks.json").read_text(encoding="utf-8"))
        task = next(t for t in tasks if t["task_id"] == row["task_id"])
        runtime = {k: _relocate(v, data_root) if isinstance(v, str) else v
                   for k, v in (bundle.get("runtime") or {}).items()}
        rubric = data_root / "rubrics" / row["task_id"] / "verifier"
        execution = next((t.get("execution", {}) for t in bundle.get("tasks", [])
                          if isinstance(t, dict) and t.get("task_id") == row["task_id"]), {})
        reference_calls = execution.get("tool_calls") if isinstance(execution, dict) else None
        cases.append(AgentCase(
            case_id=row["case_id"],
            case_name=f"{row['environment_id']}__{row['task_id']}",
            task=task,
            environment=bundle["environment"],
            initial_state=(source_run / task["initial_state"]).resolve(),
            runtime=runtime,
            rubric_dir=rubric if rubric.is_dir() else None,
            source_run=source_run,
            reference_calls=reference_calls if isinstance(reference_calls, list)
            else task.get("reference", {}).get("tool_calls", []),
        ))
    return cases
