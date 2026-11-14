"""Collect benchmark scores from an ``outputs/`` tree into one table.

    python -m micius_eval.report outputs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def collect(root: Path) -> dict[str, dict[str, str]]:
    table: dict[str, dict[str, str]] = {}
    for summary in sorted(root.glob("*/*/summary.json")):
        bench, model = summary.parent.parent.name, summary.parent.name
        data = json.loads(summary.read_text())
        if bench == "semibench":
            m = data["overall"]
            cell = f"{m['accuracy']:.2f} ({m['correct']}/{m['total']})"
        elif bench == "semiconductor_agent":
            cell = f"{data['pass_rate']:.2f} ({data['pass']}/{data['tasks']})"
            if data.get("verification_error") or data.get("unverified"):
                cell += " *incomplete*"
        elif bench == "pcb_agent":
            cell = f"{data['score']:.2f} ({data['solved']}/{data['tasks']})"
            if data.get("reviewer_errors"):
                cell += " *incomplete*"
        else:
            continue
        table.setdefault(model, {})[bench] = cell
    return table


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")
    table = collect(root)
    cols = [("semibench", "SemiBench"), ("semiconductor_agent", "Semiconductor Agent"), ("pcb_agent", "PCB Agent")]
    print("| Model | " + " | ".join(name for _, name in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for model, row in table.items():
        print(f"| {model} | " + " | ".join(row.get(key, "-") for key, _ in cols) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
