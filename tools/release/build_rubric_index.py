#!/usr/bin/env python3
"""[Maintainers] Index the frozen rubric packages available for every evaluation task.

    python tools/release/build_rubric_index.py --input-root <taskgen_inputs_eval200/python_profile> \
        --verifier-batches <.../verifier_batches> --output rubric_index.json
"""
import argparse
import json
from pathlib import Path

PATTERNS = ["*/verifiers/*/*/direct/generation/verifier/rubrics.json", "*/verifier/rubrics.json",
            "verifiers/*/*/direct/generation/verifier/rubrics.json", "*/*/verifier/rubrics.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--verifier-batches", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    need = [r["task_id"] for r in json.loads((a.input_root / "validation_id_mapping.json").read_text())]
    index: dict[str, list[str]] = {}
    for batch in sorted(p for p in a.verifier_batches.iterdir() if p.is_dir()):
        for pattern in PATTERNS:
            for rubric in batch.glob(pattern):
                try:
                    provenance = json.loads((rubric.parent.parent / "generation/result.json").read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if provenance.get("status") == "generated" and isinstance(provenance.get("task_id"), str):
                    index.setdefault(provenance["task_id"], []).append(str(rubric.parent))
    found = {t: index[t] for t in need if t in index}
    a.output.write_text(json.dumps(found, indent=1) + "\n")
    print(f"{len(found)}/{len(need)} tasks have a frozen rubric; missing: {[t for t in need if t not in index]}")


if __name__ == "__main__":
    main()
