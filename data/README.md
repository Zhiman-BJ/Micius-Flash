# Benchmark data

| Directory | Benchmark | Contents |
|---|---|---|
| `semibench/semibench.jsonl` | SemiBench | 167 questions with options and gold answer sets |
| `pcb_agent/` | PCB Agent | `manifest.json` (30 tasks, per-file SHA-256), `PROMPTS.md` (all prompts), `tasks/<id>/public/` (prompt, requirements, contracts, initial KiCad project) |
| `semiconductor_agent/` | Semiconductor Agent | not in git — created by `bash scripts/download_data.sh semiconductor_agent` |

The PCB Agent public files are frozen: `python -m micius_eval.pcb_agent.run` refuses to
start if any file differs from `manifest.json`. Some initial KiCad projects come from
open-source hardware repositories and keep their original metadata.

These benchmarks are for evaluation only; please do not train on them.
