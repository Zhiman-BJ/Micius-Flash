# Semiconductor Agent

Semiconductor Agent measures whether a model can finish realistic engineering
work by calling domain tools in an executable environment: reconstructing
provenance in an AiiDA archive, checking timing sign-off reports, reducing
EELS / 4D-STEM / C–V measurement data, auditing fab equipment logs, running
RTL simulations, and so on.

| | |
|---|---|
| Tasks | 197 held-out tasks, one per environment |
| Environment | record database (SQLite) + workspace files + task-specific tools (median 26, up to 88), each tool running in a pinned Python software profile (real scientific packages such as `aiida-core`, `ase`, `pymatgen`, `kwant`, `amaranth`, `hyperspy`, ...) |
| Agent interface | business tools (function calling) + read-only `Read` / `Grep` / `Glob` on `/workspace` |
| Budget | 85 business tool calls, 262,144-token context (compaction at 50%), 131,072 output tokens per turn |
| Grading | frozen per-task rubric applied by an independent Codex verifier (`gpt-6.1-sol`, `high`) that inspects the final answer, the tool-call log and the actual final environment state |
| Metric | pass rate over all 197 tasks (fixed denominator) |

## 1. Get the environments

The environments, tool runtimes and rubrics are distributed through Hugging Face
(too large for git):

```bash
export MICIUS_DATA=/path/with/enough/space          # default: ./data
bash scripts/download_data.sh semiconductor_agent
```

This downloads the archives into `$MICIUS_DATA/semiconductor_agent`, unpacks them
and runs `python -m micius_eval.semiconductor_agent.setup_data`, which rewrites
the absolute paths inside the pre-built software profiles (uv virtual
environments) to your location. After that the directory looks like:

```
semiconductor_agent/
├── tasks.jsonl                 # 197 rows: case_id, environment_id, task_id, package_id, software_profile
├── environments/<env_id>/      # tasks.json, intermediate/step_5_bundle.json, initial/ (records.sqlite + files)
├── source_tree/                # tool delivery packages, software profiles, interpreters
└── rubrics/<task_id>/verifier/ # frozen rubric (rubrics.json, instructions.md, tools.json, coverage.json)
```

Requirements: Linux x86-64, Python ≥ 3.11 for the runner. Tools execute in their
own bundled interpreters, so the host Python version does not matter. If a
profile fails the interpreter check (old glibc), use the Docker image instead:

```bash
docker build -f docker/eval.Dockerfile -t micius-eval .
docker run --rm -it --network host -v $MICIUS_DATA:/data -e MICIUS_DATA=/data -e OPENAI_API_KEY micius-eval \
    bash scripts/run_semiconductor_agent.sh
```

## 2. Generation

```bash
python -m micius_eval.semiconductor_agent.run \
    --base-url http://127.0.0.1:30000/v1 --model Micius-Flash \
    --output-dir outputs/semiconductor_agent/Micius-Flash --workers 16
```

For every task the runner copies the initial state into a fresh task directory,
starts the task-scoped tool server in-process and runs a native function-calling
loop ([`agent.py`](../micius_eval/semiconductor_agent/agent.py)). The context
matches the one used when the evaluation trajectories were collected: the
Kimi Code CLI system prompt with a two-level listing of `/workspace` and the
runtime tool contract, the task as the first user message, then the date and
auto-permission reminders. Reasoning is kept in the history
(`reasoning_content`).

Per task the output directory holds `trajectory.json` (all messages and tool
calls), `raw/environment_tool_calls.jsonl` (exact business-tool records),
`private-state/records.sqlite` and the model workspace (final state), and
`result.json`.

| option | default | |
|---|---|---|
| `--workers` | 16 | tasks in parallel; tool calls run in subprocesses |
| `--tasks ID ...` | all | case ids, task ids or environment ids |
| `--max-tool-calls` | 85 | business tool calls; file tools are not counted |
| `--max-context` / `--max-output` | 262144 / 131072 | |
| `--compaction-ratio` | 0.5 | summarize history when the prompt exceeds this fraction of the context |
| `--reasoning-effort` / `--provider` | `xhigh` / `local` | as for SemiBench |
| `--temperature` / `--top-p` | server default | Micius-Flash ships `temperature=1.0, top_p=0.95, top_k=20` in `generation_config.json` |

## 3. Verification

```bash
npm install -g @openai/codex                          # Codex CLI
export OPENAI_API_KEY=...                             # key of the verifier endpoint
python -m micius_eval.semiconductor_agent.verify \
    --output-dir outputs/semiconductor_agent/Micius-Flash \
    --verifier-base-url https://api.openai.com/v1 --workers 8
```

Each rubric package was generated once from the public task, blind-reviewed and
frozen (hashes in `frozen_manifest.json`) before any model was graded; all
models are graded with the same packages. The verifier reconstructs the final
state (initial state + model workspace + private record database), investigates
it read-only and returns `pass`/`fail` with per-rubric scores
(`outputs/.../verification/runs/<case>/verdict.json`).

Scoring rules (as in the paper):

* tasks whose generation errored or that exceeded the tool budget count as **fail**;
* `verification_error` is an infrastructure failure, not a model failure — rerun with
  `--retry-errors` until none remain before reporting;
* the score is `pass / 197`.

Instead of `--verifier-base-url` you can pass an existing Codex home
(`--codex-home ~/.codex`). `--verifier-model` / `--verifier-reasoning-effort`
default to `gpt-6.1-sol` / `high`; other verifiers are allowed but scores are
then not comparable to the paper.

## Troubleshooting

* **`Tool runtimes ... are not relocated`** – run `python -m micius_eval.semiconductor_agent.setup_data --data-root $MICIUS_DATA/semiconductor_agent` (again after moving the directory).
* **`Could not find platform independent libraries`** – an interpreter archive is incomplete; re-download, or use the Docker image.
* **Requests fail with a proxy error** – requests to the model server bypass `HTTP(S)_PROXY` by default; set `MICIUS_USE_PROXY=1` if your endpoint must go through a proxy.
* **Verifier reports `codex-linux-sandbox: No such file or directory`** – the output directory is under `/tmp`; Codex refuses to create its sandbox helpers there. Use an output directory elsewhere.
