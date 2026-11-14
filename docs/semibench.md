# SemiBench

SemiBench evaluates semiconductor-domain knowledge and reasoning with hard,
scenario-style multiple-choice questions written by domain engineers (device
physics, process, analog/digital/RF circuits, FPGA and hardware systems,
packaging, metrology, ...).

| | |
|---|---|
| Questions | 167 (5 single-choice, 162 multiple-choice; up to 10 options A–J) |
| Selection | from a larger engineer-written pool, a question is kept only if at least 3 of 5 strong reference models answer it wrongly |
| Language | Chinese |
| Metric | accuracy with exact match of the answer **set** (no partial credit) |
| File | [`data/semibench/semibench.jsonl`](../data/semibench/semibench.jsonl) |

Each line:

```json
{"id": "v2_039abbabd23a192b4ecf4c05", "question_type": "multiple_choice",
 "question": "...", "options": {"A": "...", "...": "...", "J": "..."}, "answer": ["A", "D", "E", "F"]}
```

## Protocol

* one user turn per question (no system prompt), built by `build_prompt` in
  [`micius_eval/semibench/run.py`](../micius_eval/semibench/run.py); it asks for a
  final line `答案：<letters>`;
* thinking at the highest effort (`xhigh` for Micius-Flash), `temperature=0`,
  `top_p=1`, `max_tokens=32768` (reasoning included);
* the **last non-empty line** of the final answer is parsed; responses that are
  truncated (`finish_reason=length`), unparsable, contain duplicate letters, or
  give several letters for a single-choice question count as wrong;
* transport errors are retried (3 attempts) and are never scored.

## Run

```bash
python -m micius_eval.semibench.run \
    --base-url http://127.0.0.1:30000/v1 --model Micius-Flash \
    --output-dir outputs/semibench/Micius-Flash
```

Useful options:

| option | default | |
|---|---|---|
| `--base-url URL [URL ...]` | – | several replicas are load-balanced |
| `--reasoning-effort` | `xhigh` | sent as `chat_template_kwargs` (`--provider local`) |
| `--provider {local,openai,none}` | `local` | `openai` sends a top-level `reasoning_effort` (hosted APIs) |
| `--max-tokens` | 32768 | keep the default to compare with the paper |
| `--workers` | 32 | concurrent requests |
| `--limit N` | – | smoke test on the first N questions |

Outputs in `--output-dir`: `predictions.jsonl` (full responses incl. reasoning),
`summary.json`, `config.json`. Interrupted runs resume from `predictions.jsonl`.
