# PCB Agent

PCB Agent evaluates end-to-end hardware design with KiCad: designing a
schematic from an empty project to meet public interface / power / mode
requirements, or repairing and completing existing schematics and boards
(placement, routing, DRC clean-up, manufacturing outputs).

| | |
|---|---|
| Tasks | 30 public tasks: design layers D1/D2/D3 = 8/5/2, repair layers R1/R2/R3 = 8/3/4 |
| Inputs | [`data/pcb_agent/tasks/<id>/public/`](../data/pcb_agent/tasks) – `prompt.txt`, requirements, interface and delivery contracts, initial KiCad project; all prompts in [`PROMPTS.md`](../data/pcb_agent/PROMPTS.md) |
| Agent | PCB engineering harness (Kimi Code CLI 1.50.0 + atomic KiCad/PCB tools) in a pinned container image, KiCad 10.0.6 |
| Budget | 500 agent steps, 262,144-token context, 131,072 output tokens per request (65,536 for compaction), no wall-clock limit |
| Sampling | `temperature=1.0, top_p=0.95, top_k=20`, `xhigh` thinking for agent and compaction |
| Grading | independent three-axis review by a Codex agent (`gpt-6.1-sol`, `high`) in an isolated KiCad 10.0.6 container |
| Metric | fraction of tasks with **answer**, **structure** and **completion** all PASS |

## Requirements

* Docker (the user must be able to run `docker`), ~10 GB for the images;
* a model endpoint reachable from containers. `127.0.0.1`/`localhost` URLs are
  rewritten to `host.docker.internal`. SGLang is recommended: the harness calls
  `/v1/tokenize` to size each request exactly to the remaining context;
* for review: Codex CLI (`npm i -g @openai/codex`, the static
  `x86_64-unknown-linux-musl` binary from the package is mounted into the
  reviewer container) and an API key for the verifier model.

Images (override with `MICIUS_PCB_IMAGE` / `MICIUS_PCB_VERIFIER_IMAGE`):

```bash
docker pull ghcr.io/zhiman-bj/pcb-agent-env:v34
docker pull ghcr.io/zhiman-bj/pcb-agent-verifier:kicad-10.0.6
```

## Run

```bash
python -m micius_eval.pcb_agent.run \
    --base-url http://<host>:30000/v1 --model Micius-Flash \
    --output-dir outputs/pcb_agent/Micius-Flash --concurrency 10

export OPENAI_API_KEY=...
python -m micius_eval.pcb_agent.verify --output-dir outputs/pcb_agent/Micius-Flash --workers 4
```

or simply `bash scripts/run_pcb_agent.sh`.

Generation: every task gets a fresh container (read-only root, 3 CPUs, 4 GB),
public files mounted read-only at `/task`, initial files copied into a writable
`/workspace`. A small runtime overlay ([`adapter/`](../micius_eval/pcb_agent/adapter))
switches the harness to native streaming Chat Completions so the model's
`reasoning_content` and full tool-call history are preserved. Per task:
`tasks/<id>/workspace/` (final project + `session/` trajectory) and
`completion.json`.

Review: the final workspace (without the harness session) is frozen; the
reviewer sees only the public task, the frozen workspace and terminal metadata
— never the generator's thinking or a hidden reference design. It must back
every PASS with recorded native checks (ERC/DRC, parity, unrouted, …) or
independent functional evidence bound to file hashes; the host re-validates
these bindings (`classify` in [`verify.py`](../micius_eval/pcb_agent/verify.py)).
Generations that did not finish count as unsolved; reviewer errors are reported
separately and are retried on the next invocation.
