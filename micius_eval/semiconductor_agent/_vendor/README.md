# Vendored Agent-World runtime

`agent_world/` is the subset of the Agent-World environment framework used to
build and evaluate the Semiconductor Agent benchmark, copied verbatim except for environment-overridable path defaults (only
import-reachable modules plus their schemas and prompt files). It provides:

* `harness/` – task workspace layout, isolated tool execution in the pinned
  software profile, record store, MCP wire contract;
* `task_gen/task_eval_mcp.py` – the task-scoped tool server the agent talks to;
* `eval/` and `task_gen/test_verifier/` – the frozen-rubric Codex verifier.

`micius_eval.semiconductor_agent` adds this directory to `sys.path`; the
packages keep their original top-level names (`harness`, `task_gen`, ...).
Do not edit these files when reproducing paper numbers.
