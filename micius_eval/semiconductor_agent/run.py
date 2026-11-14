"""Stage 1 of Semiconductor Agent: let the model solve every task.

Each task gets a fresh copy of its initial state (workspace files + private
record database). Business tools run in the environment's pinned software
profile through the original Agent-World harness; the model also has
read-only ``Read`` / ``Grep`` / ``Glob`` on its workspace.

Usage::

    python -m micius_eval.semiconductor_agent.run \
        --base-url http://127.0.0.1:30000/v1 --model Micius-Flash \
        --output-dir outputs/semiconductor_agent/micius-flash --workers 32
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from ..common import ChatClient, reasoning_body, write_json
from .agent import AgentConfig, run_agent
from .data import AgentCase, check_ready, default_data_root, enable_vendor, load_cases
from .file_tools import VIRTUAL_ROOT, WorkspaceFiles

enable_vendor()
from harness.layout import create_task_run_layout  # noqa: E402
from harness.mcp_protocol import RpcError  # noqa: E402
from task_gen.task_eval_mcp import TaskEvalMcpServer  # noqa: E402


def _swap(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_swap(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: _swap(v, old, new) for k, v in value.items()}
    return value


def server_config(case: AgentCase, layout, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "workspace_root": str(layout.model_workspace),
        "public_workspace_root": str(layout.model_workspace),
        "record_database": str(layout.record_database),
        "scope_mounts": layout.scope_mounts,
        "trace": str(layout.raw / "environment_tool_calls.jsonl"),
        # The agent loop enforces the business-tool budget itself.
        "max_tool_calls": 10_000,
        "timeout": args.tool_timeout,
        "memory_limit": 128 * 1024 ** 3,
        "write_limit": 256 * 1024 ** 2,
        "process_limit": 1024,
        "tools": case.environment["tools"],
        "environment": case.environment,
        **({"binding_path": case.runtime["binding_path"]} if case.runtime.get("binding_path") else {}),
        **({"tools_path": case.runtime["tools_path"]} if case.runtime.get("tools_path") else {}),
    }


def solve(case: AgentCase, args: argparse.Namespace, client: ChatClient) -> dict[str, Any]:
    task_dir = args.output_dir / "tasks" / case.case_name
    started = time.monotonic()
    if (task_dir / "result.json").is_file():
        previous = json.loads((task_dir / "result.json").read_text())
        if previous.get("status") in ("completed", "tool_budget_exceeded", "max_turns"):
            return {**previous, "resumed": True}
    if task_dir.exists():
        shutil.rmtree(task_dir)
    try:
        layout = create_task_run_layout(case.initial_state, task_dir, environment=case.environment,
                                        workspace_name=f"{case.environment['environment_id']}_{case.task['task_id']}")
        server = TaskEvalMcpServer(server_config(case, layout, args))
        public = server.handle({"method": "tools/list"})["tools"]
        real = str(layout.model_workspace)

        def call_env_tool(name: str, arguments: dict[str, Any]):
            arguments = _swap(arguments, VIRTUAL_ROOT + "/", real + "/")
            try:
                response = server.handle({"method": "tools/call", "params": {"name": name, "arguments": arguments}})
            except RpcError as exc:
                return {"error": str(exc)}, True
            payload = response.get("structuredContent")
            if payload is None:
                text = "".join(c.get("text", "") for c in response.get("content", []))
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = text
            return _swap(payload, real, VIRTUAL_ROOT), bool(response.get("isError"))

        events = (task_dir / "agent_events.jsonl").open("a", encoding="utf-8")
        outcome = run_agent(
            client=client,
            task_text=case.task["task_text"],
            files=WorkspaceFiles(layout.model_workspace),
            env_tools=public,
            call_env_tool=call_env_tool,
            config=AgentConfig(max_tool_calls=args.max_tool_calls, max_context=args.max_context,
                               max_output=args.max_output, compaction_ratio=args.compaction_ratio,
                               temperature=args.temperature, top_p=args.top_p),
            log=lambda e: (events.write(json.dumps(e, ensure_ascii=False) + "\n"), events.flush()),
        )
        events.close()
        trajectory = {
            "case": case.case_name, "case_id": case.case_id, "task_id": case.task["task_id"],
            "environment_id": case.environment["environment_id"], "model": args.model,
            "messages": outcome.messages, "steps": outcome.steps,
            "outcome": {"status": outcome.status, "final_answer": outcome.final_answer, "error": outcome.error},
        }
        write_json(task_dir / "trajectory.json", trajectory)
        result = {
            "case": case.case_name, "case_id": case.case_id, "status": outcome.status,
            "error": outcome.error, "environment_calls": outcome.environment_calls,
            "native_calls": outcome.native_calls, "compactions": outcome.compactions,
            "turns": len(outcome.steps), "usage": outcome.usage,
            "tool_budget_stop": any(c["tool_kind"] == "environment_mcp" and c["is_error"]
                                    and "limit reached" in str(c["output"])
                                    for s in outcome.steps for c in s["tool_calls"]),
            "seconds": round(time.monotonic() - started, 1),
        }
    except Exception as exc:  # environment setup errors are reported, not fatal
        result = {"case": case.case_name, "case_id": case.case_id, "status": "error",
                  "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                  "seconds": round(time.monotonic() - started, 1)}
    write_json(task_dir / "result.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Semiconductor Agent -- generation stage.")
    p.add_argument("--base-url", nargs="+", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=None, help="Defaults to $MICIUS_DATA/semiconductor_agent.")
    p.add_argument("--tasks", nargs="*", default=None, help="Subset of case ids / task ids / environment ids.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=16, help="Tasks solved concurrently.")
    p.add_argument("--reasoning-effort", default="xhigh")
    p.add_argument("--provider", choices=["local", "openai", "none"], default="local")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--max-tool-calls", type=int, default=85)
    p.add_argument("--max-context", type=int, default=262_144)
    p.add_argument("--max-output", type=int, default=131_072)
    p.add_argument("--compaction-ratio", type=float, default=0.5)
    p.add_argument("--tool-timeout", type=int, default=300)
    p.add_argument("--request-timeout", type=float, default=3600)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = (args.data_root or default_data_root()).resolve()
    check_ready(data_root)
    cases = load_cases(data_root, set(args.tasks) if args.tasks else None)
    if args.limit:
        cases = cases[: args.limit]
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "config.json", {
        "benchmark": "Semiconductor Agent", "model": args.model, "base_urls": args.base_url,
        "data_root": str(data_root), "tasks": len(cases), "reasoning_effort": args.reasoning_effort,
        "max_tool_calls": args.max_tool_calls, "max_context": args.max_context, "max_output": args.max_output,
        "compaction_ratio": args.compaction_ratio, "temperature": args.temperature, "top_p": args.top_p,
    })
    client = ChatClient(args.base_url, args.model, api_key=args.api_key, timeout=args.request_timeout,
                        attempts=4, extra_body=reasoning_body(args.reasoning_effort, provider=args.provider))
    print(f"[Semiconductor Agent] solving {len(cases)} tasks with {args.workers} workers", flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(solve, c, args, client): c for c in cases}
        for k, future in enumerate(concurrent.futures.as_completed(futures), 1):
            r = future.result()
            results.append(r)
            print(f"[{k}/{len(cases)}] {r['case']}: {r['status']} env_calls={r.get('environment_calls')} "
                  f"({r.get('seconds')}s){' ERROR ' + r['error'][:200] if r.get('error') else ''}", flush=True)
    write_json(args.output_dir / "generation_summary.json", {
        "tasks": len(results),
        "status_counts": {s: sum(r["status"] == s for r in results) for s in sorted({r["status"] for r in results})},
        "results": sorted(results, key=lambda r: r["case"]),
    })
    print(f"Done. Next: python -m micius_eval.semiconductor_agent.verify --output-dir {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
