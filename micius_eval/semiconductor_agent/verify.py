"""Stage 2 of Semiconductor Agent: grade trajectories with the frozen rubrics.

Every task ships one rubric package that was designed and blind-reviewed once
from the public task and initial state, then frozen (hash-pinned) before any
model was graded. A Codex agent (``gpt-6.1-sol`` at ``high`` effort in the
paper) investigates the final answer, the tool-call log and the actual final
environment state in a read-only sandbox, and returns ``pass``/``fail`` plus
per-rubric scores. The benchmark score is the pass rate over all 197 tasks;
tasks that failed to run, exceeded the tool budget or could not be verified
count as failures.

Requirements: the Codex CLI (``npm i -g @openai/codex``) on PATH and either an
existing ``~/.codex`` login or an OpenAI-compatible endpoint + API key.

    python -m micius_eval.semiconductor_agent.verify \
        --output-dir outputs/semiconductor_agent/micius-flash --workers 16
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from ..common import write_json
from .data import AgentCase, check_ready, default_data_root, enable_vendor, load_cases

enable_vendor()
from eval.verifier import _copy_state, serialized_state_preparation, verify_execution  # noqa: E402
from task_gen.task_eval_mcp import bind_delivery  # noqa: E402


def codex_home(args: argparse.Namespace, out: Path) -> Path:
    if args.codex_home:
        return args.codex_home.expanduser().resolve()
    if not args.verifier_base_url:
        default = Path.home() / ".codex"
        if (default / "config.toml").is_file():
            return default
        sys.exit("No Codex configuration found: pass --codex-home, or --verifier-base-url "
                 "(and export the key named by --verifier-api-key-env).")
    home = out / ".codex-verifier"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text("\n".join([
        f'model = {json.dumps(args.verifier_model)}',
        'model_provider = "micius-verifier"',
        f'model_reasoning_effort = {json.dumps(args.verifier_reasoning_effort)}',
        '[model_providers."micius-verifier"]',
        'name = "micius-verifier"',
        f'base_url = {json.dumps(args.verifier_base_url)}',
        f'env_key = {json.dumps(args.verifier_api_key_env)}',
        'wire_api = "responses"',
        'request_max_retries = 3',
        'stream_max_retries = 3',
    ]) + "\n")
    return home


def reconstruct_actual_state(case: AgentCase, task_dir: Path, target: Path) -> None:
    """Initial state + the model's workspace + the private record database."""
    _copy_state(case.initial_state, target)
    path_map = json.loads((task_dir / "private-state/path-map.json").read_text())
    # resolve relative to the task directory so output directories can be moved
    workspace = task_dir / Path(path_map["workspace_root"]).name
    private_db = task_dir / "private-state/records.sqlite"
    db = target / "records.sqlite"
    if private_db.is_file():
        if db.exists():
            db.unlink()
        shutil.copy2(private_db, db)
    scopes = target / "filesystem_scopes"
    scopes.mkdir(exist_ok=True)
    for item in case.environment.get("filesystem_scopes", []):
        scope = str(item["scope_id"])
        destination = scopes / scope
        if destination.exists():
            shutil.rmtree(destination)
        mount = path_map["scope_mounts"][scope]
        _copy_state(workspace if mount == "." else workspace / mount, destination)


def verification_calls(trajectory: dict[str, Any], env_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls = list(env_records)
    for step in trajectory["steps"]:
        for call in step["tool_calls"]:
            if call["tool_kind"] == "native_file_tool":
                calls.append({"tool": call["name"], "arguments": call["arguments"], "result": call["output"],
                              "error": "native_file_tool_error" if call["is_error"] else None,
                              "call_id": call["call_id"]})
    return calls


def verify_one(case: AgentCase, args: argparse.Namespace, home: Path) -> dict[str, Any]:
    task_dir = args.output_dir / "tasks" / case.case_name
    result_path = args.output_dir / "verification" / f"{case.case_name}.json"
    if result_path.is_file() and not args.retry_errors:
        previous = json.loads(result_path.read_text())
        if previous.get("status") in ("verified", "not_verifiable"):
            return previous
    base = {"case": case.case_name, "case_id": case.case_id, "task_id": case.task["task_id"]}
    run = json.loads((task_dir / "result.json").read_text()) if (task_dir / "result.json").is_file() else None
    if run is None or run.get("status") != "completed" or run.get("tool_budget_stop"):
        reason = "missing" if run is None else ("tool_budget_exceeded" if run.get("tool_budget_stop") else run["status"])
        result = {**base, "status": "not_verifiable", "outcome": "fail", "reason": f"generation {reason}"}
        write_json(result_path, result)
        return result
    if case.rubric_dir is None:
        result = {**base, "status": "verification_error", "outcome": None, "error": "frozen rubric missing"}
        write_json(result_path, result)
        return result
    verifier_dir = args.output_dir / "verification" / "runs" / case.case_name
    if verifier_dir.exists():
        shutil.rmtree(verifier_dir)
    try:
        trajectory = json.loads((task_dir / "trajectory.json").read_text())
        trace = task_dir / "raw/environment_tool_calls.jsonl"
        env_records = [json.loads(l) for l in trace.read_text().splitlines() if l.strip()] if trace.is_file() else []
        bound = bind_delivery({"tools": case.environment["tools"], "environment": case.environment,
                               "max_tool_calls": 1, "timeout": 1, "memory_limit": 1, "write_limit": 1,
                               **{k: case.runtime[k] for k in ("binding_path", "tools_path") if case.runtime.get(k)}})
        environment = dict(case.environment, tools=bound["tools"])
        calls = verification_calls(trajectory, env_records)
        mutations = sum(1 for c in env_records if c.get("error") is None and isinstance(c.get("result"), dict)
                        and c["result"].get("success") is True
                        and {t["name"]: t for t in bound["tools"]}.get(c.get("tool"), {}).get("usageConditions", {}).get("sideEffects"))
        verifier_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="actual-state-", dir=verifier_dir.parent) as tmp:
            actual = Path(tmp) / "state"
            with serialized_state_preparation():
                reconstruct_actual_state(case, task_dir, actual)
            verdict = verify_execution(
                run_dir=verifier_dir,
                config={"codex_home": str(home), "model": args.verifier_model,
                        "reasoning_effort": args.verifier_reasoning_effort,
                        "timeout_seconds": args.verifier_timeout, "attempts": args.verifier_attempts,
                        "submission_preflight": True, "method": "direct_codex_rubric",
                        "agent_sandbox": "read-only", "verifier_dir": str(case.rubric_dir)},
                task=case.task, environment=environment, initial_state=case.initial_state,
                actual_state=actual, calls=calls, answer=trajectory["outcome"]["final_answer"],
                execution={"error": None, "attempts": 1, "harness": "micius_eval.semiconductor_agent",
                           "call_source": "environment tool trace followed by native file tool calls",
                           "mutation_count": mutations},
                reference_state=case.reference_state, reference_calls=case.reference_calls,
                reference_answer=case.task.get("reference", {}).get("answer", ""),
            )
        direct = verifier_dir / "direct_result.json"
        rubric = json.loads(direct.read_text()) if direct.is_file() else {}
        result = {**base, "status": "verified", "outcome": verdict.get("outcome"),
                  "summary": verdict.get("summary"), "rubric_score": rubric.get("score"),
                  "rubric_mode": rubric.get("mode"), "verdict": str(verifier_dir / "verdict.json")}
    except Exception as exc:
        result = {**base, "status": "verification_error", "outcome": None,
                  "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    finally:
        evidence = verifier_dir / "evidence"
        if evidence.is_dir() and not args.keep_evidence:
            shutil.rmtree(evidence, ignore_errors=True)
    write_json(result_path, result)
    return result


def report(results: list[dict[str, Any]], total: int) -> dict[str, Any]:
    passed = sum(r.get("outcome") == "pass" for r in results)
    errors = sum(r["status"] == "verification_error" for r in results)
    return {
        "benchmark": "Semiconductor Agent", "tasks": total, "pass": passed,
        "fail": sum(r.get("outcome") == "fail" for r in results),
        "verification_error": errors, "unverified": total - len(results),
        "pass_rate": round(100 * passed / total, 2) if total else None,
        "note": "pass_rate uses the fixed denominator; verification errors must be retried "
                "(--retry-errors) before reporting a final score.",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Semiconductor Agent -- frozen-rubric verification stage.")
    p.add_argument("--output-dir", type=Path, required=True, help="The directory used for the run stage.")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--codex-home", type=Path, default=None, help="Existing Codex home (config.toml/auth.json).")
    p.add_argument("--verifier-base-url", default=os.environ.get("VERIFIER_BASE_URL"),
                   help="OpenAI-compatible Responses API base URL for the verifier.")
    p.add_argument("--verifier-api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--verifier-model", default="gpt-6.1-sol")
    p.add_argument("--verifier-reasoning-effort", default="high")
    p.add_argument("--verifier-timeout", type=int, default=1800)
    p.add_argument("--verifier-attempts", type=int, default=3)
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--keep-evidence", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    config = json.loads((args.output_dir / "config.json").read_text())
    data_root = (args.data_root or Path(config["data_root"]) or default_data_root()).resolve()
    check_ready(data_root)
    if shutil.which("codex") is None:
        sys.exit("The Codex CLI is required for verification: npm i -g @openai/codex")
    if args.output_dir.is_relative_to("/tmp"):
        sys.exit("--output-dir must not be under /tmp: Codex refuses to create its sandbox helpers there.")
    cases = load_cases(data_root, set(args.tasks) if args.tasks else None)
    generated = {p.name for p in (args.output_dir / "tasks").iterdir()} if (args.output_dir / "tasks").is_dir() else set()
    cases = [c for c in cases if c.case_name in generated]
    home = codex_home(args, args.output_dir)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(verify_one, c, args, home) for c in cases]
        for k, future in enumerate(concurrent.futures.as_completed(futures), 1):
            r = future.result()
            results.append(r)
            print(f"[{k}/{len(cases)}] {r['case']}: {r['status']} {r.get('outcome')}", flush=True)
    summary = report(results, int(config.get("tasks") or len(cases)))
    write_json(args.output_dir / "summary.json", {**summary, "results": sorted(results, key=lambda r: r["case"])})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
