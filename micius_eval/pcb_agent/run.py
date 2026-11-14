"""Stage 1 of PCB Agent: run the agent on the 30 public tasks.

Each task runs in its own container from the pinned evaluation image. Public
task files are mounted read-only at ``/task``; the declared initial project
files are copied into a fresh writable ``/workspace``. The model is reached
through an OpenAI-compatible endpoint (SGLang recommended: the harness uses
``/v1/tokenize`` to size every request exactly to the remaining context).

Protocol used in the paper: xhigh thinking for agent and compaction,
temperature 1.0 / top_p 0.95 / top_k 20, 500 agent steps, 262,144-token
context, 131,072 output tokens per request, no wall-clock limit.

    python -m micius_eval.pcb_agent.run --base-url http://<host>:30000/v1 \
        --model Micius-Flash --output-dir outputs/pcb_agent/micius-flash --concurrency 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from . import GENERATION_IMAGE
from ..common import write_json

DATA = Path(__file__).resolve().parents[2] / "data" / "pcb_agent"
ADAPTER = Path(__file__).resolve().parent / "adapter"
ENTRY = "tools.kimi_dsflash_v7_harness"
OVERLAY = {
    "kimi_dsflash_v6_harness.py": "/pcb-agent/tools/kimi_dsflash_v6_harness.py",
    "kimi_k3v19_provider.py": "/pcb-agent/tools/kimi_k3v19_provider.py",
    "kimi_k3v19_progress.py": "/pcb-agent/tools/kimi_k3v19_progress.py",
    "kimi_cli_llm.py": "/opt/kimi/lib/python3.12/site-packages/kimi_cli/llm.py",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_join(root: Path, name: str) -> Path:
    rel = Path(name)
    if not rel.parts or rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"invalid relative path: {name}")
    return root / rel


def container_endpoint(base_url: str) -> str:
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        if not path.endswith("/v1"):
            raise ValueError("--base-url must end in /v1")
        path += "/chat/completions"
    netloc = parts.netloc
    if parts.hostname in {"localhost", "127.0.0.1", "0.0.0.0"}:
        netloc = "host.docker.internal" + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, path, "", ""))


def preflight(data: Path) -> dict:
    """Check every public file against the frozen manifest."""
    manifest = json.loads((data / "manifest.json").read_text())
    assert manifest["count"] == 30 and len(manifest["rows"]) == 30
    for row in manifest["rows"]:
        public = data / "tasks" / row["question"] / "public"
        for item in row["public_files"]:
            path = safe_join(public, item["path"])
            if not path.is_file() or sha(path) != item["sha256"]:
                raise SystemExit(f"public file changed or missing: {row['question']}/{item['path']}")
    return manifest


def prepare(data: Path, output: Path, question: str) -> dict:
    run = output / "tasks" / question
    public, workspace = run / "public", run / "workspace"
    shutil.copytree(data / "tasks" / question / "public", public)
    workspace.mkdir()
    layout_path = public / ("runtime-input-layout.json" if (public / "runtime-input-layout.json").exists() else "input-layout.json")
    layout = json.loads(layout_path.read_text())
    if layout.get("mapping"):
        members = [(k, str(Path(layout["public_package_prefix"]) / v)) for k, v in layout["mapping"].items()]
    else:
        members = [(k, k) for k in layout.get("initial_files", {})]
    for target, source in members:
        destination = safe_join(workspace, target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_join(public, source), destination)
    working = layout.get("working_directory", ".")
    (workspace if working == "." else safe_join(workspace, working)).mkdir(parents=True, exist_ok=True)
    return {"question": question, "working": working,
            "requirements_sha256": sha(public / "public_requirements.json"),
            "delivery_sha256": sha(public / "delivery-requirements.json")}


def docker(args: list[str], env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["docker", *args], capture_output=True, text=True, env=env)
    if check and result.returncode:
        raise RuntimeError(f"docker {args[0]} failed: {result.stderr.strip()[:500]}")
    return result


def run_one(output: Path, question: str, a: argparse.Namespace) -> dict:
    run = output / "tasks" / question
    if (run / "completion.json").is_file():
        return json.loads((run / "completion.json").read_text())
    if run.exists():
        shutil.rmtree(run)
    started = time.time()
    try:
        row = prepare(DATA, output, question)
        cwd = "/workspace" if row["working"] == "." else "/workspace/" + row["working"]
        name = "micius-pcb-" + hashlib.sha256((str(output) + question).encode()).hexdigest()[:16]
        docker(["rm", "-f", name], check=False)
        env = dict(os.environ, MODEL_API_KEY=a.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY")
        cmd = ["create", "--name", name, "--init", "--read-only", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--pids-limit", "512",
               "--memory", a.memory, "--memory-swap", a.memory, "--cpus", str(a.cpus),
               "--tmpfs", "/tmp:rw,nosuid,size=1g", "--user", "0:0"]
        for cap in ("CHOWN", "DAC_OVERRIDE", "FOWNER", "KILL", "SETUID", "SETGID"):
            cmd += ["--cap-add", cap]
        for filename, target in OVERLAY.items():
            cmd += ["-v", f"{ADAPTER / filename}:{target}:ro"]
        image_id = docker(["image", "inspect", a.image, "--format", "{{.Id}}"]).stdout.strip()
        cmd += ["--add-host", "host.docker.internal:host-gateway",
                "-v", f"{run / 'workspace'}:/workspace", "-v", f"{run / 'public'}:/task:ro",
                "-e", "MODEL_API_KEY", "-e", "PCB_PUBLIC_REQUIREMENTS=/task/public_requirements.json",
                "-e", "PCB_REQUIREMENTS_REQUIRED=1", "-e", "PCB_REQUIREMENTS_SHA256=" + row["requirements_sha256"],
                "-e", "PCB_RUNTIME_IMAGE=" + image_id, "-e", "PCB_TOOL_TIMEOUT_SECONDS=600",
                "-e", "HOME=/tmp/kimi-home", "--workdir", cwd, "--entrypoint", "/usr/bin/python3", a.image,
                "-m", ENTRY, "--workspace", cwd, "--task", "/task/prompt.txt", "--model", a.model,
                "--endpoint", container_endpoint(a.base_url), "--reasoning-effort", a.reasoning_effort,
                "--delivery-requirements", "/task/delivery-requirements.json",
                "--delivery-requirements-sha256", row["delivery_sha256"],
                "--max-steps", str(a.max_steps), "--max-seconds", "none", "--task-phase", "1"]
        cid = docker(cmd, env=env).stdout.strip()
        docker(["start", cid], env=env)
        rc = int(docker(["wait", cid]).stdout.strip())
        logs = docker(["logs", cid], check=False)
        (run / "container.log").write_text(logs.stdout + logs.stderr)
        state = json.loads(docker(["inspect", cid, "--format", "{{json .State}}"]).stdout)
        docker(["rm", cid], check=False)
        session = run / "workspace" / ("" if row["working"] == "." else row["working"]) / "session" / "result.json"
        native = json.loads(session.read_text()) if session.is_file() else {}
        result = {"question": question, "status": "finished", "container_exit_code": rc,
                  "oom_killed": state.get("OOMKilled"), "harness_finished": native.get("harness_finished"),
                  "harness_reason": native.get("reason"), "session_result": str(session) if session.is_file() else None,
                  "working": row["working"], "seconds": round(time.time() - started, 1)}
    except Exception as exc:
        result = {"question": question, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                  "seconds": round(time.time() - started, 1)}
    write_json(run / "completion.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PCB Agent -- generation stage.")
    p.add_argument("--base-url", required=True, help="OpenAI-compatible /v1 URL reachable from containers.")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--image", default=GENERATION_IMAGE)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"), default="xhigh")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--cpus", type=float, default=3)
    p.add_argument("--memory", default="4g")
    a = p.parse_args(argv)
    if shutil.which("docker") is None:
        sys.exit("Docker is required for PCB Agent.")
    manifest = preflight(DATA)
    questions = [r["question"] for r in manifest["rows"] if not a.tasks or r["question"] in a.tasks]
    output = a.output_dir.resolve()
    (output / "tasks").mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", {"benchmark": "PCB Agent", "model": a.model, "base_url": a.base_url,
                                        "image": a.image, "reasoning_effort": a.reasoning_effort,
                                        "max_steps": a.max_steps, "tasks": len(questions)})
    results = []
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futures = [pool.submit(run_one, output, q, a) for q in questions]
        for k, f in enumerate(as_completed(futures), 1):
            r = f.result()
            results.append(r)
            print(f"[{k}/{len(questions)}] {r['question']}: {r['status']} reason={r.get('harness_reason')} "
                  f"({r.get('seconds')}s){' ' + r['error'][:200] if r.get('error') else ''}", flush=True)
    write_json(output / "generation_summary.json", {"tasks": len(results), "results": sorted(results, key=lambda r: r["question"])})
    print(f"Done. Next: python -m micius_eval.pcb_agent.verify --output-dir {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
