"""SemiBench evaluation.

Protocol (identical to the one used for the technical report):

* one user turn per question, no system prompt;
* thinking enabled at the highest effort, ``temperature=0``, ``top_p=1``;
* the last non-empty line of the final answer must be ``答案：<letters>``;
* scoring is exact match of the answer *set* (no partial credit); truncated,
  unparsable or failed responses count as incorrect.

Usage::

    python -m micius_eval.semibench.run --base-url http://127.0.0.1:30000/v1 \
        --model Micius-Flash --output-dir outputs/semibench/micius-flash

SemiBench has 167 questions (5 single-choice, 162 multiple-choice with up to
10 options). Each question was kept only if at least 3 of 5 strong reference
models answered it wrongly.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ..common import APIError, ChatClient, read_jsonl, reasoning_body, sha256_file, write_json

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data" / "semibench" / "semibench.jsonl"
LABELS = tuple("ABCDEFGHIJ")
PARSER_VERSION = "semibench-v1"


def build_prompt(row: dict[str, Any]) -> str:
    options = row["options"]
    labels = [k for k in LABELS if k in options]
    single = row["question_type"] == "single_choice"
    rule = "本题为单选题，请选择唯一一个正确选项。" if single else "本题为多选题，请选择所有正确选项；漏选或多选均不得分。"
    letters = "、".join(labels) if single else "A–" + labels[-1]
    return (
        "你正在参加中文工程技术选择题评测。请独立分析题干和全部选项。"
        + rule
        + "\n可以先思考。最终回答必须在最后一行写成“答案：”加选项字母；"
        f"字母仅限 {letters}，多个字母按字母顺序排列且不重复。最后一行不要附加解释。\n\n"
        + row["question"]
        + "\n\n选项：\n"
        + "\n".join(f"{k}. {options[k]}" for k in labels)
    )


def extract_answer(content: str | None, finish_reason: str | None, question_type: str):
    """Return ``(sorted_letters, None)`` or ``(None, invalid_reason)``."""
    if finish_reason == "length":
        return None, "truncated_response"
    if finish_reason != "stop":
        return None, "unfinished_response"
    text = str(content or "")
    if re.search(r"</think\s*>", text, re.I):
        text = re.split(r"</think\s*>", text, flags=re.I)[-1]
    if re.search(r"<think\b", text, re.I):
        return None, "unfinished_thinking"
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    if lines and lines[-1] == "```":
        lines.pop()
    if not lines:
        return None, "missing_final_answer"
    line = re.sub(r"^#{1,6}\s+", "", lines[-1]).replace("**", "").replace("__", "").strip("`").strip()
    match = re.fullmatch(r"<(?:answer|final)>(.*?)</(?:answer|final)>", line, re.I)
    if match:
        line = match.group(1).strip()
    line = re.sub(r"^(?:最终答案|正确答案|答案|final\s*answer|answer)\s*(?:是|为)?\s*[:：]?\s*", "", line, flags=re.I)
    line = line.rstrip("。.!！").strip()
    if len(line) >= 2 and (line[0], line[-1]) in [("(", ")"), ("（", "）"), ("[", "]")]:
        line = line[1:-1].strip()
    if not re.fullmatch(r"[A-J](?:[\s,，、;；]*[A-J])*", line):
        return None, "missing_or_ambiguous_final_answer"
    answers = re.findall(r"[A-J]", line)
    if len(answers) != len(set(answers)):
        return None, "duplicate_answer_letters"
    if question_type == "single_choice" and len(answers) != 1:
        return None, "multiple_answers_for_single_choice"
    return sorted(answers), None


def evaluate_one(row: dict[str, Any], client: ChatClient, args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    base = {"id": row["id"], "question_type": row["question_type"],
            "gold": row["answer"], "parser_version": PARSER_VERSION}
    try:
        response = client.chat(
            [{"role": "user", "content": build_prompt(row)}],
            temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, stream=False,
        )
    except APIError as exc:
        return {**base, "status": "error", "prediction": None, "correct": False, "error": str(exc),
                "elapsed_seconds": round(time.monotonic() - started, 3)}
    choice = response["choices"][0]
    message = choice.get("message") or {}
    prediction, invalid = extract_answer(message.get("content"), choice.get("finish_reason"), row["question_type"])
    return {
        **base,
        "status": "invalid" if invalid else "ok",
        "invalid_reason": invalid,
        "prediction": prediction,
        "correct": invalid is None and prediction == row["answer"],
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content") or message.get("reasoning"),
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage"),
        "endpoint": response.get("_endpoint"),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(bool(r["correct"]) for r in rows)
    return {
        "total": n,
        "correct": correct,
        "accuracy": round(100 * correct / n, 2) if n else None,
        "status_counts": dict(Counter(r["status"] for r in rows)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a model on SemiBench.")
    p.add_argument("--base-url", nargs="+", required=True, help="One or more OpenAI-compatible /v1 base URLs.")
    p.add_argument("--model", required=True, help="Served model name.")
    p.add_argument("--api-key", default=None, help="Defaults to $OPENAI_API_KEY (or 'EMPTY').")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--reasoning-effort", default="xhigh",
                   help="Thinking budget (Micius-Flash: low | medium | xhigh).")
    p.add_argument("--provider", choices=["local", "openai", "none"], default="local",
                   help="How to send the reasoning effort: chat_template_kwargs (local SGLang/vLLM), "
                        "top-level reasoning_effort (hosted APIs) or not at all.")
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--attempts", type=int, default=3, help="Retries for transport errors only.")
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions (smoke test).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "benchmark": "SemiBench", "model": args.model, "base_urls": args.base_url,
        "reasoning_effort": args.reasoning_effort, "provider": args.provider, "temperature": args.temperature,
        "top_p": args.top_p, "max_tokens": args.max_tokens, "data_sha256": sha256_file(args.data),
        "limit": args.limit, "parser_version": PARSER_VERSION,
        "scoring": "exact answer set; no partial credit; invalid/error count as incorrect",
    }
    config_path = out / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if {k: v for k, v in previous.items() if k != "base_urls"} != {k: v for k, v in config.items() if k != "base_urls"}:
            sys.exit(f"{config_path} exists with a different configuration; use a new --output-dir.")
    write_json(config_path, config)

    predictions_path = out / "predictions.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if predictions_path.exists():
        for r in read_jsonl(predictions_path):
            if r["status"] != "error":  # transport errors are retried on resume
                done[r["id"]] = r
    todo = [r for r in rows if r["id"] not in done]
    print(f"[SemiBench] {len(rows)} questions, {len(done)} cached, {len(todo)} to run", flush=True)

    client = ChatClient(args.base_url, args.model, api_key=args.api_key, timeout=args.timeout,
                        attempts=args.attempts, extra_body=reasoning_body(args.reasoning_effort, provider=args.provider))
    with predictions_path.open("a", encoding="utf-8") as sink, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_one, r, client, args) for r in todo]
        for k, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            done[result["id"]] = result
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")
            sink.flush()
            if k % 10 == 0 or k == len(futures):
                m = metrics([done[r["id"]] for r in rows if r["id"] in done])
                print(f"[SemiBench] {len(done)}/{len(rows)}  acc={m['accuracy']}%  {m['status_counts']}", flush=True)

    final = [done[r["id"]] for r in rows if r["id"] in done]
    summary = {
        **config,
        "overall": metrics(final),
        "by_question_type": {t: metrics([r for r in final if r["question_type"] == t])
                             for t in ("single_choice", "multiple_choice")},
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(out / "summary.json", summary)
    print(json.dumps({"SemiBench": summary["overall"]}, ensure_ascii=False, indent=2))
    return 0 if all(r["status"] != "error" for r in final) else 1


if __name__ == "__main__":
    raise SystemExit(main())
