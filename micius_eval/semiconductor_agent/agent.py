"""Native function-calling agent loop for Semiconductor Agent tasks.

The context reproduces what the model saw when the evaluation trajectories
were collected with Kimi Code CLI: the CLI system prompt (with a two-level
listing of ``/workspace``) followed by the runtime tool contract, the task as
the first user turn, then the date and auto-permission reminders. Business
tools keep their bare names; the native file tools are ``Read``, ``Grep`` and
``Glob``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..common import APIError, ChatClient
from .file_tools import WorkspaceFiles

ASSETS = Path(__file__).resolve().parent / "assets"
NATIVE_TOOLS = json.loads((ASSETS / "native_tools.json").read_text(encoding="utf-8"))
SYSTEM_TEMPLATE = (ASSETS / "system_prompt.txt").read_text(encoding="utf-8")
DATE_REMINDER = (
    "<system-reminder>\nToday's date is {date}. The current date is restated in a reminder whenever it "
    "changes; rely on the latest such reminder for the current date. DO NOT mention this to the user "
    "explicitly.\n</system-reminder>"
)
AUTO_MODE_REMINDER = (
    "<system-reminder>\nAuto permission mode is active. Tool approvals will be handled automatically while "
    "this mode remains enabled.\n  - Continue normally without pausing for approval prompts.\n  - Do NOT "
    "call AskUserQuestion while auto mode is active. Make a reasonable decision and continue without asking "
    "the user.\n  - ExitPlanMode is also approved automatically, without the user reviewing the plan. An "
    "auto-approved plan is NOT a signal from the user to start executing — follow the user's original "
    "instructions on whether to proceed.\n</system-reminder>"
)
COMPACTION_PROMPT = (
    "The conversation is approaching the context limit. Write a first-person summary of the work so far for "
    "your own continuation: the user's request and every explicit requirement, what you have verified (with "
    "exact values, identifiers, file paths and tool results you will need), what you have changed in the "
    "environment, and what remains to be done. Do not call tools."
)


@dataclass
class AgentConfig:
    max_tool_calls: int = 85            # environment (business) tool calls
    max_turns: int = 400                # model requests, safety net
    max_context: int = 262_144
    max_output: int = 131_072
    compaction_ratio: float = 0.5
    compaction_output: int = 65_536
    temperature: float | None = None    # None = server default (generation_config)
    top_p: float | None = None
    date: str = "2026-10-22"


@dataclass
class AgentResult:
    status: str                          # completed | tool_budget_exceeded | max_turns | error
    final_answer: str
    messages: list[dict[str, Any]]
    steps: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    environment_calls: int = 0
    native_calls: int = 0
    compactions: int = 0
    error: str | None = None


def openai_tool(public_tool: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": public_tool["name"],
        "description": public_tool.get("description", ""),
        "parameters": public_tool.get("inputSchema") or {"type": "object", "properties": {}},
    }}


def build_context(task_text: str, files: WorkspaceFiles, config: AgentConfig) -> list[dict[str, Any]]:
    system = SYSTEM_TEMPLATE.replace("{directory_listing}", files.listing())
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": task_text},
        {"role": "user", "content": DATE_REMINDER.format(date=config.date)},
        {"role": "user", "content": AUTO_MODE_REMINDER},
    ]


def _shrink_last_tool_results(messages: list[dict[str, Any]], keep: int = 20_000) -> bool:
    """Truncate oversized tool results of the latest turn; True if anything changed."""
    changed = False
    for message in reversed(messages):
        if message["role"] != "tool":
            break
        text = message["content"]
        if len(text) > keep:
            message["content"] = text[:keep] + f"\n[truncated by the harness: {len(text) - keep} characters omitted; request a smaller result]"
            changed = True
    return changed


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def run_agent(
    *,
    client: ChatClient,
    task_text: str,
    files: WorkspaceFiles,
    env_tools: list[dict[str, Any]],
    call_env_tool: Callable[[str, dict[str, Any]], tuple[Any, bool]],
    config: AgentConfig,
    log: Callable[[dict[str, Any]], None] = lambda _: None,
) -> AgentResult:
    """Run until the model answers without tool calls (or a budget is hit)."""
    tools = [openai_tool(t) for t in env_tools] + NATIVE_TOOLS
    env_names = {t["name"] for t in env_tools}
    head = build_context(task_text, files, config)
    messages = list(head)
    result = AgentResult(status="error", final_answer="", messages=messages)
    usage_total: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
    last_prompt_tokens = 0
    over_budget = 0                      # business calls refused after the budget
    empty_retries = 0

    def request(msgs: list[dict[str, Any]], with_tools: bool = True, max_tokens: int | None = None) -> dict[str, Any]:
        budget = max(1024, min(config.max_output, config.max_context - last_prompt_tokens - 4096))
        params: dict[str, Any] = {"max_tokens": max_tokens or budget}
        if with_tools:
            params["tools"] = tools
        if config.temperature is not None:
            params["temperature"] = config.temperature
        if config.top_p is not None:
            params["top_p"] = config.top_p
        response = client.chat(msgs, **params)
        for key in usage_total:
            usage_total[key] += int((response.get("usage") or {}).get(key) or 0)
        return response

    for turn in range(config.max_turns):
        if last_prompt_tokens > config.compaction_ratio * config.max_context:
            try:
                summary = request(messages + [{"role": "user", "content": COMPACTION_PROMPT}], with_tools=False,
                                  max_tokens=config.compaction_output)
                text = (summary["choices"][0]["message"].get("content") or "").strip()
            except APIError as exc:
                result.error = f"compaction failed: {exc}"
                break
            messages[:] = head + [{"role": "user", "content":
                                   "<system-reminder>\nSummary of the earlier part of this session:\n\n"
                                   + text + "\n</system-reminder>"}]
            result.compactions += 1
            last_prompt_tokens = 0
            log({"event": "compaction", "turn": turn})
        try:
            response = request(messages)
        except APIError as exc:
            # Large tool results can push the prompt over the context window:
            # compact once and retry before giving up.
            overflow = any(k in str(exc).lower() for k in ("context length", "maximum context", "too long", "max_tokens"))
            if overflow and _shrink_last_tool_results(messages):
                continue
            result.error = str(exc)
            break
        last_prompt_tokens = int((response.get("usage") or {}).get("prompt_tokens") or 0) \
            + int((response.get("usage") or {}).get("completion_tokens") or 0)
        choice = response["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        tool_calls = message.get("tool_calls") or []
        assistant: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            assistant["reasoning_content"] = reasoning
        if tool_calls:
            assistant["tool_calls"] = [{"id": c.get("id") or f"call_{turn}_{k}", "type": "function",
                                        "function": {"name": c["function"]["name"],
                                                     "arguments": c["function"].get("arguments") or "{}"}}
                                       for k, c in enumerate(tool_calls)]
        messages.append(assistant)
        step = {"turn": turn, "finish_reason": choice.get("finish_reason"), "usage": response.get("usage"),
                "content": content, "reasoning_content": reasoning, "tool_calls": []}
        result.steps.append(step)

        if not tool_calls:
            if not content.strip() and empty_retries < 2:
                empty_retries += 1
                messages.pop()
                continue
            result.status = "completed"
            result.final_answer = content
            break

        for call in assistant["tool_calls"]:
            name = call["function"]["name"]
            started = time.monotonic()
            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                payload, is_error, kind = {"error": f"Invalid tool arguments: {exc}"}, True, "invalid_arguments"
                arguments = call["function"]["arguments"]
            else:
                if name in env_names:
                    if result.environment_calls >= config.max_tool_calls:
                        payload = {"error": "Tool call limit reached. No further business tool calls are "
                                            "allowed; give your final answer now."}
                        is_error, kind = True, "environment_mcp"
                        over_budget += 1
                    else:
                        result.environment_calls += 1
                        payload, is_error = call_env_tool(name, arguments)
                        kind = "environment_mcp"
                elif name in ("Read", "Grep", "Glob"):
                    result.native_calls += 1
                    payload, is_error = files.call(name, arguments)
                    kind = "native_file_tool"
                else:
                    payload, is_error, kind = {"error": f"Tool not found: {name}"}, True, "unknown"
            text = payload if isinstance(payload, str) else _dump(payload)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
            step["tool_calls"].append({"call_id": call["id"], "name": name, "tool_kind": kind,
                                       "arguments": arguments, "output": payload, "is_error": is_error,
                                       "seconds": round(time.monotonic() - started, 3)})
            log({"event": "tool", "turn": turn, "name": name, "is_error": is_error})
        if over_budget >= 2:  # the model ignored the limit notice: stop
            result.status = "tool_budget_exceeded"
            break
    else:
        result.status = "max_turns"

    result.messages = messages
    result.usage = usage_total
    return result
