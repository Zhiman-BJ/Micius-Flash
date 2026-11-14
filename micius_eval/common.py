"""Shared helpers: OpenAI-compatible chat client, endpoint pool and JSON IO."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

# Requests to a model server should never be routed through an HTTP proxy that
# happens to be configured in the shell (a common source of silent failures on
# clusters). Set MICIUS_USE_PROXY=1 to opt back in.
_OPENER = (
    urllib.request.build_opener()
    if os.environ.get("MICIUS_USE_PROXY") == "1"
    else urllib.request.build_opener(urllib.request.ProxyHandler({}))
)


class APIError(RuntimeError):
    """Transport or server error (retryable)."""


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EndpointPool:
    """Least-active load balancing over one or more OpenAI-compatible base URLs."""

    def __init__(self, base_urls: Iterable[str]):
        self.urls = [u.rstrip("/") for u in base_urls if u]
        if not self.urls:
            raise ValueError("at least one --base-url is required")
        self.active = [0] * len(self.urls)
        self.used = [0] * len(self.urls)
        self.lock = threading.Lock()

    def acquire(self) -> int:
        with self.lock:
            i = min(range(len(self.urls)), key=lambda k: (self.active[k], self.used[k], k))
            self.active[i] += 1
            self.used[i] += 1
            return i

    def release(self, i: int) -> None:
        with self.lock:
            self.active[i] -= 1


class ChatClient:
    """Minimal, dependency-free client for ``/v1/chat/completions``.

    ``extra_body`` is merged into every request, e.g. ``chat_template_kwargs``
    for SGLang / vLLM servers.
    """

    def __init__(
        self,
        base_urls: Iterable[str] | str,
        model: str,
        api_key: str | None = None,
        timeout: float = 1800,
        attempts: int = 3,
        extra_body: dict[str, Any] | None = None,
    ):
        if isinstance(base_urls, str):
            base_urls = [base_urls]
        self.pool = EndpointPool(base_urls)
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.timeout = timeout
        self.attempts = attempts
        self.extra_body = extra_body or {}

    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with _OPENER.open(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:2000]
            raise APIError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            raise APIError(f"{type(exc).__name__}: {exc}") from exc

    def chat(self, messages: list[dict[str, Any]], **params: Any) -> dict[str, Any]:
        """Return the raw response JSON. Retries transport errors only."""
        body = {"model": self.model, "messages": messages, **self.extra_body, **params}
        last: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            i = self.pool.acquire()
            try:
                response = self._post(self.pool.urls[i], body)
                if "choices" not in response:
                    raise APIError(f"malformed response: {str(response)[:500]}")
                response["_endpoint"] = self.pool.urls[i]
                return response
            except APIError as exc:
                last = exc
            finally:
                self.pool.release(i)
            if attempt < self.attempts:
                time.sleep(min(2 * attempt, 30))
        raise APIError(f"request failed after {self.attempts} attempts: {last}")

    def list_models(self) -> list[str]:
        request = urllib.request.Request(
            self.pool.urls[0] + "/models", headers={"Authorization": f"Bearer {self.api_key}"}
        )
        with _OPENER.open(request, timeout=30) as response:
            return [m["id"] for m in json.loads(response.read()).get("data", [])]


def reasoning_body(reasoning_effort: str | None, *, provider: str = "local") -> dict[str, Any]:
    """Request fields that select the thinking budget.

    ``local``  -- SGLang / vLLM serving Micius-Flash (or any server whose chat
                  template reads ``chat_template_kwargs``): passed through it.
    ``openai`` -- hosted APIs that accept a top-level ``reasoning_effort``.
    ``none``   -- send nothing.
    """
    if not reasoning_effort or provider == "none":
        return {}
    if provider == "openai":
        return {"reasoning_effort": reasoning_effort}
    return {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": reasoning_effort}}
