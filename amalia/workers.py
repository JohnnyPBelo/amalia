"""
Worker pool — a swappable set of LLM "agents", each just an OpenAI-compatible endpoint.

This is the abstraction that makes the pool swappable (the paper's "adaptive worker
selection" / Fugu's sovereignty play): add/remove/reorder workers in config, the
orchestrator only ever sees them as ordinals (Model 0, Model 1, ...).

A worker can point at:
  * a local llama.cpp server        (http://localhost:8901/v1)
  * the copilot/providers bridge    (http://localhost:4141/v1)
  * ollama                          (http://localhost:11434/v1)
  * OpenAI / Anthropic / any vendor (https://api.openai.com/v1, ...)
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import List, Optional

import httpx


@dataclass
class Worker:
    """One agent in the pool."""
    name: str                      # human label, e.g. "gpt-oss-120b" (NOT shown to orchestrator)
    model: str                     # model id sent to the endpoint
    base_url: str                  # OpenAI-compatible base, ending in /v1
    api_key: str = "none"          # env-expanded; "none" for local servers
    # generation defaults (paper sets workers to temp 0.2, 4096 max tokens)
    temperature: float = 0.2
    max_tokens: int = 4096
    # capability hint surfaced to the orchestrator as metadata (still ordinal-named)
    capabilities: str = "general"
    timeout: float = 600.0
    # wire protocol: "chat" (default, /chat/completions) or "responses" (/responses).
    # gpt-5.x models on the copilot bridge ONLY work via the Responses API (:4142).
    api_type: str = "chat"
    # reasoning effort for Responses-API models (none/low/medium/high/xhigh).
    reasoning_effort: Optional[str] = None

    def __post_init__(self):
        if self.api_key.startswith("env:"):
            self.api_key = os.environ.get(self.api_key[4:], "none")
        if self.api_type not in ("chat", "responses"):
            raise ValueError(f"api_type must be 'chat' or 'responses', got {self.api_type!r}")


@dataclass
class WorkerResult:
    ok: bool
    text: str
    worker_name: str
    error: Optional[str] = None
    usage: dict = field(default_factory=dict)


def _extract_responses_text(data: dict) -> str:
    """Pull the assistant text out of a Responses-API payload.

    Defends against the several shapes the Responses API returns:
      * a top-level `output_text` convenience field (string)
      * `output` list with a `message` block whose `content` has output_text/text parts
      * reasoning-only blocks (skipped)
    """
    # 1) convenience field
    ot = data.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot
    if isinstance(ot, list) and ot:
        joined = "".join(x for x in ot if isinstance(x, str))
        if joined.strip():
            return joined
    # 2) walk the output blocks
    parts: List[str] = []
    for blk in data.get("output", []):
        if blk.get("type") == "message":
            for c in blk.get("content", []):
                if c.get("type") in ("output_text", "text") and c.get("text"):
                    parts.append(c["text"])
    return "".join(parts)


class WorkerPool:
    def __init__(self, workers: List[Worker]):
        if not workers:
            raise ValueError("worker pool cannot be empty")
        self.workers = workers

    @property
    def n(self) -> int:
        return len(self.workers)

    def ordinal_listing(self) -> str:
        """The 'AVAILABLE LANGUAGE MODELS' block — ordinals + capability hint only.

        Deliberately brand-free (paper: avoid bias from known model names).
        """
        lines = []
        for i, w in enumerate(self.workers):
            lines.append(f"Model {i}: skills = {w.capabilities}")
        return "\n".join(lines)

    async def call(self, idx: int, messages: List[dict],
                   client: httpx.AsyncClient) -> WorkerResult:
        w = self.workers[idx]
        if w.api_type == "responses":
            return await self._call_responses(w, messages, client)
        return await self._call_chat(w, messages, client)

    async def _call_chat(self, w: Worker, messages: List[dict],
                         client: httpx.AsyncClient) -> WorkerResult:
        headers = {"Content-Type": "application/json"}
        if w.api_key and w.api_key != "none":
            headers["Authorization"] = f"Bearer {w.api_key}"
        payload = {
            "model": w.model,
            "messages": messages,
            "temperature": w.temperature,
            "max_tokens": w.max_tokens,
        }
        try:
            r = await client.post(f"{w.base_url.rstrip('/')}/chat/completions",
                                  json=payload, headers=headers, timeout=w.timeout)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"] or ""
            return WorkerResult(ok=True, text=text, worker_name=w.name,
                                usage=data.get("usage", {}))
        except Exception as e:  # noqa: BLE001 — surface any transport/HTTP error as a failed step
            return WorkerResult(ok=False, text="", worker_name=w.name, error=repr(e))

    async def _call_responses(self, w: Worker, messages: List[dict],
                              client: httpx.AsyncClient) -> WorkerResult:
        """Call the OpenAI Responses API (used by gpt-5.x via the :4142 shim)."""
        headers = {"Content-Type": "application/json"}
        if w.api_key and w.api_key != "none":
            headers["Authorization"] = f"Bearer {w.api_key}"
        # Convert chat messages -> Responses `input` items. A system message becomes
        # `instructions`; user/assistant turns become input_text content blocks.
        instructions = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        input_items = []
        for m in messages:
            if m["role"] == "system":
                continue
            ctype = "output_text" if m["role"] == "assistant" else "input_text"
            input_items.append({"role": m["role"],
                                "content": [{"type": ctype, "text": m["content"]}]})
        payload = {
            "model": w.model,
            "input": input_items,
            "max_output_tokens": w.max_tokens,
        }
        if instructions:
            payload["instructions"] = instructions
        if w.reasoning_effort:
            payload["reasoning"] = {"effort": w.reasoning_effort}
        try:
            r = await client.post(f"{w.base_url.rstrip('/')}/responses",
                                  json=payload, headers=headers, timeout=w.timeout)
            r.raise_for_status()
            data = r.json()
            text = _extract_responses_text(data)
            return WorkerResult(ok=True, text=text, worker_name=w.name,
                                usage=data.get("usage", {}))
        except Exception as e:  # noqa: BLE001
            return WorkerResult(ok=False, text="", worker_name=w.name, error=repr(e))

    async def health(self) -> List[dict]:
        """Ping every worker to verify reachability.

        chat workers -> GET /models; responses workers -> a tiny /responses probe
        (the :4142 shim may not expose /models for every model).
        """
        out = []
        async with httpx.AsyncClient() as client:
            for i, w in enumerate(self.workers):
                headers = {}
                if w.api_key and w.api_key != "none":
                    headers["Authorization"] = f"Bearer {w.api_key}"
                try:
                    if w.api_type == "responses":
                        res = await self.call(i, [{"role": "user", "content": "ping"}], client)
                        out.append({"idx": i, "name": w.name, "ok": res.ok,
                                    "api_type": "responses",
                                    **({"error": res.error} if res.error else {})})
                    else:
                        r = await client.get(f"{w.base_url.rstrip('/')}/models",
                                             headers=headers, timeout=15.0)
                        out.append({"idx": i, "name": w.name, "ok": r.status_code == 200,
                                    "api_type": "chat", "status": r.status_code})
                except Exception as e:  # noqa: BLE001
                    out.append({"idx": i, "name": w.name, "ok": False, "error": repr(e)})
        return out
