"""
OpenAI-compatible server — exposes `amalia-v1` as a single model.

Endpoints:
  GET  /v1/models               -> advertises `amalia-v1`
  POST /v1/chat/completions     -> runs the Conductor over the worker pool
  GET  /health                  -> pings every worker

Any tool that speaks the OpenAI API (Hermes provider, OpenClaw model, Codex, Cursor,
curl) can point at this endpoint and transparently get multi-agent orchestration.

Set AMALIA_DEBUG=1 to attach the full workflow trace under `choices[0].message.amalia_trace`.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import load_config
from .conductor import Conductor

CONFIG_PATH = os.environ.get("AMALIA_CONFIG", "config.yaml")
DEBUG = os.environ.get("AMALIA_DEBUG", "0") == "1"

app = FastAPI(title="amalia-v1", version="0.1.0")
_cfg = None
_conductor: Optional[Conductor] = None


def _ensure_loaded():
    global _cfg, _conductor
    if _conductor is None:
        _cfg = load_config(CONFIG_PATH)
        _conductor = Conductor(_cfg.orchestrator, _cfg.pool, _cfg.conductor)
    return _cfg, _conductor


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "amalia-v1"
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    # amalia knobs (optional, ignored by generic clients)
    amalia_max_recursion: Optional[int] = None


def _flatten_query(messages: List[ChatMessage]) -> str:
    """Collapse the incoming chat into a single user question for the Conductor.

    System messages are prepended as instructions; the last user turn is the question.
    Prior turns are included as lightweight context.
    """
    sys = [m.content for m in messages if m.role == "system"]
    convo = [f"{m.role.upper()}: {m.content}" for m in messages if m.role != "system"]
    parts = []
    if sys:
        parts.append("INSTRUCTIONS:\n" + "\n".join(sys))
    parts.append("\n".join(convo))
    return "\n\n".join(parts).strip()


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "amalia-v1", "object": "model", "owned_by": "amalia-v1",
                  "created": int(time.time())}],
    }


@app.get("/health")
async def health():
    _, conductor = _ensure_loaded()
    pool_health = await conductor.pool.health()
    return {"status": "ok", "orchestrator": conductor.orchestrator.model,
            "pool": pool_health}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    cfg, conductor = _ensure_loaded()
    if req.stream:
        raise HTTPException(400, "streaming not yet supported by amalia-v1")

    if req.amalia_max_recursion is not None:
        conductor.cfg.max_recursion = int(req.amalia_max_recursion)

    question = _flatten_query(req.messages)
    if not question:
        raise HTTPException(400, "no user content in messages")

    trace = await conductor.run(question)

    message = {"role": "assistant", "content": trace.final_answer}
    if DEBUG:
        message["amalia_trace"] = {
            "orchestrator": trace.orchestrator_model,
            "used_fallback": trace.used_fallback,
            "total_worker_calls": trace.total_worker_calls,
            "workflows": [
                {"model_id": w.model_id, "subtasks": w.subtasks, "access_list": w.access_list}
                for w in trace.workflows
            ],
            "rounds": [
                {"final_len": len(r.final_answer), "n_worker_calls": r.n_worker_calls,
                 "steps": [{"idx": s.idx, "model_id": s.model_id, "worker": s.worker_name,
                            "ok": s.ok, "error": s.error} for s in r.steps]}
                for r in trace.rounds
            ],
        }

    return JSONResponse({
        "id": f"chatcmpl-amalia-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "amalia-v1",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                  "total_tokens": 0, "amalia_worker_calls": trace.total_worker_calls},
    })
