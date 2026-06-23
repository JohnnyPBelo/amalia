"""Config loader — builds the Worker pool + Conductor settings from a YAML file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import yaml

from .workers import Worker, WorkerPool
from .conductor import ConductorConfig


@dataclass
class AmaliaConfig:
    orchestrator: Worker
    pool: WorkerPool
    conductor: ConductorConfig
    host: str = "0.0.0.0"
    port: int = 8900


def _worker_from_dict(d: dict) -> Worker:
    return Worker(
        name=d["name"],
        model=d["model"],
        base_url=d["base_url"],
        api_key=d.get("api_key", "none"),
        temperature=d.get("temperature", 0.2),
        max_tokens=d.get("max_tokens", 4096),
        capabilities=d.get("capabilities", "general"),
        timeout=d.get("timeout", 600.0),
        api_type=d.get("api_type", "chat"),
        reasoning_effort=d.get("reasoning_effort"),
    )


def load_config(path: str) -> AmaliaConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    orch = _worker_from_dict(raw["orchestrator"])
    pool = WorkerPool([_worker_from_dict(w) for w in raw["pool"]])

    c = raw.get("conductor", {})
    cc = ConductorConfig(
        max_steps=c.get("max_steps", 5),
        max_recursion=c.get("max_recursion", 1),
        parse_retries=c.get("parse_retries", 2),
        orchestrator_temperature=c.get("orchestrator_temperature", 1.0),
        orchestrator_max_tokens=c.get("orchestrator_max_tokens", 1024),
        fallback_worker=c.get("fallback_worker", 0),
    )

    srv = raw.get("server", {})
    return AmaliaConfig(orchestrator=orch, pool=pool, conductor=cc,
                      host=srv.get("host", "0.0.0.0"), port=srv.get("port", 8900))
