"""
Conductor — ties the pieces together.

Flow (paper Sec. 3.1 + 3.2):
  1. Ask the orchestrator LLM for a workflow (CoT + three Python lists).
  2. Parse it. If parsing fails (paper's format-reward=0), retry up to `parse_retries`
     times, then fall back to a 1-step "just answer it" workflow on the strongest worker.
  3. Execute the workflow via WorkflowEngine (chain / tree / best-of-N, parallel waves).
  4. If recursion is enabled, feed the result back to the orchestrator (recursion prompt).
     It either emits three empty lists (return as-is) or a new refine/verify workflow.
     Repeat up to `max_recursion` rounds — a tunable test-time-compute axis.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional

import httpx

from .parser import parse_workflow, Workflow, WorkflowParseError
from .prompts import build_conductor_prompt, build_recursion_prompt, DEFAULT_FEWSHOT
from .workers import Worker, WorkerPool
from .engine import WorkflowEngine, WorkflowResult


@dataclass
class ConductorConfig:
    max_steps: int = 5
    max_recursion: int = 1          # 0 = no recursion; >0 = test-time scaling rounds
    parse_retries: int = 2
    orchestrator_temperature: float = 1.0   # paper trains/decodes Conductor at temp 1.0
    orchestrator_max_tokens: int = 1024     # paper: max completion length 1024
    fallback_worker: int = 0        # used if the orchestrator never yields a valid workflow
    few_shot: str = DEFAULT_FEWSHOT


@dataclass
class ConductorTrace:
    final_answer: str
    rounds: List[WorkflowResult] = field(default_factory=list)
    workflows: List[Workflow] = field(default_factory=list)
    total_worker_calls: int = 0
    used_fallback: bool = False
    orchestrator_model: str = ""


class Conductor:
    def __init__(self, orchestrator: Worker, pool: WorkerPool,
                 config: Optional[ConductorConfig] = None):
        self.orchestrator = orchestrator
        self.pool = pool
        self.engine = WorkflowEngine(pool)
        self.cfg = config or ConductorConfig()

    async def _ask_orchestrator(self, system_or_user: str,
                                client: httpx.AsyncClient,
                                history: Optional[List[dict]] = None) -> str:
        """One completion from the orchestrator LLM."""
        w = self.orchestrator
        headers = {"Content-Type": "application/json"}
        if w.api_key and w.api_key != "none":
            headers["Authorization"] = f"Bearer {w.api_key}"
        messages = history or [{"role": "user", "content": system_or_user}]
        payload = {
            "model": w.model,
            "messages": messages,
            "temperature": self.cfg.orchestrator_temperature,
            "max_tokens": self.cfg.orchestrator_max_tokens,
        }
        r = await client.post(f"{w.base_url.rstrip('/')}/chat/completions",
                              json=payload, headers=headers, timeout=w.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

    async def _get_workflow(self, prompt: str, client: httpx.AsyncClient,
                            history: Optional[List[dict]] = None) -> Optional[Workflow]:
        """Ask + parse, with retries. Returns None if all attempts fail to parse."""
        last_err = None
        for _ in range(self.cfg.parse_retries + 1):
            completion = await self._ask_orchestrator(prompt, client, history)
            try:
                return parse_workflow(completion, n_models=self.pool.n,
                                      max_steps=self.cfg.max_steps)
            except WorkflowParseError as e:
                last_err = e
                continue
        return None

    def _fallback_workflow(self, user_question: str) -> Workflow:
        return Workflow(model_id=[self.cfg.fallback_worker],
                        subtasks=["Answer the user's question directly and completely."],
                        access_list=[[]], raw="<fallback>")

    async def run(self, user_question: str) -> ConductorTrace:
        trace = ConductorTrace(final_answer="", orchestrator_model=self.orchestrator.model)
        async with httpx.AsyncClient() as client:
            # ---- round 0: initial workflow ------------------------------------
            prompt = build_conductor_prompt(
                user_question=user_question,
                available_models=self.pool.ordinal_listing(),
                max_steps=self.cfg.max_steps,
                few_shot=self.cfg.few_shot,
            )
            wf = await self._get_workflow(prompt, client)
            if wf is None or wf.is_empty():
                wf = self._fallback_workflow(user_question)
                trace.used_fallback = True

            trace.workflows.append(wf)
            result = await self.engine.execute(wf, user_question)
            trace.rounds.append(result)
            trace.total_worker_calls += result.n_worker_calls
            trace.final_answer = result.final_answer

            # ---- recursion rounds: refine / verify ----------------------------
            for _ in range(self.cfg.max_recursion):
                rec_prompt = (
                    build_conductor_prompt(
                        user_question=user_question,
                        available_models=self.pool.ordinal_listing(),
                        max_steps=self.cfg.max_steps,
                        few_shot=self.cfg.few_shot,
                    )
                    + "\n\n"
                    + build_recursion_prompt(trace.final_answer, max_steps=self.cfg.max_steps)
                )
                rec_wf = await self._get_workflow(rec_prompt, client)
                if rec_wf is None or rec_wf.is_empty():
                    break  # orchestrator is satisfied — return current answer as-is
                trace.workflows.append(rec_wf)
                rec_result = await self.engine.execute(rec_wf, user_question)
                trace.rounds.append(rec_result)
                trace.total_worker_calls += rec_result.n_worker_calls
                if rec_result.final_answer.strip():
                    trace.final_answer = rec_result.final_answer

        return trace
