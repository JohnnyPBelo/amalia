"""
WorkflowEngine — executes a parsed Conductor Workflow over the worker pool.

This is the core. Given model_id / subtasks / access_list, it:

  1. Builds a DAG: step i depends on the steps named in its access_list.
  2. Executes in topological waves — independent steps (e.g. tree leaves or
     best-of-N attempts) run in PARALLEL; dependent steps wait for their refs.
  3. Each worker sees: the original user question (step 0 only, paper-faithful) +
     the (subtask, response) pairs from the steps in its access_list, as prior
     conversational turns.
  4. The FINAL step's output is the workflow's answer.
  5. Optional recursion: the orchestrator inspects the answer and may emit a new
     workflow (refine/verify) or three empty lists to return as-is — a tunable
     test-time-compute axis (paper Sec. 3.2).

Topologies are emergent from access_list, not hard-coded:
  * chain        -> access_list = [[], ["all"], ["all"], ...]
  * best-of-N    -> N independent attempts + a final picker referencing [0..N-1]
  * tree         -> several [] leaves + an aggregator referencing the leaf indices
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import httpx

from .parser import Workflow, AccessEntry
from .workers import WorkerPool, WorkerResult


@dataclass
class StepTrace:
    idx: int
    model_id: int
    worker_name: str
    subtask: str
    response: str
    ok: bool
    error: Optional[str] = None


@dataclass
class WorkflowResult:
    final_answer: str
    steps: List[StepTrace] = field(default_factory=list)
    recursion_round: int = 0
    n_worker_calls: int = 0
    error: Optional[str] = None


def _deps_of(entry: AccessEntry, step_idx: int) -> List[int]:
    """Resolve an access_list entry to concrete prior-step indices.

    Accepts the normalized "all" string, the raw ["all"] form, or a list of ints,
    so the engine is robust whether or not the workflow came through the parser.
    """
    if entry == "all" or entry == ["all"]:
        return list(range(step_idx))
    if isinstance(entry, list):
        return [d for d in entry if isinstance(d, int)]
    return []


def _build_context(step_idx: int, access: AccessEntry, subtasks: List[str],
                   responses: Dict[int, str], user_question: str) -> List[dict]:
    """Assemble the message list for a worker call.

    Paper: the first selected model is prompted with the user question + its subtask.
    Each following model receives the (subtask, response) history named in its access_list,
    then its own subtask.
    """
    messages: List[dict] = []
    deps = _deps_of(access, step_idx)
    for d in deps:
        # represent a prior step as a user(subtask)/assistant(response) turn
        messages.append({"role": "user", "content": subtasks[d]})
        messages.append({"role": "assistant", "content": responses.get(d, "")})

    if step_idx == 0:
        # root step always sees the original user question
        content = f"{user_question}\n\n---\nYour task: {subtasks[step_idx]}"
    else:
        content = subtasks[step_idx]
    messages.append({"role": "user", "content": content})
    return messages


class WorkflowEngine:
    def __init__(self, pool: WorkerPool):
        self.pool = pool

    async def execute(self, wf: Workflow, user_question: str) -> WorkflowResult:
        """Run a single (non-recursive) workflow with wave-based parallelism."""
        if wf.is_empty():
            return WorkflowResult(final_answer="", error="empty workflow")

        responses: Dict[int, str] = {}
        traces: List[Optional[StepTrace]] = [None] * wf.n_steps
        n_calls = 0
        remaining = set(range(wf.n_steps))

        async with httpx.AsyncClient() as client:
            # topological waves: a step is ready when all its deps already have responses
            while remaining:
                ready = [
                    i for i in remaining
                    if all(d in responses for d in _deps_of(wf.access_list[i], i))
                ]
                if not ready:
                    # cycle or dangling ref (parser should prevent this) — bail safely
                    return WorkflowResult(
                        final_answer=responses.get(wf.n_steps - 1, ""),
                        steps=[t for t in traces if t], n_worker_calls=n_calls,
                        error="unresolvable dependency wave (cycle?)",
                    )

                coros = []
                for i in ready:
                    msgs = _build_context(i, wf.access_list[i], wf.subtasks,
                                          responses, user_question)
                    coros.append(self.pool.call(wf.model_id[i], msgs, client))
                results: List[WorkerResult] = await asyncio.gather(*coros)
                n_calls += len(results)

                for i, res in zip(ready, results):
                    responses[i] = res.text
                    traces[i] = StepTrace(
                        idx=i, model_id=wf.model_id[i], worker_name=res.worker_name,
                        subtask=wf.subtasks[i], response=res.text, ok=res.ok, error=res.error,
                    )
                    remaining.discard(i)

        final = responses.get(wf.n_steps - 1, "")
        return WorkflowResult(final_answer=final,
                              steps=[t for t in traces if t],
                              n_worker_calls=n_calls)
