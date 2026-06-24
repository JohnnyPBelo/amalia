"""
Training-Free GRPO for the Amalia Conductor.

Idea (Training-Free GRPO, arXiv:2510.19807-style): instead of gradient updates,
optimize in *context space*. Each iteration:

  1. For each task, sample G workflows (rollouts) from the Conductor at temp 1.0.
  2. Execute each, score with the verifiable reward (format + correctness).
  3. Within the group, contrast successes vs failures and ask the orchestrator LLM
     to extract a short, general "experience" (a semantic group-advantage) — what
     routing/decomposition choice made the winners win.
  4. Append distilled experiences to an experience library.
  5. The library is injected into the Conductor prompt on subsequent iterations.

No weights are touched. The artifact is `experiences.json` — a learned prior over
*how to orchestrate*, portable across orchestrator models.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import httpx

from ..conductor import Conductor, ConductorConfig
from ..workers import Worker, WorkerPool
from ..config import load_config
from .tasks import Task, get_tasks, extract_final


EXPERIENCE_HEADER = (
    "\nLEARNED ORCHESTRATION EXPERIENCES (apply these when they fit; they were "
    "distilled from what worked on similar tasks):\n"
)

_VAGUE_WORDS = ("properly", "effectively", "thoroughly", "appropriate", "necessary",
                "correctly", "accurately", "comprehensive")
# Strong structural mechanisms — a useful experience must name at least one of these.
# NOTE: the bare token "access_list" is deliberately NOT here: the entire problem
# domain is about access_lists, so "use access_lists effectively" is a platitude.
# A *concrete* access_list is named by actual brackets ("[0,1]", "[]"), which the
# "[" key below captures. We require the structure to be named, not just referenced.
_MECHANISM_KEYS = ("step", "chain", "tree", "best-of", "best of", "aggregat",
                   "[", "leaf", "leaves", "single", "parallel")


def is_useful_experience(line: str) -> bool:
    """A distilled experience is kept only if it names a concrete STRUCTURAL mechanism
    (a topology, a step count, a literal access_list like [0,1]) and isn't padded with
    vague filler. Mirrors the failure mode we observed where the 7B orchestrator emitted
    platitudes like 'ensure subtasks are verified properly' or 'use access_lists
    effectively' — note that neither 'verify' nor a bare 'access_list' mention is a
    structural mechanism (every workflow can claim both)."""
    if not line:
        return False
    lower = line.lower()
    # Reject the orchestrator's "no structural difference" signal, even when it's
    # embedded mid/end-of-line (observed: "... vs [[], 'all'] (NONE)").
    if "none" in lower and ("(none)" in lower or lower.startswith("none")
                            or lower.strip().endswith("none")):
        return False
    # Reject self-comparisons "X vs X" — comparing a thing to itself carries no signal
    # (observed: "access_list choice [[], 'all'] vs [[], 'all']").
    if " vs " in lower:
        a, b = lower.split(" vs ", 1)
        if _norm_exp(a) and _norm_exp(a) == _norm_exp(b):
            return False
    has_mechanism = any(k in lower for k in _MECHANISM_KEYS)
    vague_count = sum(lower.count(v) for v in _VAGUE_WORDS)
    return has_mechanism and vague_count < 2


def _norm_exp(line: str) -> str:
    """Normalize an experience to its STRUCTURAL SIGNATURE for near-duplicate and
    self-comparison detection. We keep only the tokens that carry structural meaning —
    bracket layout and digits — and drop all prose, quotes, verbs and filler. So:

      "Set access_list=[[], 'all'] instead of multiple empty lists."  ->  "[[]all]"
      "set access_list to [[],'all']"                                 ->  "[[]all]"
      "access_list choice [ [], 'all' ]"                              ->  "[[]all]"

    Two experiences whose structural core is identical collapse to the same key.
    """
    import re
    s = line.lower()
    # keep only brackets, digits, and the literal tokens 'all'/'none' that name a
    # concrete access_list value; strip everything else (prose, quotes, spaces, commas).
    s = s.replace("all", "\x00").replace("none", "\x01")   # protect value words
    s = re.sub(r"[^\[\]\d\x00\x01]", "", s)                # keep only structure
    s = s.replace("\x00", "all").replace("\x01", "none")
    return s


def dedup_experiences(experiences: List[str]) -> List[str]:
    """Drop near-duplicate experiences (keep first occurrence), e.g.
    'Set access_list=[[], all]' vs 'set access_list to [[],all]'."""
    seen, out = set(), []
    for e in experiences:
        k = _norm_exp(e)
        if k and k not in seen:
            seen.add(k)
            out.append(e)
    return out


@dataclass
class Rollout:
    workflow_repr: str
    final_answer: str
    reward: float
    n_calls: int


@dataclass
class TrainState:
    experiences: List[str] = field(default_factory=list)
    history: List[dict] = field(default_factory=list)  # per-iteration metrics

    def experience_block(self, max_items: int = 12) -> str:
        if not self.experiences:
            return ""
        items = self.experiences[-max_items:]
        return EXPERIENCE_HEADER + "\n".join(f"- {e}" for e in items) + "\n"

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"experiences": self.experiences, "history": self.history}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainState":
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            d = json.load(f)
        return cls(experiences=d.get("experiences", []), history=d.get("history", []))


class TrainingFreeGRPO:
    def __init__(self, orchestrator: Worker, pool: WorkerPool,
                 base_cfg: ConductorConfig, state: Optional[TrainState] = None):
        self.orchestrator = orchestrator
        self.pool = pool
        self.base_cfg = base_cfg
        self.state = state or TrainState()

    def _conductor(self, few_shot_extra: str) -> Conductor:
        """A Conductor whose few-shot block is augmented with learned experiences."""
        from ..prompts import DEFAULT_FEWSHOT
        cfg = ConductorConfig(
            max_steps=self.base_cfg.max_steps,
            max_recursion=0,                 # rollouts are single-shot for clean credit
            parse_retries=self.base_cfg.parse_retries,
            orchestrator_temperature=1.0,    # exploration during training
            orchestrator_max_tokens=self.base_cfg.orchestrator_max_tokens,
            fallback_worker=self.base_cfg.fallback_worker,
            few_shot=DEFAULT_FEWSHOT + few_shot_extra,
        )
        return Conductor(self.orchestrator, self.pool, cfg)

    async def _rollout(self, task: Task, conductor: Conductor) -> Rollout:
        trace = await conductor.run(task.prompt())
        wf = trace.workflows[0] if trace.workflows else None
        wf_repr = (f"model_id={wf.model_id} access_list={wf.access_list}" if wf else "<none>")
        reward = 1.0 if task.check(trace.final_answer) else 0.0
        return Rollout(workflow_repr=wf_repr, final_answer=extract_final(trace.final_answer),
                       reward=reward, n_calls=trace.total_worker_calls)

    async def _distill(self, task: Task, rollouts: List[Rollout],
                       client: httpx.AsyncClient) -> Optional[str]:
        """Ask the orchestrator LLM to extract one general experience from the group."""
        wins = [r for r in rollouts if r.reward > 0]
        losses = [r for r in rollouts if r.reward <= 0]
        if not wins or not losses:
            return None  # no contrast -> no signal (mirrors GRPO zero-advantage group)

        def fmt(rs):
            return "\n".join(f"  * {r.workflow_repr} -> FINAL={r.final_answer!r}" for r in rs[:4])

        prompt = (
            "You are improving an LLM ORCHESTRATOR that routes a question across worker "
            "models by emitting model_id/subtasks/access_list workflows.\n\n"
            f"Task domain: {task.domain}\nQuestion: {task.question}\n\n"
            f"WORKFLOWS THAT SUCCEEDED:\n{fmt(wins)}\n\n"
            f"WORKFLOWS THAT FAILED:\n{fmt(losses)}\n\n"
            "Compare the SUCCESSFUL vs FAILED workflows above. State the ONE concrete "
            "structural difference that made the successes win. Your rule MUST reference "
            "a specific mechanism: a number of steps, a topology (single/chain/tree/"
            "best-of-N), or a concrete access_list choice (e.g. 'aggregator reads [0,1]', "
            "'leaves use []'). Avoid vague words like 'properly', 'effectively', "
            "'thoroughly', 'appropriate'. Max 22 words, start with a verb. If there is no "
            "clear structural difference, reply exactly: NONE."
        )
        w = self.orchestrator
        headers = {"Content-Type": "application/json"}
        if w.api_key and w.api_key != "none":
            headers["Authorization"] = f"Bearer {w.api_key}"
        try:
            r = await client.post(f"{w.base_url.rstrip('/')}/chat/completions",
                                  json={"model": w.model, "temperature": 0.7, "max_tokens": 80,
                                        "messages": [{"role": "user", "content": prompt}]},
                                  headers=headers, timeout=w.timeout)
            r.raise_for_status()
            text = (r.json()["choices"][0]["message"]["content"] or "").strip()
            line = text.splitlines()[0].strip(" -*").strip() if text else ""
            return line if is_useful_experience(line) else None
        except Exception:  # noqa: BLE001
            return None

    async def iteration(self, tasks: List[Task], group_size: int = 4) -> dict:
        conductor = self._conductor(self.state.experience_block())
        total_reward, n = 0.0, 0
        new_exps: List[str] = []

        async with httpx.AsyncClient() as client:
            for task in tasks:
                rollouts = await asyncio.gather(*[self._rollout(task, conductor)
                                                  for _ in range(group_size)])
                grp_reward = sum(r.reward for r in rollouts) / len(rollouts)
                total_reward += grp_reward
                n += 1
                exp = await self._distill(task, list(rollouts), client)
                if exp and exp not in self.state.experiences and exp not in new_exps:
                    # near-duplicate guard: don't add a trivial rewording of one we have
                    existing_keys = {_norm_exp(e) for e in self.state.experiences + new_exps}
                    if _norm_exp(exp) not in existing_keys:
                        new_exps.append(exp)

        self.state.experiences.extend(new_exps)
        metrics = {"mean_group_reward": round(total_reward / max(n, 1), 4),
                   "n_tasks": n, "new_experiences": len(new_exps),
                   "total_experiences": len(self.state.experiences)}
        self.state.history.append(metrics)
        return metrics
