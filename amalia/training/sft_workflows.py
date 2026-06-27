"""Supervised workflow examples for conductor-policy warmup.

These are not answer demonstrations. They teach the Qwen policy to emit compact,
parseable Conductor workflows for the fixed 3-worker Amalia stack:

- Model 0: general/coding
- Model 1: math/calculation
- Model 2: verification
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..prompts import DEFAULT_FEWSHOT, build_conductor_prompt
from .curriculum import get_curriculum_tasks
from .tasks import Task, get_tasks

SFT_POOL_LISTING = (
    "Model 0: skills = general problem solving, coding, step-by-step reasoning\n"
    "Model 1: skills = arithmetic, math, number theory, precise calculation\n"
    "Model 2: skills = verification, checking answers, catching errors"
)
SFT_MAX_STEPS = 5


@dataclass(frozen=True)
class SFTWorkflowExample:
    task: Task
    prompt: str
    completion: str


def make_sft_prompt(task: Task) -> str:
    return build_conductor_prompt(
        user_question=task.question,
        available_models=SFT_POOL_LISTING,
        max_steps=SFT_MAX_STEPS,
        few_shot=DEFAULT_FEWSHOT,
    )


def _completion(model_ids: List[int], subtasks: List[str], access_list: List[list[int]]) -> str:
    """Render a canonical single-code-block workflow completion."""
    return (
        "Use a compact workflow with one solver and one verifier.\n"
        "```python\n"
        f"model_id = {model_ids!r}\n"
        f"subtasks = {subtasks!r}\n"
        f"access_list = {access_list!r}\n"
        "```"
    )


def workflow_for_task(task: Task) -> SFTWorkflowExample:
    """Return a hand-authored canonical workflow for a verifiable task."""
    if task.domain == "code":
        model_ids = [0, 2]
        subtasks = [
            "Solve the programming/string task directly. End with FINAL: <answer>.",
            "Verify the previous answer against the original question. If correct, repeat it exactly as FINAL: <answer>; otherwise correct it and end with FINAL: <answer>.",
        ]
    else:
        model_ids = [1, 2]
        subtasks = [
            "Solve the math/reasoning problem carefully. Show concise calculation and end with FINAL: <answer>.",
            "Independently verify the previous calculation and the final answer. If correct, repeat it exactly as FINAL: <answer>; otherwise correct it and end with FINAL: <answer>.",
        ]
    access_list = [[], [0]]
    return SFTWorkflowExample(
        task=task,
        prompt=make_sft_prompt(task),
        completion=_completion(model_ids, subtasks, access_list),
    )


def get_sft_examples(task_source: str = "seed+curriculum") -> List[SFTWorkflowExample]:
    if task_source == "seed":
        tasks = get_tasks()
    elif task_source == "curriculum":
        tasks = get_curriculum_tasks("train")
    elif task_source == "seed+curriculum":
        tasks = get_tasks() + get_curriculum_tasks("train")
    else:
        raise ValueError(f"unknown SFT task source: {task_source!r}")
    return [workflow_for_task(t) for t in tasks]


def example_to_text(example: SFTWorkflowExample) -> str:
    """Flatten prompt+completion for causal-LM SFT."""
    return example.prompt.rstrip() + "\n\n" + example.completion.strip()


def build_sft_records(task_source: str = "seed+curriculum") -> List[dict]:
    return [
        {
            "id": ex.task.id,
            "domain": ex.task.domain,
            "tier": ex.task.tier,
            "text": example_to_text(ex),
        }
        for ex in get_sft_examples(task_source)
    ]
