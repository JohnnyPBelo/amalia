"""
Workflow parser — extracts the Conductor's three Python lists (model_id, subtasks,
access_list) from its completion.

The paper's "format reward" sets r=0 when these three lists cannot be parsed. We mirror
that: parsing either succeeds into a validated Workflow, or raises WorkflowParseError
(which the engine treats as a failed/empty workflow and can retry or fall back).

Robustness notes:
  * The orchestrator emits a chain-of-thought THEN the lists, usually inside a ```python
    block. It may also continue with stray `Human:` turns or repeated/empty code blocks
    after a valid workflow. We scan python blocks from latest to earliest and return
    the first one that validates; fall back to scanning raw text.
  * We use ast.literal_eval (never exec/eval) so a malicious or malformed completion
    cannot run code.
  * access_list entries may be: []  |  ["all"]  |  [int, int, ...].
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List, Union

AccessEntry = Union[List[int], str]  # a list of step indices, or the literal "all"


class WorkflowParseError(ValueError):
    """Raised when the three lists cannot be parsed/validated (paper's r=0 condition)."""


@dataclass
class Workflow:
    model_id: List[int]
    subtasks: List[str]
    access_list: List[AccessEntry]
    raw: str = field(default="", repr=False)

    @property
    def n_steps(self) -> int:
        return len(self.model_id)

    def is_empty(self) -> bool:
        """Recursion 'return as-is' signal: three empty lists."""
        return self.n_steps == 0 and not self.subtasks and not self.access_list


_PY_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
_NAME_EQ = {name: re.compile(rf"{name}\s*=\s*\[") for name in ("model_id", "subtasks", "access_list")}


def _extract_bracketed(snippet: str, name: str) -> str:
    """Find `name = [ ... ]` and return the FULL balanced-bracket literal.

    A non-greedy regex breaks on nested lists like [[], ["all"]] (stops at the first
    ']'). We instead locate the opening '[' after the name and scan forward tracking
    bracket depth, while respecting string literals so a ']' inside a subtask string
    doesn't fool the counter.
    """
    m = _NAME_EQ[name].search(snippet)
    if not m:
        raise WorkflowParseError(f"could not find `{name}` assignment")
    start = m.end() - 1  # index of the opening '['
    depth = 0
    in_str = None  # current quote char if inside a string
    esc = False
    for i in range(start, len(snippet)):
        ch = snippet[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return snippet[start:i + 1]
    raise WorkflowParseError(f"`{name}` bracket never closed")


def _literal(snippet: str, name: str):
    literal_str = _extract_bracketed(snippet, name)
    try:
        return ast.literal_eval(literal_str)
    except (ValueError, SyntaxError) as e:
        raise WorkflowParseError(f"`{name}` is not a valid Python literal: {e}") from e


def _parse_candidate(search_space: str, completion: str, n_models: int, max_steps: int) -> Workflow:
    model_id = _literal(search_space, "model_id")
    subtasks = _literal(search_space, "subtasks")
    access_list = _literal(search_space, "access_list")

    # --- structural validation -------------------------------------------------
    if not (isinstance(model_id, list) and isinstance(subtasks, list) and isinstance(access_list, list)):
        raise WorkflowParseError("all three must be lists")

    # Empty workflow (recursion 'return as-is') is valid.
    if len(model_id) == 0 and len(subtasks) == 0 and len(access_list) == 0:
        return Workflow([], [], [], raw=completion)

    if not (len(model_id) == len(subtasks) == len(access_list)):
        raise WorkflowParseError(
            f"length mismatch: model_id={len(model_id)} subtasks={len(subtasks)} "
            f"access_list={len(access_list)}"
        )

    if len(model_id) > max_steps:
        raise WorkflowParseError(f"{len(model_id)} steps exceeds max_steps={max_steps}")

    norm_access: List[AccessEntry] = []
    for i, (mid, sub, acc) in enumerate(zip(model_id, subtasks, access_list)):
        if not isinstance(mid, int):
            raise WorkflowParseError(f"step {i}: model_id must be int, got {type(mid).__name__}")
        if not (0 <= mid < n_models):
            raise WorkflowParseError(f"step {i}: model_id {mid} out of range [0,{n_models})")
        if not isinstance(sub, str) or not sub.strip():
            raise WorkflowParseError(f"step {i}: subtask must be a non-empty string")

        # access entry: "all" | ["all"] | [ints]
        if acc == "all" or acc == ["all"]:
            norm_access.append("all")
        elif isinstance(acc, list):
            if len(acc) == 1 and acc[0] == "all":
                norm_access.append("all")
            else:
                for ref in acc:
                    if not isinstance(ref, int) or not (0 <= ref < i):
                        raise WorkflowParseError(
                            f"step {i}: access ref {ref!r} must be an int index of a PRIOR step (0..{i-1})"
                        )
                norm_access.append(list(acc))
        else:
            raise WorkflowParseError(f"step {i}: access_list entry must be a list or 'all', got {acc!r}")

    return Workflow(model_id=list(model_id), subtasks=list(subtasks),
                    access_list=norm_access, raw=completion)


def parse_workflow(completion: str, n_models: int, max_steps: int = 5) -> Workflow:
    """Parse + validate a Conductor completion into a Workflow.

    Raises WorkflowParseError on any structural problem (the paper's format-reward=0 case).
    """
    # Prefer the latest VALID python code block. The model often emits a good workflow
    # and then continues with stray `Human:` turns, repeated examples, or an empty
    # ```python block. Trying only the last block turns those continuations into false
    # parse failures; trying every block keeps the format reward aligned with intent.
    blocks = _PY_BLOCK.findall(completion)
    errors = []
    for block in reversed(blocks):
        try:
            return _parse_candidate(block, completion, n_models, max_steps)
        except WorkflowParseError as e:
            errors.append(str(e))

    # No valid fenced block — fall back to scanning raw text.
    try:
        return _parse_candidate(completion, completion, n_models, max_steps)
    except WorkflowParseError as e:
        if errors:
            raise WorkflowParseError(f"no valid workflow block found; latest errors: {errors[:3]}") from e
        raise
