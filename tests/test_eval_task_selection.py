"""Unit tests for eval task-source selection (no model load)."""
import importlib.util
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "eval_lora_policy", Path(__file__).resolve().parents[1] / "scripts" / "eval_lora_policy.py"
)
_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_eval)


def test_select_heldout_ids_filters_non_seed_sources():
    tasks = _eval.select_tasks("heldout", "H_mul_1,H_reverse")
    assert [t.id for t in tasks] == ["H_mul_1", "H_reverse"]


def test_select_task_limit_applies_after_ids():
    tasks = _eval.select_tasks("seed+heldout", "mul1,H_mul_1,H_reverse", limit=2)
    assert [t.id for t in tasks] == ["mul1", "H_mul_1"]


def test_select_unknown_id_raises():
    with pytest.raises(ValueError, match="unknown task id"):
        _eval.select_tasks("heldout", "mul1")
