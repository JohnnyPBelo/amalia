"""Tests for supervised workflow warmup examples."""

from amalia.parser import parse_workflow
from amalia.training.sft_workflows import build_sft_records, get_sft_examples, workflow_for_task
from amalia.training.tasks import Task, num_eq, str_eq


def test_sft_examples_cover_seed_plus_train_curriculum_without_heldout():
    examples = get_sft_examples("seed+curriculum")
    ids = {ex.task.id for ex in examples}
    assert len(examples) >= 90
    assert "mul1" in ids
    assert "T1_mul_1" in ids
    assert "H_mul_1" not in ids


def test_sft_completions_parse_into_valid_nonempty_workflows():
    for ex in get_sft_examples("seed+curriculum"):
        wf = parse_workflow(ex.completion, n_models=3, max_steps=5)
        assert not wf.is_empty(), ex.task.id
        assert all(model_id in {0, 1, 2} for model_id in wf.model_id), ex.task.id
        assert len(wf.model_id) == len(wf.subtasks) == len(wf.access_list)


def test_sft_routes_math_and_reasoning_to_math_then_verifier():
    task = Task("math_probe", "Compute 123 * 456.", num_eq(56088), "math")
    ex = workflow_for_task(task)
    wf = parse_workflow(ex.completion, n_models=3, max_steps=5)
    assert wf.model_id == [1, 2]
    assert wf.access_list == [[], [0]]


def test_sft_routes_code_to_general_then_verifier():
    task = Task("code_probe", "Is 'level' a palindrome? Answer yes or no.", str_eq("yes"), "code")
    ex = workflow_for_task(task)
    wf = parse_workflow(ex.completion, n_models=3, max_steps=5)
    assert wf.model_id == [0, 2]
    assert wf.access_list == [[], [0]]


def test_sft_completion_is_single_python_code_block_and_tells_final_worker_to_final():
    ex = get_sft_examples("seed")[0]
    assert ex.completion.count("```python") == 1
    assert ex.completion.rstrip().endswith("```")
    assert "FINAL: <answer>" in ex.completion


def test_build_sft_records_flattens_prompt_and_completion_text():
    records = build_sft_records("seed")
    assert records
    row = records[0]
    assert {"id", "domain", "tier", "text"}.issubset(row)
    assert "AVAILABLE LANGUAGE MODELS" in row["text"]
    assert "```python" in row["text"]
    assert "access_list" in row["text"]
