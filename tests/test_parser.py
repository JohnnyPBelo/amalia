"""Parser unit tests — no network. Run: pytest tests/test_parser.py -v"""
import pytest
from amalia.parser import parse_workflow, WorkflowParseError


def test_chain_with_nested_lists():
    c = '```python\nmodel_id = [2, 0]\nsubtasks = ["a", "b"]\naccess_list = [[], ["all"]]\n```'
    w = parse_workflow(c, n_models=3)
    assert w.model_id == [2, 0]
    assert w.access_list == [[], "all"]
    assert w.n_steps == 2


def test_tree_topology():
    c = '```python\nmodel_id = [0, 1, 2]\nsubtasks = ["x", "y", "z"]\naccess_list = [[], [], [0, 1]]\n```'
    w = parse_workflow(c, n_models=3)
    assert w.access_list == [[], [], [0, 1]]


def test_empty_workflow_is_recursion_return():
    w = parse_workflow("model_id = []\nsubtasks = []\naccess_list = []", n_models=3)
    assert w.is_empty()


def test_prefers_last_python_block():
    c = ('Here is an example:\n```python\nmodel_id = [9]\n```\n'
         'Now my answer:\n```python\nmodel_id = [1]\nsubtasks = ["go"]\naccess_list = [[]]\n```')
    w = parse_workflow(c, n_models=3)
    assert w.model_id == [1]


def test_bracket_inside_string_does_not_fool_parser():
    c = 'model_id=[0]\nsubtasks=["return a list like [1,2,3] please"]\naccess_list=[[]]'
    w = parse_workflow(c, n_models=3)
    assert w.subtasks == ["return a list like [1,2,3] please"]


def test_no_code_block_raw_scan():
    c = 'model_id = [0, 1]\nsubtasks = ["a", "b"]\naccess_list = [[], ["all"]]'
    w = parse_workflow(c, n_models=2)
    assert w.n_steps == 2


@pytest.mark.parametrize("c,n,why", [
    ('model_id=[5]\nsubtasks=["x"]\naccess_list=[[]]', 3, "out of range"),
    ('model_id=[0,1]\nsubtasks=["a","b"]\naccess_list=[[1],[]]', 3, "forward ref"),
    ('model_id=[0,1]\nsubtasks=["a"]\naccess_list=[[]]', 3, "length mismatch"),
    ('model_id=[0]\nsubtasks=[""]\naccess_list=[[]]', 3, "empty subtask"),
    ('subtasks=["a"]\naccess_list=[[]]', 3, "missing model_id"),
    ('model_id=[0,1,2,3,4,5]\nsubtasks=["a","b","c","d","e","f"]\naccess_list=[[],[],[],[],[],[]]', 3, "exceeds max_steps"),
])
def test_invalid_workflows_rejected(c, n, why):
    with pytest.raises(WorkflowParseError):
        parse_workflow(c, n_models=n, max_steps=5)


def test_all_as_bare_string_normalized():
    c = 'model_id=[0,1]\nsubtasks=["a","b"]\naccess_list=[[], "all"]'
    w = parse_workflow(c, n_models=2)
    assert w.access_list == [[], "all"]
