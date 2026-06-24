"""Tests for the training task checkers + final-answer extraction. No network."""
from amalia.training.tasks import extract_final, SEED_TASKS, comma_pair_eq, exact_num_set, num_eq, str_eq, all_in


def test_extract_final_basic():
    assert extract_final("blah\nFINAL: 42") == "42"


def test_extract_final_last_wins():
    assert extract_final("FINAL: 1\nmore\nFINAL: 2") == "2"


def test_extract_final_unwraps_boxed():
    assert extract_final(r"FINAL: \boxed{0,1}") == "0,1"


def test_extract_final_strips_latex_and_markdown():
    assert extract_final(r"FINAL: \[ 36 \]") == "36"
    assert extract_final("FINAL: **yes**") == "yes"
    assert extract_final("FINAL: `0,1`") == "0,1"


def test_extract_final_fallback_last_line():
    assert extract_final("no marker here\nthe answer is 5") == "the answer is 5"


def test_num_eq_finds_number():
    assert num_eq(1081)("FINAL: 1081") is True
    assert num_eq(1081)("FINAL: 1000") is False


def test_num_eq_ignores_commas_in_thousands():
    assert num_eq(1081)("FINAL: 1,081") is True


def test_str_eq_substring():
    assert str_eq("yes")("FINAL: Yes, balanced") is True
    assert str_eq("yes")("FINAL: no") is False


def test_yes_no_str_eq_uses_word_boundaries():
    assert str_eq("yes")("FINAL: yesterday") is False
    assert str_eq("no")("FINAL: unknown") is False
    assert str_eq("no")("FINAL: no") is True


def test_exact_pair_and_set_checkers_reject_extras():
    assert comma_pair_eq(1, 2)("FINAL: 1,2") is True
    assert comma_pair_eq(1, 2)("FINAL: 11,23") is False
    assert exact_num_set([83, 89, 97])("FINAL: 83, 89, 97") is True
    assert exact_num_set([83, 89, 97])("FINAL: 83, 89, 91, 97") is False


def test_all_in_requires_every_item():
    chk = all_in(["53", "59", "61", "67"])
    assert chk("FINAL: 53, 59, 61, 67") is True
    assert chk("FINAL: 53, 59, 61") is False


def test_seed_tasks_have_unique_ids():
    ids = [t.id for t in SEED_TASKS]
    assert len(ids) == len(set(ids))


def test_two_sum_boxed_now_passes():
    """Regression: the baseline run failed two_sum only due to \\boxed wrapping."""
    t = {x.id: x for x in SEED_TASKS}["two_sum"]
    assert t.check(r"FINAL: \boxed{0,1}") is True
