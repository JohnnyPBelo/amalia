"""Tests for the experience-quality filter (the fix for vague distillations)."""
from amalia.training.grpo_free import (
    is_useful_experience, dedup_experiences, _norm_exp, TrainState,
)


def test_rejects_embedded_none():
    # observed leak from the frontier run: "(NONE)" embedded mid/end-of-line
    assert is_useful_experience("access_list choice [ [], 'all' ] vs [ [], 'all' ] (NONE)") is False
    assert is_useful_experience("Use a single step (NONE)") is False
    assert is_useful_experience("no clear structural difference, none") is False


def test_rejects_self_comparison():
    # comparing a thing to itself carries no signal (observed real leak)
    assert is_useful_experience("access_list choice [ [], 'all' ] vs [ [], 'all' ]") is False
    assert is_useful_experience("used [0,1] vs used [0,1]") is False
    # but a genuine A-vs-B contrast with a mechanism is still kept
    assert is_useful_experience("Used 'all' in access_list vs used []") is True


def test_dedup_near_duplicates():
    # the two real ones from the frontier run that differ only by wording
    exps = [
        "Set access_list=[[], 'all'] instead of multiple empty lists.",
        "set access_list to [[],'all']",
        "Use a tree: two leaves then aggregator reading [0,1]",
    ]
    out = dedup_experiences(exps)
    assert len(out) == 2, out
    assert out[0].startswith("Set access_list")
    assert "tree" in out[2 - 1]


def test_norm_exp_collapses_wording():
    assert _norm_exp("Set access_list=[[], 'all']") == _norm_exp("set access_list to [[],'all']")


def test_rejects_none():
    assert is_useful_experience("NONE") is False
    assert is_useful_experience("none, no clear difference") is False


def test_rejects_empty():
    assert is_useful_experience("") is False


def test_rejects_vague_platitudes():
    # the exact failure mode observed in the first training run
    assert is_useful_experience("Ensure all subtasks are decomposed and verified properly") is False
    assert is_useful_experience("Verify all subtasks and access required information thoroughly") is False
    assert is_useful_experience("Ensure models access necessary data correctly and accurately") is False


def test_rejects_bare_access_list_mention():
    # the subtle hole: "access_list" is the whole domain, so mentioning it without
    # naming a concrete bracket is still a platitude. These 4 leaked through the
    # first filter and must now be dropped.
    assert is_useful_experience("Use specific access_list settings to route and verify subtasks effectively") is False
    assert is_useful_experience("Ensure access_list validation matches across subtasks for correctness verification") is False
    assert is_useful_experience("Ensure access_lists restrict model access to necessary information for accurate verification") is False
    assert is_useful_experience("Ensure consistent access_list usage across models for accurate verification and decomposition") is False


def test_accepts_concrete_mechanism():
    assert is_useful_experience("Add a verifier step that reads access_list [0] to recheck the answer") is True
    assert is_useful_experience("Use a tree: two independent leaves then an aggregator reading [0,1]") is True
    assert is_useful_experience("Route arithmetic to a single step; chains added errors") is True


def test_accepts_mechanism_even_with_one_vague_word():
    assert is_useful_experience("Add a verifier step to properly recheck arithmetic") is True


def test_experience_block_formatting():
    s = TrainState(experiences=["Add a verifier step reading [0]", "Use single-step for arithmetic"])
    block = s.experience_block()
    assert "LEARNED ORCHESTRATION EXPERIENCES" in block
    assert "- Add a verifier step reading [0]" in block
    assert "- Use single-step for arithmetic" in block


def test_experience_block_empty_when_none():
    assert TrainState().experience_block() == ""
