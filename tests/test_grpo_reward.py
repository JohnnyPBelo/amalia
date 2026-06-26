"""Unit tests for GRPO reward profile scoring."""

import pytest

from amalia.training.grpo_real import score_exec_result, score_format_result


def test_binary_reward_profile_preserves_legacy_signal():
    assert score_exec_result(True, n_worker_calls=5, model_ids=[0, 1, 2], profile="binary") == 1.0
    assert score_exec_result(False, n_worker_calls=1, model_ids=[0], profile="binary") == -0.1


def test_beyond_fugu_v1_rewards_correct_minimal_verified_workflow():
    assert score_exec_result(True, n_worker_calls=2, model_ids=[0, 2], profile="beyond_fugu_v1") == pytest.approx(1.15)


def test_beyond_fugu_v1_penalizes_extra_calls_but_keeps_correctness_dominant():
    assert score_exec_result(True, n_worker_calls=5, model_ids=[0, 1, 2, 0, 2], profile="beyond_fugu_v1") == pytest.approx(0.98)


def test_beyond_fugu_v1_penalizes_failed_extra_calls():
    assert score_exec_result(False, n_worker_calls=4, model_ids=[0, 1, 2, 0], profile="beyond_fugu_v1") == pytest.approx(-0.33)


def test_unknown_reward_profile_raises():
    with pytest.raises(ValueError, match="unknown reward profile"):
        score_exec_result(True, n_worker_calls=1, model_ids=[0], profile="wat")


def test_strict_format_profile_keeps_valid_workflow_at_full_reward():
    text = 'model_id = [1, 2]\nsubtasks = ["solve", "verify"]\naccess_list = [[], [0]]'
    assert score_format_result(text, profile="strict_v2") == 1.0


def test_strict_format_profile_penalizes_model_metadata_ramble():
    text = "Model 3: skills = advanced math\nModel 4: skills = random\n[1, 2, 3] [4, 5, 6] [7, 8, 9]"
    assert score_format_result(text, profile="legacy") == 0.5
    assert score_format_result(text, profile="strict_v2") == -0.2


def test_strict_format_profile_gives_tiny_breadcrumb_for_contract_like_bad_output():
    text = 'model_id = [0, 2]\nsubtasks = ["solve", "verify"]\naccess_list = [[bad], [0]]'
    assert score_format_result(text, profile="strict_v2") == pytest.approx(0.15)
