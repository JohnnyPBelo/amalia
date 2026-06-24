"""Tests for the expanded deterministic curriculum task set."""
from amalia.training.curriculum import get_curriculum_tasks


def test_curriculum_splits_nonempty_and_disjoint():
    train = get_curriculum_tasks("train")
    heldout = get_curriculum_tasks("heldout")
    assert len(train) >= 40
    assert len(heldout) >= 10
    train_ids = {t.id for t in train}
    heldout_ids = {t.id for t in heldout}
    assert train_ids.isdisjoint(heldout_ids)
    assert len(train_ids) == len(train)
    assert len(heldout_ids) == len(heldout)


def test_curriculum_answer_keys_accept_obvious_final_values():
    # Spot-check generated deterministic answers and checker normalization.
    by_id = {t.id: t for t in get_curriculum_tasks("all")}
    assert by_id["T1_mul_1"].check("FINAL: 2368")
    assert by_id["T1_gcd_1"].check("FINAL: 21")
    assert by_id["T1_lcm_1"].check("FINAL: 1260")
    assert by_id["T1_fib_18"].check("FINAL: 2584")
    assert by_id["H_mul_1"].check("FINAL: 100{,}283")
    assert by_id["H_discount"].check("FINAL: $149.60")
    assert by_id["H_reverse"].check("FINAL: noitacifirev")


def test_curriculum_rejects_wrong_sentinels():
    by_id = {t.id: t for t in get_curriculum_tasks("all")}
    assert by_id["T2_balanced_bad"].check("FINAL: unknown") is False
    assert by_id["T2_substring_signal"].check("FINAL: yesterday") is False
    assert by_id["T2_anagram_evils"].check("FINAL: yesterday") is False
    assert by_id["T2_two_sum_2"].check("FINAL: 11,23") is False
    assert by_id["H_primes"].check("FINAL: 83, 89, 91, 97") is False


def test_curriculum_domains_and_tiers_present():
    tasks = get_curriculum_tasks("all")
    domains = {t.domain for t in tasks}
    tiers = {t.tier for t in tasks}
    assert {"math", "reasoning", "code"}.issubset(domains)
    assert {"T1", "T2", "H"}.issubset(tiers)
