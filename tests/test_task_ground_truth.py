"""Independently verify every task's ground-truth answer.

A wrong answer key here is *silent poison*: the model would be punished for being
right. So we recompute each expected answer from first principles (NOT by importing
the task's stored value) and assert the task's own checker accepts it.
"""
from math import comb, factorial, gcd

import pytest

from amalia.training.tasks import get_tasks

_TASKS = {t.id: t for t in get_tasks()}


def _fib(n: int) -> int:
    """nth Fibonacci (1-indexed) with the 1,1,2,3,5,... convention used in the
    task prompts: term 1 = 1, term 2 = 1, term 3 = 2, ... term 15 = 610."""
    seq = [1, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[n - 1]


# id -> a string the checker MUST accept, computed independently of the task file.
EXPECTED = {
    "mul1": str(23 * 47),
    "mul2": str(17 * 19 * 3),
    "bat_ball": "0.05",
    "speed": "80",
    "pct": str(0.15 * 240),
    "primes_50_70": "53, 59, 61, 67",
    "gcd": str(gcd(48, 180)),
    "fib": str(_fib(10)),
    "palindrome": "yes",
    "two_sum": "0,1",
    "balanced": "yes",
    "count_vowels": str(sum("orchestration".count(v) for v in "aeiou")),
    "mul_big": str(127 * 893),
    "compound": str((45 + 67) * 12 - 89),
    "digit_sum": str(sum(int(d) for d in "9876")),
    "seconds_2_5h": str(int(2.5 * 3600)),
    "area_rect": str(12 * 7),
    "avg_4": str(int((10 + 20 + 30 + 40) / 4)),
    "pct_nested": str(int(0.3 * 0.3 * 1000)),
    "age": "5",
    "coins": str(int((205 - 5 * 26) / 5)),
    "discount": str(int(80 * 0.75 * 0.9)),
    "trains_meet": str(int(300 / (60 + 90))),
    "work_days": str(int(4 * 6 / 3)),
    "lcm": str(12 * 18 // gcd(12, 18)),
    "factorial6": str(factorial(6)),
    "sum_1_100": str(sum(range(1, 101))),
    "primes_below_20": str(len([n for n in range(2, 20) if all(n % d for d in range(2, n))])),
    "fib15": str(_fib(15)),
    "gcd_big": str(gcd(252, 105)),
    "handshakes": str(comb(10, 2)),
    "choose_3_7": str(comb(7, 3)),
    "arrange_cat": str(factorial(3)),
    "pairs_5": str(comb(5, 2)),
    "reverse_hello": "olleh",
    "anagram": "yes",
    "count_a_banana": str("banana".count("a")),
    "substring_gram": "yes",
    "distinct_mississippi": str(len(set("mississippi"))),
}


def test_every_task_has_a_ground_truth_check():
    """No task may ship without an independent verification entry."""
    missing = sorted(set(_TASKS) - set(EXPECTED))
    assert not missing, f"tasks lacking an independent ground-truth check: {missing}"


@pytest.mark.parametrize("task_id", sorted(EXPECTED))
def test_checker_accepts_correct_answer(task_id):
    task = _TASKS[task_id]
    answer = EXPECTED[task_id]
    # the model ends with 'FINAL: <answer>'; simulate that exact shape
    assert task.check(f"FINAL: {answer}"), (
        f"task {task_id!r}: checker rejected the verified-correct answer {answer!r}"
    )


@pytest.mark.parametrize("task_id", sorted(EXPECTED))
def test_checker_rejects_wrong_answer(task_id):
    """Sanity: the checker must not accept an obviously wrong sentinel."""
    task = _TASKS[task_id]
    # 'zzznope' is wrong for every string task; -987654 is wrong for every numeric one
    assert not task.check("FINAL: zzznope")
    assert not task.check("FINAL: -987654")
