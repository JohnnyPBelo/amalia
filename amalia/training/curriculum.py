"""Deterministic curriculum tasks for stronger Amalia GRPO runs.

The original 39-task seed set is a smoke test. This module expands it into train / heldout
splits with more variety while staying fully verifiable (no judge model, no network).

Design goals:
- deterministic answer keys (tests can recompute/check them)
- train/heldout separation by numbers/templates
- enough format + execution signal to train orchestration without saturating at 97% immediately
"""
from __future__ import annotations

from typing import List

from .tasks import Task, comma_pair_eq, exact_num_set, exact_str_eq, num_eq, str_eq


def _fib(n: int) -> int:
    a, b = 1, 1
    if n <= 2:
        return 1
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // _gcd(a, b)


def _train_math() -> List[Task]:
    tasks: List[Task] = []
    for i, (a, b) in enumerate([(37, 64), (83, 29), (142, 57), (316, 23), (91, 88),
                                (204, 39), (127, 893), (512, 76), (909, 12), (65, 407)], 1):
        tasks.append(Task(f"T1_mul_{i}", f"Compute {a} * {b}.", num_eq(a * b), "math", tier="T1"))
    for i, (base, p1, p2) in enumerate([(1000, 30, 30), (480, 25, 12), (1250, 18, 40),
                                        (720, 15, 20), (999, 33, 10)], 1):
        ans = base * (p1 / 100) * (p2 / 100)
        tasks.append(Task(f"T1_pct_nested_{i}", f"What is {p2}% of {p1}% of {base}?", num_eq(ans), "math", tier="T1"))
    for i, (price, d1, d2) in enumerate([(80, 25, 10), (150, 20, 15), (240, 35, 8), (99, 30, 20)], 1):
        ans = price * (1 - d1 / 100) * (1 - d2 / 100)
        tasks.append(Task(f"T1_discount_{i}",
                          f"An item costs ${price}. It is discounted {d1}%, then a further {d2}% is taken off the reduced price. What is the final price in dollars?",
                          num_eq(ans, tol=1e-2), "reasoning", tier="T1"))
    for i, (a, b) in enumerate([(252, 105), (462, 1078), (144, 360), (1001, 143)], 1):
        tasks.append(Task(f"T1_gcd_{i}", f"What is the greatest common divisor of {a} and {b}?", num_eq(_gcd(a, b)), "math", tier="T1"))
        tasks.append(Task(f"T1_lcm_{i}", f"What is the least common multiple of {a} and {b}?", num_eq(_lcm(a, b)), "math", tier="T1"))
    for n in [12, 13, 14, 15, 16, 17, 18]:
        tasks.append(Task(f"T1_fib_{n}", f"What is the {n}th Fibonacci number, counting 1, 1, 2, 3, 5, ...?", num_eq(_fib(n)), "reasoning", tier="T1"))
    return tasks


def _train_word_reasoning() -> List[Task]:
    return [
        Task("T2_age_1", "Alice is 4 times as old as Bob. In 6 years, Alice will be twice as old as Bob. How old is Bob now?", num_eq(3), "reasoning", tier="T2"),
        Task("T2_age_2", "Mia is 5 years older than Leo. In 3 years, Mia will be twice Leo's age. How old is Leo now?", num_eq(2), "reasoning", tier="T2"),
        Task("T2_coins_1", "You have 34 coins, all nickels and quarters, worth $5.50 total. How many quarters do you have?", num_eq(19), "reasoning", tier="T2"),
        Task("T2_coins_2", "You have 41 coins, all dimes and quarters, worth $6.50 total. How many quarters do you have?", num_eq(16), "reasoning", tier="T2"),
        Task("T2_trains_1", "Two vehicles are 420 km apart and travel toward each other at 80 km/h and 60 km/h. After how many hours do they meet?", num_eq(3), "reasoning", tier="T2"),
        Task("T2_work_1", "If 6 workers finish a job in 10 days, how many days do 4 workers need at the same rate?", num_eq(15), "reasoning", tier="T2"),
        Task("T2_mixture_1", "A 20 liter solution is 30% salt. How many liters of pure water must be added to make it 20% salt?", num_eq(10), "reasoning", tier="T2"),
        Task("T2_handshake_12", "In a room of 12 people, everyone shakes hands exactly once with every other person. How many handshakes occur?", num_eq(66), "reasoning", tier="T2"),
        Task("T2_choose_4_9", "How many ways are there to choose 4 items from 9 distinct items?", num_eq(126), "reasoning", tier="T2"),
        Task("T2_paths_grid", "How many shortest paths are there from the bottom-left to the top-right of a 3 by 4 grid if you can only move right or up?", num_eq(35), "reasoning", tier="T2"),
    ]


def _train_code_string() -> List[Task]:
    return [
        Task("T2_reverse_orchestrate", "Reverse the string 'orchestrate'.", exact_str_eq("etartsehcro"), "code", tier="T2"),
        Task("T2_count_r_strawberry", "How many times does the letter 'r' appear in 'strawberry'?", num_eq(3), "code", tier="T2"),
        Task("T2_distinct_banana", "How many distinct characters are in the string 'banana'?", num_eq(3), "code", tier="T2"),
        Task("T2_substring_signal", "Does the word 'misconfiguration' contain the substring 'config'? Answer yes or no.", str_eq("yes"), "code", tier="T2"),
        Task("T2_anagram_evils", "Are 'vile' and 'evil' anagrams of each other? Answer yes or no.", str_eq("yes"), "code", tier="T2"),
        Task("T2_balanced_bad", "Are the brackets in the string '([)]' balanced? Answer yes or no.", str_eq("no"), "code", tier="T2"),
        Task("T2_two_sum_2", "Given nums=[3,2,4] and target=6, return the two 0-based indices whose values sum to target, as a comma-separated pair like 1,2.", comma_pair_eq(1, 2), "code", tier="T2"),
        Task("T2_python_result_1", "In Python, what is the value of len(set('abracadabra'))?", num_eq(5), "code", tier="T2"),
    ]


def _heldout() -> List[Task]:
    return [
        Task("H_mul_1", "Compute 289 * 347.", num_eq(100283), "math", tier="H"),
        Task("H_mul_2", "Compute 671 * 84.", num_eq(56364), "math", tier="H"),
        Task("H_discount", "A jacket costs $220. It is discounted 15%, then another 20% off the reduced price. What is the final price?", num_eq(149.6, tol=1e-2), "reasoning", tier="H"),
        Task("H_age", "Nora is 2 times as old as Eli. In 9 years, Nora will be 1.5 times Eli's age. How old is Eli now?", num_eq(9), "reasoning", tier="H"),
        Task("H_coins", "You have 50 coins, all nickels and dimes, worth $3.60. How many dimes do you have?", num_eq(22), "reasoning", tier="H"),
        Task("H_work", "If 9 workers complete a task in 8 days, how many days do 6 workers need?", num_eq(12), "reasoning", tier="H"),
        Task("H_gcd", "What is the greatest common divisor of 714 and 168?", num_eq(42), "math", tier="H"),
        Task("H_lcm", "What is the least common multiple of 21 and 34?", num_eq(714), "math", tier="H"),
        Task("H_fib", "What is the 19th Fibonacci number, counting 1, 1, 2, 3, 5, ...?", num_eq(4181), "reasoning", tier="H"),
        Task("H_paths", "How many shortest paths are there across a 4 by 4 grid from one corner to the opposite corner, moving only right or up?", num_eq(70), "reasoning", tier="H"),
        Task("H_reverse", "Reverse the string 'verification'.", exact_str_eq("noitacifirev"), "code", tier="H"),
        Task("H_distinct", "How many distinct characters are in the string 'committee'?", num_eq(6), "code", tier="H"),
        Task("H_substring", "Does the word 'counterexample' contain the substring 'example'? Answer yes or no.", str_eq("yes"), "code", tier="H"),
        Task("H_balanced", "Are the brackets in the string '{[()]}' balanced? Answer yes or no.", str_eq("yes"), "code", tier="H"),
        Task("H_primes", "List all prime numbers strictly between 80 and 100.", exact_num_set([83, 89, 97]), "reasoning", tier="H"),
    ]


def get_curriculum_tasks(split: str = "train") -> List[Task]:
    """Return deterministic curriculum tasks.

    split:
      train   -> training curriculum (seed set excluded by caller if desired)
      heldout -> never used for training; use for checkpoint selection
      all     -> train + heldout
    """
    train = _train_math() + _train_word_reasoning() + _train_code_string()
    heldout = _heldout()
    if split == "train":
        return train
    if split == "heldout":
        return heldout
    if split == "all":
        return train + heldout
    raise ValueError(f"unknown curriculum split: {split!r}")
