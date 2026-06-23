"""
Verifiable task set for Training-Free GRPO.

Each Task carries a `check(final_answer_text) -> bool`. Tasks instruct the system to
end with a `FINAL: <answer>` line so the reward is computable without a judge model.
The checker is tolerant: it scans the FINAL line first, then the whole text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List

FINAL_INSTRUCTION = "\n\nThink/solve as needed, then end your reply with a single line exactly: FINAL: <answer>"


@dataclass
class Task:
    id: str
    question: str
    check: Callable[[str], bool]
    domain: str = "general"

    def prompt(self) -> str:
        return self.question + FINAL_INSTRUCTION


def extract_final(text: str) -> str:
    if not text:
        return ""
    m = re.findall(r"FINAL:\s*(.+)", text, re.IGNORECASE)
    cand = m[-1].strip() if m else ""
    if not cand:
        # fallback: last non-empty line
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        cand = lines[-1] if lines else ""
    # unwrap common LaTeX/markdown answer wrappers: \boxed{0,1}, $...$, **...**, `...`
    cand = re.sub(r"\\boxed\s*\{([^}]*)\}", r"\1", cand)
    cand = re.sub(r"\\[\[\]()]", "", cand)          # \[ \] \( \)
    cand = cand.replace("$", "").replace("`", "").replace("**", "")
    return cand.strip()


def _nums(s: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))]


def num_eq(expected: float, tol: float = 1e-6) -> Callable[[str], bool]:
    """True if the expected number appears in the FINAL line (or, as fallback, the text)."""
    def chk(text: str) -> bool:
        fin = extract_final(text)
        for n in _nums(fin):
            if abs(n - expected) <= tol:
                return True
        return False
    return chk


def str_eq(expected: str) -> Callable[[str], bool]:
    exp = expected.strip().lower()
    def chk(text: str) -> bool:
        return exp in extract_final(text).strip().lower()
    return chk


def all_in(expected_items: List[str]) -> Callable[[str], bool]:
    items = [e.strip().lower() for e in expected_items]
    def chk(text: str) -> bool:
        fin = extract_final(text).lower()
        return all(it in fin for it in items)
    return chk


# --- seed task set: deterministic, verifiable, spans domains so routing matters ---
SEED_TASKS: List[Task] = [
    # arithmetic / math
    Task("mul1", "What is 23 * 47?", num_eq(1081), "math"),
    Task("mul2", "Compute 17 * 19 * 3.", num_eq(969), "math"),
    Task("bat_ball",
         "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
         "How much does the ball cost in dollars?", num_eq(0.05), "math"),
    Task("speed",
         "A train travels 120 km in 1.5 hours, then 80 km in 1 hour. "
         "What is its average speed in km/h over the whole trip?", num_eq(80.0), "math"),
    Task("pct", "What is 15% of 240?", num_eq(36), "math"),
    # number theory / reasoning
    Task("primes_50_70",
         "List all prime numbers strictly between 50 and 70.",
         all_in(["53", "59", "61", "67"]), "reasoning"),
    Task("gcd", "What is the greatest common divisor of 48 and 180?", num_eq(12), "math"),
    Task("fib", "What is the 10th Fibonacci number, counting 1, 1, 2, 3, ...?", num_eq(55), "reasoning"),
    # short code-reasoning (answer is the computed result, checkable)
    Task("palindrome",
         "Is the string 'racecar' a palindrome? Answer yes or no.", str_eq("yes"), "code"),
    Task("two_sum",
         "Given nums=[2,7,11,15] and target=9, return the two 0-based indices whose values sum to target, "
         "as a comma-separated pair like 0,1.", str_eq("0,1"), "code"),
    Task("balanced",
         "Are the brackets in the string '([{}])' balanced? Answer yes or no.", str_eq("yes"), "code"),
    Task("count_vowels",
         "How many vowels (a,e,i,o,u) are in the word 'orchestration'?", num_eq(5), "code"),
]


def get_tasks(ids: List[str] | None = None) -> List[Task]:
    if ids is None:
        return list(SEED_TASKS)
    by_id = {t.id: t for t in SEED_TASKS}
    return [by_id[i] for i in ids if i in by_id]
