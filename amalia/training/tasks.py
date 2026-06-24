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
    tier: str = "seed"

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
    # Normalize common math/markdown formats before extracting numbers:
    #   113{,}411 -> 113411, 3{,}600 -> 3600, $54 -> 54
    s = re.sub(r"\{\s*,\s*\}", "", s)
    s = s.replace(",", "").replace("{", "").replace("}", "")
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]


def num_eq(expected: float, tol: float = 1e-6) -> Callable[[str], bool]:
    """True if the expected number appears in the FINAL line (or, as fallback, the text)."""
    def chk(text: str) -> bool:
        # Prefer the FINAL line, but fall back to the whole text. Some worker
        # outputs are explanatory and put the correct numeric answer before the
        # last line; training/eval should not punish a correct answer just for
        # formatting noise.
        for scope in (extract_final(text), text or ""):
            for n in _nums(scope):
                if abs(n - expected) <= tol:
                    return True
        return False
    return chk


def str_eq(expected: str) -> Callable[[str], bool]:
    exp = expected.strip().lower()
    def chk(text: str) -> bool:
        fin = extract_final(text).strip().lower()
        whole = (text or "").strip().lower()
        if exp in {"yes", "no"}:
            # Avoid substring false positives like yes<-yesterday or no<-unknown.
            pat = rf"\b{re.escape(exp)}\b"
            return re.search(pat, fin) is not None or re.search(pat, whole) is not None
        return exp in fin or exp in whole
    return chk


def exact_str_eq(expected: str) -> Callable[[str], bool]:
    """Case-insensitive exact final-answer match (after light wrapper stripping)."""
    exp = expected.strip().lower()
    def chk(text: str) -> bool:
        return extract_final(text).strip().lower() == exp
    return chk


def comma_pair_eq(a: int, b: int) -> Callable[[str], bool]:
    """Exact checker for index-pair tasks; avoids 1,2 matching 11,23."""
    expected = [a, b]
    def chk(text: str) -> bool:
        nums = [int(x) for x in re.findall(r"-?\d+", extract_final(text))]
        return nums == expected
    return chk


def exact_num_set(expected_items: List[int]) -> Callable[[str], bool]:
    """Exact set checker for list tasks (e.g. primes); rejects extra wrong items."""
    expected = sorted(int(x) for x in expected_items)
    def chk(text: str) -> bool:
        nums = sorted({int(x) for x in re.findall(r"-?\d+", extract_final(text))})
        return nums == expected
    return chk


def all_in(expected_items: List[str]) -> Callable[[str], bool]:
    items = [e.strip().lower() for e in expected_items]
    def chk(text: str) -> bool:
        fin = extract_final(text).lower()
        whole = (text or "").lower()
        return all(it in fin for it in items) or all(it in whole for it in items)
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

    # ---- expanded set (Phase 2, step 3): more tasks to beat eval variance, with
    # several multi-step "trap" problems that reward decomposition + verification
    # (single-step intuition fails) so a frontier worker pool still produces signal. ----

    # arithmetic — large/compound (single-shot error-prone -> verify step helps)
    Task("mul_big", "What is 127 * 893?", num_eq(113411), "math"),
    Task("compound", "Compute (45 + 67) * 12 - 89.", num_eq(1255), "math"),
    Task("digit_sum", "What is the sum of the digits of 9876?", num_eq(30), "math"),
    Task("seconds_2_5h", "How many seconds are there in 2.5 hours?", num_eq(9000), "math"),
    Task("area_rect", "What is the area of a rectangle that is 12 by 7?", num_eq(84), "math"),
    Task("avg_4", "What is the average of 10, 20, 30, and 40?", num_eq(25), "math"),
    Task("pct_nested", "What is 30% of 30% of 1000?", num_eq(90), "math"),

    # word problems — trap-prone, multi-step (decomposition + verification matter)
    Task("age",
         "Alice is 3 times as old as Bob. In 5 years, Alice will be twice as old as Bob. "
         "How old is Bob now?", num_eq(5), "reasoning"),
    Task("coins",
         "You have 26 coins, all nickels and dimes, worth $2.05 in total. "
         "How many dimes do you have?", num_eq(15), "reasoning"),
    Task("discount",
         "A shirt costs $80. It is discounted 25%, then a further 10% is taken off the "
         "reduced price. What is the final price in dollars?", num_eq(54), "reasoning"),
    Task("trains_meet",
         "Two trains are 300 km apart and travel toward each other at 60 km/h and 90 km/h. "
         "After how many hours do they meet?", num_eq(2), "reasoning"),
    Task("work_days",
         "If 4 workers build a wall in 6 days, how many days do 3 workers need to build "
         "the same wall at the same rate?", num_eq(8), "reasoning"),

    # number theory / sequences
    Task("lcm", "What is the least common multiple of 12 and 18?", num_eq(36), "math"),
    Task("factorial6", "What is 6! (six factorial)?", num_eq(720), "math"),
    Task("sum_1_100", "What is the sum of all integers from 1 to 100 inclusive?", num_eq(5050), "math"),
    Task("primes_below_20", "How many prime numbers are there below 20?", num_eq(8), "reasoning"),
    Task("fib15", "What is the 15th Fibonacci number, counting 1, 1, 2, 3, 5, ...?", num_eq(610), "reasoning"),
    Task("gcd_big", "What is the greatest common divisor of 252 and 105?", num_eq(21), "math"),

    # combinatorics / logic
    Task("handshakes",
         "In a room of 10 people, everyone shakes hands exactly once with every other "
         "person. How many handshakes happen in total?", num_eq(45), "reasoning"),
    Task("choose_3_7", "How many ways are there to choose 3 items from 7 distinct items?", num_eq(35), "reasoning"),
    Task("arrange_cat", "How many distinct ways can the letters in the word 'CAT' be arranged?", num_eq(6), "reasoning"),
    Task("pairs_5", "In a group of 5 people, how many unique pairs can be formed?", num_eq(10), "reasoning"),

    # string / code reasoning
    Task("reverse_hello",
         "Reverse the string 'hello'. Answer with only the reversed string.", str_eq("olleh"), "code"),
    Task("anagram",
         "Are 'listen' and 'silent' anagrams of each other? Answer yes or no.", str_eq("yes"), "code"),
    Task("count_a_banana",
         "How many times does the letter 'a' appear in the word 'banana'?", num_eq(3), "code"),
    Task("substring_gram",
         "Does the word 'programming' contain the substring 'gram'? Answer yes or no.", str_eq("yes"), "code"),
    Task("distinct_mississippi",
         "How many distinct characters are in the string 'mississippi'?", num_eq(4), "code"),
]


def get_tasks(ids: List[str] | None = None) -> List[Task]:
    if ids is None:
        return list(SEED_TASKS)
    by_id = {t.id: t for t in SEED_TASKS}
    return [by_id[i] for i in ids if i in by_id]
