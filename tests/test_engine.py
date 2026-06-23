"""Engine tests with FAKE workers — validates wave scheduling, tree parallelism,
access-list context assembly, and graceful degradation. No network/LLM.

Run: pytest tests/test_engine.py -v
"""
import asyncio
import time
import pytest

from amalia.parser import Workflow
from amalia.workers import WorkerPool, Worker, WorkerResult
from amalia.engine import WorkflowEngine, _build_context


class FakePool(WorkerPool):
    """A pool that records calls and returns deterministic, optionally-delayed text."""
    def __init__(self, n=3, delay=0.0, fail_idx=None):
        self.workers = [Worker(f"fake{i}", "fake", "http://fake/v1") for i in range(n)]
        self.delay = delay
        self.fail_idx = fail_idx
        self.calls = []  # (idx, messages)

    async def call(self, idx, messages, client):
        self.calls.append((idx, messages))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_idx is not None and idx == self.fail_idx:
            return WorkerResult(ok=False, text="", worker_name=self.workers[idx].name, error="boom")
        # echo which step text it saw + its own marker
        return WorkerResult(ok=True, text=f"resp-from-{idx}", worker_name=self.workers[idx].name)


def test_build_context_root_sees_question():
    msgs = _build_context(0, [], ["solve it"], {}, "QUESTION")
    assert msgs[-1]["role"] == "user"
    assert "QUESTION" in msgs[-1]["content"]
    assert "solve it" in msgs[-1]["content"]


def test_build_context_access_list_injects_prior_turns():
    responses = {0: "ANSWER_ZERO"}
    msgs = _build_context(1, [0], ["sub0", "sub1"], responses, "Q")
    # should contain user(sub0)/assistant(ANSWER_ZERO) then user(sub1)
    assert msgs[0]["content"] == "sub0"
    assert msgs[1]["content"] == "ANSWER_ZERO"
    assert msgs[-1]["content"] == "sub1"


def test_chain_executes_in_order():
    pool = FakePool(n=3)
    eng = WorkflowEngine(pool)
    wf = Workflow([0, 1], ["a", "b"], [[], ["all"]])
    res = asyncio.run(eng.execute(wf, "Q"))
    assert res.final_answer == "resp-from-1"
    assert res.n_worker_calls == 2


def test_tree_leaves_run_in_parallel():
    # 2 leaves with 0.5s delay each + 1 aggregator. If parallel, wall time ~1.0s
    # (0.5 for the leaf wave + 0.5 for the aggregator wave), not 1.5s (serial).
    pool = FakePool(n=3, delay=0.5)
    eng = WorkflowEngine(pool)
    wf = Workflow([0, 1, 2], ["a", "b", "agg"], [[], [], [0, 1]])
    t0 = time.time()
    res = asyncio.run(eng.execute(wf, "Q"))
    dt = time.time() - t0
    assert res.n_worker_calls == 3
    assert dt < 1.3, f"leaves did not run in parallel (took {dt:.2f}s)"
    assert res.final_answer == "resp-from-2"


def test_aggregator_sees_both_leaves():
    pool = FakePool(n=3)
    eng = WorkflowEngine(pool)
    wf = Workflow([0, 1, 2], ["a", "b", "agg"], [[], [], [0, 1]])
    asyncio.run(eng.execute(wf, "Q"))
    # the aggregator call (idx=2) must include both leaf responses
    agg_call = [m for (i, m) in pool.calls if i == 2][0]
    joined = " ".join(x["content"] for x in agg_call)
    assert "resp-from-0" in joined and "resp-from-1" in joined


def test_graceful_degradation_on_worker_failure():
    # leaf 0 fails; aggregator still runs and returns (degrades, no crash)
    pool = FakePool(n=3, fail_idx=0)
    eng = WorkflowEngine(pool)
    wf = Workflow([0, 1, 2], ["a", "b", "agg"], [[], [], [0, 1]])
    res = asyncio.run(eng.execute(wf, "Q"))
    assert res.final_answer == "resp-from-2"
    step0 = [s for s in res.steps if s.idx == 0][0]
    assert step0.ok is False and step0.error == "boom"


def test_best_of_n_topology():
    pool = FakePool(n=2)
    eng = WorkflowEngine(pool)
    wf = Workflow([0, 0, 0, 1], ["t1", "t2", "t3", "pick"], [[], [], [], [0, 1, 2]])
    res = asyncio.run(eng.execute(wf, "Q"))
    assert res.n_worker_calls == 4
    assert res.final_answer == "resp-from-1"
