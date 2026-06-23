"""
Training-Free GRPO runner.

Usage:
  python -m amalia.training.run --config config.yaml --iters 3 --group 4 \
      --out experiences.json [--eval-only]

  --eval-only : just measure pass-rate on the seed tasks with current experiences.
"""
from __future__ import annotations

import argparse
import asyncio
import json

from ..config import load_config
from .tasks import get_tasks
from .grpo_free import TrainingFreeGRPO, TrainState


async def evaluate(trainer: TrainingFreeGRPO, tasks, group_size: int = 1) -> dict:
    """Deterministic-ish eval: 1 rollout/task at the configured (low) temp."""
    conductor = trainer._conductor(trainer.state.experience_block())
    conductor.cfg.orchestrator_temperature = 0.3
    passed, rows = 0, []
    for t in tasks:
        r = await trainer._rollout(t, conductor)
        ok = r.reward > 0
        passed += ok
        rows.append({"id": t.id, "domain": t.domain, "ok": ok,
                     "final": r.final_answer, "workflow": r.workflow_repr})
    return {"pass_rate": round(passed / len(tasks), 4), "passed": passed,
            "total": len(tasks), "rows": rows}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--out", default="experiences.json")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    state = TrainState.load(args.out)
    trainer = TrainingFreeGRPO(cfg.orchestrator, cfg.pool, cfg.conductor, state)
    tasks = get_tasks()

    if args.eval_only:
        res = await evaluate(trainer, tasks)
        print(json.dumps({"experiences": len(state.experiences), **res}, indent=2))
        return

    print(f"[train] baseline eval (experiences={len(state.experiences)}) ...")
    base = await evaluate(trainer, tasks)
    print(f"  baseline pass_rate = {base['pass_rate']}  ({base['passed']}/{base['total']})")

    for i in range(args.iters):
        m = await trainer.iteration(tasks, group_size=args.group)
        state.save(args.out)
        print(f"[iter {i+1}/{args.iters}] mean_group_reward={m['mean_group_reward']} "
              f"new_exp={m['new_experiences']} total_exp={m['total_experiences']}")

    final = await evaluate(trainer, tasks)
    print(f"[train] final pass_rate = {final['pass_rate']}  ({final['passed']}/{final['total']})")
    print(f"[train] delta = {round(final['pass_rate'] - base['pass_rate'], 4)}")
    print(f"[train] experiences saved to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
