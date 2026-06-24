"""A/B eval: does the clean experience library help, hurt, or do nothing?

Multi-seed to beat the eval noise we measured (this task set swings 0.667<->0.917
on temperature alone), PLUS a per-task breakdown so we can see *which* tasks the
experiences flip — an aggregate mean can hide a real effect on a few hard tasks.

Runs against the FRONTIER pool (config.yaml) by default so it's a valid comparison
with the training run that produced the +0.0513 delta. The WITH-exp condition loads
the cleaned, versioned experiences.json; NO-exp uses an empty library.

Usage:
  python scripts/ab_eval.py --config config.yaml --exp experiences.json --seeds 5
"""
import argparse
import asyncio
import os
import statistics
import sys

# Bootstrap: ensure project root is importable regardless of cwd / how we're invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amalia.config import load_config
from amalia.training.grpo_free import TrainingFreeGRPO, TrainState
from amalia.training.run import evaluate
from amalia.training.tasks import get_tasks


async def multi_eval(cfg, state, label, tasks, seeds):
    trainer = TrainingFreeGRPO(cfg.orchestrator, cfg.pool, cfg.conductor, state)
    rates = []
    per_task = {t.id: 0 for t in tasks}
    for s in range(seeds):
        r = await evaluate(trainer, tasks)
        rates.append(r["pass_rate"])
        for row in r["rows"]:
            per_task[row["id"]] += int(row["ok"])
        print(f"  {label} seed{s + 1}: {r['pass_rate']}  ({r['passed']}/{r['total']})", flush=True)
    m = statistics.mean(rates)
    sd = statistics.pstdev(rates) if len(rates) > 1 else 0.0
    print(f"{label}: rates={rates} mean={m:.4f} sd={sd:.4f}", flush=True)
    return m, sd, per_task


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")        # frontier pool by default
    ap.add_argument("--exp", default="experiences.json")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tasks = get_tasks()
    empty = TrainState()
    trained = TrainState.load(args.exp)
    print(f"Config={args.config} | exp={args.exp} ({len(trained.experiences)} experiences) "
          f"| seeds={args.seeds} | tasks={len(tasks)}", flush=True)
    for i, e in enumerate(trained.experiences):
        print(f"   exp{i + 1}: {e}", flush=True)
    print("\nMulti-seed A/B:", flush=True)

    m0, sd0, pt0 = await multi_eval(cfg, empty, "NO-exp  ", tasks, args.seeds)
    m1, sd1, pt1 = await multi_eval(cfg, trained, "WITH-exp", tasks, args.seeds)

    delta = m1 - m0
    pooled = (((sd0 ** 2) + (sd1 ** 2)) / 2) ** 0.5
    print(f"\nNO-exp  : mean={m0:.4f} sd={sd0:.4f}", flush=True)
    print(f"WITH-exp: mean={m1:.4f} sd={sd1:.4f}", flush=True)
    print(f"DELTA (multi-seed): {delta:+.4f}", flush=True)
    if pooled > 0:
        print(f"effect size delta/pooled_sd = {delta / pooled:+.2f}", flush=True)

    print("\nPer-task flips (pass count across seeds, NO -> WITH):", flush=True)
    flips = 0
    for t in tasks:
        a, b = pt0[t.id], pt1[t.id]
        if a != b:
            flips += 1
            arrow = "HELP" if b > a else "HURT"
            print(f"  [{arrow}] {t.id:22s} {a}/{args.seeds} -> {b}/{args.seeds}", flush=True)
    if not flips:
        print("  (none — every task passed/failed identically in both conditions)", flush=True)

    verdict = ("experiences HELP" if delta > 0.02
               else "experiences HURT" if delta < -0.02
               else "no clear effect (within noise)")
    print("\nVERDICT:", verdict, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
