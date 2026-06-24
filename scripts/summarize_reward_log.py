#!/usr/bin/env python3
"""Summarize Amalia GRPO exec-reward telemetry JSONL.

Usage:
  python scripts/summarize_reward_log.py path/to/reward.jsonl
"""
from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_reward_log.py reward.jsonl", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        print("no rows")
        return 0

    rewards = [float(r.get("reward", 0.0)) for r in rows]
    ok = [r for r in rows if r.get("ok") is True]
    parsed = [r for r in rows if "workflow" in r]
    by_domain = collections.defaultdict(list)
    by_task = collections.defaultdict(list)
    model_use = collections.Counter()
    topo = collections.Counter()
    latencies = []

    for r in rows:
        if r.get("domain"):
            by_domain[r["domain"]].append(float(r.get("reward", 0.0)))
        if r.get("task_id"):
            by_task[r["task_id"]].append(float(r.get("reward", 0.0)))
        wf = r.get("workflow") or {}
        mids = wf.get("model_id") or []
        model_use.update(mids)
        if mids:
            accesses = wf.get("access_list") or []
            if len(mids) == 1:
                topo["single"] += 1
            elif any(isinstance(a, list) and not a for a in accesses[1:]):
                topo["tree_or_best_of_n"] += 1
            else:
                topo["chain"] += 1
        if isinstance(r.get("latency_s"), (int, float)):
            latencies.append(float(r["latency_s"]))

    print(f"rows: {len(rows)}")
    print(f"mean_reward: {statistics.mean(rewards):.4f}")
    print(f"ok_rate: {len(ok) / len(rows):.4f} ({len(ok)}/{len(rows)})")
    if latencies:
        print(f"latency_s mean/p50/max: {statistics.mean(latencies):.2f} / "
              f"{statistics.median(latencies):.2f} / {max(latencies):.2f}")
    print("\nby_domain:")
    for d, vals in sorted(by_domain.items()):
        print(f"  {d:10s} n={len(vals):4d} mean_reward={statistics.mean(vals):+.3f}")
    print("\nmodel_use (workflow steps):")
    for mid, n in sorted(model_use.items()):
        print(f"  Model {mid}: {n}")
    print("\ntopologies:")
    for k, n in topo.most_common():
        print(f"  {k}: {n}")
    print("\nworst_tasks:")
    task_scores = [(statistics.mean(v), k, len(v)) for k, v in by_task.items()]
    for mean, tid, n in sorted(task_scores)[:10]:
        print(f"  {tid:24s} n={n:3d} mean_reward={mean:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
