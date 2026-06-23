"""A/B eval: does the experience library help, hurt, or do nothing? Multi-seed to beat noise."""
import asyncio, statistics, sys, os
# Bootstrap: ensure project root is importable regardless of cwd / how we're invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amalia.config import load_config
from amalia.training.grpo_free import TrainingFreeGRPO, TrainState
from amalia.training.run import evaluate
from amalia.training.tasks import get_tasks

cfg = load_config('config.train.yaml')
tasks = get_tasks()

async def multi_eval(state, label, seeds=3):
    trainer = TrainingFreeGRPO(cfg.orchestrator, cfg.pool, cfg.conductor, state)
    rates = []
    for s in range(seeds):
        r = await evaluate(trainer, tasks)
        rates.append(r['pass_rate'])
        print(f'  {label} seed{s+1}: {r["pass_rate"]}', flush=True)
    m = statistics.mean(rates)
    print(f'{label}: rates={rates} mean={m:.3f}', flush=True)
    return m

async def main():
    empty = TrainState()
    trained = TrainState.load('experiences.json')
    print('Multi-seed A/B (3x each):', flush=True)
    m0 = await multi_eval(empty, 'NO-exp ', 3)
    m1 = await multi_eval(trained, 'WITH-exp', 3)
    print(f'\nDELTA (multi-seed): {m1-m0:+.3f}', flush=True)
    verdict = 'experiences HELP' if m1 > m0 + 0.02 else ('experiences HURT' if m1 < m0 - 0.02 else 'no clear effect (within noise)')
    print('VERDICT:', verdict, flush=True)

asyncio.run(main())
