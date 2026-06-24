# Amalia Beyond Fugu / Fable — Technical Ambition Plan

> Goal: make Amalia more than a Conductor clone: a **sovereign, self-improving,
> test-time-scaling orchestration system** that can beat single frontier models not by
> being one larger model, but by learning when/how to assemble the right society of models.
>
> Non-negotiable rule: **no benchmark claims without measured, reproducible eval.**
> The target is to beat Fable 5 / Fugu Ultra by miles, but every claim must survive
> held-out eval, multi-seed variance, and ablations.

---

## 1. Current state (2026-06-24)

Implemented / verified:

- Conductor-style workflow emission: `model_id`, `subtasks`, `access_list`.
- Balanced parser + DAG execution with parallel topological waves.
- Recursion as a capped test-time compute axis.
- Mixed worker protocols:
  - Claude / Gemini via `copilot-api` chat bridge `:4141`.
  - GPT-5.5 via Responses shim `:4142`, with `reasoning_effort=xhigh`.
- Real GRPO on the A9 iGPU (Qwen2.5-7B policy + LoRA, ROCm):
  - 96GB VRAM exposed; no gradient checkpointing needed.
  - Current run: `grpo_7b_gpt55_xhigh_200_20260624_162020`.
  - Pool map aligned across training and runtime:
    - Model 0: Claude Opus 4.8 — general/coding/step-by-step.
    - Model 1: Gemini 3.1 Pro — math/science/precise calculation.
    - Model 2: GPT-5.5 xhigh — verifier/checker/error catcher.

Immediate issue fixed while training ran: `config.yaml` was out of ordinal alignment
with the training pool. Runtime now matches the policy's learned capability map.

---

## 2. What would make this genuinely beyond Fugu

Fugu/Conductor's core idea is *learned orchestration*. To go beyond it, Amalia needs
not just the same algorithm, but a stronger loop around it:

### A. Stronger reward than Conductor's binary correctness

The paper reward is mostly:

- format parseable?
- final answer correct?

That is sparse. Amalia should add shaped, verifiable rewards:

1. **Format reward** — parseable non-empty workflow.
2. **Execution correctness** — final answer passes deterministic checker.
3. **Verifier-use reward** — bonus when a verification step actually catches or corrects
   a wrong prior answer; zero bonus for decorative verification.
4. **Topology reward** — reward the *minimal* topology that solves the task; penalize
   unnecessary worker calls.
5. **Cost/latency reward** — answer quality per token/second, not raw quality only.
6. **Robustness reward** — re-run the same workflow under one worker failure / degraded
   worker and reward graceful recovery.
7. **Self-consistency reward** — independent branches must converge; disagreements should
   trigger a verifier/arbiter step.

Formula sketch:

```text
R = 1.00 * correctness
  + 0.20 * format
  + 0.15 * verifier_caught_error
  + 0.10 * disagreement_resolved
  - 0.05 * unnecessary_worker_call
  - 0.02 * latency_bucket
  - 0.20 * fallback_or_empty_final
```

### B. Curriculum, not one tiny seed set

The current 39 tasks are useful for proof-of-life, not for beating Fable 5.
We need tiers:

| Tier | Purpose | Examples |
|---|---|---|
| T0 smoke | format + basic exec | arithmetic, short string/code checks |
| T1 traps | forces decomposition/verification | coin/age/work-rate traps, adversarial wording |
| T2 coding | real tool-like code reasoning | LiveCodeBench slices, BigCodeBench mini tasks |
| T3 science/math | frontier reasoning | GPQA-D slices, AIME-style, proof checks |
| T4 agentic | where orchestration should shine | multi-file debugging, repo tasks, test-fix loops |
| T5 adversarial ops | sovereignty/failure | provider outage, wrong-worker injection, latency/cost constraints |

Training should start T0/T1 and increasingly mix T2–T5 once format is stable.

### C. Adaptive worker selection like the paper, but harder

Train with randomized worker subsets and degraded workers:

- Drop one worker randomly per batch.
- Swap Model 2 between GPT-5.5, Claude, Gemini, local model.
- Inject capability lies in hints for a small fraction of batches.
- Randomly mark a worker as expensive/slow/unavailable.

The policy must learn **capability inference from outcomes**, not just memorize
`Model 1 = math`.

### D. Recursion as an explicit search budget

Fugu's recursion is test-time scaling. Amalia should expose a quality ladder:

| Mode | Recursion | Branching | Target |
|---|---:|---:|---|
| fast | 0 | low | latency-sensitive calls |
| standard | 1 | moderate | default |
| deep | 2 | verification-heavy | hard coding/math |
| insane | 3+ | budget-capped search | benchmark / research |

For benchmark claims against Fable 5, compare both:

- **latency-normalized** performance, and
- **quality-max** performance under a published budget.

### E. A learned verifier, not just a worker hint

Today Model 2 is GPT-5.5 xhigh with a "verify" capability hint. Next: train the
orchestrator to create verification subtasks that are *specific*:

Bad:
```text
Verify the answer.
```

Good:
```text
Check the arithmetic in step 1 by independently deriving the equation; if it
conflicts, return corrected FINAL only.
```

This is where Amalia can beat simple routers: the policy must learn the **shape of
verification**, not just route to a verifier.

### F. Evaluation bigger than pass-rate

A pass-rate alone hides whether the system is doing something new. Track:

- pass_rate by domain and difficulty tier
- worker-call count
- topology class: single / chain / tree / best-of-N / recursive
- verifier involvement and verifier correction rate
- fallback rate
- latency and token cost
- per-task flips vs baseline
- bootstrap confidence interval, not one seed

---

## 3. Experimental ladder

### Run 0 — current run (already launched)

Purpose: prove the 7B policy can train with a heterogeneous Opus/Gemini/GPT-5.5-xhigh
pool and execution reward.

Success criteria:

- completes 200 steps without OOM or bridge collapse
- checkpoints at 25-step intervals
- non-zero KL and gradients when reward variance exists
- final eval beats the base prompt-only orchestrator on held-out T0/T1

### Run 1 — shaped reward

Add `--reward-profile beyond_fugu_v1`:

- correctness + format + topology/cost penalties
- verifier-use bonus
- failure-robustness mini-probe on a subset of batches

Success: better pass-rate **and** fewer unnecessary worker calls than Run 0.

### Run 2 — curriculum

Add task tiers and sampling weights:

```text
steps 0-50:   T0/T1 format + traps
steps 50-150: T1/T2/T3 mixed
steps 150+:   T2/T3/T4/T5 hard mix
```

Success: no regression on easy tasks; measurable gains on T2/T3.

### Run 3 — adaptive worker selection

Randomize worker subsets/capability hints/degraded workers.

Success: model retains performance when a provider is removed or swapped.

### Run 4 — benchmark-quality eval

Run against external/held-out benchmark slices:

- LiveCodeBench mini
- BigCodeBench mini
- GPQA-D subset
- AIME-style subset
- repo-debug tasks from real codebases

Compare:

1. best single worker
2. static MoA / fixed aggregator
3. prompt-only Amalia
4. Run 0/1/2/3 checkpoints
5. Fugu-style public numbers where comparable

---

## 4. Architecture additions to implement

1. **Training telemetry JSONL**
   - log each completion's parsed workflow, task id, rewards, final answer, worker calls.
   - required to know *why* a run improves.

2. **Checkpoint evaluator**
   - automatically evaluate every `checkpoint-*` vs baseline.
   - choose best checkpoint by held-out score, not last checkpoint.

3. **Task tiers / split files**
   - `tasks_core.py`, `tasks_hard.py`, `tasks_benchmark.py` or one JSONL schema.
   - strict train/held-out separation.

4. **Reward profiles**
   - `format_only`, `exec_binary`, `beyond_fugu_v1`.
   - selectable CLI flag.

5. **Runtime quality modes**
   - `amalia_mode=fast|standard|deep|insane`, mapping to recursion/decode/budget.

6. **Ablation harness**
   - same task, same seed, compare worker pool variants and topology constraints.

---

## 5. Honesty constraints

We do **not** claim "beats Fable 5" until:

- benchmark set is held-out and not used for reward design;
- baseline includes best single worker and prompt-only Amalia;
- eval has multi-seed or bootstrap CI;
- runtime budget is published;
- failures are reported, not hidden.

The goal is maximal ambition with zero fake numbers.

---

## 6. Immediate next actions after current run

1. Run eval on Run 0 checkpoint(s).
2. Add telemetry JSONL before Run 1.
3. Add held-out hard task set.
4. Implement shaped reward profile.
5. Launch Run 1 with curriculum + shaped reward.
6. Compare Run 0 vs Run 1 vs prompt-only vs best single worker.

If Run 0 is flat, do **not** panic: it proves the pipe. The leap comes from shaped
reward + curriculum + adaptive subsets, not from one 39-task seed set.
