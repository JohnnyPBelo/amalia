# amalia-v1 — build plan & roadmap

## Provenance
Clean-room reimplementation of the **Conductor** orchestration algorithm
(arXiv:2512.04388) — the engine behind Sakana's Fugu-Ultra — for local,
training-free use over the providers we already have. Trinity (arXiv:2512.04695)
and the Fugu technical report (arXiv:2606.21228) informed the design.

## Design decision: the Conductor branch, not the Trinity/selection-head branch
- Trinity/Fugu-base route via a **selection head over hidden states** → needs a
  local orchestrator with weight access; can't wrap API-only workers.
- The **Conductor** routes purely through prompting (input → text → parse) → works
  over any OpenAI-compatible endpoint. This is what makes it plug-and-play.
- It also expresses the full coordination space (chain / best-of-N / tree) and
  recursion, so we lose nothing by picking it.

## Phase 1 — training-free MVP  ✅ DONE
- [x] `prompts.py` — Conductor system + recursion prompts (verbatim from paper) + few-shot.
- [x] `parser.py` — balanced-bracket parser for model_id/subtasks/access_list (+ validation = paper's format reward).
- [x] `workers.py` — swappable WorkerPool over OpenAI-compatible endpoints; ordinal-only naming.
- [x] `engine.py` — DAG executor with **topological waves** (parallel tree leaves / best-of-N), access-list context assembly, graceful degradation.
- [x] `conductor.py` — orchestrate → parse (+retries/fallback) → execute → **recursion** rounds.
- [x] `server.py` — OpenAI-compatible `amalia-v1` endpoint (+ `AMALIA_DEBUG` trace).
- [x] Local orchestrator: Qwen2.5-7B-Instruct (paper's exact base) on llama.cpp Vulkan.
- [x] E2E verified: bat-and-ball (chain+verify), primes (tree), is_balanced (chain+recursion → correct, executable code).
- [x] 20/20 unit tests (parser + engine waves, parallelism, degradation).
- [x] `INTEGRATION.md` — Hermes custom provider + OpenClaw model provider.

## Phase 2 — optional fine-tune (sharpen the orchestrator)
Paper shows prompt+few-shot already induces orchestration; GRPO sharpens it.

### 2a. Training-Free GRPO  ❌ NEGATIVE RESULT (honest) — does not beat baseline on this task set
Optimize in *context space*, no gradients, runs on the A9 with no H100s.
- [x] Verifiable task set with reward = format + correctness (`amalia/training/tasks.py`, **39 tasks**, FINAL: answer convention, LaTeX/markdown-tolerant checkers, multi-step "trap" problems for routing signal). Every ground-truth independently verified in `tests/test_task_ground_truth.py`.
- [x] Group-rollout loop: G workflows/task → score → contrast wins vs losses → orchestrator LLM distills ONE transferable "experience" (semantic group-advantage) → experience library injected into the Conductor prompt (`amalia/training/grpo_free.py`).
- [x] Runner + eval harness (`amalia/training/run.py`): baseline → iterate → final pass-rate + delta; experiences persisted to `experiences.json`.
- [x] **Experience-quality filter** (`is_useful_experience`) + structural near-dup dedup (`_norm_exp`/`dedup_experiences`): rejects platitudes, embedded `NONE`, and self-comparisons that the 7B distiller emits. Covered by `tests/test_experience_filter.py`.
- [x] **Homogeneous-pool run (all-Qwen): delta = 0.0** — root-caused to (a) no real worker diversity, (b) distiller emitting vague platitudes.
- [x] **Frontier-pool run: baseline 0.7949 (31/39) → final 0.8462 (33/39), delta = +0.0513** — but this turned out to be **regression to the mean**, see below.
- [x] **Multi-seed A/B validation (5 seeds × 2 conditions × 39 tasks, frontier pool) — the +0.0513 was NOISE:**
  - NO-exp (empty library):  mean **0.8667** sd 0.0377  → the training run's "baseline 0.7949" was a ~1.9σ unlucky seed.
  - WITH-exp (3 clean exps):  mean **0.8256** sd 0.0594
  - delta = **−0.0410**, Welch t=−1.17 (df 6.8), Cohen's d=−0.74, 95% CI **[−0.139, +0.057] crosses 0**.
  - Verdict: **not statistically significant; experiences do NOT help, and trend slightly hurts.**
  - Per-task net (passes gained−lost across seeds): reasoning **−5**, math **−2**, code **−1** (total −8). The single-concept `access_list=[[], 'all']` rule *helped* a few pure-structure tasks (handshakes, pairs_5, speed +2 each) but *hurt* multi-step word problems (coins −3, age −2, trains_meet −2) — injecting a routing platitude distracts the orchestrator from decomposing.
- **Root cause of the negative result:** (1) eval ceiling too high — frontier workers already solve ~87% single-shot, so there's little headroom for orchestration to win; (2) the 7B distiller only ever produces one concept (`access_list`), so the "learned prior" is narrow and off-target for reasoning tasks.
- [ ] **If revisited:** need harder tasks where single-shot fails (true multi-hop/compositional), a stronger distiller (frontier model, not the 7B), and per-domain experiences. Otherwise skip to 2b.

### 2b. Real GRPO (optional, if 2a plateaus)
- [ ] Real GRPO via TRL on Qwen2.5-7B. Paper: 2×H100, KL=0, 64 rollouts/q. On A9 use ROCm (torch 2.5.1+rocm6.2 confirmed) + LoRA to fit. See skill `fine-tuning-with-trl`.
- [ ] Adaptive worker selection: train on random k-of-N pool subsets → generalize to swapped pools.
- [ ] Recursion fine-tune: instantiate one recursion call for half the batch (paper Sec 3.2).

## Phase 3 — hardening / productionization
- [ ] Streaming passthrough on the final step (`stream: true`).
- [ ] Caching of (query → workflow) to skip the orchestrator call on repeats.
- [ ] Responses-API worker adapter (so gpt-5.x can join the pool via the bridge).
- [ ] Cost/latency budget guard (cap worker calls per request).
- [ ] Optional Trinity-style local selection-head variant for a latency-first `fugu-fast`.
- [ ] Package: `pip install amalia-v1`, `fugu serve` console-script.

## Known gotchas (discovered during build)
- The :4141 copilot bridge only serves `/v1/chat/completions` — **gpt-5.x models 500** there (need Responses API). Use Claude/Gemini/GPT-4o in the pool, or run local llama.cpp workers.
- Non-greedy regex breaks on nested `access_list` like `[[], ["all"]]`; parser uses a balanced-bracket scanner that respects string literals.
- llama.cpp on the A9 = **Vulkan**, not ROCm (no SDK installed). See skill `strix-halo-local-inference`.
