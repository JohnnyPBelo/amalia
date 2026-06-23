# Amalia

**Multi-agent orchestration as a single model.** Amalia is a local,
training-free implementation of a *learned-orchestrator* pattern: you call **one**
OpenAI-compatible endpoint (`amalia-v1`) and a small orchestrator LLM decides — per
query — which frontier models to involve, how to split the work, and how to combine
or verify their outputs.

Point any OpenAI-compatible tool (an IDE, an agent framework, your own scripts) at
Amalia and it transparently gains multi-agent orchestration with a one-line config
change.

```
client ──► Amalia :8900 (OpenAI-compatible, model "amalia-v1")
              │
              ▼
        Orchestrator LLM  (local Qwen2.5-7B)
              │  emits  model_id / subtasks / access_list   (a workflow, in natural language)
              ▼
        WorkflowEngine  ── parse ─► validate ─► execute (chain │ tree │ best-of-N)
              │                                      │
              │  recursion (≤ N rounds, refine/verify)
              └──────────────◄───────────  Worker Pool (swappable)
                                            Model 0,1,2... = any OpenAI/Responses endpoint
```

## Why

Different frontier models specialize in different things. Amalia turns a **pool** of
them into one collectively-smarter system, while exposing a single stable interface.
The pool is **swappable**: add, remove, or reorder workers in config and the
orchestrator adapts without retraining — it only ever sees them as ordinals
(`Model 0`, `Model 1`, …) plus a capability hint. If a provider disappears, drop it
and restart; nothing else changes.

## What works today

- **Topologies** emergent from the orchestrator's `access_list`, not hard-coded:
  chain / sequential, best-of-N, and **tree** (parallel leaves + an aggregator chosen
  per task).
- **Recursion** — the orchestrator inspects the result and may refine/verify or return
  as-is. A tunable test-time-compute axis.
- **Mixed wire protocols** in one pool: `chat` (`/chat/completions`) and `responses`
  (`/responses`, for reasoning models like GPT-5.x).
- **Parallel execution** of independent steps (tree leaves / best-of-N attempts) via
  topological waves.
- **Graceful degradation** — a failed worker doesn't crash the workflow.
- **OpenAI-compatible server** — drop-in for Hermes, OpenClaw, Codex, Cursor, curl.

## Quick start

```bash
git clone <this-repo> amalia && cd amalia
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) start a local orchestrator (Qwen2.5-7B) and the Amalia server
./run.sh                 # uses config.yaml

# 2) call it like any OpenAI endpoint
curl -s http://localhost:8900/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"amalia-v1","messages":[{"role":"user","content":"Write and verify a Python is_prime(n)."}]}'
```

Health + reachability of the pool:

```bash
curl -s http://localhost:8900/health | python3 -m json.tool
```

## Configuration

Everything lives in `config.yaml`: the `orchestrator`, the `pool` (each worker's
`base_url`, `model`, `api_type`, `capabilities`), and the `conductor` knobs
(`max_steps`, `max_recursion`, `parse_retries`, `fallback_worker`). See
[`INTEGRATION.md`](INTEGRATION.md) for wiring Amalia into Hermes and OpenClaw, and
[`PLAN.md`](PLAN.md) for the roadmap.

Set `AMALIA_DEBUG=1` to attach the full workflow trace
(`choices[0].message.amalia_trace`): the emitted `model_id`/`subtasks`/`access_list`,
which workers ran, and per-step ok/error.

## Status

Phase 1 (training-free MVP) is complete and verified end-to-end. Phase 2 (optional
GRPO fine-tune of the orchestrator) is in progress — see `PLAN.md`.

## Credit / lineage

Amalia is an independent, clean-room implementation inspired by Sakana AI's published
research on learned orchestration. It reimplements ideas — it is **not** affiliated
with Sakana AI and contains none of their code or weights.

- Nielsen et al., *Learning to Orchestrate Agents in Natural Language with the Conductor*, ICLR 2026 (arXiv:2512.04388)
- Xu et al., *Trinity: An Evolved LLM Coordinator*, ICLR 2026 (arXiv:2512.04695)
- *Sakana Fugu Technical Report* (arXiv:2606.21228)

## License

MIT — see [LICENSE](LICENSE).
