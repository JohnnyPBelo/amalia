# Integrating amalia-v1 into Hermes & OpenClaw

`amalia-v1` exposes a single OpenAI-compatible endpoint (`http://localhost:8900/v1`,
model id `amalia-v1`). Both Hermes and OpenClaw treat it as just another provider —
no code changes, just config. That is the whole point: **plug-and-play multi-agent
orchestration**.

Start it first:

```bash
cd ~/projects/amalia-v1
source .venv/bin/activate
AMALIA_CONFIG=config.yaml python -m uvicorn fugu.server:app --host 0.0.0.0 --port 8900
# (or config.e2e.yaml to use the :4141 frontier pool)
```

Sanity check:

```bash
curl -s http://localhost:8900/health | python3 -m json.tool
```

---

## Hermes — add as a custom provider

Hermes already uses this exact shape for `qwen36-moe-local`. Add a sibling under
`providers:` in `~/.hermes/config.yaml`:

```yaml
providers:
  amalia-v1:
    name: "Fugu Local (Conductor orchestrator)"
    api_key: "no-key-required"
    api_mode: chat_completions
    base_url: http://localhost:8900/v1
    default_model: amalia-v1
    models:
      amalia-v1:
        context_length: 32768
```

Then either set it as the main model:

```bash
hermes config set model.provider amalia-v1
hermes config set model.default  amalia-v1
```

…or, better, keep it as a **delegated worker model** so only heavy subtasks pay the
orchestration cost. In a `delegate_task` call or a cron job, pin
`provider=custom:amalia-v1`, `model=amalia-v1`.

Verify:

```bash
hermes -m amalia-v1 --provider custom:amalia-v1 -z "Write and verify a Python is_prime(n)."
```

> Note the `custom:` prefix when referencing the provider on the CLI / in cron `model`
> overrides — Hermes namespaces config-defined providers under `custom:<name>`.

---

## OpenClaw — add as a model provider

Edit `~/.openclaw/openclaw.json` under `models.providers` (merge mode is already on):

```json
{
  "models": {
    "mode": "merge",
    "providers": {
      "amalia-v1": {
        "baseUrl": "http://localhost:8900/v1",
        "apiKey": "no-key-required",
        "api": "openai-completions",
        "models": ["amalia-v1"]
      }
    }
  }
}
```

Validate + reload:

```bash
openclaw doctor          # or: openclaw doctor --repair
# then restart the gateway
```

Point an agent (or a subagent role) at `amalia-v1` as its model. Any agent so
configured transparently gets Conductor orchestration over the worker pool.

---

## Tuning knobs

All in the `conductor:` block of the active `config.yaml`:

| Knob | Effect |
|---|---|
| `max_steps` | Max workflow steps the orchestrator may emit (paper: 5). |
| `max_recursion` | Refine/verify rounds. 0 = off (fast). Higher = more test-time compute. |
| `parse_retries` | Retries if the orchestrator emits an unparseable workflow before falling back. |
| `fallback_worker` | Worker used if the orchestrator never yields a valid workflow. |

Per-request override (any OpenAI client can pass extra fields):

```json
{"model": "amalia-v1", "messages": [...], "amalia_max_recursion": 2}
```

Set `AMALIA_DEBUG=1` to attach the full workflow trace (`choices[0].message.amalia_trace`):
which workers ran, the emitted `model_id`/`subtasks`/`access_list`, and per-step ok/error.

---

## Swapping the pool (the "sovereignty" knob)

Edit the `pool:` list in `config.yaml`. Add/remove/reorder workers freely — the
orchestrator only ever sees them as ordinals (`Model 0/1/2`) plus a capability hint,
so it adapts without any retraining. Point workers at local llama.cpp servers, the
:4141 bridge, ollama, or any vendor API. If a provider goes away, drop it from the
list and restart; nothing else changes.
