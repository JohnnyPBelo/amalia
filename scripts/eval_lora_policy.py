#!/usr/bin/env python3
"""Evaluate a trained Amalia GRPO LoRA policy against verifiable tasks.

This evaluates the *policy checkpoint* directly with Transformers/PEFT, then executes
its parsed Conductor workflow against the configured worker pool. It is the missing
link between a GRPO training run and an honest post-training score: the runtime
llama.cpp server may still be serving the base model, while this script loads the
trained LoRA adapter.

Examples:
  # Evaluate the latest adapter after training finishes
  HSA_OVERRIDE_GFX_VERSION=11.0.0 .venv-train/bin/python scripts/eval_lora_policy.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapter grpo_out/grpo_7b_gpt55_xhigh_200_20260624_162020 \
    --config config.yaml --out eval_trained.jsonl

  # Baseline: same HF base policy, no adapter
  HSA_OVERRIDE_GFX_VERSION=11.0.0 .venv-train/bin/python scripts/eval_lora_policy.py \
    --model Qwen/Qwen2.5-7B-Instruct --config config.yaml --out eval_base.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root import works when run as scripts/eval_lora_policy.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# gfx1151 (Radeon 8060S) needs the gfx1100 kernel override before importing torch.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from amalia.config import load_config  # noqa: E402
from amalia.engine import WorkflowEngine  # noqa: E402
from amalia.parser import WorkflowParseError, parse_workflow  # noqa: E402
from amalia.prompts import DEFAULT_FEWSHOT, build_conductor_prompt  # noqa: E402
from amalia.training.tasks import get_tasks  # noqa: E402


def load_policy(model_name: str, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def generate_workflow(tok, model, prompt: str, max_new_tokens: int, temperature: float) -> str:
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


async def eval_one(tok, model, cfg, task, args) -> dict:
    prompt = build_conductor_prompt(
        user_question=task.question,
        available_models=cfg.pool.ordinal_listing(),
        max_steps=cfg.conductor.max_steps,
        few_shot=DEFAULT_FEWSHOT,
    )
    t0 = time.time()
    completion = generate_workflow(tok, model, prompt, args.max_new_tokens, args.temperature)
    row = {
        "id": task.id,
        "domain": task.domain,
        "question": task.question,
        "completion_preview": completion[:1000],
        "parse_ok": False,
        "ok": False,
        "latency_s": None,
    }
    try:
        wf = parse_workflow(completion, n_models=cfg.pool.n, max_steps=cfg.conductor.max_steps)
    except WorkflowParseError as e:
        row.update({"error": f"parse_error: {e}", "latency_s": round(time.time() - t0, 3)})
        return row
    row.update({
        "parse_ok": True,
        "workflow": {"model_id": wf.model_id, "subtasks": wf.subtasks, "access_list": wf.access_list},
    })
    if wf.is_empty():
        row.update({"error": "empty_workflow", "latency_s": round(time.time() - t0, 3)})
        return row
    engine = WorkflowEngine(cfg.pool)
    try:
        result = await engine.execute(wf, task.prompt())
    except Exception as e:  # noqa: BLE001
        row.update({"error": f"execution_error: {e!r}", "latency_s": round(time.time() - t0, 3)})
        return row
    ok = task.check(result.final_answer)
    row.update({
        "ok": ok,
        "final_answer": result.final_answer,
        "n_worker_calls": result.n_worker_calls,
        "steps": [{"idx": s.idx, "model_id": s.model_id, "worker": s.worker_name,
                   "ok": s.ok, "error": s.error} for s in result.steps],
        "latency_s": round(time.time() - t0, 3),
    })
    return row


async def main_async(args) -> int:
    cfg = load_config(args.config)
    tasks = get_tasks(args.ids.split(",") if args.ids else None)
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"loading policy model={args.model} adapter={args.adapter or '<none>'}", flush=True)
    tok, model = load_policy(args.model, args.adapter)
    print(f"eval tasks={len(tasks)} pool={cfg.pool.ordinal_listing()!r}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    rows = []
    with out_path.open("w", encoding="utf-8") as f:
        for i, task in enumerate(tasks, 1):
            row = await eval_one(tok, model, cfg, task, args)
            rows.append(row)
            passed += int(row.get("ok") is True)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            status = "OK" if row.get("ok") else "FAIL"
            print(f"[{i:02d}/{len(tasks):02d}] {status:4s} {task.id:24s} calls={row.get('n_worker_calls')} "
                  f"lat={row.get('latency_s')}s", flush=True)
    summary = {
        "adapter": args.adapter,
        "model": args.model,
        "tasks": len(tasks),
        "passed": passed,
        "pass_rate": round(passed / len(tasks), 4) if tasks else 0.0,
        "out": str(out_path),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit for base policy baseline)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="eval_lora_policy.jsonl")
    ap.add_argument("--ids", default="", help="comma-separated task ids; default = all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
