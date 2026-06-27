#!/usr/bin/env python3
"""Manual supervised warmup for the Amalia conductor policy.

This intentionally avoids TRL's SFTTrainer because the first 7B run on the ROCm
stack produced non-finite LoRA weights after a few optimizer steps. The manual
loop is small and auditable:

- loss is computed only on the workflow completion tokens;
- prompt tokens are masked with -100;
- every optimizer step checks loss, gradient norm, and trainable LoRA weights for
  NaN/Inf before saving a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# gfx1151 (Radeon 8060S) needs the gfx1100 kernel override before importing torch.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

from .sft_workflows import SFTWorkflowExample, get_sft_examples  # noqa: E402


@dataclass(frozen=True)
class EncodedExample:
    task_id: str
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]


def encode_example(tok, example: SFTWorkflowExample, max_seq_length: int) -> EncodedExample:
    """Tokenize prompt+completion while masking prompt tokens from the loss."""
    prompt = example.prompt.rstrip() + "\n\n"
    completion = example.completion.strip() + (tok.eos_token or "")
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tok(completion, add_special_tokens=False)["input_ids"]
    if len(completion_ids) >= max_seq_length:
        raise ValueError(
            f"completion for {example.task.id} has {len(completion_ids)} tokens, "
            f">= max_seq_length={max_seq_length}"
        )
    # Keep the whole target; if truncation is needed, trim old prompt context.
    room_for_prompt = max_seq_length - len(completion_ids)
    prompt_ids = prompt_ids[-room_for_prompt:]
    input_ids = prompt_ids + completion_ids
    labels = ([-100] * len(prompt_ids)) + completion_ids
    return EncodedExample(
        task_id=example.task.id,
        input_ids=input_ids,
        labels=labels,
        attention_mask=[1] * len(input_ids),
    )


def collate_batch(batch: list[EncodedExample], pad_token_id: int):
    import torch

    max_len = max(len(x.input_ids) for x in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for ex in batch:
        pad = max_len - len(ex.input_ids)
        input_ids.append(ex.input_ids + [pad_token_id] * pad)
        labels.append(ex.labels + [-100] * pad)
        attention_mask.append(ex.attention_mask + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def batched(items: list[EncodedExample], batch_size: int) -> Iterable[list[EncodedExample]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def trainable_tensors(model):
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def assert_trainable_finite(model, where: str) -> None:
    import torch

    bad = []
    for name, param in trainable_tensors(model):
        data = param.detach()
        if not torch.isfinite(data).all():
            bad.append(name)
            if len(bad) >= 5:
                break
    if bad:
        raise FloatingPointError(f"non-finite trainable weights at {where}: {bad}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task-source", choices=["seed", "curriculum", "seed+curriculum"], default="seed+curriculum")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--adam-eps", type=float, default=1e-6)
    ap.add_argument("--max-grad-norm", type=float, default=0.1)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-seq-length", type=int, default=1536)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--torch-dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--attn-implementation", choices=["default", "eager"], default="eager")
    ap.add_argument("--filter-nonfinite-examples", action="store_true",
                    help="drop examples whose initial forward/loss is NaN/Inf on this ROCm stack")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    examples = get_sft_examples(args.task_source)
    if args.smoke:
        examples = examples[:8]
        if args.max_steps < 0:
            args.max_steps = 2
        args.epochs = 1

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    encoded = [encode_example(tok, ex, args.max_seq_length) for ex in examples]

    dtype = torch.float32 if args.torch_dtype == "fp32" else torch.bfloat16
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": {"": 0} if torch.cuda.is_available() else None,
    }
    if args.attn_implementation != "default":
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.train()
    assert_trainable_finite(model, "init")

    filtered_ids: list[str] = []
    if args.filter_nonfinite_examples:
        print("[sft-manual] screening examples for finite initial loss ...", flush=True)
        kept: list[EncodedExample] = []
        model.eval()
        with torch.no_grad():
            for ex in encoded:
                batch = collate_batch([ex], tok.pad_token_id)
                batch = {k: v.to(model.device) for k, v in batch.items()}
                out_probe = model(**batch)
                finite = torch.isfinite(out_probe.loss).item() and torch.isfinite(out_probe.logits).all().item()
                if finite:
                    kept.append(ex)
                else:
                    filtered_ids.append(ex.task_id)
                    print(f"[sft-manual] filtered nonfinite example {ex.task_id}", flush=True)
        model.train()
        encoded = kept
        if not encoded:
            raise RuntimeError("all SFT examples were filtered as non-finite")

    total_micro_batches = math.ceil(len(encoded) / args.batch_size) * args.epochs
    planned_steps = math.ceil(total_micro_batches / args.grad_accum)
    if args.max_steps > 0:
        planned_steps = min(planned_steps, args.max_steps)
    warmup_steps = max(1, int(planned_steps * args.warmup_ratio)) if planned_steps > 1 else 0

    opt = torch.optim.AdamW(
        [p for _, p in trainable_tensors(model)],
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=planned_steps)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": args.model,
        "out": str(out),
        "task_source": args.task_source,
        "records": len(encoded),
        "epochs": args.epochs,
        "planned_steps": planned_steps,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "adam_eps": args.adam_eps,
        "max_grad_norm": args.max_grad_norm,
        "warmup_steps": warmup_steps,
        "max_seq_length": args.max_seq_length,
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation,
        "filter_nonfinite_examples": args.filter_nonfinite_examples,
        "filtered_ids": filtered_ids,
        "filtered_count": len(filtered_ids),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "seed": args.seed,
        "min_len": min(len(x.input_ids) for x in encoded),
        "max_len": max(len(x.input_ids) for x in encoded),
    }
    (out / "SFT-MANUAL-CONFIG.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[sft-manual] CONFIG " + json.dumps(meta), flush=True)

    step = 0
    micro = 0
    running_loss = 0.0
    opt.zero_grad(set_to_none=True)
    try:
        for epoch in range(args.epochs):
            order = encoded[:]
            random.Random(args.seed + epoch).shuffle(order)
            for batch_examples in batched(order, args.batch_size):
                batch = collate_batch(batch_examples, tok.pad_token_id)
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss before backward at micro={micro} step={step}: {loss.item()}")
                (loss / args.grad_accum).backward()
                running_loss += float(loss.detach().cpu())
                micro += 1
                if micro % args.grad_accum != 0:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for _, p in trainable_tensors(model)],
                    max_norm=args.max_grad_norm,
                    error_if_nonfinite=True,
                )
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                assert_trainable_finite(model, f"after_step_{step}")
                avg_loss = running_loss / args.grad_accum
                running_loss = 0.0
                print(json.dumps({
                    "step": step,
                    "epoch": epoch,
                    "loss": round(avg_loss, 6),
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "lr": sched.get_last_lr()[0],
                }), flush=True)

                if not args.smoke and args.save_steps > 0 and step % args.save_steps == 0:
                    ckpt = out / f"checkpoint-{step}"
                    model.save_pretrained(ckpt)
                    tok.save_pretrained(ckpt)
                    print(f"[sft-manual] saved checkpoint {ckpt}", flush=True)
                if args.max_steps > 0 and step >= args.max_steps:
                    raise StopIteration
    except StopIteration:
        pass

    # If the last partial grad accumulation had gradients, take one final safe step.
    if micro % args.grad_accum:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for _, p in trainable_tensors(model)],
            max_norm=args.max_grad_norm,
            error_if_nonfinite=True,
        )
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        assert_trainable_finite(model, f"after_final_partial_step_{step}")
        print(json.dumps({
            "step": step,
            "partial": True,
            "grad_norm": float(grad_norm.detach().cpu()),
            "lr": sched.get_last_lr()[0],
        }), flush=True)

    if not args.smoke:
        assert_trainable_finite(model, "pre_save_final")
        model.save_pretrained(out)
        tok.save_pretrained(out)
        print(f"[sft-manual] saved final {out}", flush=True)
    print("[sft-manual] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
