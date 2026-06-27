#!/usr/bin/env python3
"""Supervised warmup for the Amalia conductor policy.

This trains the policy to emit compact, parseable workflows before GRPO. It is not
answer supervision: targets are the Conductor routing lists for the fixed 3-worker
Amalia stack.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# gfx1151 (Radeon 8060S) needs the gfx1100 kernel override before importing torch.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

from .sft_workflows import build_sft_records  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="sft_out")
    ap.add_argument("--task-source", choices=["seed", "curriculum", "seed+curriculum"], default="seed+curriculum")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--torch-dtype", choices=["bf16", "fp32"], default="bf16",
                    help="bf16 is faster; fp32 is safer if ROCm/PEFT adapters go non-finite")
    ap.add_argument("--max-grad-norm", type=float, default=0.3)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--stop-on-nonfinite", action="store_true",
                    help="abort immediately if logged loss/grad_norm becomes NaN/Inf")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    records = build_sft_records(args.task_source)
    if args.smoke:
        records = records[:8]
        if args.max_steps < 0:
            args.max_steps = 2
        args.epochs = 1.0

    ds = Dataset.from_list(records)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if args.torch_dtype == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    cfg = SFTConfig(
        output_dir=args.out,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="no" if args.smoke else "steps",
        save_steps=args.save_steps,
        report_to=[],
        bf16=args.torch_dtype == "bf16",
        fp16=False,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
    )

    print("[sft] CONFIG " + json.dumps({
        "model": args.model,
        "out": args.out,
        "task_source": args.task_source,
        "records": len(records),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_seq_length": args.max_seq_length,
        "torch_dtype": args.torch_dtype,
        "max_grad_norm": args.max_grad_norm,
        "warmup_ratio": args.warmup_ratio,
        "stop_on_nonfinite": args.stop_on_nonfinite,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "seed": args.seed,
    }), flush=True)

    class NonFiniteGuardCallback(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kwargs):  # noqa: ANN001
            if not args.stop_on_nonfinite:
                return control
            import math
            logs = logs or {}
            for key in ("loss", "grad_norm"):
                value = logs.get(key)
                if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                    print(f"[sft] EARLY_STOP nonfinite {key}={value} step={state.global_step}", flush=True)
                    control.should_training_stop = True
                    return control
            return control

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
        callbacks=[NonFiniteGuardCallback()] if args.stop_on_nonfinite else None,
    )
    print("[sft] starting training ...", flush=True)
    trainer.train()
    if not args.smoke:
        trainer.save_model(args.out)
        print(f"[sft] saved to {args.out}")
    print("[sft] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
