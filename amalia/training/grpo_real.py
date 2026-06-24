"""
Real GRPO (Phase 2b) for the Amalia orchestrator — gradient-based, on-device (ROCm).

Unlike Phase 2a (Training-Free GRPO, which optimized in *context space*), this trains
the orchestrator's WEIGHTS with TRL's GRPOTrainer + LoRA. The policy being optimized is
the orchestrator: given a task, it must emit a valid Conductor workflow (3 Python lists).

Reward design (verifiable, no judge model):
  * format reward  — the completion parses into a valid (model_id, subtasks, access_list)
    workflow under the pool size / max_steps constraints. This is the paper's format
    reward and is the dominant early-training signal (teaches the output grammar).
  * (optional) execution reward — actually run the workflow against the worker pool and
    check the final answer. Gated behind --exec-reward because it needs the bridge up and
    is slow (N generations x workers per step). Off by default for the smoke test.

Memory strategy for the A9 (Strix Halo iGPU, unified RAM):
  * LoRA (peft) so we never materialize full 7B optimizer states.
  * gradient checkpointing + bf16.
  * small num_generations + short completions.
  * model size is a CLI arg so we validate the loop on Qwen2.5-0.5B (already cached)
    before committing to a bigger policy.

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python -m amalia.training.grpo_real \
      --model Qwen/Qwen2.5-0.5B --steps 10 --num-generations 4 --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import List

# gfx1151 (Radeon 8060S) needs the gfx1100 kernel override — set BEFORE importing torch.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import torch  # noqa: E402

from .tasks import get_tasks, Task  # noqa: E402
from ..parser import parse_workflow, WorkflowParseError  # noqa: E402
from ..prompts import build_conductor_prompt, DEFAULT_FEWSHOT  # noqa: E402


# A small, fixed ordinal listing so the policy learns the grammar against a stable pool.
# (3 ordinal workers, brand-free, exactly as the Conductor sees them at inference.)
SMOKE_POOL_LISTING = (
    "Model 0: skills = general problem solving, coding, step-by-step reasoning\n"
    "Model 1: skills = arithmetic, math, number theory, precise calculation\n"
    "Model 2: skills = verification, checking answers, catching errors"
)
N_POOL = 3
MAX_STEPS = 5


def make_prompt(task: Task) -> str:
    """The training prompt: ask the orchestrator for a workflow for this task."""
    return build_conductor_prompt(
        user_question=task.question,
        available_models=SMOKE_POOL_LISTING,
        max_steps=MAX_STEPS,
        few_shot=DEFAULT_FEWSHOT,
    )


def format_reward(completions: List[str], **kwargs) -> List[float]:
    """Reward = does the completion parse into a valid workflow?

    Graded, not binary, so the policy gets a gradient toward well-formedness:
      1.0  parses cleanly into a valid, non-empty workflow
      0.5  contains all three bracketed lists but fails strict validation
      0.0  no parseable 3-list structure at all
    """
    rewards = []
    for c in completions:
        text = c if isinstance(c, str) else c[0]["content"]
        try:
            wf = parse_workflow(text, n_models=N_POOL, max_steps=MAX_STEPS)
            rewards.append(1.0 if not wf.is_empty() else 0.5)
        except WorkflowParseError:
            # partial credit if it at least emitted three bracketed lists
            n_lists = len(re.findall(r"\[.*?\]", text, re.DOTALL))
            rewards.append(0.5 if n_lists >= 3 else 0.0)
    return rewards


# ---- execution reward (optional, --exec-reward): the real signal -----------------
# Build a prompt->task map so the reward can recover the verifiable answer checker for
# whatever task generated each completion. Workers run via the FRONTIER BRIDGE, so this
# costs no GPU RAM (only the policy is on the iGPU); it IS slow (network per rollout).
_PROMPT_TO_TASK = {make_prompt(t): t for t in get_tasks()}
_EXEC_POOL = None  # lazily built WorkerPool for execution reward
REWARD_LOG_PATH = os.environ.get("AMALIA_REWARD_LOG")


def _append_reward_log(row: dict) -> None:
    """Best-effort JSONL telemetry for understanding *why* GRPO learns.

    Reward logging must never kill training: if disk/logging fails, skip it and let
    the RL loop continue. This is for analysis, not correctness.
    """
    if not REWARD_LOG_PATH:
        return
    try:
        os.makedirs(os.path.dirname(REWARD_LOG_PATH) or ".", exist_ok=True)
        with open(REWARD_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _get_exec_pool():
    global _EXEC_POOL
    if _EXEC_POOL is None:
        from ..workers import Worker, WorkerPool
        # 3 frontier workers matching SMOKE_POOL_LISTING ordinals (0 general, 1 math, 2 verify).
        _EXEC_POOL = WorkerPool([
            Worker(name="m0", model="claude-opus-4.8", base_url="http://localhost:4141/v1",
                   api_type="chat", capabilities="general", temperature=0.2, max_tokens=1024),
            Worker(name="m1", model="gemini-3.1-pro-preview", base_url="http://localhost:4141/v1",
                   api_type="chat", capabilities="math", temperature=0.1, max_tokens=1024),
            Worker(name="m2", model="gpt-5.5", base_url="http://localhost:4142/v1",
                   api_type="responses", capabilities="verify", temperature=0.1,
                   max_tokens=2048, reasoning_effort="xhigh"),
        ])
    return _EXEC_POOL


def exec_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Reward = execute the generated workflow on the worker pool; +1 if the final
    answer passes the task's verifiable checker, else a small shaped penalty.

    Composed with format_reward by the trainer (multi-objective): format teaches the
    grammar early, exec teaches *good* orchestration. Unparseable -> 0 (no execution).
    """
    import asyncio
    from ..engine import WorkflowEngine
    pool = _get_exec_pool()
    engine = WorkflowEngine(pool)

    async def run_one(prompt: str, completion: str) -> float:
        t0 = time.time()
        text = completion if isinstance(completion, str) else completion[0]["content"]
        task = _PROMPT_TO_TASK.get(prompt)
        if task is None:
            _append_reward_log({"ts": t0, "task_id": None, "reward": 0.0,
                                "reason": "unknown_prompt"})
            return 0.0
        try:
            wf = parse_workflow(text, n_models=N_POOL, max_steps=MAX_STEPS)
        except WorkflowParseError:
            _append_reward_log({"ts": t0, "task_id": task.id, "domain": task.domain,
                                "reward": 0.0, "reason": "parse_error",
                                "completion_preview": text[:500]})
            return 0.0
        if wf.is_empty():
            _append_reward_log({"ts": t0, "task_id": task.id, "domain": task.domain,
                                "reward": 0.0, "reason": "empty_workflow"})
            return 0.0
        try:
            result = await engine.execute(wf, task.prompt())
        except Exception:  # noqa: BLE001 — a bad workflow shouldn't kill the step
            _append_reward_log({"ts": t0, "task_id": task.id, "domain": task.domain,
                                "reward": -0.2, "reason": "execution_exception",
                                "workflow": {"model_id": wf.model_id,
                                             "subtasks": wf.subtasks,
                                             "access_list": wf.access_list},
                                "latency_s": round(time.time() - t0, 3)})
            return -0.2
        ok = task.check(result.final_answer)
        reward = 1.0 if ok else -0.1
        _append_reward_log({
            "ts": t0,
            "task_id": task.id,
            "domain": task.domain,
            "reward": reward,
            "ok": ok,
            "final_answer": result.final_answer,
            "n_worker_calls": result.n_worker_calls,
            "workflow": {"model_id": wf.model_id,
                         "subtasks": wf.subtasks,
                         "access_list": wf.access_list},
            "steps": [{"idx": s.idx, "model_id": s.model_id, "worker": s.worker_name,
                       "ok": s.ok, "error": s.error} for s in result.steps],
            "latency_s": round(time.time() - t0, 3),
        })
        return reward

    async def run_all():
        return await asyncio.gather(*[run_one(p, c) for p, c in zip(prompts, completions)])

    return asyncio.run(run_all())


def build_dataset(repeat: int = 8):
    """Prompt-only dataset (GRPO needs just prompts; reward comes from the funcs)."""
    from datasets import Dataset
    tasks = get_tasks()
    prompts = [make_prompt(t) for t in tasks] * repeat
    return Dataset.from_dict({"prompt": prompts})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", default="grpo_out")
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the loop")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--exec-reward", action="store_true",
                    help="add the execution reward (runs workflows on the frontier bridge; "
                         "this is the real orchestration signal but needs :4141 up and is slow)")
    ap.add_argument("--save-steps", type=int, default=25,
                    help="checkpoint every N steps (cheap insurance for long iGPU runs)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint in --out")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="trade speed for memory (OFF by default — the A9 iGPU has 96GB VRAM, "
                         "so we keep activations and skip the backward recompute for ~30-40%% speedup)")
    ap.add_argument("--reward-log", default=os.environ.get("AMALIA_REWARD_LOG"),
                    help="optional JSONL path for exec-reward telemetry (workflows, answers, rewards)")
    args = ap.parse_args()
    global REWARD_LOG_PATH
    REWARD_LOG_PATH = args.reward_log

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer, GRPOConfig

    print(f"[grpo] torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"[grpo] loading policy: {args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False
    # CRITICAL for peft + gradient checkpointing on ROCm: without this the checkpointed
    # forward produces activations with requires_grad=False and the backward pass dies
    # with "element 0 of tensors does not require grad". Only needed WITH grad checkpointing.
    if not args.no_lora and args.grad_checkpointing:
        model.enable_input_require_grads()

    peft_config = None
    if not args.no_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    cfg = GRPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.num_generations,  # one prompt's group per step
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=1024,
        learning_rate=args.lr,
        logging_steps=1,
        max_steps=args.steps,
        gradient_checkpointing=args.grad_checkpointing,
        bf16=True,
        save_strategy="no" if args.smoke else "steps",
        save_steps=args.save_steps,
        report_to=[],
        temperature=1.0,
    )

    # reward_funcs: format always on (teaches grammar); exec optional (real signal).
    reward_funcs = [format_reward]
    if args.exec_reward:
        reward_funcs.append(exec_reward)

    ds = build_dataset(repeat=4 if args.smoke else 16)
    # config dump (pre-flight check #3: reproducibility metadata at step 0)
    import json as _json
    print("[grpo] CONFIG " + _json.dumps({
        "model": args.model, "steps": args.steps, "num_generations": args.num_generations,
        "lr": args.lr, "lora": not args.no_lora, "exec_reward": args.exec_reward,
        "reward_funcs": [f.__name__ for f in reward_funcs], "save_steps": args.save_steps,
        "torch": torch.__version__, "dataset_prompts": len(ds),
        "reward_log": REWARD_LOG_PATH,
    }), flush=True)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    print("[grpo] starting training ...", flush=True)
    trainer.train(resume_from_checkpoint=args.resume or None)
    if not args.smoke:
        trainer.save_model(args.out)
        print(f"[grpo] saved to {args.out}")
    print("[grpo] DONE")


if __name__ == "__main__":
    main()
