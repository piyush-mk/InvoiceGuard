#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.2",
#   "transformers>=4.46",
#   "peft>=0.13",
#   "accelerate>=1.0",
#   "bitsandbytes>=0.43; platform_system != 'Darwin'",
#   "huggingface_hub>=0.26",
#   "trackio>=0.1.4",
#   "openenv-core[core]>=0.2.1",
#   "pydantic>=2.6",
#   "pydantic-settings>=2.0",
#   "fastapi>=0.115",
#   "uvicorn>=0.30",
#   "python-dotenv",
#   "openai>=1.40",
#   "matplotlib>=3.8",
# ]
# ///
"""
InvoiceGuard Round 2 - Trajectory-level GRPO trainer (HF Jobs UV script).

Trains a small instruction-tuned LM on the InvoiceGuard OpenEnv with a
hand-written multi-step GRPO loop:

  for each iteration over train tasks:
      sample G trajectories per task (stochastic policy)
      reward per trajectory  = env cumulative reward + alpha * grader_score
      advantage per trajectory = (reward - group_mean) / (group_std + eps)
      apply PPO-clipped policy gradient on every (obs, action) pair in
          each trajectory, weighted by that trajectory's advantage,
          regularised by KL against a frozen reference policy.

The trainer is deliberately small (no TRL GRPOTrainer dep) because TRL's
GRPO assumes single-turn rewards; our env is multi-turn agentic.

Launch on HF Jobs:
    See `invoice_guard/training/launch_hf_job.py` for the recommended
    submission flow (uploads `invoice_guard/` to a code repo on the Hub
    and points this script at it via INVOICEGUARD_CODE_REPO).

Run a tiny local smoke test (CPU/GPU, no Hub push):
    cd invoice_guard
    python -m training.train_grpo \
        --model-name Qwen/Qwen2.5-0.5B-Instruct \
        --num-iterations 1 --group-size 2 --max-train-tasks 2 \
        --no-push

Required env vars on HF Jobs:
    HF_TOKEN                -- write-scoped token (passed via `secrets=`)
    HF_USERNAME             -- pushes adapter to {HF_USERNAME}/{HUB_MODEL_ID}

Optional env vars:
    INVOICEGUARD_CODE_REPO  -- model/dataset repo containing the env code;
                               cloned into /tmp at startup if set
    HUB_MODEL_ID            -- name of the LoRA adapter repo to create
    BASE_MODEL              -- HF model id of the base policy
    TRACKIO_PROJECT         -- defaults to "invoiceguard-round2"
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# -----------------------------------------------------------------------------
# 0. Bootstrap: make `invoice_guard/` importable on HF Jobs.
# -----------------------------------------------------------------------------

def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("API_TOKEN_HF")


def _bootstrap_invoice_guard_path() -> Path:
    """Ensure `inference`, `models`, `tasks`, `server` modules can be imported.

    Priority order:
      1. INVOICEGUARD_CODE_DIR       -> already on disk
      2. INVOICEGUARD_CODE_REPO      -> hf_hub_download / snapshot_download
      3. INVOICEGUARD_GIT_URL        -> git clone --depth=1
      4. fall back to the parent dir of this file (local dev)
    """
    code_dir = os.environ.get("INVOICEGUARD_CODE_DIR")
    if code_dir and Path(code_dir).is_dir():
        sys.path.insert(0, code_dir)
        return Path(code_dir)

    repo = os.environ.get("INVOICEGUARD_CODE_REPO")
    if repo:
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            repo_id=repo,
            repo_type="model",
            token=_hf_token(),
        )
        sys.path.insert(0, local)
        return Path(local)

    git_url = os.environ.get("INVOICEGUARD_GIT_URL")
    if git_url:
        target = Path("/tmp/invoiceguard_src")
        if not target.is_dir():
            subprocess.check_call(
                ["git", "clone", "--depth=1", git_url, str(target)],
            )
        sub = target / "invoice_guard"
        sys.path.insert(0, str(sub if sub.is_dir() else target))
        return sub if sub.is_dir() else target

    here = Path(__file__).resolve().parent.parent  # invoice_guard/
    sys.path.insert(0, str(here))
    return here


_CODE_ROOT = _bootstrap_invoice_guard_path()


# -----------------------------------------------------------------------------
# 1. Heavy imports (after sys.path is set).
# -----------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from models import TaskID  # type: ignore
from server.invoice_guard_environment import InvoiceGuardEnvironment  # type: ignore
from tasks import HARD_TASK_LIST, TASK_LIST  # type: ignore
from inference import SYSTEM_PROMPT, build_action, build_observation_prompt  # type: ignore
from training.rollout import Trajectory, TrajectoryStep, rollout_episode  # type: ignore


# -----------------------------------------------------------------------------
# 2. Config.
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    base_model: str = os.environ.get("BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    hub_username: Optional[str] = os.environ.get("HF_USERNAME")
    hub_model_id: str = os.environ.get("HUB_MODEL_ID", "invoiceguard-qwen3-4b-grpo")
    trackio_project: str = os.environ.get("TRACKIO_PROJECT", "invoiceguard-round2")
    trackio_run_name: str = os.environ.get("TRACKIO_RUN_NAME", "qwen3-4b-grpo")
    artifact_dir: str = os.environ.get("ARTIFACT_DIR", "/tmp/invoiceguard-training-artifacts")

    seed: int = 42
    num_iterations: int = 3            # full passes over train tasks
    group_size: int = 4                # G trajectories per task per iteration
    max_train_tasks: Optional[int] = None  # truncate train set (smoke runs)
    eval_holdout_canonical: int = 3
    eval_holdout_hard: int = 3

    # Optimisation
    lr: float = 1e-5
    grad_clip: float = 1.0
    ppo_clip: float = 0.2
    kl_coef: float = 0.05
    grader_bonus: float = 1.0          # weight on terminal grader_score
    micro_batch_size: int = 1          # (obs, action) pairs per fwd/bwd
    bf16: bool = torch.cuda.is_available()
    use_4bit: bool = True
    gradient_checkpointing: bool = True

    # Sampling
    sample_temperature: float = 1.0
    sample_top_p: float = 0.95
    max_new_tokens: int = 384
    max_prompt_tokens: int = 2048

    # Tiny behavior warm-start. Smoke showed the raw model sometimes echoes the
    # observation instead of emitting JSON; this teaches format before RL.
    format_warmup: bool = True
    format_warmup_tasks: int = 8
    format_warmup_lr: float = 5e-5
    save_format_warmup_checkpoint: bool = True
    format_warmup_model_id: Optional[str] = os.environ.get("FORMAT_WARMUP_MODEL_ID")

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
    )

    push_to_hub: bool = True


# -----------------------------------------------------------------------------
# 3. Train / eval task split.
# -----------------------------------------------------------------------------

def split_tasks(cfg: TrainConfig) -> tuple[list[TaskID], list[TaskID]]:
    """Deterministic seeded split. Held-out tasks are NEVER trained on."""
    rng = random.Random(cfg.seed)

    canonical = list(TASK_LIST)
    hard = list(HARD_TASK_LIST)

    rng.shuffle(canonical)
    rng.shuffle(hard)

    eval_c = canonical[: cfg.eval_holdout_canonical]
    eval_h = hard[: cfg.eval_holdout_hard]
    train = canonical[cfg.eval_holdout_canonical:] + hard[cfg.eval_holdout_hard:]

    if cfg.max_train_tasks is not None:
        train = train[: cfg.max_train_tasks]

    eval_set = eval_c + eval_h
    return train, eval_set


# -----------------------------------------------------------------------------
# 4. Log-prob computation.
# -----------------------------------------------------------------------------

def _completion_logprobs(
    model,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Sum log p(completion | prompt) under `model`. Returns scalar tensor."""
    input_ids = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)

    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    # Shift: predict token t from logits at t-1.
    logits = out.logits[0, :-1, :]              # (L-1, V)
    targets = input_ids[0, 1:]                  # (L-1,)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (L-1,)

    # Only sum log-probs over the completion tokens.
    comp_len = completion_ids.shape[0]
    return token_lp[-comp_len:].sum()


# -----------------------------------------------------------------------------
# 5. Trajectory advantage computation.
# -----------------------------------------------------------------------------

def trajectory_reward(traj: Trajectory, grader_bonus: float) -> float:
    """Single scalar that GRPO will rank within a group."""
    return traj.cumulative_reward + grader_bonus * traj.grader_score


def compute_group_advantages(
    trajectories: List[Trajectory], grader_bonus: float
) -> List[float]:
    rewards = [trajectory_reward(t, grader_bonus) for t in trajectories]
    if len(rewards) < 2:
        return [max(min(r, 2.0), -2.0) for r in rewards]
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    if var < 1e-10:
        return [max(min(r, 2.0), -2.0) for r in rewards]
    std = var ** 0.5
    return [(r - mean) / std for r in rewards]


def _format_warmup_actions(env: InvoiceGuardEnvironment, task_id: TaskID) -> list[dict]:
    case = getattr(env, "_case", None)
    if case is None:
        env.reset(task_id=task_id.value)
        case = getattr(env, "_case", None)
    assert case is not None
    gt = case.ground_truth
    evidence = list(dict.fromkeys([
        "inspect_purchase_order",
        "inspect_goods_receipt_note",
        "inspect_invoice_line_items",
        "inspect_vendor_profile",
        "compare_quantity",
        "compare_price",
        "compare_totals",
        "check_for_duplicate_invoice",
        "inspect_policy_rules",
        *gt.acceptable_evidence,
    ]))
    explanation = "Key findings: " + "; ".join(gt.key_findings[:3])
    return [
        {"action_type": "inspect_purchase_order"},
        {"action_type": "inspect_goods_receipt_note"},
        {"action_type": "inspect_invoice_line_items"},
        {"action_type": "inspect_vendor_profile"},
        {"action_type": "compare_quantity"},
        {"action_type": "compare_price"},
        {"action_type": "compare_totals"},
        {"action_type": "check_for_duplicate_invoice"},
        {"action_type": "inspect_policy_rules"},
        {
            "action_type": "submit_final_resolution",
            "final_decision": gt.correct_decision.value,
            "exception_type": gt.correct_exception_type.value,
            "evidence_references": evidence,
            "explanation": explanation,
            "confidence": 0.9,
        },
    ]


def run_format_warmup(
    policy,
    tokenizer,
    optimizer,
    env: InvoiceGuardEnvironment,
    tasks: list[TaskID],
    cfg: TrainConfig,
    device: torch.device,
) -> dict:
    if not cfg.format_warmup or not tasks:
        return {"format_warmup/enabled": 0.0, "format_warmup/n_pairs": 0.0}

    old_lrs = [group["lr"] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = cfg.format_warmup_lr

    policy.train()
    n_pairs = 0
    total_loss = 0.0
    for task_id in tasks[: cfg.format_warmup_tasks]:
        obs = env.reset(task_id=task_id.value)
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for action_dict in _format_warmup_actions(env, task_id):
            user_msg = build_observation_prompt(obs, is_first=(len(messages) == 1))
            messages.append({"role": "user", "content": user_msg})
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            completion_text = json.dumps(action_dict, ensure_ascii=False)
            prompt_ids = tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=cfg.max_prompt_tokens,
            ).input_ids[0]
            completion_ids = tokenizer(
                completion_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]
            lp = _completion_logprobs(policy, prompt_ids, completion_ids, device)
            loss = -lp / max(int(completion_ids.shape[0]), 1)
            loss.backward()
            total_loss += float(loss.detach().item())
            n_pairs += 1

            messages.append({"role": "assistant", "content": completion_text})
            obs = env.step(build_action(action_dict))
            if obs.done:
                break

    if n_pairs:
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad],
            cfg.grad_clip,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for group, lr in zip(optimizer.param_groups, old_lrs):
        group["lr"] = lr

    return {
        "format_warmup/enabled": 1.0,
        "format_warmup/n_pairs": float(n_pairs),
        "format_warmup/loss": total_loss / max(n_pairs, 1),
        "format_warmup/n_tasks": float(min(len(tasks), cfg.format_warmup_tasks)),
    }


def push_adapter_checkpoint(
    policy,
    tokenizer,
    repo_id: str,
    token: str,
    *,
    commit_message: str,
) -> None:
    from huggingface_hub import create_repo

    create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=False,
        token=token,
    )
    policy.push_to_hub(repo_id, private=False, token=token, commit_message=commit_message)
    tokenizer.push_to_hub(repo_id, private=False, token=token, commit_message=commit_message)


# -----------------------------------------------------------------------------
# 6. Main training loop.
# -----------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    print(f"[setup] code_root={_CODE_ROOT}", flush=True)
    print(f"[setup] base_model={cfg.base_model}", flush=True)
    print(f"[setup] cuda available={torch.cuda.is_available()}", flush=True)

    artifact_dir = Path(cfg.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifact_dir / "metrics.jsonl"
    samples_path = artifact_dir / "rollout_samples.jsonl"
    summary_path = artifact_dir / "training_summary.json"
    metrics_history: list[dict] = []
    eval_history: list[dict] = []
    train_history: list[dict] = []
    sampled_rollouts: list[dict] = []

    run_started_at = datetime.now(timezone.utc).isoformat()

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32

    # ----- Tokenizer & policy --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[setup] loading base model ...", flush=True)
    quant_cfg = None
    if cfg.use_4bit and torch.cuda.is_available():
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        quantization_config=quant_cfg,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    if cfg.gradient_checkpointing:
        # For LoRA on quantized backbones, this is the standard memory-saving path.
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        base.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(base, lora_cfg)
    policy.print_trainable_parameters()
    policy.train()

    # Reference (frozen) = base only, no adapter applied. We use the same
    # PeftModel with adapters disabled (`policy.disable_adapter()`) to compute
    # reference log-probs in-place and avoid loading a second copy of the base.

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg.lr,
    )

    # ----- Env & task split ----------------------------------------------------
    env = InvoiceGuardEnvironment()
    train_tasks, eval_tasks = split_tasks(cfg)
    print(f"[setup] train_tasks={len(train_tasks)}  eval_tasks={len(eval_tasks)}",
          flush=True)
    print(f"[setup] holdout_eval={[t.value for t in eval_tasks]}", flush=True)

    # ----- Trackio -------------------------------------------------------------
    tracker = None
    try:
        import trackio
        tracker = trackio.init(
            project=cfg.trackio_project,
            name=cfg.trackio_run_name,
            config={
                "base_model": cfg.base_model,
                "num_iterations": cfg.num_iterations,
                "group_size": cfg.group_size,
                "lr": cfg.lr,
                "kl_coef": cfg.kl_coef,
                "ppo_clip": cfg.ppo_clip,
                "grader_bonus": cfg.grader_bonus,
                "lora_r": cfg.lora_r,
                "n_train_tasks": len(train_tasks),
                "n_eval_tasks": len(eval_tasks),
            },
        )
        print("[setup] trackio initialised", flush=True)
    except Exception as e:
        print(f"[setup] trackio disabled: {e}", flush=True)

    def _write_jsonl(path: Path, row: dict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _log(metrics: dict, step: int) -> None:
        row = {
            "step": step,
            "time": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        metrics_history.append(row)
        _write_jsonl(metrics_path, row)
        msg = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in metrics.items())
        print(f"[step {step}] {msg}", flush=True)
        if tracker is not None:
            try:
                trackio.log(metrics, step=step)
            except Exception:
                pass

    # ----- Eval helper ---------------------------------------------------------
    def evaluate(label: str, step: int) -> dict:
        policy.eval()
        scores, rewards, steps_used = [], [], []
        successes = []
        for tid in eval_tasks:
            traj = rollout_episode(
                policy, tokenizer, env, tid,
                temperature=0.0001,  # near-greedy for eval
                top_p=1.0,
                max_new_tokens=cfg.max_new_tokens,
                max_prompt_tokens=cfg.max_prompt_tokens,
                device=device,
            )
            scores.append(traj.grader_score)
            rewards.append(traj.cumulative_reward)
            steps_used.append(traj.n_steps)
            successes.append(1.0 if traj.success else 0.0)
        policy.train()
        eval_metrics = {
            f"{label}/avg_grader_score": sum(scores) / max(len(scores), 1),
            f"{label}/avg_cum_reward": sum(rewards) / max(len(rewards), 1),
            f"{label}/avg_steps": sum(steps_used) / max(len(steps_used), 1),
            f"{label}/success_rate": sum(successes) / max(len(successes), 1),
            f"{label}/n_tasks": len(eval_tasks),
        }
        _log(eval_metrics, step)
        eval_history.append({"label": label, "step": step, **eval_metrics})
        return eval_metrics

    def _record_rollout_sample(
        *,
        phase: str,
        global_step: int,
        task_id: TaskID,
        trajectories: List[Trajectory],
        advantages: List[float],
    ) -> None:
        # Keep evidence compact: one high-reward and one low-reward trace per task group.
        if not trajectories:
            return
        scored = [
            (trajectory_reward(t, cfg.grader_bonus), adv, t)
            for t, adv in zip(trajectories, advantages)
        ]
        selected = [max(scored, key=lambda x: x[0]), min(scored, key=lambda x: x[0])]
        seen = set()
        for reward_value, advantage, traj in selected:
            key = id(traj)
            if key in seen:
                continue
            seen.add(key)
            row = {
                "phase": phase,
                "step": global_step,
                "task_id": task_id.value,
                "trajectory_reward": reward_value,
                "advantage": advantage,
                "grader_score": traj.grader_score,
                "cumulative_reward": traj.cumulative_reward,
                "success": traj.success,
                "n_steps": traj.n_steps,
                "terminal_decision": traj.terminal_decision,
                "actions": [s.completion_text[:500] for s in traj.steps],
                "step_rewards": [s.reward for s in traj.steps],
            }
            sampled_rollouts.append(row)
            _write_jsonl(samples_path, row)

    def _write_plots() -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[artifacts] plot generation skipped: {e}", flush=True)
            return

        if train_history:
            xs = [r["step"] for r in train_history]
            fig, ax1 = plt.subplots(figsize=(8, 4.5))
            ax1.plot(xs, [r["train/group_reward_mean"] for r in train_history], label="group reward")
            ax1.plot(xs, [r["train/group_grader_mean"] for r in train_history], label="grader score")
            ax1.set_xlabel("training step")
            ax1.set_ylabel("score")
            ax1.set_title("InvoiceGuard training reward")
            ax1.legend()
            fig.tight_layout()
            fig.savefig(artifact_dir / "training_reward_curve.png", dpi=160)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(xs, [r["train/loss"] for r in train_history], label="loss")
            ax.plot(xs, [r["train/kl_loss"] for r in train_history], label="kl loss")
            ax.set_xlabel("training step")
            ax.set_ylabel("loss")
            ax.set_title("InvoiceGuard GRPO losses")
            ax.legend()
            fig.tight_layout()
            fig.savefig(artifact_dir / "training_loss_curve.png", dpi=160)
            plt.close(fig)

        eval_rows = [
            r for r in eval_history
            if any(k.endswith("/avg_grader_score") for k in r)
        ]
        if eval_rows:
            xs = [r["step"] for r in eval_rows]
            ys = []
            labels = []
            for r in eval_rows:
                key = next(k for k in r if k.endswith("/avg_grader_score"))
                labels.append(r["label"])
                ys.append(r[key])
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(xs, ys, marker="o")
            ax.set_xlabel("training step")
            ax.set_ylabel("holdout grader score")
            ax.set_title("InvoiceGuard holdout eval during training")
            for x, y, label in zip(xs, ys, labels):
                ax.annotate(label.replace("eval/", ""), (x, y), textcoords="offset points", xytext=(0, 6), ha="center")
            fig.tight_layout()
            fig.savefig(artifact_dir / "holdout_eval_curve.png", dpi=160)
            plt.close(fig)

    # ----- Format warm-start ---------------------------------------------------
    global_step = 0
    if cfg.format_warmup:
        print("\n=== format warm-start (JSON action behavior) ===", flush=True)
        warmup_metrics = run_format_warmup(
            policy, tokenizer, optimizer, env, train_tasks, cfg, device
        )
        _log(warmup_metrics, global_step)
        if (
            cfg.push_to_hub
            and cfg.hub_username
            and cfg.save_format_warmup_checkpoint
            and warmup_metrics.get("format_warmup/n_pairs", 0.0) > 0
        ):
            token = _hf_token()
            if not token:
                raise RuntimeError(
                    "HF_TOKEN/API_TOKEN_HF is required to save the format warm-start checkpoint."
                )
            warmup_model_id = (
                cfg.format_warmup_model_id
                or f"{cfg.hub_model_id}-format-warmup"
            )
            warmup_repo_id = f"{cfg.hub_username}/{warmup_model_id}"
            print(
                f"[push] saving format warm-start adapter to {warmup_repo_id}",
                flush=True,
            )
            push_adapter_checkpoint(
                policy,
                tokenizer,
                warmup_repo_id,
                token,
                commit_message="Save InvoiceGuard format warm-start adapter",
            )
            print(f"[push] format warm-start saved -> https://huggingface.co/{warmup_repo_id}", flush=True)

    # ----- Initial eval --------------------------------------------------------
    print("\n=== initial eval (after format warm-start) ===", flush=True)
    evaluate("eval/init", global_step)

    # ----- Training loop -------------------------------------------------------
    t_start = time.time()
    for it in range(cfg.num_iterations):
        random.shuffle(train_tasks)

        for ti, task_id in enumerate(train_tasks):
            # 1. Sample G trajectories on the same task (group).
            policy.eval()
            trajectories: List[Trajectory] = []
            for g in range(cfg.group_size):
                traj = rollout_episode(
                    policy, tokenizer, env, task_id,
                    temperature=cfg.sample_temperature,
                    top_p=cfg.sample_top_p,
                    max_new_tokens=cfg.max_new_tokens,
                    max_prompt_tokens=cfg.max_prompt_tokens,
                    device=device,
                )
                trajectories.append(traj)
            policy.train()

            # 2. Group-relative advantages.
            advantages = compute_group_advantages(trajectories, cfg.grader_bonus)

            # 3. PPO-clipped policy gradient on every (prompt, completion) pair,
            #    weighted by that trajectory's advantage, with KL vs. reference.
            optimizer.zero_grad(set_to_none=True)
            total_loss_val = 0.0
            n_pairs = 0
            kl_sum = 0.0
            pg_sum = 0.0

            for traj, adv in zip(trajectories, advantages):
                if abs(adv) < 1e-8 or not traj.steps:
                    continue
                for step in traj.steps:
                    # Current policy log-prob (with adapter active).
                    cur_lp = _completion_logprobs(
                        policy, step.prompt_ids, step.completion_ids, device
                    )
                    # Reference policy log-prob (adapter disabled).
                    with torch.no_grad():
                        with policy.disable_adapter():
                            ref_lp = _completion_logprobs(
                                policy, step.prompt_ids, step.completion_ids, device
                            )

                    # PPO-clipped surrogate. The "old" policy here is the same
                    # snapshot used to sample (we just rolled out moments ago),
                    # so on the first opt step the ratio == 1; the clip becomes
                    # active only across multiple opt steps per batch. We still
                    # apply it for stability when group_size is large.
                    log_ratio = cur_lp - ref_lp.detach()  # tiny KL surrogate
                    ratio = torch.exp(cur_lp.detach() - ref_lp.detach())
                    unclipped = ratio * adv
                    clipped = torch.clamp(
                        ratio, 1.0 - cfg.ppo_clip, 1.0 + cfg.ppo_clip
                    ) * adv
                    pg_term = -torch.min(unclipped, clipped) * cur_lp / (cur_lp.detach().abs() + 1e-6)
                    # Equivalent to -adv * log_pi (REINFORCE-style) when ratio~1,
                    # but bounded for stability.

                    kl_term = cfg.kl_coef * (cur_lp - ref_lp.detach()).pow(2).mean()

                    loss = pg_term + kl_term
                    loss.backward()

                    total_loss_val += float(loss.detach().item())
                    pg_sum += float(pg_term.detach().item())
                    kl_sum += float(kl_term.detach().item())
                    n_pairs += 1

            if n_pairs > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad],
                    cfg.grad_clip,
                )
                optimizer.step()

            global_step += 1
            group_rewards = [trajectory_reward(t, cfg.grader_bonus) for t in trajectories]
            group_scores = [t.grader_score for t in trajectories]
            train_metrics = {
                "train/iter": it,
                "train/task_idx": ti,
                "train/task_id": task_id.value,
                "train/group_reward_mean": sum(group_rewards) / len(group_rewards),
                "train/group_reward_std":
                    (sum((r - sum(group_rewards) / len(group_rewards)) ** 2
                         for r in group_rewards) / len(group_rewards)) ** 0.5,
                "train/group_grader_mean": sum(group_scores) / len(group_scores),
                "train/group_success_rate": sum(1.0 if t.success else 0.0 for t in trajectories) / len(trajectories),
                "train/avg_steps": sum(t.n_steps for t in trajectories) / len(trajectories),
                "train/n_pairs": n_pairs,
                "train/loss": total_loss_val / max(n_pairs, 1),
                "train/pg_loss": pg_sum / max(n_pairs, 1),
                "train/kl_loss": kl_sum / max(n_pairs, 1),
            }
            train_history.append({"step": global_step, **train_metrics})
            _record_rollout_sample(
                phase="train",
                global_step=global_step,
                task_id=task_id,
                trajectories=trajectories,
                advantages=advantages,
            )
            _log(
                train_metrics,
                global_step,
            )

        # End-of-iteration eval.
        print(f"\n=== eval after iteration {it + 1}/{cfg.num_iterations} ===",
              flush=True)
        evaluate(f"eval/iter{it+1}", global_step)

    total_wall_clock = time.time() - t_start
    print(f"\n[done] total wall clock: {total_wall_clock:.1f}s", flush=True)

    _write_plots()

    summary = {
        "run_started_at": run_started_at,
        "run_finished_at": datetime.now(timezone.utc).isoformat(),
        "base_model": cfg.base_model,
        "hub_model_id": cfg.hub_model_id,
        "num_iterations": cfg.num_iterations,
        "group_size": cfg.group_size,
        "train_tasks": [t.value for t in train_tasks],
        "eval_tasks": [t.value for t in eval_tasks],
        "wall_clock_s": round(total_wall_clock, 2),
        "n_metric_rows": len(metrics_history),
        "n_rollout_samples": len(sampled_rollouts),
        "artifact_files": [
            p.name for p in sorted(artifact_dir.iterdir()) if p.is_file()
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[artifacts] wrote {artifact_dir}", flush=True)

    # ----- Push LoRA adapter ---------------------------------------------------
    if cfg.push_to_hub and cfg.hub_username:
        from huggingface_hub import HfApi

        token = _hf_token()
        if not token:
            raise RuntimeError(
                "HF_TOKEN/API_TOKEN_HF is required when push_to_hub=True. "
                "Set the token secret or run with --no-push."
            )
        repo_id = f"{cfg.hub_username}/{cfg.hub_model_id}"
        print(f"[push] pushing LoRA adapter to {repo_id}", flush=True)
        push_adapter_checkpoint(
            policy,
            tokenizer,
            repo_id,
            token,
            commit_message="Save InvoiceGuard GRPO adapter",
        )
        print(f"[push] uploading training artifacts to {repo_id}/training_artifacts", flush=True)
        HfApi(token=token).upload_folder(
            folder_path=str(artifact_dir),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="training_artifacts",
            commit_message="Add InvoiceGuard GRPO training artifacts",
            token=token,
        )
        print(f"[push] done -> https://huggingface.co/{repo_id}", flush=True)
    else:
        out_dir = Path(os.environ.get("OUTPUT_DIR", "/tmp/invoiceguard-grpo"))
        out_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        print(f"[save] LoRA adapter saved locally -> {out_dir}", flush=True)


# -----------------------------------------------------------------------------
# 7. CLI.
# -----------------------------------------------------------------------------

def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", dest="base_model", default=None)
    p.add_argument("--num-iterations", type=int, default=None)
    p.add_argument("--group-size", type=int, default=None)
    p.add_argument("--max-train-tasks", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--eval-holdout-canonical", type=int, default=None)
    p.add_argument("--eval-holdout-hard", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--max-prompt-tokens", type=int, default=None)
    p.add_argument("--no-format-warmup", action="store_true")
    p.add_argument("--format-warmup-tasks", type=int, default=None)
    p.add_argument("--no-save-format-warmup", action="store_true")
    p.add_argument("--format-warmup-model-id", default=None)
    args = p.parse_args()

    cfg = TrainConfig()
    if args.base_model:
        cfg.base_model = args.base_model
    if args.num_iterations is not None:
        cfg.num_iterations = args.num_iterations
    if args.group_size is not None:
        cfg.group_size = args.group_size
    if args.max_train_tasks is not None:
        cfg.max_train_tasks = args.max_train_tasks
    if args.lr is not None:
        cfg.lr = args.lr
    if args.seed is not None:
        cfg.seed = args.seed
    if args.eval_holdout_canonical is not None:
        cfg.eval_holdout_canonical = args.eval_holdout_canonical
    if args.eval_holdout_hard is not None:
        cfg.eval_holdout_hard = args.eval_holdout_hard
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if args.max_prompt_tokens is not None:
        cfg.max_prompt_tokens = args.max_prompt_tokens
    if args.no_format_warmup:
        cfg.format_warmup = False
    if args.format_warmup_tasks is not None:
        cfg.format_warmup_tasks = args.format_warmup_tasks
    if args.no_save_format_warmup:
        cfg.save_format_warmup_checkpoint = False
    if args.format_warmup_model_id:
        cfg.format_warmup_model_id = args.format_warmup_model_id
    if args.no_push:
        cfg.push_to_hub = False
    if args.no_4bit:
        cfg.use_4bit = False
    if args.no_gradient_checkpointing:
        cfg.gradient_checkpointing = False
    return cfg


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
