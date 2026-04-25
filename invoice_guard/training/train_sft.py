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
# ]
# ///
"""InvoiceGuard supervised trace fine-tuning backup run.

This is intentionally separate from GRPO. It trains a LoRA adapter on
environment-generated expert traces so we have a deterministic supervised
fallback artifact while online RL jobs are running.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("API_TOKEN_HF")


def _bootstrap_invoice_guard_path() -> Path:
    code_dir = os.environ.get("INVOICEGUARD_CODE_DIR")
    if code_dir and Path(code_dir).is_dir():
        sys.path.insert(0, code_dir)
        return Path(code_dir)

    repo = os.environ.get("INVOICEGUARD_CODE_REPO")
    if repo:
        from huggingface_hub import snapshot_download

        local = snapshot_download(repo_id=repo, repo_type="model", token=_hf_token())
        sys.path.insert(0, local)
        return Path(local)

    here = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(here))
    return here


_CODE_ROOT = _bootstrap_invoice_guard_path()

import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, create_repo
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from inference import SYSTEM_PROMPT, build_action, build_observation_prompt  # type: ignore
from models import TaskID  # type: ignore
from server.invoice_guard_environment import InvoiceGuardEnvironment  # type: ignore
from tasks import HARD_TASK_LIST, TASK_LIST  # type: ignore
from training.rollout import rollout_episode  # type: ignore


@dataclass
class SftConfig:
    base_model: str = os.environ.get("BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    hub_username: Optional[str] = os.environ.get("HF_USERNAME")
    hub_model_id: str = os.environ.get("HUB_MODEL_ID", "invoiceguard-qwen3-4b-sft")
    trackio_project: str = os.environ.get("TRACKIO_PROJECT", "invoiceguard-round2")
    trackio_run_name: str = os.environ.get("TRACKIO_RUN_NAME", "qwen3-4b-sft")
    artifact_dir: str = os.environ.get("ARTIFACT_DIR", "/tmp/invoiceguard-sft-artifacts")

    seed: int = 42
    num_epochs: int = 4
    max_train_tasks: Optional[int] = None
    eval_holdout_canonical: int = 3
    eval_holdout_hard: int = 3
    eval_every_epoch: bool = True
    submit_only: bool = False
    min_investigation_steps: int = 0

    lr: float = 5e-5
    grad_clip: float = 1.0
    max_prompt_tokens: int = 2048
    max_new_tokens: int = 384
    bf16: bool = torch.cuda.is_available()
    use_4bit: bool = True
    gradient_checkpointing: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    push_to_hub: bool = True


def split_tasks(cfg: SftConfig) -> tuple[list[TaskID], list[TaskID]]:
    rng = random.Random(cfg.seed)
    canonical = list(TASK_LIST)
    hard = list(HARD_TASK_LIST)
    rng.shuffle(canonical)
    rng.shuffle(hard)
    eval_tasks = (
        canonical[: cfg.eval_holdout_canonical]
        + hard[: cfg.eval_holdout_hard]
    )
    train_tasks = (
        canonical[cfg.eval_holdout_canonical:]
        + hard[cfg.eval_holdout_hard:]
    )
    if cfg.max_train_tasks is not None:
        train_tasks = train_tasks[: cfg.max_train_tasks]
    return train_tasks, eval_tasks


_ALL_INVESTIGATION_ACTIONS = [
    {"action_type": "inspect_purchase_order"},
    {"action_type": "inspect_goods_receipt_note"},
    {"action_type": "inspect_invoice_line_items"},
    {"action_type": "inspect_vendor_profile"},
    {"action_type": "compare_quantity"},
    {"action_type": "compare_price"},
    {"action_type": "compare_totals"},
    {"action_type": "check_for_duplicate_invoice"},
    {"action_type": "inspect_policy_rules"},
]


def _expert_actions(
    env: InvoiceGuardEnvironment,
    task_id: TaskID,
    max_investigation_steps: int = 9,
) -> list[dict]:
    case = getattr(env, "_case", None)
    if case is None:
        env.reset(task_id=task_id.value)
        case = getattr(env, "_case", None)
    assert case is not None
    gt = case.ground_truth
    investigation = _ALL_INVESTIGATION_ACTIONS[:max_investigation_steps]
    used_names = [a["action_type"] for a in investigation]
    evidence = list(dict.fromkeys([*used_names, *gt.acceptable_evidence]))
    return [
        *investigation,
        {
            "action_type": "submit_final_resolution",
            "final_decision": gt.correct_decision.value,
            "exception_type": gt.correct_exception_type.value,
            "evidence_references": evidence,
            "explanation": "Key findings: " + "; ".join(gt.key_findings[:3]),
            "confidence": 0.9,
        },
    ]


TRACE_LENGTHS = [3, 5, 7, 9]
SUBMIT_LOSS_WEIGHT = 5.0


def build_sft_examples(
    tokenizer,
    env: InvoiceGuardEnvironment,
    tasks: list[TaskID],
    max_prompt_tokens: int,
) -> list[dict]:
    examples: list[dict] = []
    for task_id in tasks:
        for n_inv in TRACE_LENGTHS:
            obs = env.reset(task_id=task_id.value)
            messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
            for action_dict in _expert_actions(env, task_id, max_investigation_steps=n_inv):
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
                    max_length=max_prompt_tokens,
                ).input_ids[0]
                comp_enc = tokenizer(
                    completion_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids[0]
                eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
                if eos_id is not None and eos_id != tokenizer.unk_token_id:
                    completion_ids = torch.cat([comp_enc, torch.tensor([eos_id])])
                else:
                    completion_ids = comp_enc
                examples.append({
                    "task_id": task_id.value,
                    "action_type": action_dict["action_type"],
                    "prompt_ids": prompt_ids,
                    "completion_ids": completion_ids,
                    "completion_text": completion_text,
                    "trace_inv_steps": n_inv,
                })
                messages.append({"role": "assistant", "content": completion_text})
                obs = env.step(build_action(action_dict))
                if obs.done:
                    break
    return examples


def completion_loss(
    model,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    device: torch.device,
    weight: float = 1.0,
) -> torch.Tensor:
    input_ids = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[0, :-1, :]
    targets = input_ids[0, 1:]
    logprobs = F.log_softmax(logits.float(), dim=-1)
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    comp_len = completion_ids.shape[0]
    return -token_lp[-comp_len:].mean() * weight


def main() -> None:
    cfg = _parse_args()
    token = _hf_token()
    if cfg.push_to_hub and (not token or not cfg.hub_username):
        raise RuntimeError("HF_TOKEN and HF_USERNAME are required when pushing SFT output.")

    print(f"[setup] code_root={_CODE_ROOT}", flush=True)
    print(f"[setup] base_model={cfg.base_model}", flush=True)
    print(f"[setup] cuda available={torch.cuda.is_available()}", flush=True)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32

    artifact_dir = Path(cfg.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifact_dir / "sft_metrics.jsonl"
    summary_path = artifact_dir / "sft_summary.json"

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        token=token,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    if cfg.gradient_checkpointing:
        if cfg.use_4bit:
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
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    env = InvoiceGuardEnvironment()
    train_tasks, eval_tasks = split_tasks(cfg)
    examples = build_sft_examples(tokenizer, env, train_tasks, cfg.max_prompt_tokens)
    if cfg.submit_only:
        examples = [ex for ex in examples if ex["action_type"] == "submit_final_resolution"]
        print(f"[setup] submit-only mode: filtered to {len(examples)} submit examples", flush=True)
    if cfg.min_investigation_steps > 0:
        before = len(examples)
        examples = [ex for ex in examples if ex.get("trace_inv_steps", 0) >= cfg.min_investigation_steps]
        print(f"[setup] min_investigation_steps={cfg.min_investigation_steps}: {before} -> {len(examples)} examples", flush=True)
    print(f"[setup] train_tasks={len(train_tasks)} eval_tasks={len(eval_tasks)} examples={len(examples)}", flush=True)

    tracker = None
    try:
        import trackio
        tracker = trackio.init(
            project=cfg.trackio_project,
            name=cfg.trackio_run_name,
            config={
                "base_model": cfg.base_model,
                "hub_model_id": cfg.hub_model_id,
                "num_epochs": cfg.num_epochs,
                "n_train_tasks": len(train_tasks),
                "n_eval_tasks": len(eval_tasks),
                "n_examples": len(examples),
                "lr": cfg.lr,
                "lora_r": cfg.lora_r,
            },
        )
        print("[setup] trackio initialised", flush=True)
    except Exception as e:
        print(f"[setup] trackio disabled: {e}", flush=True)

    def log(row: dict) -> None:
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **row}) + "\n")
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        if tracker is not None:
            try:
                trackio.log(row, step=int(row.get("step", 0)))
            except Exception:
                pass

    def evaluate(epoch: int) -> dict:
        model.eval()
        scores, rewards, successes, steps = [], [], [], []
        for task_id in eval_tasks:
            traj = rollout_episode(
                model,
                tokenizer,
                env,
                task_id,
                temperature=0.0001,
                top_p=1.0,
                max_new_tokens=cfg.max_new_tokens,
                max_prompt_tokens=cfg.max_prompt_tokens,
                device=device,
            )
            scores.append(traj.grader_score)
            rewards.append(traj.cumulative_reward)
            successes.append(1.0 if traj.success else 0.0)
            steps.append(traj.n_steps)
        model.train()
        return {
            "step": epoch,
            "eval/avg_grader_score": sum(scores) / max(len(scores), 1),
            "eval/avg_cum_reward": sum(rewards) / max(len(rewards), 1),
            "eval/success_rate": sum(successes) / max(len(successes), 1),
            "eval/avg_steps": sum(steps) / max(len(steps), 1),
        }

    t_start = time.time()
    global_step = 0
    best_eval_score = -1.0
    best_epoch_dir: Optional[Path] = None
    for epoch in range(cfg.num_epochs):
        random.shuffle(examples)
        total_loss = 0.0
        model.train()
        for i, ex in enumerate(examples, 1):
            w = SUBMIT_LOSS_WEIGHT if ex["action_type"] == "submit_final_resolution" else 1.0
            loss = completion_loss(model, ex["prompt_ids"], ex["completion_ids"], device, weight=w)
            loss.backward()
            total_loss += float(loss.detach().item())
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if i % 25 == 0:
                log({"step": global_step, "train/epoch": epoch + 1, "train/example": i, "train/loss": total_loss / i})
        log({"step": global_step, "train/epoch": epoch + 1, "train/loss": total_loss / max(len(examples), 1)})
        if cfg.eval_every_epoch:
            eval_result = evaluate(epoch + 1)
            log(eval_result)
            score = eval_result.get("eval/avg_grader_score", 0.0)
            if score > best_eval_score:
                best_eval_score = score
                best_epoch_dir = artifact_dir / f"best_epoch_{epoch+1}"
                best_epoch_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(best_epoch_dir))
                tokenizer.save_pretrained(str(best_epoch_dir))
                print(f"[checkpoint] new best epoch {epoch+1}: score={score:.4f}", flush=True)

    summary = {
        "run_finished_at": datetime.now(timezone.utc).isoformat(),
        "base_model": cfg.base_model,
        "hub_model_id": cfg.hub_model_id,
        "num_epochs": cfg.num_epochs,
        "train_tasks": [t.value for t in train_tasks],
        "eval_tasks": [t.value for t in eval_tasks],
        "n_examples": len(examples),
        "wall_clock_s": round(time.time() - t_start, 2),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if cfg.push_to_hub and cfg.hub_username:
        assert token is not None
        repo_id = f"{cfg.hub_username}/{cfg.hub_model_id}"
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False, token=token)
        print(f"[push] pushing SFT adapter to {repo_id}", flush=True)
        model.push_to_hub(repo_id, private=False, token=token, commit_message="Save InvoiceGuard SFT adapter")
        tokenizer.push_to_hub(repo_id, private=False, token=token, commit_message="Save InvoiceGuard SFT tokenizer")
        HfApi(token=token).upload_folder(
            folder_path=str(artifact_dir),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="sft_artifacts",
            token=token,
            commit_message="Add InvoiceGuard SFT artifacts",
        )
        print(f"[push] done -> https://huggingface.co/{repo_id}", flush=True)

        if best_epoch_dir and best_epoch_dir.exists():
            best_repo = f"{repo_id}-best"
            create_repo(repo_id=best_repo, repo_type="model", exist_ok=True, private=False, token=token)
            HfApi(token=token).upload_folder(
                folder_path=str(best_epoch_dir),
                repo_id=best_repo,
                repo_type="model",
                token=token,
                commit_message=f"Best epoch checkpoint (score={best_eval_score:.4f})",
            )
            print(f"[push] best checkpoint -> https://huggingface.co/{best_repo}", flush=True)


def _parse_args() -> SftConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", dest="base_model", default=None)
    p.add_argument("--hub-model-id", default=None)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--max-train-tasks", type=int, default=None)
    p.add_argument("--eval-holdout-canonical", type=int, default=None)
    p.add_argument("--eval-holdout-hard", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--max-prompt-tokens", type=int, default=None)
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--submit-only", action="store_true")
    p.add_argument("--min-investigation-steps", type=int, default=None)
    args = p.parse_args()

    cfg = SftConfig()
    if args.base_model:
        cfg.base_model = args.base_model
    if args.hub_model_id:
        cfg.hub_model_id = args.hub_model_id
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.max_train_tasks is not None:
        cfg.max_train_tasks = args.max_train_tasks
    if args.eval_holdout_canonical is not None:
        cfg.eval_holdout_canonical = args.eval_holdout_canonical
    if args.eval_holdout_hard is not None:
        cfg.eval_holdout_hard = args.eval_holdout_hard
    if args.lr is not None:
        cfg.lr = args.lr
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if args.max_prompt_tokens is not None:
        cfg.max_prompt_tokens = args.max_prompt_tokens
    if args.no_push:
        cfg.push_to_hub = False
    if args.no_4bit:
        cfg.use_4bit = False
    if args.submit_only:
        cfg.submit_only = True
    if args.min_investigation_steps is not None:
        cfg.min_investigation_steps = args.min_investigation_steps
    return cfg


if __name__ == "__main__":
    main()
