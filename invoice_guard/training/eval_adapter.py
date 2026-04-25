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
#   "openenv-core[core]>=0.2.1",
#   "pydantic>=2.6",
#   "pydantic-settings>=2.0",
#   "fastapi>=0.115",
#   "uvicorn>=0.30",
#   "python-dotenv",
#   "openai>=1.40",
# ]
# ///
"""Evaluate a LoRA adapter on InvoiceGuard tasks and upload JSON artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from server.invoice_guard_environment import InvoiceGuardEnvironment  # type: ignore
from tasks import HARD_TASK_LIST, TASK_LIST  # type: ignore
from training.rollout import rollout_episode  # type: ignore


def _task_slice(name: str):
    if name == "canonical":
        return list(TASK_LIST)
    if name == "hard":
        return list(HARD_TASK_LIST)
    return list(TASK_LIST) + list(HARD_TASK_LIST)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507"))
    p.add_argument("--adapter-repo", required=True)
    p.add_argument("--slice", choices=["canonical", "hard", "all"], default="all")
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--max-prompt-tokens", type=int, default=2048)
    p.add_argument("--artifact-dir", default="/tmp/invoiceguard-adapter-eval")
    args = p.parse_args()

    token = _hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN/API_TOKEN_HF is required for adapter eval upload.")

    print(f"[setup] code_root={_CODE_ROOT}", flush=True)
    print(f"[setup] base_model={args.base_model}", flush=True)
    print(f"[setup] adapter_repo={args.adapter_repo}", flush=True)
    print(f"[setup] cuda available={torch.cuda.is_available()}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    quant_cfg = None
    if torch.cuda.is_available():
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        quantization_config=quant_cfg,
        token=token,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, args.adapter_repo, token=token)
    model.eval()

    tasks = _task_slice(args.slice)
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    env = InvoiceGuardEnvironment()
    rows = []
    for i, task_id in enumerate(tasks, 1):
        print(f"[eval] {i}/{len(tasks)} {task_id.value}", flush=True)
        traj = rollout_episode(
            model,
            tokenizer,
            env,
            task_id,
            temperature=0.0001,
            top_p=1.0,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            device=device,
        )
        rows.append({
            "task_id": task_id.value,
            "grader_score": traj.grader_score,
            "cumulative_reward": traj.cumulative_reward,
            "success": traj.success,
            "n_steps": traj.n_steps,
            "terminal_decision": traj.terminal_decision,
            "actions": [step.completion_text for step in traj.steps],
            "step_rewards": [step.reward for step in traj.steps],
        })

    summary = {
        "run_finished_at": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter_repo": args.adapter_repo,
        "slice": args.slice,
        "n_tasks": len(rows),
        "avg_grader_score": sum(r["grader_score"] for r in rows) / max(len(rows), 1),
        "avg_cumulative_reward": sum(r["cumulative_reward"] for r in rows) / max(len(rows), 1),
        "success_rate": sum(1.0 if r["success"] else 0.0 for r in rows) / max(len(rows), 1),
        "avg_steps": sum(r["n_steps"] for r in rows) / max(len(rows), 1),
    }

    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "adapter_eval_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "adapter_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    HfApi(token=token).upload_folder(
        folder_path=str(out_dir),
        repo_id=args.adapter_repo,
        repo_type="model",
        path_in_repo=f"eval_artifacts/{args.slice}",
        token=token,
        commit_message=f"Add InvoiceGuard adapter eval results ({args.slice})",
    )
    print(f"[push] eval artifacts uploaded to https://huggingface.co/{args.adapter_repo}", flush=True)


if __name__ == "__main__":
    main()
