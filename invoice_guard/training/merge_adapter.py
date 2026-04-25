#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.2",
#   "transformers>=4.46",
#   "peft>=0.13",
#   "accelerate>=1.0",
#   "huggingface_hub>=0.26",
#   "safetensors>=0.4",
# ]
# ///
"""Merge a PEFT LoRA adapter into its base model and push the merged model."""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
from huggingface_hub import create_repo
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("API_TOKEN_HF")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507"))
    p.add_argument("--adapter-repo", required=True)
    p.add_argument("--merged-repo", required=True)
    args = p.parse_args()

    token = _hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN/API_TOKEN_HF is required to push merged model.")

    print(f"[setup] base_model={args.base_model}", flush=True)
    print(f"[setup] adapter_repo={args.adapter_repo}", flush=True)
    print(f"[setup] merged_repo={args.merged_repo}", flush=True)
    print(f"[setup] cuda available={torch.cuda.is_available()}", flush=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base.config.pad_token_id = tokenizer.pad_token_id

    peft_model = PeftModel.from_pretrained(base, args.adapter_repo, token=token)
    merged = peft_model.merge_and_unload()

    create_repo(repo_id=args.merged_repo, repo_type="model", exist_ok=True, private=False, token=token)
    print("[push] pushing merged model", flush=True)
    merged.push_to_hub(
        args.merged_repo,
        private=False,
        safe_serialization=True,
        token=token,
        commit_message="Save merged InvoiceGuard model",
    )
    tokenizer.push_to_hub(
        args.merged_repo,
        private=False,
        token=token,
        commit_message="Save merged InvoiceGuard tokenizer",
    )
    print(f"[push] done -> https://huggingface.co/{args.merged_repo}", flush=True)


if __name__ == "__main__":
    main()
