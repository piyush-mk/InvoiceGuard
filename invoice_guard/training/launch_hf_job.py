"""
Submit the InvoiceGuard GRPO training as a Hugging Face Jobs UV job.

What this does:
  1. Bundles the local `invoice_guard/` source folder and uploads it to a
     dedicated *code* repo on the Hub (default: {user}/invoiceguard-code).
     The training script clones it back inside the Job container so the env,
     tasks, models, grader, etc. are available.
  2. Reads `train_grpo.py` from disk and submits it inline via `run_uv_job`.
  3. Sets `INVOICEGUARD_CODE_REPO`, `HF_USERNAME`, etc. as env vars in the
     job, plus passes HF_TOKEN as a secret so push-to-hub works.

Usage:
    cd invoice_guard
    python training/launch_hf_job.py \
        --hf-username <your-username> \
        --flavor a10g-large \
        --timeout 4h \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --num-iterations 3 --group-size 4

Requires:
    - `pip install huggingface_hub` locally
    - `hf auth login` already done
    - HF Pro / Team / Enterprise plan (Jobs require a paid plan)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent  # invoice_guard/
TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_grpo.py"


def upload_code(hf_username: str, code_repo_name: str) -> str:
    from huggingface_hub import HfApi, create_repo

    repo_id = f"{hf_username}/{code_repo_name}"
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    api = HfApi()
    print(f"[upload] {REPO_DIR} -> {repo_id}", flush=True)
    api.upload_folder(
        folder_path=str(REPO_DIR),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=[
            "outputs/**",
            ".venv/**",
            "__pycache__/**",
            "*.pyc",
            ".env",
            ".env.example",
        ],
        commit_message="Sync InvoiceGuard code for GRPO training job",
    )
    print(f"[upload] done -> https://huggingface.co/{repo_id}", flush=True)
    return repo_id


def submit_job(args: argparse.Namespace, code_repo_id: str) -> None:
    from huggingface_hub import run_uv_job

    # `run_uv_job`'s Python API expects a local path or URL. Passing the script
    # contents directly makes Jobs try to spawn a command whose "filename" is
    # the entire source file. Use the script we just uploaded to the code repo.
    script_url = (
        f"https://huggingface.co/{code_repo_id}/resolve/main/training/train_grpo.py"
    )

    job = run_uv_job(
        script=script_url,
        flavor=args.flavor,
        timeout=args.timeout,
        secrets={"HF_TOKEN": "$HF_TOKEN"},
        env={
            "INVOICEGUARD_CODE_REPO": code_repo_id,
            "HF_USERNAME": args.hf_username,
            "HUB_MODEL_ID": args.hub_model_id,
            "BASE_MODEL": args.base_model,
            "TRACKIO_PROJECT": args.trackio_project,
            "TRACKIO_RUN_NAME": args.run_name,
        },
        script_args=[
            "--num-iterations", str(args.num_iterations),
            "--group-size", str(args.group_size),
        ] + (["--max-train-tasks", str(args.max_train_tasks)]
             if args.max_train_tasks else []),
    )
    print("\n[submit] job submitted!", flush=True)
    print(f"  job id : {getattr(job, 'id', job)}", flush=True)
    url = getattr(job, "url", None)
    if url:
        print(f"  monitor: {url}", flush=True)
    print(f"  flavor : {args.flavor}", flush=True)
    print(f"  timeout: {args.timeout}", flush=True)
    print(f"  trackio: project={args.trackio_project}  run={args.run_name}",
          flush=True)
    print("\nCheck status later with `hf jobs ps` or `hf jobs logs <id>`.",
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-username", required=True)
    p.add_argument("--code-repo-name", default="invoiceguard-code")
    p.add_argument("--hub-model-id", default="invoiceguard-qwen25-7b-grpo")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--flavor", default="a10g-large")
    p.add_argument("--timeout", default="4h")
    p.add_argument("--num-iterations", type=int, default=3)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-train-tasks", type=int, default=None)
    p.add_argument("--trackio-project", default="invoiceguard-round2")
    p.add_argument("--run-name", default="qwen25-7b-grpo")
    p.add_argument("--skip-upload", action="store_true",
                   help="Reuse the existing code repo (no re-upload).")
    args = p.parse_args()

    code_repo_id = f"{args.hf_username}/{args.code_repo_name}"
    if not args.skip_upload:
        code_repo_id = upload_code(args.hf_username, args.code_repo_name)

    submit_job(args, code_repo_id)


if __name__ == "__main__":
    main()
