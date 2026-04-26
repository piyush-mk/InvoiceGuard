# InvoiceGuard Space Blog

This file is intentionally separate from the Space `README.md` to satisfy hackathon submission guidance.

## What this environment does

InvoiceGuard is an OpenEnv environment for enterprise accounts payable exception resolution.  
The agent reads invoice, PO, GRN, vendor, and policy evidence; investigates with tools; and submits a final decision.

## What we trained

- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- SFT strategy: submit-focused LoRA training
- RL strategy: GRPO warm-started from best SFT checkpoint

## Baseline -> SFT -> GRPO

- Local baseline (no training): score `0.137`, success `0%`
- Best SFT checkpoint: score `0.729`, success `75%`
- GRPO warm-start best checkpoint (iter2): score `0.775`, success `75%`

The core behavior change is that the untrained model keeps investigating until timeout, while trained checkpoints learn to submit `submit_final_resolution` with grounded evidence.

## Artifacts and evidence

- Training curves and metrics are committed in `invoice_guard/outputs/training_runs/`
- Full job artifacts are in `invoice_guard/outputs/job_reports/`
- Public reproducibility notebook: `notebooks/InvoiceGuard_Round2_GRPO_Reproducibility.ipynb`

## Links

- HF Space: https://huggingface.co/spaces/piyush-mk/invoice-guard
- Main repo README: https://github.com/piyush-mk/InvoiceGuard/blob/main/README.md
- Full project blog: https://github.com/piyush-mk/InvoiceGuard/blob/main/BLOG.md
