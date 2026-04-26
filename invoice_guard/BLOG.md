# InvoiceGuard: How I Trained a 4B Model for AP Exception Resolution

Live demo webpage: [piyush-mk.github.io/InvoiceGuard](https://piyush-mk.github.io/InvoiceGuard/)

## Why I Built This

I built InvoiceGuard to model a real enterprise workflow: three-way invoice matching in accounts payable. In this workflow, a reviewer must check invoices against POs, GRNs, vendor risk signals, and internal policy before deciding whether to approve, hold, reject, or escalate.

I chose this because it is practical, high impact, and naturally agentic. A model cannot solve it with one-shot pattern matching. It has to investigate, compare evidence, and then commit to a final decision.

## How I Built the Environment

I implemented InvoiceGuard as an OpenEnv environment with 22 tasks (12 canonical + 10 hard variants). I designed the tasks to cover both straightforward cases and deceptive traps like split-invoice behavior, false duplicate signals, and retroactive policy edge cases.

Each episode gives the agent a step budget and a structured action space:
- inspection actions (`inspect_purchase_order`, `inspect_goods_receipt_note`, etc.),
- comparison actions (`compare_price`, `compare_quantity`, `compare_totals`),
- and a terminal action: `submit_final_resolution`.

The grader is deterministic and scores six dimensions: decision correctness, exception type, evidence quality, investigation quality, explanation quality, and efficiency.

## Baselines: The First Surprise

My API baselines looked strong. Qwen3-4B via router reached around `0.83` on canonical and `0.75` on hard slices.

But my local training baseline was very different: under my real training constraints, the same model dropped to `0.137`. It investigated correctly but almost never submitted a final resolution. That became my Round 2 target: teach the model to conclude.

## What Failed First

### Full-Trace SFT (4-bit, then bf16)

I started with full expert traces and standard LoRA SFT. It did not work:
- 4-bit full-trace SFT stayed near `0.155`.
- bf16 full-trace SFT also stayed near `0.155`.

The pattern was clear in rollouts: the model learned to keep investigating because most training tokens were investigation actions, not resolution actions.

### Early GRPO

I then tried GRPO too early. I hit low-variance trajectory batches (`group_reward_std=0.0`) and weak policy updates. The model still did not reliably learn completion behavior.

## The Turning Point: Submit-Focused SFT

I reframed the problem: the model already knew how to investigate, but it lacked termination behavior. So I trained submit behavior directly.

### Submit-only SFT (v5b)

This gave the first meaningful jump:

| Epoch | Score | Success |
|------|------|---------|
| 1 | 0.650 | 50% |
| 2 | 0.625 | 50% |
| 10 | 0.518 | 50% |

It improved behavior, but still submitted early in some trajectories.

### Submit-only SFT with deeper context (v5c)

I filtered training examples so submission examples came after deeper investigations. This was the breakthrough:

| Best epoch | Score | Success | Avg steps |
|-----------|-------|---------|-----------|
| 13 | **0.729** | **75%** | **3.0** |

### Best-checkpoint SFT (v5d)

I added explicit best-checkpoint saving and got:

| Best epoch | Score | Success |
|-----------|-------|---------|
| 9 | 0.704 | 75% |

This produced a stable checkpointed SFT variant for warm-start RL.

## SFT + GRPO

From the best SFT checkpoint, I ran warm-started GRPO (`v6c-stable`):
- warm-start init: `0.704`
- best checkpoint (iter2): `0.775`
- final iter3: `0.720`

This confirmed two practical lessons for this environment:
1. warm-started RL can improve reward quality on top of SFT,
2. best checkpoint selection matters more than taking the final iteration by default.

I also found that iteration speed mattered more than chasing bigger models. Working with a 4B model let me run many cycles, inspect failures quickly, and fix the reward and termination behavior step by step. In practice, compute budgeting and stable QLoRA-style training constraints were more useful than trying to stretch to larger models with fewer successful runs.

## Key Bugs I Had to Fix

1. **Missing EOS token in SFT targets** (`<|im_end|>`) caused generation spillover and parser failure.
2. **Qwen thinking blocks** consumed generation budget and interfered with stable action extraction.
3. **JSON extraction fragility** required robust first-object parsing.
4. **GRPO numerical instability** needed stable ratio handling and finite-loss guards.
5. **HF job/runtime plumbing issues** required script and secret handling hardening.

Without these fixes, training quality looked randomly bad even when the core idea was correct.

## Final Snapshot

| Configuration | Score | Success |
|--------------|-------|---------|
| Local baseline (no training) | 0.137 | 0% |
| Best SFT (v5c) | 0.729 | 75% |
| Best GRPO checkpoint (v6c iter2) | 0.775 | 75% |

From local baseline to best SFT, I got about **5.3x** score improvement. Warm-started GRPO improved that further at its best checkpoint.

## Curves and Evidence

### End-to-End Progression
![Round 2 Progression Score](./outputs/training_runs/round2_progression_eval_score.png)

### Stage Comparison
![Round 2 Stage Comparison](./outputs/training_runs/round2_stage_comparison.png)

### SFT Eval Score
![Eval Grader Score](./outputs/training_runs/sft_eval_grader_score.png)

### SFT Training Loss
![Training Loss](./outputs/training_runs/sft_training_loss.png)

### SFT Success Rate
![Success Rate](./outputs/training_runs/sft_success_rate.png)

### Average Steps
![Average Steps](./outputs/training_runs/sft_avg_steps.png)

### GRPO Signals
![GRPO Training Signals](./outputs/training_runs/round2_grpo_training_signals.png)

### GRPO Loss Components
![GRPO Loss Components](./outputs/training_runs/round2_grpo_loss_components_log.png)

## Conclusion

I started with a good environment and strong API baselines, but my local trainable setup exposed a major behavior gap: the model investigated but did not finish. The core win was not a bigger model or more compute. The core win was changing the training objective to target the missing behavior (`submit_final_resolution`) and fixing pipeline-level bugs that were masking real progress.

InvoiceGuard now demonstrates the full story I wanted: realistic enterprise tasks, deterministic grading, measurable post-training gains, and reproducible artifacts.

## What I Would Change Next

If I had one more iteration cycle, I would:
- add adaptive checkpoint selection during GRPO (automatic early-stop on holdout regression),
- expand hard-task curriculum during SFT before RL,
- add uncertainty-aware submission behavior (confidence calibration),
- increase cross-case memory signals for split-invoice and identity-mismatch families,
- and publish a larger all-task adapter eval sweep for every saved checkpoint.

## Artifacts and Links

- HF Space: [piyush-mk/invoice-guard](https://huggingface.co/spaces/piyush-mk/invoice-guard)
- YouTube demo: [youtu.be/sZv2oE8gL5A](https://youtu.be/sZv2oE8gL5A)
- Training code: [piyush-mk/invoiceguard-code](https://huggingface.co/piyush-mk/invoiceguard-code)
- Best SFT checkpoint: [piyush-mk/invoiceguard-qwen3-4b-sft-v5d-submit-deep-best](https://huggingface.co/piyush-mk/invoiceguard-qwen3-4b-sft-v5d-submit-deep-best)
- Completed all-task adapter eval run: [HF Job 69ed929cd70108f37acdf80b](https://huggingface.co/jobs/piyush-mk/69ed929cd70108f37acdf80b)
- Repo outputs: `invoice_guard/outputs/training_runs/` and `invoice_guard/outputs/job_reports/`
