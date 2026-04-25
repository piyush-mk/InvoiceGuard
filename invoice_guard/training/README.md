# InvoiceGuard Round 2 — Trajectory-level GRPO

This package trains a small instruction-tuned LM (default `Qwen/Qwen3-4B-Instruct-2507`)
on the InvoiceGuard OpenEnv with a hand-written multi-step GRPO loop:

- Sample **G trajectories** per training task with the current (stochastic) policy.
- Trajectory reward = env cumulative reward + `grader_bonus * grader_score`.
- **Group-relative advantage** = z-score within the G trajectories of one task.
- **PPO-clipped policy gradient** on every `(observation → action)` pair in
  each trajectory, weighted by that trajectory's advantage, regularised by
  KL against a **frozen reference** (the same model with the LoRA adapter
  disabled — no second copy of the base model in memory).
- LoRA on attention projections; everything else frozen.

We deliberately do **not** use TRL's `GRPOTrainer` — it assumes a single-turn
reward, but our env is multi-turn agentic. The whole loop is in
[`train_grpo.py`](./train_grpo.py) and is ~300 lines.

## Files

| File | Purpose |
| --- | --- |
| `rollout.py` | Drives the local `InvoiceGuardEnvironment` with an HF model; reuses `inference.py`'s prompt/parse helpers so trajectories are IO-identical to the OpenAI baseline. |
| `train_grpo.py` | The trainer. Self-contained PEP 723 UV script — runnable both as a Hugging Face Jobs payload and locally for smoke tests. |
| `launch_hf_job.py` | Uploads the `invoice_guard/` source to a Hub code repo, then submits `train_grpo.py` to HF Jobs with the right env vars and secrets. |

## Train / eval split

Split is deterministic from `--seed` (default 42):

- **Holdout (never trained on):** `eval_holdout_canonical=3` canonical + `eval_holdout_hard=3` hard tasks.
- **Train:** the remaining 9 canonical + 7 hard = 16 tasks.

The end-of-iteration eval inside the trainer reports the average grader score
on the holdout, so you can see the learning curve in Trackio. The full
benchmark for the README plots is produced separately by
[`eval_round2.py`](../eval_round2.py).

## Quick local smoke test (no GPU, no Hub push)

```powershell
cd invoice_guard
..\.venv\Scripts\python -m training.train_grpo `
    --model-name Qwen/Qwen3-4B-Instruct-2507 `
    --num-iterations 1 --group-size 2 --max-train-tasks 2 --no-push
```

This uses the in-tree env (no clone), runs one iteration over 2 tasks with
G=2 trajectories each, and saves the LoRA adapter to `/tmp/invoiceguard-grpo`.
On CPU it's slow but verifies the wiring. Use a small base model.

## Launch on Hugging Face Jobs

Prereqs: HF Pro/Team/Enterprise plan, `hf auth login` done locally,
`pip install huggingface_hub`.

```powershell
cd invoice_guard
..\.venv\Scripts\python training\launch_hf_job.py `
    --hf-username <your-username> `
    --flavor a10g-large `
    --timeout 4h `
    --base-model Qwen/Qwen3-4B-Instruct-2507 `
    --num-iterations 3 --group-size 4
```

What happens:

1. The launcher uploads `invoice_guard/` to `{your-username}/invoiceguard-code`
   (creates the repo if needed; ignores `outputs/`, `.venv/`, `.env`).
2. `train_grpo.py` is submitted to HF Jobs from the uploaded Hub script URL.
3. Inside the container the script `snapshot_download`s the code repo and adds
   it to `sys.path`, then runs the GRPO loop.
4. Trackio dashboard appears at `https://huggingface.co/spaces/<user>/trackio`
   (project `invoiceguard-round2`, run `qwen3-4b-grpo`).
5. On completion the LoRA adapter is pushed to
   `{your-username}/invoiceguard-qwen3-4b-grpo`.

## After training

Run the Round 2 benchmark against the trained model:

```powershell
$env:API_BASE_URL = "<endpoint serving your-username/invoiceguard-qwen3-4b-grpo>"
$env:MODEL_NAME = "<your-username>/invoiceguard-qwen3-4b-grpo"
..\.venv\Scripts\python eval_round2.py --slice all --model-tag trained_qwen3_4b_grpo
..\.venv\Scripts\python eval_round2.py --compare `
    outputs\baseline_scores\hard__clean_qwen3_4b.json `
    outputs\round2\hard__trained_qwen3_4b_grpo.json
```

The compare output is what the README before/after plot is built from in
Stage H.

## Cost estimate

| Setting | Approx | Notes |
| --- | --- | --- |
| `l40s` / `a100-large` × 2-4h | varies | Default — Qwen3-4B-Instruct, 3 iter × 16 tasks × G=4 = 192 trajectories per epoch. |
| `a10g-large` × 3-5h | lower cost | May require shorter prompts/generation and 4-bit LoRA. |
| Smoke test (CPU) | $0 | Uses in-tree env, no Hub push. |
