# InvoiceGuard -- Three-Way Invoice Matching Environment

An [OpenEnv](https://github.com/meta-pytorch/openenv) environment that simulates accounts payable exception resolution. An AI agent investigates multi-document business cases -- invoices, purchase orders, goods receipt notes, vendor profiles, and company policies -- to detect discrepancies, classify exception types, and render correct decisions.

**Hugging Face Space:** [piyush-mk/invoice-guard](https://huggingface.co/spaces/piyush-mk/invoice-guard)

## Motivation

Three-way invoice matching is one of the most common and error-prone tasks in enterprise finance. Accounts payable teams manually compare invoices against purchase orders and goods receipt notes to detect overbilling, partial shipments, duplicate submissions, and price variances. This environment turns that real-world workflow into a structured evaluation benchmark where an AI agent must gather evidence through sequential investigation actions and reach a correct, policy-compliant decision.

## Tasks

| Task ID | Description | Difficulty | Expected Decision | Exception Type |
|---------|-------------|------------|-------------------|----------------|
| `task_1_clean_match` | All documents align within tolerance | Easy | `approve_for_payment` | `clean_match` |
| `task_2_partial_receipt` | Billed quantity exceeds received quantity | Moderate | `place_on_hold` | `partial_receipt` |
| `task_3_price_variance` | Unit price exceeds PO price beyond tolerance | Moderate | `escalate_for_supervisor_review` | `price_mismatch` |
| `task_4_duplicate_invoice` | Previously processed invoice resubmitted | Hard | `reject_invoice` | `duplicate_invoice` |
| `task_5_mixed_discrepancy` | Invoice with both price variance and partial receipt; conflicting signals | Hard | `escalate_for_supervisor_review` | `price_mismatch` |
| `task_6_false_positive_duplicate` | Invoice looks like a duplicate but is a legitimate recurring order for a different PO | Hard | `approve_for_payment` | `clean_match` |
| `task_7_retroactive_price` | Vendor applied a price increase retroactively; PO predates the effective date | Hard | `escalate_for_supervisor_review` | `price_mismatch` |
| `task_8_split_invoice_pattern` | Supplier splits large order into sub-threshold invoices to dodge auto-approval | Hard | `escalate_for_supervisor_review` | `policy_violation` |
| `task_9_clean_from_risky_vendor` | Clean invoice from high-risk vendor with 5 prior incidents -- false-positive trap | Hard | `approve_for_payment` | `clean_match` |
| `task_10_rounding_false_alarm` | Invoice total off by $0.01 due to line-item rounding -- all else matches perfectly | Hard | `approve_for_payment` | `clean_match` |
| `task_11_authorized_overship` | GRN shows 110 received vs 100 ordered, but PO amendment authorized 10% overship | Hard | `approve_for_payment` | `clean_match` |
| `task_12_corrected_resubmission` | Corrected invoice (INV-R1) looks like a duplicate of rejected original | Hard | `approve_for_payment` | `clean_match` |

Each task includes fully synthetic business documents with deterministic ground truth and a multi-criteria grader. Tasks 5-8 test ambiguity, temporal reasoning, and cross-case pattern detection. Tasks 9-12 are false-positive traps where surface signals mislead toward rejection but deeper investigation reveals the correct answer is approval.

In addition to the 12 canonical tasks, there are 8 variant tasks (20 total) covering edge cases like multi-line clean matches, missing receipts, over-receipts, within-tolerance variances, total mismatches, corrected invoice traps, and policy violations.

## Action Space

The agent has 12 available actions divided into investigation, proposal, and terminal categories.

### Investigation Actions (provide `action_type` only)

| Action | Description |
|--------|-------------|
| `inspect_invoice_line_items` | Reveal detailed invoice line items (codes, quantities, prices, totals) |
| `inspect_purchase_order` | Reveal purchase order details (ordered quantities, agreed prices) |
| `inspect_goods_receipt_note` | Reveal goods receipt note (received/accepted/rejected quantities) |
| `inspect_vendor_profile` | Reveal vendor risk tier, duplicate history, escalation thresholds |
| `inspect_policy_rules` | Reveal company matching tolerances and escalation rules |
| `check_for_duplicate_invoice` | Search case history for similar/processed invoices |
| `compare_quantity` | Compare billed vs ordered vs received quantities per line item |
| `compare_price` | Compare billed unit prices vs PO-agreed prices per line item |
| `compare_totals` | Verify subtotal consistency, PO total match, tax, and grand total |
| `summarize_findings` | Get a numbered summary of all collected findings |

### Proposal Action

| Action | Description |
|--------|-------------|
| `propose_exception_type` | Declare the suspected exception type (with `exception_type` field) |

### Terminal Action

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `submit_final_resolution` | `final_decision`, `exception_type`, `evidence_references`, `explanation` | End the episode with a decision |

### Action JSON Format

```json
{"action_type": "inspect_purchase_order"}
```

```json
{
  "action_type": "submit_final_resolution",
  "final_decision": "escalate_for_supervisor_review",
  "exception_type": "price_mismatch",
  "evidence_references": ["inspect_purchase_order", "compare_price", "inspect_policy_rules"],
  "explanation": "Price variance of 10% exceeds 5% tolerance, requiring supervisor escalation per company policy."
}
```

## Observation Space

Each step returns an `InvoiceGuardObservation` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `case_id` | `str` | Unique case identifier |
| `task_id` | `str` | Which task is being evaluated |
| `difficulty` | `str` | `easy`, `moderate`, or `hard` |
| `invoice_summary` | `str` | One-line invoice overview (supplier, amount, PO ref) |
| `goal` | `str` | Natural language description of the agent's objective |
| `available_actions` | `list[str]` | Actions the agent can take |
| `revealed_documents` | `list[str]` | Documents the agent has already inspected |
| `findings` | `list[str]` | Accumulated investigation findings |
| `remaining_steps` | `int` | Steps left before timeout |
| `last_action_result` | `str` | Detailed output from the most recent action |
| `last_action_error` | `bool` | Whether the last action had an error |
| `warnings` | `list[str]` | System warnings (e.g., low steps remaining) |
| `reward` | `float` | Reward signal for the last action |
| `done` | `bool` | Whether the episode has ended |
| `metadata` | `dict` | Grader results (on episode end) |

## Reward Design

The environment provides dense, per-step rewards:

| Event | Reward |
|-------|--------|
| Reveal a new document | +0.05 |
| Useful comparison finding discrepancy | +0.10 |
| Confirm no issue (clean comparison) | +0.02 |
| Propose correct exception type | +0.15 |
| Propose wrong exception type | -0.05 |
| Summarize findings | +0.03 |
| Repeat an already-seen action | -0.02 |
| Submit correct final decision | +0.30 |
| Submit wrong final decision | -0.20 |
| Correct exception type on resolution | +0.15 |

## Grading

Episodes are scored by a deterministic grader on six weighted criteria (total = 1.0):

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Decision correctness | 0.35 | Exact match = 1.0, partial credit for related decisions |
| Exception type | 0.20 | Correct classification of the exception |
| Evidence sufficiency | 0.15 | Did the agent inspect the right documents? |
| Investigation quality | 0.10 | Breadth of document review and findings |
| Explanation quality | 0.10 | Cites specific numbers, references policy, uses correct terminology |
| Efficiency | 0.10 | Completing within step budget without waste |

## Decisions

| Decision | When to use |
|----------|-------------|
| `approve_for_payment` | All matches are clean and within tolerance |
| `place_on_hold` | Billed quantity exceeds received quantity |
| `reject_invoice` | Duplicate invoice or fraudulent submission |
| `escalate_for_supervisor_review` | Price/total variance exceeds tolerance, high-value invoice |

## Setup & Usage

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Docker (for containerized deployment)

### Local Development

```bash
cd invoice_guard

# Install dependencies with uv
uv sync

# Start the server
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000

# Validate
openenv validate
```

### Running the Baseline Agent

```bash
cd invoice_guard

# Create .env from the example
cp .env.example .env
# Edit .env with your API key and model

# Run inference
uv run python inference.py
```

### Docker

```bash
cd invoice_guard

# Build
docker build -t invoiceguard .

# Run
docker run -p 8000:8000 invoiceguard

# Run with hackathon resource constraints
docker run --cpus=2 --memory=8g -p 8000:8000 invoiceguard

# Validate against running container
openenv validate --url http://localhost:8000
```

### Deploy to Hugging Face Spaces

```bash
cd invoice_guard
openenv push --repo-id piyush-mk/invoice-guard
```

## Baseline Scores

### Canonical Tasks (12 tasks)

| Model | Type | Avg Score | Decision Correct Rate |
|-------|------|-----------|----------------------|
| **gpt-4o** | API | **0.95** | 100% |
| **gpt-4.1-mini** | API | **0.89** | 92% |
| **gpt-5.1** | API | **0.83** | 83% |
| **Qwen3-4B-Instruct-2507** | Open-weight | **0.83** | 83% |
| **Qwen2.5-7B-Instruct** | Open-weight | **0.70** | 58% |

### Hard Tasks (10 tasks -- Round 2)

| Model | Type | Avg Score | Decision Correct Rate |
|-------|------|-----------|----------------------|
| **gpt-5.1** | API | **0.76** | 70% |
| **Qwen3-4B-Instruct-2507** | Open-weight | **0.75** | 70% |
| **gpt-4.1-mini** | API | **0.74** | 70% |
| **Qwen2.5-7B-Instruct** | Open-weight | **0.66** | 50% |
| **gpt-4o** | API | **0.67** | 60% |

The hard tasks are designed with a strong performance gap (~40-55% on the worst tasks). Even gpt-4o drops to 0.67 on the hard slice, confirming the tasks meaningfully challenge frontier models.

Full per-task breakdowns are in `invoice_guard/outputs/baseline_scores/` and `invoice_guard/outputs/round2/`.

---

## Round 2: Training an Open-Weight Agent

### Approach

We fine-tune **Qwen/Qwen3-4B-Instruct-2507** using 4-bit quantized LoRA on the InvoiceGuard environment. The training pipeline has two stages:

1. **SFT Warm-Start** -- supervised fine-tuning on expert traces generated from environment ground truth. Teaches the model proper JSON action formatting for the InvoiceGuard action space.
2. **GRPO (Group Relative Policy Optimization)** -- trajectory-level RL where the model interacts with the live environment, generates rollouts, and is optimized using the grader's reward signal.

### Key Technical Challenges Solved

- **Qwen3 Thinking Mode:** Qwen3's default `<think>...</think>` blocks consumed the token budget and broke JSON action parsing. Fixed by disabling thinking mode via `enable_thinking=False` in the chat template and stripping residual thinking blocks during decoding.
- **Token Budget:** Initial `max_new_tokens=96` was too small for structured JSON actions with explanations. Increased to 384.
- **Special Token Handling:** `skip_special_tokens=True` was stripping `<think>` tags before our regex could remove thinking content. Fixed by decoding with `skip_special_tokens=False`, then explicitly stripping thinking blocks and special tokens.

### Baseline (Before Training)

| Slice | Avg Score | Decision Correct |
|-------|-----------|-----------------|
| Canonical (12 tasks) | 0.83 | 10/12 |
| Hard (10 tasks) | 0.75 | 7/10 |

### Trained Model Results

*Results will be updated when the current v3 training jobs complete.*

### Training Infrastructure

- **Primary:** Hugging Face Jobs (L40S / A100 GPUs)
- **Reproducibility:** Colab/Kaggle notebook at `notebooks/InvoiceGuard_Round2_GRPO_Reproducibility.ipynb`
- **Artifacts:** LoRA adapters, metrics, reward curves, and rollout samples pushed to Hugging Face Hub

See [`invoice_guard/training/README.md`](invoice_guard/training/README.md) for full training documentation.

### Hub Artifacts

| Artifact | Hub Location |
|----------|-------------|
| Training code | [piyush-mk/invoiceguard-code](https://huggingface.co/piyush-mk/invoiceguard-code) |
| Deployed environment | [piyush-mk/invoice-guard](https://huggingface.co/spaces/piyush-mk/invoice-guard) |

---

## Reproducibility

### Reproduce Baselines

```bash
cd invoice_guard
uv sync
cp .env.example .env
# Edit .env with your API key
uv run python inference.py
```

### Reproduce Training

Open `notebooks/InvoiceGuard_Round2_GRPO_Reproducibility.ipynb` in Colab or Kaggle, set `HF_TOKEN` in secrets, and run cells top-to-bottom. The notebook pulls all code from the Hub and pushes trained artifacts back -- no local clone needed.

## Project Structure

```
InvoiceGuard/
|---- README.md
|---- SUBMISSION.md
|---- .gitignore
|---- notebooks/
|   |---- InvoiceGuard_Round2_GRPO_Reproducibility.ipynb
|---- invoice_guard/                # OpenEnv project root
|   |---- openenv.yaml              # OpenEnv manifest
|   |---- pyproject.toml            # Dependencies (managed by uv)
|   |---- uv.lock                   # Locked dependencies
|   |---- Dockerfile                # Container image definition
|   |---- models.py                 # All data models (Action, Observation, State, entities)
|   |---- client.py                 # InvoiceGuardEnv client (EnvClient subclass)
|   |---- inference.py              # Baseline LLM agent + response parsing
|   |---- eval_round2.py            # Round 2 evaluation harness
|   |---- .env.example              # Environment variable template
|   |---- outputs/
|   |   |---- baseline_scores/      # Qwen3-4B, Qwen2.5-7B baseline JSONs
|   |   |---- round2/               # GPT baseline JSONs
|   |---- tasks/
|   |   |---- definitions.py        # 12 canonical + 8 variant task builders
|   |   |---- hard_definitions.py   # 10 hard-mode tasks (Round 2)
|   |---- graders/
|   |   |---- scoring.py            # Deterministic 6-criterion grader
|   |---- server/
|   |   |---- app.py                # FastAPI application (HTTP + WebSocket)
|   |   |---- invoice_guard_environment.py
|   |---- training/
|   |   |---- README.md             # Training documentation
|   |   |---- train_grpo.py         # Trajectory GRPO training script
|   |   |---- train_sft.py          # SFT training script
|   |   |---- rollout.py            # Agentic rollout helper
|   |   |---- launch_hf_job.py      # HF Jobs launcher
|   |   |---- eval_adapter.py       # LoRA adapter evaluation
|   |   |---- merge_adapter.py      # Merge LoRA into base model
|   |---- tests/                    # Unit tests
```

## License

This project was created for the OpenEnv Hackathon 2026.
