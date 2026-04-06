---
title: Invoice Guard Environment Server
emoji: 📋
colorFrom: purple
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# InvoiceGuard -- Three-Way Invoice Matching Environment

An OpenEnv environment that simulates accounts payable exception resolution. An AI agent investigates multi-document business cases -- invoices, purchase orders, goods receipt notes, vendor profiles, and company policies -- to detect discrepancies, classify exception types, and render correct decisions.

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

Each task includes fully synthetic business documents with deterministic ground truth and a multi-criteria grader. Tasks 5 and 6 are designed to challenge frontier models with ambiguity and conflicting evidence.

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

Episodes are scored by a deterministic grader on five weighted criteria (total = 1.0):

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Decision correctness | 0.40 | Exact match = 1.0, partial credit for related decisions |
| Exception type | 0.20 | Correct classification of the exception |
| Evidence sufficiency | 0.20 | Did the agent inspect the right documents? |
| Investigation quality | 0.10 | Breadth of document review and findings |
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
openenv push
# or: openenv push --namespace my-org --private
```

## Baseline Scores

### gpt-4o-mini

| Task | Score | Decision | Steps |
|------|-------|----------|-------|
| `task_1_clean_match` | 0.95 | `approve_for_payment` | 9 |
| `task_2_partial_receipt` | 0.75 | `place_on_hold` | 8 |
| `task_3_price_variance` | 0.75 | `escalate_for_supervisor_review` | 8 |
| `task_4_duplicate_invoice` | 0.98 | `reject_invoice` | 8 |
| `task_5_mixed_discrepancy` | 0.78 | `escalate_for_supervisor_review` | 9 |
| `task_6_false_positive_duplicate` | 0.35 | `reject_invoice` (incorrect) | 10 |

Average: **0.76**

### gpt-4o

| Task | Score | Decision | Steps |
|------|-------|----------|-------|
| `task_1_clean_match` | 0.95 | `approve_for_payment` | 8 |
| `task_2_partial_receipt` | 0.78 | `place_on_hold` | 7 |
| `task_3_price_variance` | 0.78 | `escalate_for_supervisor_review` | 7 |
| `task_4_duplicate_invoice` | 0.86 | `reject_invoice` | 9 |
| `task_5_mixed_discrepancy` | 0.78 | `escalate_for_supervisor_review` | 7 |
| `task_6_false_positive_duplicate` | 0.95 | `approve_for_payment` | 10 |

Average: **0.85**

Task 6 demonstrates strong model discrimination: gpt-4o correctly identifies the false-positive duplicate as a legitimate invoice, while gpt-4o-mini falls for the trap.

## Project Structure

```
invoice_guard/
|---- openenv.yaml              # OpenEnv manifest
|---- pyproject.toml             # Dependencies (managed by uv)
|---- uv.lock                   # Locked dependencies
|---- Dockerfile                # Container image definition
|---- models.py                 # All data models (Action, Observation, State, entities)
|---- client.py                 # InvoiceGuardEnv client (EnvClient subclass)
|---- inference.py              # Baseline LLM agent script
|---- .env.example              # Environment variable template
|---- tasks/
|   |---- __init__.py
|   |---- definitions.py        # Synthetic case templates and ground truth
|---- graders/
|   |---- __init__.py
|   |---- scoring.py            # Deterministic multi-criteria grader
|---- server/
    |---- __init__.py
    |---- app.py                 # FastAPI application (HTTP + WebSocket)
    |---- invoice_guard_environment.py  # Core Environment implementation
```
