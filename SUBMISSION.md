# InvoiceGuard -- Final Submission Summary

**Hackathon:** OpenEnv Hackathon 2026
**Team / Author:** piyush-mk
**Repository:** [github.com/piyush-mk/InvoiceGuard](https://github.com/piyush-mk/InvoiceGuard)
**HF Space:** [huggingface.co/spaces/piyush-mk/invoice-guard](https://huggingface.co/spaces/piyush-mk/invoice-guard)

---

## 1. The Idea

### Problem we targeted

Three-way invoice matching is one of the most common, error-prone, and costly tasks in enterprise finance. Every accounts-payable (AP) team in the world has to reconcile three documents for every purchase:

1. **Invoice** (what the supplier is billing)
2. **Purchase Order** (what was agreed to buy)
3. **Goods Receipt Note** (what actually arrived in the warehouse)

When any of these disagree -- wrong quantity, wrong price, missing receipt, suspected duplicate, policy violation -- the invoice becomes an "exception" that a human analyst has to investigate and resolve. Enterprise teams spend thousands of hours per month on this. It is a natural fit for agentic LLMs: structured documents, clear decision framework, deterministic ground truth.

### Why this is a good OpenEnv benchmark

1. **Real-world utility** -- directly maps to a multi-billion-dollar enterprise workflow.
2. **Multi-step reasoning** -- the agent must gather evidence from 5+ document sources before it can decide.
3. **No tool shortcut** -- the LLM cannot just "know" the answer; it has to actively investigate.
4. **Deterministic grading** -- every case has a ground-truth decision and exception type.
5. **Ambiguity and traps** -- moves well beyond simple matching into cases where surface signals mislead.

### What InvoiceGuard actually simulates

The agent is dropped into a synthetic AP case. On each episode it:

- Sees an invoice summary (supplier, amount, PO ref)
- Is given a natural-language goal describing what to investigate
- Has access to 12 distinct actions (inspect documents, run comparisons, check duplicates, propose exception type, submit final resolution)
- Has a limited step budget (8--12 steps depending on difficulty)
- Must submit a four-way resolution: `final_decision`, `exception_type`, `evidence_references`, `explanation`

The environment then grades the episode on six weighted criteria.

---

## 2. What We Built

### Project scope at final submission

| Component | Count | Details |
|-----------|-------|---------|
| Canonical tasks | 12 | 1 easy, 2 moderate, 9 hard |
| Variant tasks | 8 | Edge cases for each canonical family |
| **Total tasks** | **20** | Available via the `TaskID` enum |
| Investigation actions | 10 | Document inspection + comparisons |
| Proposal actions | 1 | `propose_exception_type` (cheap hypothesis test) |
| Terminal actions | 1 | `submit_final_resolution` |
| Grader criteria | 6 | Weighted rubric (see below) |
| Unit tests | 43 | Covering models, grader, environment |
| Benchmark models run | 5 | gpt-4.1-mini, gpt-5.4-mini, gpt-4.1, gpt-5.4, Nemotron-3-Super (partial) |

### Task catalog

| Task | Difficulty | Expected Decision | What it tests |
|------|-----------|-------------------|---------------|
| `task_1_clean_match` | easy | `approve_for_payment` | Baseline happy path |
| `task_2_partial_receipt` | moderate | `place_on_hold` | Quantity discrepancy |
| `task_3_price_variance` | moderate | `escalate_for_supervisor_review` | Price above tolerance |
| `task_4_duplicate_invoice` | hard | `reject_invoice` | Duplicate detection |
| `task_5_mixed_discrepancy` | hard | `escalate_for_supervisor_review` | Both price and receipt issues; conflicting signals |
| `task_6_false_positive_duplicate` | hard | `approve_for_payment` | Looks duplicate, is actually legit recurring order |
| `task_7_retroactive_price` | hard | `escalate_for_supervisor_review` | Temporal reasoning: PO date vs price change date |
| `task_8_split_invoice_pattern` | hard | `escalate_for_supervisor_review` | Cross-case detection of approval-threshold evasion |
| `task_9_clean_from_risky_vendor` | hard | `approve_for_payment` | Vendor-bias trap -- high risk vendor, clean invoice |
| `task_10_rounding_false_alarm` | hard | `approve_for_payment` | $0.01 rounding difference, well within tolerance |
| `task_11_authorized_overship` | hard | `approve_for_payment` | 10% overship was pre-authorized via PO amendment |
| `task_12_corrected_resubmission` | hard | `approve_for_payment` | Looks like duplicate, is actually a corrected resubmission |

Tasks 9--12 are **false-positive traps**: surface signals all push toward rejection, but deeper investigation reveals approval is correct. These are the tasks where frontier models fail most often.

### Action space

```
inspect_invoice_line_items   -- Reveal invoice line items
inspect_purchase_order       -- Reveal PO details
inspect_goods_receipt_note   -- Reveal GRN
inspect_vendor_profile       -- Reveal vendor risk tier, history
inspect_policy_rules         -- Reveal tolerances and escalation rules
check_for_duplicate_invoice  -- Search case history for duplicates
compare_quantity             -- Compare billed vs ordered vs received
compare_price                -- Compare billed vs PO prices
compare_totals               -- Verify subtotal, tax, grand total
summarize_findings           -- Get a numbered summary of findings so far
propose_exception_type       -- Test a hypothesis with small +/- reward
submit_final_resolution      -- Terminal: commit to a decision
```

### Grader rubric (deterministic, weights sum to 1.0)

| Criterion | Weight | What it measures |
|-----------|--------|------------------|
| Decision correctness | 0.35 | Exact match = 1.0; related decisions get partial credit |
| Exception type | 0.20 | Correct classification of the exception type |
| Evidence sufficiency | 0.15 | Did the agent inspect the right documents? |
| Investigation quality | 0.10 | Breadth of documents reviewed and findings |
| Explanation quality | 0.10 | Cites specific numbers, policy terms, uses correct terminology |
| Efficiency | 0.10 | Finishing within step budget without wasted actions |

### Observation shape

```python
class InvoiceGuardObservation(Observation):
    case_id: str
    task_id: str
    difficulty: str
    invoice_summary: str
    goal: str                          # Self-documenting task goal
    available_actions: list[str]
    suggested_next_actions: list[str]  # Context-aware hints, updates per step
    revealed_documents: list[str]
    findings: list[str]                # Accumulating evidence list
    remaining_steps: int
    last_action_result: str
    last_action_error: bool
    warnings: list[str]
    reward: float
    done: bool
    metadata: dict
    grader_result: dict                # Populated on episode end
```

---

## 3. Design Choices That Paid Off

### Self-explaining environment (reduces prompt dependency)

Rather than having the inference script carry all the task instructions in its system prompt, the environment itself serves a detailed `goal` string in the first observation. Any compliant OpenAI client can drive the environment without custom knowledge -- the prompt stays lean and generic.

### Dense, shaped rewards

Every action produces a reward signal:

| Event | Reward |
|-------|--------|
| Reveal a new document | +0.05 |
| Useful comparison finding a discrepancy | +0.08 to +0.10 |
| Clean (no-issue) comparison | +0.02 to +0.03 |
| Propose correct exception type | +0.15 |
| Propose wrong exception type | -0.05 |
| Repeat an already-taken action | -0.02 |
| Correct final decision | +0.30 |
| Wrong final decision | -0.20 |
| Correct exception type on resolution | +0.15 |

This keeps RL-style training signals usable while still producing a clean grader score in [0, 1].

### False-positive traps

Tasks 9--12 are designed specifically to punish lazy pattern-matching. The trap works like this: the agent sees one or two "suspicious" signals (risky vendor, similar invoice number, quantity overship, total off by cents) and the easy path is to escalate or reject. But the *correct* answer is to approve -- the signals are noise, and approving saves the business real money.

### Confidence scoring

Agents can optionally attach a `confidence` field to `submit_final_resolution`. If they're highly confident (>=0.8) AND correct, they get a small +0.05 bonus. If they're highly confident AND wrong, they get a -0.05 penalty. This rewards calibrated self-assessment without forcing it.

### Six-dimensional grading

Moving from a 5-criterion to a 6-criterion rubric (adding `explanation_score`) encourages agents to actually explain their reasoning, not just submit the right enum value. The grader parses the explanation for numbers, policy terminology, and key findings.

---

## 4. Benchmark Results

All runs use the same inference script (`inference.py`), temperature 0.0, JSON response format, against all 12 canonical tasks. Each model was run with its appropriate API endpoint and key.

### Full benchmark table (local mode, 12 tasks)

| Model | Avg | task_1 | task_2 | task_3 | task_4 | task_5 | task_6 | task_7 | task_8 | task_9 | task_10 | task_11 | task_12 |
|-------|-----|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|---------|---------|
| **gpt-4.1-mini** | **0.87** | 0.95 | 0.78 | 0.75 | 0.95 | 0.78 | 0.95 | 0.75 | 0.75 | 0.95 | 0.95 | 0.98 | 0.95 |
| **gpt-5.4-mini** | **0.87** | 0.98 | 0.95 | 0.73 | 0.98 | 0.75 | 0.98 | 0.75 | 0.50 | 0.95 | 0.98 | 0.98 | 0.95 |
| **gpt-4.1** | **0.79** | 0.95 | 0.75 | 0.75 | 0.47 | 0.78 | 0.95 | 0.78 | 0.75 | 0.40 | 0.95 | 0.95 | 0.95 |
| **gpt-5.4** | **0.78** | 0.95 | 0.75 | 0.70 | 0.47 | 0.75 | 0.95 | 0.78 | 0.75 | 0.40 | 0.95 | 0.95 | 0.95 |

### Live HF Space run (gpt-4.1-mini)

Same inference script, but running against the deployed HF Space via WebSocket instead of local environment. This simulates exactly what the judges see.

- **Average score:** 0.83
- **Tasks passed:** 11/12
- **Only failure:** task_9 (the vendor-bias trap) -- 0.40

### Nemotron-3-Super (via OpenRouter free tier)

Partial run only -- the free tier is very slow (~60s per task vs ~10s on OpenAI) and the episode was aborted after 3 tasks. Results so far:

| Task | Score | Decision |
|------|-------|----------|
| task_1_clean_match | 0.95 | `approve_for_payment` |
| task_2_partial_receipt | 0.75 | `place_on_hold` |
| task_3_price_variance | 0.75 | `escalate_for_supervisor_review` |

All three decisions correct -- tracks well with gpt-4.1-mini. Extrapolating the pattern suggests Nemotron would also land near 0.80--0.85 avg if the full run completed.

### Key findings from the benchmarks

1. **Mini models beat their full-size counterparts** by ~9 points on this benchmark. The full models over-think and get trapped into escalating when approval is correct.
2. **Task 9 (risky vendor, clean invoice) is the sharpest trap.** Both gpt-4.1 and gpt-5.4 consistently escalate when they should approve, scoring 0.40.
3. **Task 4 (duplicate invoice)** tripped both full-size models -- they escalated instead of rejecting (0.47).
4. **Tasks 1, 6, 10, 11, 12 are consistently solved** by all models -- these are the "happy paths" that confirm the environment isn't broken.
5. **No single model is dominant across all tasks** -- different models fail in different ways, making the benchmark useful for discrimination.

### Score-variance sanity check

| Model | Avg | Std-dev across tasks | Range |
|-------|-----|----------------------|-------|
| gpt-4.1-mini | 0.87 | 0.10 | 0.75--0.98 |
| gpt-5.4-mini | 0.87 | 0.16 | 0.50--0.98 |
| gpt-4.1 | 0.79 | 0.19 | 0.40--0.95 |
| gpt-5.4 | 0.78 | 0.19 | 0.40--0.95 |

Score variance is healthy: there is no model that gets 1.0 on everything (saturation) or 0.0 on everything (broken env). Every model gets a mix of passes and fails.

---

## 5. Hackathon Compliance

Every item from the hackathon guidelines was explicitly verified.

| Requirement | Our implementation |
|-------------|---------------------|
| `API_BASE_URL` env var with default | `os.getenv("API_BASE_URL", "https://api.openai.com/v1")` |
| `MODEL_NAME` env var with default | `os.getenv("MODEL_NAME", "gpt-4.1-mini")` |
| `HF_TOKEN` is the primary API key | `API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")` |
| `LOCAL_IMAGE_NAME` for docker mode | Read via `os.getenv("LOCAL_IMAGE_NAME")`, switches to `from_docker_image` |
| `inference.py` at project root | `invoice_guard/inference.py` |
| OpenAI client for all LLM calls | `from openai import OpenAI`; single `client.chat.completions.create` call site |
| `[START]` / `[STEP]` / `[END]` stdout format | Matches sample exactly, 2-decimal formatting |
| Reward and rewards at 2 decimals | `f"{r:.2f}"` everywhere |
| `done` / `success` lowercase | `str(bool).lower()` everywhere |
| `error` is `last_action_error` or `null` | Handled in `log_step` |
| Score in [0, 1] per task | Grader clamps to [0, 1] |
| Runtime < 20min | Full 12-task run takes ~2 min on `gpt-4.1-mini` |
| Runs on vcpu=2, memory=8gb | Dockerfile sets `--cpus=2 --memory=8g` |
| Pre-validation script passes | Checked: `/reset` returns 200, Dockerfile exists, `openenv validate` OK |
| HF Space is live | [piyush-mk/invoice-guard](https://huggingface.co/spaces/piyush-mk/invoice-guard) |

---

## 6. Project Structure

```
InvoiceGuard/                              # Git repo root
|-- README.md                              # Public-facing docs
|-- SUBMISSION.md                          # This file
|-- .gitignore                             # Ignores .venv, .env, references/
|-- notebooks/
|   |-- InvoiceGuard_Round2_GRPO_Reproducibility.ipynb  # Training notebook
|-- invoice_guard/                         # OpenEnv project root
|   |-- openenv.yaml                       # Environment manifest
|   |-- pyproject.toml                     # Dependencies (managed by uv)
|   |-- uv.lock                            # Locked dependencies
|   |-- Dockerfile                         # Container image (HF Spaces target)
|   |-- README.md                          # HF-Space-visible docs with YAML frontmatter
|   |-- .env.example                       # Env var template
|   |-- .openenvignore                     # Files excluded from `openenv push`
|   |-- models.py                          # Action/Observation/State + business entities
|   |-- client.py                          # InvoiceGuardEnv EnvClient subclass
|   |-- inference.py                       # Baseline LLM agent + response parsing
|   |-- eval_round2.py                     # Round 2 evaluation harness
|   |-- outputs/
|   |   |-- baseline_scores/               # Qwen3-4B, Qwen2.5-7B baseline JSONs
|   |   |-- round2/                        # GPT model baseline JSONs
|   |-- tasks/
|   |   |-- definitions.py                 # 12 canonical + 8 variant task builders
|   |   |-- hard_definitions.py            # 10 hard-mode tasks (Round 2)
|   |-- graders/
|   |   |-- scoring.py                     # Deterministic 6-criterion grader
|   |-- server/
|   |   |-- app.py                         # FastAPI create_app() wire-up
|   |   |-- invoice_guard_environment.py   # Core Environment implementation
|   |-- training/
|   |   |-- README.md                      # Training documentation
|   |   |-- train_grpo.py                  # Trajectory GRPO script
|   |   |-- train_sft.py                   # SFT training script
|   |   |-- rollout.py                     # Agentic rollout helper
|   |   |-- launch_hf_job.py               # HF Jobs launcher
|   |   |-- eval_adapter.py               # LoRA adapter evaluation
|   |   |-- merge_adapter.py              # Merge LoRA into base model
|   |-- tests/
|       |-- test_models.py                 # 14 model tests
|       |-- test_grader.py                 # 10 grader tests
|       |-- test_environment.py            # 19 environment tests
```

---

## 7. Development Timeline

1. **Scaffold** -- created OpenEnv project structure, manifest, Dockerfile, models.
2. **Core environment** -- implemented `reset`/`step`, action handlers, document reveal logic.
3. **Tasks 1--4** -- canonical cases with deterministic ground truth.
4. **First baseline** -- ran `gpt-4o-mini` locally; avg ~0.76.
5. **Improvement plan** -- added variant tasks, tuned reward shaping, improved grader.
6. **Tasks 5--8** -- harder cases (ambiguity, temporal reasoning, cross-case detection).
7. **Prompt reduction** -- moved instructions from the script into environment observations.
8. **Confidence + suggested_next_actions** -- environment became more self-documenting.
9. **6-dimensional grader** -- added explanation quality criterion.
10. **Tests** -- added 43 unit tests; all passing.
11. **Docker + HF Spaces deployment** -- fixed encoding issues, deployed successfully.
12. **Judge simulation** -- wrote `test_live_space.py` to replicate judge workflow.
13. **Tasks 9--12** -- added 4 more false-positive trap tasks.
14. **Multi-model benchmarks** -- ran gpt-4.1-mini, gpt-5.4-mini, gpt-4.1, gpt-5.4, Nemotron.
15. **Spec alignment** -- switched to `HF_TOKEN` as primary key, default model `gpt-4.1-mini`.
16. **Round 1 deploy** -- pushed to HF Spaces, verified with pre-validation.
17. **Round 2: Hard tasks** -- added 10 hard-mode tasks targeting cross-case fraud, temporal policy, contradictory evidence, partial observability, multi-party approvals.
18. **Round 2: Broader baselines** -- benchmarked gpt-4o, gpt-4.1-mini, gpt-5.1, Qwen3-4B, Qwen2.5-7B across canonical and hard slices.
19. **Round 2: Training pipeline** -- built trajectory GRPO + SFT training scripts, rollout helper, HF Jobs launcher, adapter eval/merge tools.
20. **Round 2: Qwen3 debugging** -- diagnosed and fixed thinking mode, token budget, and special token decoding issues.
21. **Round 2: Reproducibility notebook** -- created Colab/Kaggle notebook for judges to reproduce training.
22. **Round 2: HF Space update** -- pushed latest environment code (including hard tasks) to deployed Space.

---

## 8. What Judges Will See

### Environment (HF Space)
1. Hit `POST /reset` on our live HF Space -- it returns a valid initial observation in under 1s.
2. Hit `POST /step` with actions -- the environment advances correctly and returns shaped rewards.
3. Access `grader_result` in the final observation (populated as a direct field so it survives the WebSocket protocol).
4. See 22 tasks (12 canonical + 10 hard) all produce scores in [0, 1].
5. See the mandatory `[START]` / `[STEP]` / `[END]` logs emitted to stdout.

### Training Evidence
6. Baseline JSON scores for Qwen3-4B, Qwen2.5-7B, gpt-4o, gpt-4.1-mini, gpt-5.1 in `invoice_guard/outputs/`.
7. Training code in `invoice_guard/training/` (GRPO + SFT scripts, rollout helper, adapter eval/merge).
8. Reproducibility notebook at `notebooks/InvoiceGuard_Round2_GRPO_Reproducibility.ipynb`.
9. Trained LoRA adapters and training artifacts (metrics, curves, rollouts) on Hugging Face Hub.

---

## 9. How to Reproduce Our Results

### Environment + Baselines

```bash
git clone https://github.com/piyush-mk/InvoiceGuard.git
cd InvoiceGuard/invoice_guard
uv sync
cp .env.example .env
# Edit .env: set HF_TOKEN, MODEL_NAME to gpt-4.1-mini

uv run python inference.py          # Run baseline inference
uv run pytest tests/ -v             # Run unit tests
uv run openenv validate             # Validate environment
```

### Training (Round 2)

Open `notebooks/InvoiceGuard_Round2_GRPO_Reproducibility.ipynb` in Colab or Kaggle with a GPU accelerator, set `HF_TOKEN` in secrets, and run cells top-to-bottom. The notebook pulls all code from the Hub and pushes trained artifacts back -- no local clone needed.

---

## 10. Round 2: Training an Open-Weight Agent

### Model selection

We chose **Qwen/Qwen3-4B-Instruct-2507** as our trainable model. At 4B parameters it fits comfortably in 4-bit quantized LoRA on a single L40S/A100 GPU while still being capable enough to follow the multi-step investigation protocol.

### Training approach

The training pipeline has two stages:

1. **SFT warm-start** -- supervised fine-tuning on expert traces generated from environment ground truth. Each task's optimal action sequence is known deterministically, so we can generate high-quality training data without any LLM inference. This teaches the model proper JSON action formatting for InvoiceGuard's 12 action types.

2. **GRPO (Group Relative Policy Optimization)** -- trajectory-level reinforcement learning where the model interacts with the live InvoiceGuard environment. For each task, we generate multiple rollouts, compute the 6-criterion grader score as the reward, and optimize using group relative advantages. This stage teaches domain reasoning beyond what SFT can provide.

### Key technical challenges

**Qwen3 thinking mode:** Qwen3-4B-Instruct generates `<think>...</think>` blocks by default. These consumed most of the token budget and broke JSON action parsing. We solved this by:
- Passing `enable_thinking=False` to the chat template
- Decoding with `skip_special_tokens=False` to preserve tags for regex stripping
- Explicitly removing `<think>` blocks and residual special tokens before parsing

**Token budget:** The initial `max_new_tokens=96` was too small for structured JSON actions containing evidence references and explanations. Increasing to 384 resolved truncation issues.

**Quantization trade-offs:** 4-bit quantization enables single-GPU training but introduces quality degradation. The SFT warm-start partially compensates by establishing strong action formatting priors before RL fine-tuning.

### Baseline scores (Qwen3-4B, before training)

| Task Slice | Avg Score | Decisions Correct |
|-----------|-----------|-------------------|
| Canonical (12 tasks) | 0.83 | 10/12 (83%) |
| Hard (10 tasks) | 0.75 | 7/10 (70%) |

### Trained model results

*Results will be updated when the current training jobs complete.*

### Training infrastructure

| Component | Details |
|-----------|---------|
| Primary training | Hugging Face Jobs (L40S / A100 GPUs) |
| Reproducibility | Colab/Kaggle notebook (`notebooks/InvoiceGuard_Round2_GRPO_Reproducibility.ipynb`) |
| Artifacts | LoRA adapters + training_artifacts/ pushed to Hugging Face Hub |
| Monitoring | Trackio real-time metrics dashboard |
| Training code | [piyush-mk/invoiceguard-code](https://huggingface.co/piyush-mk/invoiceguard-code) |

### Broader baselines (Round 2)

We benchmarked across both API and open-weight models to contextualize training improvement:

| Model | Canonical Avg | Hard Avg |
|-------|--------------|----------|
| gpt-4o | 0.95 | 0.67 |
| gpt-4.1-mini | 0.89 | 0.74 |
| gpt-5.1 | 0.83 | 0.76 |
| Qwen3-4B-Instruct-2507 | 0.83 | 0.75 |
| Qwen2.5-7B-Instruct | 0.70 | 0.66 |

The hard task slice is designed with a strong performance gap -- even gpt-4o drops to 0.67, confirming the tasks meaningfully challenge frontier models.

---

## 11. Future Work

Things we deliberately scoped out but would naturally extend:

- **Multi-currency** -- all tasks currently in USD; adding EUR / GBP / INR with FX conversion.
- **Real OCR documents** -- generate PDF invoices and run through a document-parsing action.
- **Larger model training** -- scale to 7B+ models with multi-GPU training for stronger domain reasoning.
- **Agent memory** -- per-vendor memory that persists across episodes for split-invoice detection.
- **Curriculum learning** -- progressively increase task difficulty during RL training.

---

## 12. License and Attribution

Built for the OpenEnv Hackathon 2026. Uses:

- OpenEnv (Meta / PyTorch) -- environment protocol and server harness
- FastAPI / Pydantic -- HTTP and data modeling
- OpenAI Python client -- LLM API calls (OpenAI-compatible, works with HF router and OpenRouter)
- Hugging Face Transformers + PEFT + BitsAndBytes -- model loading, LoRA, quantization
- Hugging Face Hub + Jobs -- artifact storage, cloud training
- Trackio -- training metrics monitoring
- python-dotenv -- .env loading
