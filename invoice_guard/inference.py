"""
InvoiceGuard — Baseline Inference Script.

Runs a baseline LLM agent against all four tasks and reports per-task
and overall grader scores. Uses the OpenAI API client as required by
the hackathon guidelines.

Environment variables:
    API_BASE_URL   — LLM API endpoint (default: https://api.openai.com/v1)
    MODEL_NAME     — model identifier (default: gpt-4o-mini)
    HF_TOKEN       — Hugging Face / API key
    OPENAI_API_KEY — API key for the LLM provider
"""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from models import (
    ActionType, DecisionType, ExceptionType, InvoiceGuardAction, TaskID,
)
from tasks import get_task_case, TASK_LIST
from server.invoice_guard_environment import InvoiceGuardEnvironment

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")
API_KEY = os.getenv("OPENAI_API_KEY", HF_TOKEN)


SYSTEM_PROMPT = """You are a senior accounts payable analyst. You review supplier invoice cases by comparing the invoice against the purchase order and goods receipt note.

WORKFLOW:
1. Inspect documents (purchase order, goods receipt note, line items)
2. Run comparisons (quantity, price, totals) and check for duplicates
3. Review vendor profile and company policy if needed
4. Submit your final resolution

AVAILABLE ACTIONS — respond with a JSON object containing "action_type":

Investigation actions (just provide action_type):
  inspect_invoice_line_items — reveal invoice line item details
  inspect_purchase_order — reveal purchase order details
  inspect_goods_receipt_note — reveal goods receipt note details
  inspect_vendor_profile — reveal vendor risk and tolerance info
  inspect_policy_rules — reveal company matching policies
  check_for_duplicate_invoice — search case history for duplicates
  compare_quantity — compare billed vs ordered vs received quantities
  compare_price — compare billed price vs PO agreed price
  compare_totals — verify subtotal, tax, and total consistency
  summarize_findings — get a summary of collected findings
  propose_exception_type — declare the suspected exception type

Terminal action (ends the episode):
  submit_final_resolution — provide all of the following fields:
    "final_decision": one of "approve_for_payment" | "place_on_hold" | "reject_invoice" | "escalate_for_supervisor_review"
    "exception_type": one of "clean_match" | "quantity_mismatch" | "price_mismatch" | "total_amount_mismatch" | "partial_receipt" | "missing_receipt" | "duplicate_invoice" | "tax_variance" | "policy_violation" | "mixed_discrepancy"
    "evidence_references": list of action names that support your decision
    "explanation": one-sentence justification

RULES:
- Respond with ONLY a JSON object. No markdown, no commentary.
- For investigation actions: {"action_type": "inspect_purchase_order"}
- For resolution: {"action_type": "submit_final_resolution", "final_decision": "...", "exception_type": "...", "evidence_references": [...], "explanation": "..."}
- Investigate before resolving. Do NOT guess without evidence.
- Always inspect the purchase order and goods receipt note at minimum.
"""


def build_observation_prompt(obs) -> str:
    """Format observation as a readable prompt for the LLM."""
    parts = [
        f"Case: {obs.case_id} | Difficulty: {obs.difficulty} | Steps remaining: {obs.remaining_steps}",
        f"Invoice: {obs.invoice_summary}",
    ]

    if obs.revealed_documents:
        parts.append(f"Documents reviewed: {', '.join(obs.revealed_documents)}")

    if obs.findings:
        parts.append("Findings:")
        for i, f in enumerate(obs.findings, 1):
            parts.append(f"  {i}. {f}")

    if obs.last_action_result:
        parts.append(f"Last result: {obs.last_action_result}")

    if obs.warnings:
        parts.append(f"Warnings: {'; '.join(obs.warnings)}")

    return "\n".join(parts)


def parse_llm_response(response_text: str) -> dict:
    """Extract a JSON object from the LLM response."""
    text = response_text.strip()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    return {"action_type": "summarize_findings"}


def build_action(params: dict) -> InvoiceGuardAction:
    """Build a typed InvoiceGuardAction from parsed LLM output."""
    action_type = params.get("action_type", "summarize_findings")

    try:
        ActionType(action_type)
    except ValueError:
        action_type = "summarize_findings"

    kwargs = {"action_type": action_type}

    if params.get("final_decision"):
        try:
            kwargs["final_decision"] = DecisionType(params["final_decision"])
        except ValueError:
            pass

    if params.get("exception_type"):
        try:
            kwargs["exception_type"] = ExceptionType(params["exception_type"])
        except ValueError:
            pass

    if params.get("evidence_references"):
        kwargs["evidence_references"] = list(params["evidence_references"])

    if params.get("explanation"):
        kwargs["explanation"] = str(params["explanation"])

    return InvoiceGuardAction(**kwargs)


def run_episode(env, client, task_id: TaskID) -> dict:
    """Run one full episode against a task and return grading results."""
    obs = env.reset(task_id=task_id.value)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_reward = 0.0
    steps = 0

    while not obs.done:
        user_msg = build_observation_prompt(obs)
        messages.append({"role": "user", "content": user_msg})

        try:
            api_kwargs = {
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 512,
            }

            # Use JSON mode for reliable structured output when supported.
            # Falls back gracefully if the provider doesn't support it.
            try:
                api_kwargs["response_format"] = {"type": "json_object"}
                response = client.chat.completions.create(**api_kwargs)
            except Exception:
                del api_kwargs["response_format"]
                response = client.chat.completions.create(**api_kwargs)

            assistant_msg = response.choices[0].message.content or ""

        except Exception as e:
            print(f"  LLM API error: {e}")
            assistant_msg = '{"action_type": "summarize_findings"}'

        messages.append({"role": "assistant", "content": assistant_msg})

        params = parse_llm_response(assistant_msg)
        action = build_action(params)

        obs = env.step(action)
        total_reward += obs.reward if obs.reward else 0.0
        steps += 1

    grader_result = obs.metadata.get("grader_result", {})
    return {
        "task_id": task_id.value,
        "steps": steps,
        "grader_score": grader_result.get("score", 0.0),
        "total_reward": total_reward,
        "decision": env.state.final_decision,
        "exception_type": env.state.final_exception_type,
        "grader_breakdown": grader_result,
    }


def main():
    print("=" * 60)
    print("InvoiceGuard — Baseline Inference")
    print("=" * 60)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Model:        {MODEL_NAME}")
    print(f"Tasks:        {len(TASK_LIST)}")
    print()

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = InvoiceGuardEnvironment()

    results = []
    for task_id in TASK_LIST:
        case = get_task_case(task_id)
        print(f"Running {task_id.value} ({case.difficulty.value})...")
        start = time.time()

        result = run_episode(env, client, task_id)
        elapsed = time.time() - start

        print(f"  Score: {result['grader_score']:.4f} | "
              f"Steps: {result['steps']} | "
              f"Decision: {result['decision']} | "
              f"Time: {elapsed:.1f}s")
        results.append(result)

    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    scores = [r["grader_score"] for r in results]
    for r in results:
        print(f"  {r['task_id']:30s}  score={r['grader_score']:.4f}  "
              f"decision={r['decision']}")
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\n  Average score: {avg:.4f}")
    print(f"  Total tasks:   {len(scores)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
