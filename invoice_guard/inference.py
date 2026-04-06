"""
InvoiceGuard -- Baseline Inference Script.

Runs a baseline LLM agent against all canonical tasks and reports per-task
and overall grader scores. Uses the OpenAI API client as required by
the hackathon guidelines.

STDOUT FORMAT (mandatory):
    [START] task=<task_name> env=invoice_guard model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables (mandatory):
    API_BASE_URL       -- LLM API endpoint (default: https://api.openai.com/v1)
    MODEL_NAME         -- model identifier (default: gpt-4o-mini)
    HF_TOKEN           -- Hugging Face token / primary API key
    OPENAI_API_KEY     -- Fallback API key for the LLM provider
    LOCAL_IMAGE_NAME   -- Docker image name (uses from_docker_image when set)
"""

import asyncio
import json
import os
import sys
import time
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from models import (
    ActionType, DecisionType, ExceptionType, InvoiceGuardAction, TaskID,
)
from tasks import get_task_case, TASK_LIST

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or ""
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")

BENCHMARK = "invoice_guard"


# -- Mandatory stdout logging --------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# -- System prompt --------------------------------------------------------


SYSTEM_PROMPT = """You are a senior accounts payable analyst. You will be given an invoice case to investigate and resolve.

The environment tells you your goal, available actions, and decision options. Read the goal carefully.

WORKFLOW:
1. Investigate: inspect documents (PO, GRN, vendor profile, policy rules), run comparisons (quantity, price, totals), check for duplicates.
2. Resolve: submit_final_resolution with your decision, exception type, evidence references, and explanation.

Complete a thorough investigation before resolving. Inspect at least: purchase order, goods receipt note, compare quantity, compare price, policy rules, duplicate check, and vendor profile.

RESPONSE FORMAT:
- Respond with ONLY a valid JSON object. No markdown, no commentary.
- Investigation example: {"action_type": "inspect_purchase_order"}
- Resolution example: {"action_type": "submit_final_resolution", "final_decision": "approve_for_payment", "exception_type": "clean_match", "evidence_references": ["inspect_purchase_order", "compare_quantity"], "explanation": "All documents match within tolerance."}

RULES:
- Pay close attention to POLICY findings -- they tell you when escalation is required.
- When multiple issues exist, escalation takes priority over hold.
- Check PO references carefully before concluding an invoice is a duplicate.
- Include all investigation actions you performed in evidence_references.
- Cite specific numbers in your explanation.
- NEVER repeat an action you already took.
- When remaining_steps is 3 or fewer, submit immediately with what you have.
"""


# -- Prompt building ------------------------------------------------------


def build_observation_prompt(obs, is_first: bool = False) -> str:
    """Format observation as a readable prompt for the LLM."""
    parts = [
        f"Case: {obs.case_id} | Difficulty: {obs.difficulty} | Steps remaining: {obs.remaining_steps}",
        f"Invoice: {obs.invoice_summary}",
    ]

    if is_first and obs.goal:
        parts.append(f"\n{obs.goal}")

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

    if obs.remaining_steps <= 2:
        parts.append(
            ">>> YOU MUST submit_final_resolution NOW. "
            "No more investigation. Decide based on what you have. <<<"
        )

    return "\n".join(parts)


# -- LLM response parsing ------------------------------------------------


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


# -- Observation extraction helpers ---------------------------------------


def _obs_from_step_result(result):
    """Extract observation from an EnvClient StepResult, copying reward/done."""
    obs = result.observation
    obs.reward = result.reward
    obs.done = result.done
    return obs


# -- Episode runner (local, synchronous) ----------------------------------


def run_episode_local(env, client: OpenAI, task_id: TaskID) -> dict:
    """Run one full episode against the local environment."""
    obs = env.reset(task_id=task_id.value)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards: List[float] = []
    steps = 0
    score = 0.0
    success = False
    last_decision = None
    last_exception = None

    log_start(task=task_id.value, env=BENCHMARK, model=MODEL_NAME)

    try:
        while not obs.done:
            user_msg = build_observation_prompt(obs, is_first=(steps == 0))
            messages.append({"role": "user", "content": user_msg})

            try:
                api_kwargs = {
                    "model": MODEL_NAME,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 512,
                }
                try:
                    api_kwargs["response_format"] = {"type": "json_object"}
                    response = client.chat.completions.create(**api_kwargs)
                except Exception:
                    del api_kwargs["response_format"]
                    response = client.chat.completions.create(**api_kwargs)

                assistant_msg = response.choices[0].message.content or ""

            except Exception as e:
                print(f"[DEBUG] LLM API error: {e}", flush=True)
                assistant_msg = '{"action_type": "summarize_findings"}'

            messages.append({"role": "assistant", "content": assistant_msg})

            params = parse_llm_response(assistant_msg)
            if params.get("final_decision"):
                last_decision = params["final_decision"]
            if params.get("exception_type"):
                last_exception = params["exception_type"]
            action = build_action(params)

            obs = env.step(action)
            reward = obs.reward if obs.reward else 0.0
            rewards.append(reward)
            steps += 1

            error_str = None
            if obs.last_action_error:
                error_str = obs.last_action_result

            log_step(
                step=steps,
                action=action.action_type.value,
                reward=reward,
                done=obs.done,
                error=error_str,
            )

        grader_data = getattr(obs, "grader_result", None) or obs.metadata.get("grader_result", {})
        score = grader_data.get("score", 0.0) if isinstance(grader_data, dict) else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.5

    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)

    return {
        "task_id": task_id.value,
        "steps": steps,
        "grader_score": score,
        "total_reward": sum(rewards),
        "rewards": rewards,
        "decision": last_decision,
        "exception_type": last_exception,
        "grader_breakdown": grader_data,
    }


# -- Episode runner (Docker, asynchronous) --------------------------------


async def run_episode_docker(env, client: OpenAI, task_id: TaskID) -> dict:
    """Run one full episode against a Docker-based environment via EnvClient."""
    result = await env.reset(task_id=task_id.value)
    obs = _obs_from_step_result(result)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards: List[float] = []
    steps = 0
    score = 0.0
    success = False
    last_decision = None
    last_exception = None

    log_start(task=task_id.value, env=BENCHMARK, model=MODEL_NAME)

    try:
        while not obs.done:
            user_msg = build_observation_prompt(obs, is_first=(steps == 0))
            messages.append({"role": "user", "content": user_msg})

            try:
                api_kwargs = {
                    "model": MODEL_NAME,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 512,
                }
                try:
                    api_kwargs["response_format"] = {"type": "json_object"}
                    response = client.chat.completions.create(**api_kwargs)
                except Exception:
                    del api_kwargs["response_format"]
                    response = client.chat.completions.create(**api_kwargs)

                assistant_msg = response.choices[0].message.content or ""

            except Exception as e:
                print(f"[DEBUG] LLM API error: {e}", flush=True)
                assistant_msg = '{"action_type": "summarize_findings"}'

            messages.append({"role": "assistant", "content": assistant_msg})

            params = parse_llm_response(assistant_msg)
            if params.get("final_decision"):
                last_decision = params["final_decision"]
            if params.get("exception_type"):
                last_exception = params["exception_type"]
            action = build_action(params)

            result = await env.step(action)
            obs = _obs_from_step_result(result)
            reward = obs.reward if obs.reward else 0.0
            rewards.append(reward)
            steps += 1

            error_str = None
            if obs.last_action_error:
                error_str = obs.last_action_result

            log_step(
                step=steps,
                action=action.action_type.value,
                reward=reward,
                done=obs.done,
                error=error_str,
            )

        grader_data = getattr(obs, "grader_result", None) or obs.metadata.get("grader_result", {})
        score = grader_data.get("score", 0.0) if isinstance(grader_data, dict) else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.5

    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)

    return {
        "task_id": task_id.value,
        "steps": steps,
        "grader_score": score,
        "total_reward": sum(rewards),
        "rewards": rewards,
        "decision": last_decision,
        "exception_type": last_exception,
        "grader_breakdown": grader_data,
    }


# -- Main -----------------------------------------------------------------


def _print_header():
    print("=" * 60, flush=True)
    print("InvoiceGuard -- Baseline Inference", flush=True)
    print("=" * 60, flush=True)
    print(f"API Base URL: {API_BASE_URL}", flush=True)
    print(f"Model:        {MODEL_NAME}", flush=True)
    print(f"Tasks:        {len(TASK_LIST)}", flush=True)
    mode = f"docker ({LOCAL_IMAGE_NAME})" if LOCAL_IMAGE_NAME else "local"
    print(f"Mode:         {mode}", flush=True)
    print(flush=True)


def _print_results(results):
    print(flush=True)
    print("=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    scores = [r["grader_score"] for r in results]
    for r in results:
        print(
            f"  {r['task_id']:30s}  score={r['grader_score']:.4f}  "
            f"decision={r['decision']}",
            flush=True,
        )
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\n  Average score: {avg:.4f}", flush=True)
    print(f"  Total tasks:   {len(scores)}", flush=True)
    print("=" * 60, flush=True)


async def main_docker():
    """Run inference against a Docker container via EnvClient."""
    from client import InvoiceGuardEnv

    _print_header()
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await InvoiceGuardEnv.from_docker_image(LOCAL_IMAGE_NAME)

    results = []
    try:
        for task_id in TASK_LIST:
            start = time.time()
            result = await run_episode_docker(env, llm_client, task_id)
            elapsed = time.time() - start
            print(
                f"  >> {task_id.value}: score={result['grader_score']:.4f} "
                f"steps={result['steps']} decision={result['decision']} "
                f"time={elapsed:.1f}s",
                flush=True,
            )
            results.append(result)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

    _print_results(results)


def main_local():
    """Run inference directly against the local environment."""
    from server.invoice_guard_environment import InvoiceGuardEnvironment

    _print_header()
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = InvoiceGuardEnvironment()

    results = []
    for task_id in TASK_LIST:
        start = time.time()
        result = run_episode_local(env, llm_client, task_id)
        elapsed = time.time() - start
        print(
            f"  >> {task_id.value}: score={result['grader_score']:.4f} "
            f"steps={result['steps']} decision={result['decision']} "
            f"time={elapsed:.1f}s",
            flush=True,
        )
        results.append(result)

    _print_results(results)


if __name__ == "__main__":
    if LOCAL_IMAGE_NAME:
        asyncio.run(main_docker())
    else:
        main_local()
