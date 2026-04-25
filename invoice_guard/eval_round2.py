"""
Round 2 evaluation harness for InvoiceGuard.

Runs a chosen task slice (canonical / hard / all) against any
OpenAI-compatible model and writes a structured JSON report containing,
for every task: grader score, six grader sub-component scores, decision,
exception type, steps used, shortcut-penalty flag, and the per-step
reward_components log emitted by the environment.

Designed so baseline_*.json (Stage F) and trained_*.json (Stage G) share
the SAME schema and can be diffed with `--compare A.json B.json`.

Usage examples (PowerShell):
    # Baseline run on hard slice with default model from env vars
    uv run python eval_round2.py --slice hard --model-tag baseline

    # Run both slices, write to outputs/round2/
    uv run python eval_round2.py --slice all --model-tag baseline

    # Compare baseline vs trained
    uv run python eval_round2.py --compare outputs/round2/hard__baseline.json outputs/round2/hard__trained.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

# Reuse the existing inference helpers so prompts/parsing stay identical.
from inference import (  # type: ignore
    API_BASE_URL,
    API_KEY,
    MODEL_NAME,
    run_episode_local,
)
from models import TaskID  # type: ignore
from server.invoice_guard_environment import InvoiceGuardEnvironment  # type: ignore
from tasks import TASK_LIST, HARD_TASK_LIST  # type: ignore


load_dotenv()

OUT_DIR_DEFAULT = Path(__file__).parent / "outputs" / "round2"

# Grader sub-component keys we always extract for the report.
COMPONENT_KEYS = [
    "decision_score",
    "exception_type_score",
    "evidence_score",
    "investigation_score",
    "explanation_score",
    "efficiency_score",
]


def _slice_tasks(slice_name: str) -> List[TaskID]:
    if slice_name == "canonical":
        return list(TASK_LIST)
    if slice_name == "hard":
        return list(HARD_TASK_LIST)
    if slice_name == "all":
        return list(TASK_LIST) + list(HARD_TASK_LIST)
    raise SystemExit(f"Unknown --slice: {slice_name!r}")


def _run_slice(
    slice_name: str,
    model_tag: str,
    out_dir: Path,
) -> Path:
    """Run one slice end-to-end and write the JSON report. Returns path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    task_ids = _slice_tasks(slice_name)

    print("=" * 70, flush=True)
    print(f"InvoiceGuard Round 2 eval | slice={slice_name} | tasks={len(task_ids)}", flush=True)
    print(f"Model: {MODEL_NAME} | Base URL: {API_BASE_URL}", flush=True)
    print(f"Tag:   {model_tag} | Out dir: {out_dir}", flush=True)
    print("=" * 70, flush=True)

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = InvoiceGuardEnvironment()

    per_task: list[dict] = []
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    for task_id in task_ids:
        start = time.time()
        # `run_episode_local` already handles the conversation loop and stdout
        # in the hackathon-mandated [START]/[STEP]/[END] format.
        result = run_episode_local(env, llm, task_id)
        elapsed = time.time() - start

        grader_breakdown = result.get("grader_breakdown") or {}
        components = {k: float(grader_breakdown.get(k, 0.0)) for k in COMPONENT_KEYS}

        # Pull richer metadata from the env directly (last terminal obs went
        # back through the inference loop but we still hold env.state).
        s = env.state
        reward_components = list(getattr(s, "reward_components", []))

        per_task.append(
            {
                "task_id": result["task_id"],
                "decision": result.get("decision"),
                "exception_type": result.get("exception_type"),
                "steps": result["steps"],
                "grader_score": result["grader_score"],
                "components": components,
                "shortcut_penalty_applied": any(
                    rc.get("penalties", {}).get("shortcut") for rc in reward_components
                ),
                "documents_revealed": list(s.documents_revealed),
                "actions_taken": list(s.actions_taken),
                "reward_components": reward_components,
                "wall_clock_s": round(elapsed, 2),
            }
        )
        print(
            f"  >> {result['task_id']:38s} "
            f"score={result['grader_score']:.4f}  "
            f"steps={result['steps']:>2}  "
            f"decision={result.get('decision')}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    total_elapsed = time.time() - t0
    avg_score = (
        sum(t["grader_score"] for t in per_task) / len(per_task) if per_task else 0.0
    )
    component_avgs = {
        k: round(sum(t["components"][k] for t in per_task) / max(len(per_task), 1), 4)
        for k in COMPONENT_KEYS
    }
    decision_correct = sum(
        1 for t in per_task if t["components"]["decision_score"] >= 0.99
    )

    report = {
        "schema_version": 1,
        "slice": slice_name,
        "model_tag": model_tag,
        "model_name": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "started_at": started_at,
        "wall_clock_s": round(total_elapsed, 2),
        "n_tasks": len(per_task),
        "summary": {
            "avg_score": round(avg_score, 4),
            "decision_correct": decision_correct,
            "decision_correct_rate": round(decision_correct / max(len(per_task), 1), 4),
            "component_avgs": component_avgs,
            "shortcut_episodes": sum(
                1 for t in per_task if t["shortcut_penalty_applied"]
            ),
        },
        "tasks": per_task,
    }

    out_path = out_dir / f"{slice_name}__{model_tag}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("-" * 70, flush=True)
    print(f"Avg score:           {avg_score:.4f}", flush=True)
    print(f"Decision correct:    {decision_correct}/{len(per_task)}", flush=True)
    print(f"Component averages:  {component_avgs}", flush=True)
    print(f"Wrote report:        {out_path}", flush=True)
    print("=" * 70, flush=True)

    return out_path


def _compare(a_path: Path, b_path: Path) -> None:
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))

    print("=" * 78, flush=True)
    print(f"COMPARE  A: {a_path.name}  ({a['model_tag']})", flush=True)
    print(f"         B: {b_path.name}  ({b['model_tag']})", flush=True)
    print("=" * 78, flush=True)

    a_by = {t["task_id"]: t for t in a["tasks"]}
    b_by = {t["task_id"]: t for t in b["tasks"]}
    keys = list(a_by.keys())

    print(f"{'task':40s} {'A':>7s} {'B':>7s} {'delta':>8s}", flush=True)
    print("-" * 78, flush=True)
    for k in keys:
        if k not in b_by:
            continue
        sa = a_by[k]["grader_score"]
        sb = b_by[k]["grader_score"]
        d = sb - sa
        marker = "  +" if d > 0.01 else ("  -" if d < -0.01 else "   ")
        print(f"{k:40s} {sa:7.4f} {sb:7.4f} {d:+8.4f}{marker}", flush=True)

    print("-" * 78, flush=True)
    print(
        f"{'AVERAGE':40s} {a['summary']['avg_score']:7.4f} "
        f"{b['summary']['avg_score']:7.4f} "
        f"{b['summary']['avg_score'] - a['summary']['avg_score']:+8.4f}",
        flush=True,
    )
    for ck in COMPONENT_KEYS:
        ac = a["summary"]["component_avgs"][ck]
        bc = b["summary"]["component_avgs"][ck]
        print(f"  {ck:38s} {ac:7.4f} {bc:7.4f} {bc - ac:+8.4f}", flush=True)
    print("=" * 78, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="InvoiceGuard Round 2 eval harness.")
    p.add_argument("--slice", choices=["canonical", "hard", "all"], default="hard")
    p.add_argument("--model-tag", default="baseline",
                   help="Tag used in the output filename (e.g. baseline, trained, qwen3b-grpo).")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), type=Path, default=None,
                   help="Compare two report JSONs and print a delta table.")
    args = p.parse_args()

    if args.compare:
        _compare(args.compare[0], args.compare[1])
        return

    if not API_KEY:
        print(
            "WARNING: no API key found in env (HF_TOKEN / API_KEY / OPENAI_API_KEY). "
            "LLM calls will fail; this run will only verify the harness wiring.",
            file=sys.stderr,
            flush=True,
        )

    if args.slice == "all":
        _run_slice("canonical", args.model_tag, args.out_dir)
        _run_slice("hard", args.model_tag, args.out_dir)
    else:
        _run_slice(args.slice, args.model_tag, args.out_dir)


if __name__ == "__main__":
    main()
