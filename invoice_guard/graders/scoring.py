"""
Deterministic grading logic for InvoiceGuard episodes.

Scores 0.0–1.0 with partial credit across five rubric dimensions:
  - Final decision correctness  (0.40)
  - Exception type correctness  (0.20)
  - Evidence sufficiency         (0.20)
  - Investigation quality        (0.10)
  - Efficiency                   (0.10)
"""

from typing import List, Optional

try:
    from ..models import (
        CaseData, DecisionType, ExceptionType, GraderResult, InvoiceGuardState,
    )
except ImportError:
    from models import (
        CaseData, DecisionType, ExceptionType, GraderResult, InvoiceGuardState,
    )


W_DECISION = 0.40
W_EXCEPTION = 0.20
W_EVIDENCE = 0.20
W_INVESTIGATION = 0.10
W_EFFICIENCY = 0.10


def grade_episode(case: CaseData, env_state: InvoiceGuardState) -> GraderResult:
    """
    Grade a completed episode deterministically.

    Returns a GraderResult with score in [0.0, 1.0] and rubric breakdown.
    """
    gt = case.ground_truth
    breakdown = {}

    # ── 1. Decision correctness (0.40) ──────────────────────────────────
    decision_score = 0.0
    if env_state.final_decision:
        agent_decision = env_state.final_decision
        correct_decision = gt.correct_decision.value

        if agent_decision == correct_decision:
            decision_score = 1.0
        else:
            decision_score = _partial_decision_credit(
                agent_decision, correct_decision
            )
    breakdown["decision"] = {
        "agent": env_state.final_decision,
        "correct": gt.correct_decision.value,
        "score": decision_score,
    }

    # ── 2. Exception type correctness (0.20) ────────────────────────────
    exception_score = 0.0
    if env_state.final_exception_type:
        if env_state.final_exception_type == gt.correct_exception_type.value:
            exception_score = 1.0
        elif env_state.proposed_exception == gt.correct_exception_type.value:
            exception_score = 0.5
    elif env_state.proposed_exception == gt.correct_exception_type.value:
        exception_score = 0.3

    breakdown["exception_type"] = {
        "agent_final": env_state.final_exception_type,
        "agent_proposed": env_state.proposed_exception,
        "correct": gt.correct_exception_type.value,
        "score": exception_score,
    }

    # ── 3. Evidence sufficiency (0.20) ──────────────────────────────────
    evidence_score = _score_evidence(
        agent_evidence=env_state.final_evidence,
        actions_taken=env_state.actions_taken,
        acceptable_evidence=gt.acceptable_evidence,
    )
    breakdown["evidence"] = {
        "agent_evidence": env_state.final_evidence,
        "actions_taken": env_state.actions_taken,
        "acceptable": gt.acceptable_evidence,
        "score": evidence_score,
    }

    # ── 4. Investigation quality (0.10) ─────────────────────────────────
    investigation_score = _score_investigation(
        actions_taken=env_state.actions_taken,
        documents_revealed=env_state.documents_revealed,
        acceptable_evidence=gt.acceptable_evidence,
    )
    breakdown["investigation"] = {
        "documents_revealed": env_state.documents_revealed,
        "score": investigation_score,
    }

    # ── 5. Efficiency (0.10) ────────────────────────────────────────────
    efficiency_score = _score_efficiency(
        steps_used=env_state.step_count,
        max_steps=case.max_steps,
        repeated_counts=env_state.repeated_action_counts,
    )
    breakdown["efficiency"] = {
        "steps_used": env_state.step_count,
        "max_steps": case.max_steps,
        "repeated_actions": env_state.repeated_action_counts,
        "score": efficiency_score,
    }

    # ── Weighted total ──────────────────────────────────────────────────
    total = (
        W_DECISION * decision_score
        + W_EXCEPTION * exception_score
        + W_EVIDENCE * evidence_score
        + W_INVESTIGATION * investigation_score
        + W_EFFICIENCY * efficiency_score
    )
    total = round(min(max(total, 0.0), 1.0), 4)

    return GraderResult(
        score=total,
        decision_score=round(decision_score, 4),
        exception_type_score=round(exception_score, 4),
        evidence_score=round(evidence_score, 4),
        investigation_score=round(investigation_score, 4),
        efficiency_score=round(efficiency_score, 4),
        breakdown=breakdown,
    )


def _partial_decision_credit(agent: str, correct: str) -> float:
    """
    Give partial credit for 'close' but wrong decisions.
    e.g. escalate when hold was correct is better than approve when hold was correct.
    """
    RELATED_DECISIONS = {
        "place_on_hold": {"escalate_for_supervisor_review": 0.2, "reject_invoice": 0.15},
        "escalate_for_supervisor_review": {"place_on_hold": 0.2, "reject_invoice": 0.15},
        "reject_invoice": {"escalate_for_supervisor_review": 0.2, "place_on_hold": 0.1},
        "approve_for_payment": {},
    }
    related = RELATED_DECISIONS.get(correct, {})
    return related.get(agent, 0.0)


def _score_evidence(
    agent_evidence: List[str],
    actions_taken: List[str],
    acceptable_evidence: List[str],
) -> float:
    if not acceptable_evidence:
        return 1.0

    cited_set = set(agent_evidence)
    taken_set = set(actions_taken)
    total = 0.0

    for required in acceptable_evidence:
        if required in cited_set:
            total += 1.0
        elif required in taken_set:
            total += 0.7

    return min(total / len(acceptable_evidence), 1.0)


def _score_investigation(
    actions_taken: List[str],
    documents_revealed: List[str],
    acceptable_evidence: List[str],
) -> float:
    if not acceptable_evidence:
        return 1.0

    relevant_actions = 0
    for action in actions_taken:
        if action in acceptable_evidence:
            relevant_actions += 1

    coverage = min(relevant_actions / len(acceptable_evidence), 1.0)

    doc_bonus = min(len(documents_revealed) * 0.15, 0.3)

    return min(coverage + doc_bonus, 1.0)


def _score_efficiency(
    steps_used: int,
    max_steps: int,
    repeated_counts: dict,
) -> float:
    if max_steps == 0:
        return 1.0

    step_ratio = steps_used / max_steps
    total_repeats = sum(max(0, v - 1) for v in repeated_counts.values())

    if step_ratio <= 0.5:
        base = 1.0
    elif step_ratio <= 0.75:
        base = 0.8
    elif step_ratio <= 1.0:
        base = 0.5
    else:
        base = 0.2

    repeat_penalty = min(total_repeats * 0.1, 0.4)

    return max(base - repeat_penalty, 0.0)
