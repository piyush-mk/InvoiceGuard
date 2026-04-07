"""Tests for the InvoiceGuard environment."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import (
    ActionType,
    DecisionType,
    ExceptionType,
    InvoiceGuardAction,
    TaskID,
)
from server.invoice_guard_environment import InvoiceGuardEnvironment


def _make_env():
    return InvoiceGuardEnvironment()


class TestReset:
    def test_reset_returns_observation(self):
        env = _make_env()
        obs = env.reset(task_id="task_1_clean_match")
        assert obs.case_id != ""
        assert obs.task_id == "task_1_clean_match"
        assert obs.difficulty == "easy"
        assert obs.done is False

    def test_reset_provides_goal(self):
        env = _make_env()
        obs = env.reset(task_id="task_1_clean_match")
        assert "accounts payable" in obs.goal.lower()
        assert len(obs.goal) > 100

    def test_reset_provides_suggested_actions(self):
        env = _make_env()
        obs = env.reset(task_id="task_1_clean_match")
        assert len(obs.suggested_next_actions) > 0
        assert "inspect_purchase_order" in obs.suggested_next_actions

    def test_reset_has_available_actions(self):
        env = _make_env()
        obs = env.reset(task_id="task_1_clean_match")
        assert len(obs.available_actions) == 12
        assert "submit_final_resolution" in obs.available_actions

    def test_reset_clean_state(self):
        env = _make_env()
        obs = env.reset(task_id="task_1_clean_match")
        assert obs.revealed_documents == []
        assert obs.findings == []
        assert obs.remaining_steps > 0

    def test_reset_sequential_tasks(self):
        env = _make_env()
        obs1 = env.reset()
        obs2 = env.reset()
        assert obs1.task_id != obs2.task_id or obs1.case_id != obs2.case_id

    def test_all_canonical_tasks_load(self):
        env = _make_env()
        canonical = [
            "task_1_clean_match",
            "task_2_partial_receipt",
            "task_3_price_variance",
            "task_4_duplicate_invoice",
            "task_5_mixed_discrepancy",
            "task_6_false_positive_duplicate",
        ]
        for tid in canonical:
            obs = env.reset(task_id=tid)
            assert obs.task_id == tid
            assert obs.done is False


class TestStep:
    def test_investigation_action(self):
        env = _make_env()
        env.reset(task_id="task_1_clean_match")
        action = InvoiceGuardAction(action_type=ActionType.inspect_purchase_order)
        obs = env.step(action)
        assert obs.done is False
        assert "purchase_order" in obs.revealed_documents
        assert obs.reward > 0

    def test_repeated_action_penalty(self):
        env = _make_env()
        env.reset(task_id="task_1_clean_match")
        action = InvoiceGuardAction(action_type=ActionType.inspect_purchase_order)
        obs1 = env.step(action)
        obs2 = env.step(action)
        assert obs2.reward < obs1.reward

    def test_step_decrements_remaining(self):
        env = _make_env()
        obs = env.reset(task_id="task_1_clean_match")
        initial = obs.remaining_steps
        action = InvoiceGuardAction(action_type=ActionType.inspect_purchase_order)
        obs = env.step(action)
        assert obs.remaining_steps == initial - 1

    def test_suggested_actions_update_after_step(self):
        env = _make_env()
        env.reset(task_id="task_1_clean_match")
        action = InvoiceGuardAction(action_type=ActionType.inspect_purchase_order)
        obs = env.step(action)
        assert "inspect_purchase_order" not in obs.suggested_next_actions

    def test_findings_accumulate(self):
        env = _make_env()
        env.reset(task_id="task_1_clean_match")
        a1 = InvoiceGuardAction(action_type=ActionType.inspect_purchase_order)
        env.step(a1)
        a2 = InvoiceGuardAction(action_type=ActionType.inspect_goods_receipt_note)
        env.step(a2)
        a3 = InvoiceGuardAction(action_type=ActionType.compare_quantity)
        obs = env.step(a3)
        assert len(obs.findings) > 0


class TestSubmission:
    def test_correct_resolution(self):
        env = _make_env()
        env.reset(task_id="task_1_clean_match")
        env.step(InvoiceGuardAction(action_type=ActionType.inspect_purchase_order))
        env.step(InvoiceGuardAction(action_type=ActionType.inspect_goods_receipt_note))
        env.step(InvoiceGuardAction(action_type=ActionType.compare_quantity))

        action = InvoiceGuardAction(
            action_type=ActionType.submit_final_resolution,
            final_decision=DecisionType.approve_for_payment,
            exception_type=ExceptionType.clean_match,
            evidence_references=["inspect_purchase_order", "compare_quantity"],
            explanation="All quantities and prices match within tolerance.",
        )
        obs = env.step(action)
        assert obs.done is True
        assert obs.reward > 0
        assert obs.grader_result.get("score", 0) > 0.5

    def test_wrong_decision_lower_score(self):
        env = _make_env()
        env.reset(task_id="task_1_clean_match")
        env.step(InvoiceGuardAction(action_type=ActionType.inspect_purchase_order))

        action = InvoiceGuardAction(
            action_type=ActionType.submit_final_resolution,
            final_decision=DecisionType.reject_invoice,
            exception_type=ExceptionType.duplicate_invoice,
            evidence_references=["inspect_purchase_order"],
            explanation="Rejecting invoice.",
        )
        obs = env.step(action)
        assert obs.done is True
        assert obs.grader_result.get("score", 1) < 0.5

    def test_missing_fields_returns_error(self):
        env = _make_env()
        env.reset(task_id="task_1_clean_match")
        action = InvoiceGuardAction(
            action_type=ActionType.submit_final_resolution,
        )
        obs = env.step(action)
        assert obs.last_action_error is True or obs.last_action_result != ""


class TestState:
    def test_state_accessible(self):
        env = _make_env()
        env.reset(task_id="task_1_clean_match")
        state = env.state
        assert state.task_id == "task_1_clean_match"
        assert state.step_count == 0

    def test_state_tracks_actions(self):
        env = _make_env()
        env.reset(task_id="task_1_clean_match")
        env.step(InvoiceGuardAction(action_type=ActionType.inspect_purchase_order))
        env.step(InvoiceGuardAction(action_type=ActionType.compare_quantity))
        state = env.state
        assert "inspect_purchase_order" in state.actions_taken
        assert "compare_quantity" in state.actions_taken
        assert state.step_count == 2


class TestTaskDiversity:
    def test_scores_in_valid_range(self):
        """Verify all tasks produce grader scores in [0, 1]."""
        env = _make_env()
        for tid in [
            "task_1_clean_match",
            "task_2_partial_receipt",
            "task_3_price_variance",
            "task_4_duplicate_invoice",
            "task_5_mixed_discrepancy",
            "task_6_false_positive_duplicate",
        ]:
            env.reset(task_id=tid)
            env.step(InvoiceGuardAction(action_type=ActionType.inspect_purchase_order))
            obs = env.step(
                InvoiceGuardAction(
                    action_type=ActionType.submit_final_resolution,
                    final_decision=DecisionType.approve_for_payment,
                    exception_type=ExceptionType.clean_match,
                    evidence_references=["inspect_purchase_order"],
                    explanation="Approving.",
                )
            )
            assert obs.done is True
            score = obs.grader_result.get("score", -1)
            assert 0.0 <= score <= 1.0
