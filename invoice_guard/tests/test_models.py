"""Tests for InvoiceGuard data models."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import (
    ActionType,
    DecisionType,
    ExceptionType,
    TaskID,
    Difficulty,
    InvoiceGuardAction,
    InvoiceGuardObservation,
    InvoiceGuardState,
    GraderResult,
    Invoice,
    InvoiceLineItem,
    PurchaseOrder,
    POLineItem,
    GoodsReceiptNote,
    GRNLineItem,
    VendorProfile,
    CompanyPolicy,
    CaseData,
    CaseHistory,
    GroundTruth,
)


class TestEnums:
    def test_action_types_count(self):
        assert len(ActionType) == 12

    def test_decision_types(self):
        expected = {
            "approve_for_payment",
            "place_on_hold",
            "reject_invoice",
            "escalate_for_supervisor_review",
        }
        assert {d.value for d in DecisionType} == expected

    def test_exception_types_include_key_variants(self):
        values = {e.value for e in ExceptionType}
        assert "clean_match" in values
        assert "duplicate_invoice" in values
        assert "price_mismatch" in values
        assert "partial_receipt" in values
        assert "mixed_discrepancy" in values

    def test_task_ids_minimum_12_canonical(self):
        canonical = [t for t in TaskID if not any(c in t.value for c in ["b_", "c_"])]
        assert len(canonical) >= 12

    def test_difficulty_levels(self):
        assert set(d.value for d in Difficulty) == {"easy", "moderate", "hard"}


class TestAction:
    def test_minimal_action(self):
        a = InvoiceGuardAction(action_type=ActionType.inspect_purchase_order)
        assert a.action_type == ActionType.inspect_purchase_order
        assert a.final_decision is None
        assert a.evidence_references == []

    def test_resolution_action(self):
        a = InvoiceGuardAction(
            action_type=ActionType.submit_final_resolution,
            final_decision=DecisionType.approve_for_payment,
            exception_type=ExceptionType.clean_match,
            evidence_references=["inspect_purchase_order", "compare_quantity"],
            explanation="All documents match.",
        )
        assert a.final_decision == DecisionType.approve_for_payment
        assert len(a.evidence_references) == 2


class TestObservation:
    def test_defaults(self):
        obs = InvoiceGuardObservation()
        assert obs.case_id == ""
        assert obs.available_actions == []
        assert obs.suggested_next_actions == []
        assert obs.remaining_steps == 0
        assert obs.done is False
        assert obs.grader_result == {}

    def test_populated(self):
        obs = InvoiceGuardObservation(
            case_id="CASE-001",
            task_id="task_1_clean_match",
            difficulty="easy",
            remaining_steps=10,
            suggested_next_actions=["inspect_purchase_order"],
        )
        assert obs.case_id == "CASE-001"
        assert obs.suggested_next_actions == ["inspect_purchase_order"]


class TestState:
    def test_defaults(self):
        s = InvoiceGuardState()
        assert s.step_count == 0
        assert s.actions_taken == []
        assert s.is_finalized is False
        assert s.cumulative_reward == 0.0

    def test_mutation(self):
        s = InvoiceGuardState()
        s.actions_taken.append("inspect_purchase_order")
        s.step_count = 1
        assert len(s.actions_taken) == 1


class TestGraderResult:
    def test_valid_score(self):
        r = GraderResult(score=0.85)
        assert r.score == 0.85
        assert r.explanation_score == 0.0

    def test_score_bounds(self):
        r = GraderResult(score=0.0)
        assert r.score == 0.0
        r = GraderResult(score=1.0)
        assert r.score == 1.0


class TestBusinessEntities:
    def test_invoice_creation(self):
        inv = Invoice(
            invoice_number="INV-001",
            supplier_name="Test Corp",
            supplier_id="SUP-001",
            invoice_date="2024-01-01",
            po_reference="PO-001",
            line_items=[
                InvoiceLineItem(
                    item_code="ITM-A",
                    description="Widget",
                    quantity_billed=10,
                    unit_price_billed=5.0,
                    line_total_billed=50.0,
                )
            ],
            subtotal=50.0,
            tax=5.0,
            total_amount=55.0,
        )
        assert inv.currency == "USD"
        assert len(inv.line_items) == 1

    def test_company_policy_defaults(self):
        p = CompanyPolicy()
        assert p.quantity_tolerance_pct == 5.0
        assert p.price_tolerance_pct == 5.0
        assert p.duplicate_check_enabled is True
