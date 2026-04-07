"""Tests for the InvoiceGuard grading system."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import (
    CaseData,
    CaseHistory,
    CompanyPolicy,
    DecisionType,
    Difficulty,
    ExceptionType,
    GRNLineItem,
    GoodsReceiptNote,
    GroundTruth,
    Invoice,
    InvoiceGuardState,
    InvoiceLineItem,
    POLineItem,
    PurchaseOrder,
    TaskID,
    VendorProfile,
)
from graders.scoring import grade_episode, _score_explanation


def _make_case(
    correct_decision=DecisionType.approve_for_payment,
    correct_exception=ExceptionType.clean_match,
    acceptable_evidence=None,
    key_findings=None,
    max_steps=10,
):
    return CaseData(
        case_id="TEST-001",
        task_id=TaskID.task_1_clean_match,
        difficulty=Difficulty.easy,
        max_steps=max_steps,
        invoice=Invoice(
            invoice_number="INV-001",
            supplier_name="Test",
            supplier_id="SUP-001",
            invoice_date="2024-01-01",
            po_reference="PO-001",
            line_items=[
                InvoiceLineItem(
                    item_code="A",
                    description="Widget",
                    quantity_billed=10,
                    unit_price_billed=5.0,
                    line_total_billed=50.0,
                )
            ],
            subtotal=50.0,
            tax=5.0,
            total_amount=55.0,
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-001",
            supplier_id="SUP-001",
            order_date="2024-01-01",
            line_items=[
                POLineItem(
                    item_code="A",
                    description="Widget",
                    ordered_quantity=10,
                    unit_price_ordered=5.0,
                    line_total_ordered=50.0,
                )
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-001",
            po_reference="PO-001",
            receipt_date="2024-01-05",
            line_items=[
                GRNLineItem(
                    item_code="A",
                    quantity_received=10,
                    accepted_quantity=10,
                )
            ],
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-001",
            supplier_name="Test",
            risk_tier="low",
        ),
        company_policy=CompanyPolicy(),
        case_history=CaseHistory(),
        ground_truth=GroundTruth(
            correct_decision=correct_decision,
            correct_exception_type=correct_exception,
            acceptable_evidence=acceptable_evidence or [
                "inspect_purchase_order",
                "compare_quantity",
                "compare_price",
            ],
            key_findings=key_findings or [
                "quantities match within tolerance",
                "prices match within 5% tolerance",
            ],
        ),
    )


def _make_state(
    decision=None,
    exception_type=None,
    actions=None,
    evidence=None,
    explanation="",
    step_count=5,
):
    s = InvoiceGuardState()
    s.step_count = step_count
    s.final_decision = decision
    s.final_exception_type = exception_type
    s.actions_taken = actions or []
    s.final_evidence = evidence or []
    s.final_explanation = explanation
    s.documents_revealed = [
        a.replace("inspect_", "").replace("_", " ")
        for a in (actions or [])
        if a.startswith("inspect_")
    ]
    s.repeated_action_counts = {}
    for a in (actions or []):
        s.repeated_action_counts[a] = s.repeated_action_counts.get(a, 0) + 1
    return s


class TestGradeEpisode:
    def test_perfect_score(self):
        case = _make_case()
        state = _make_state(
            decision="approve_for_payment",
            exception_type="clean_match",
            actions=["inspect_purchase_order", "compare_quantity", "compare_price"],
            evidence=["inspect_purchase_order", "compare_quantity", "compare_price"],
            explanation="All quantities and prices match within 5% tolerance per company policy.",
            step_count=4,
        )
        result = grade_episode(case, state)
        assert result.score >= 0.85
        assert result.decision_score == 1.0
        assert result.exception_type_score == 1.0

    def test_wrong_decision_zero_decision_score(self):
        case = _make_case()
        state = _make_state(
            decision="reject_invoice",
            exception_type="clean_match",
            actions=["inspect_purchase_order"],
            evidence=["inspect_purchase_order"],
            explanation="Rejecting.",
        )
        result = grade_episode(case, state)
        assert result.decision_score == 0.0

    def test_partial_decision_credit(self):
        case = _make_case(correct_decision=DecisionType.place_on_hold)
        state = _make_state(
            decision="escalate_for_supervisor_review",
            exception_type="partial_receipt",
            actions=["inspect_purchase_order"],
            evidence=["inspect_purchase_order"],
        )
        result = grade_episode(case, state)
        assert 0.0 < result.decision_score < 1.0

    def test_no_decision_scores_zero(self):
        case = _make_case()
        state = _make_state(decision=None, step_count=10)
        result = grade_episode(case, state)
        assert result.decision_score == 0.0
        assert result.score < 0.5

    def test_score_in_bounds(self):
        case = _make_case()
        state = _make_state(
            decision="approve_for_payment",
            exception_type="clean_match",
            actions=["inspect_purchase_order"],
            evidence=["inspect_purchase_order"],
        )
        result = grade_episode(case, state)
        assert 0.0 <= result.score <= 1.0

    def test_all_rubric_dimensions_present(self):
        case = _make_case()
        state = _make_state(
            decision="approve_for_payment",
            exception_type="clean_match",
            actions=["inspect_purchase_order"],
            evidence=["inspect_purchase_order"],
            explanation="Matches.",
        )
        result = grade_episode(case, state)
        assert result.decision_score >= 0
        assert result.exception_type_score >= 0
        assert result.evidence_score >= 0
        assert result.investigation_score >= 0
        assert result.explanation_score >= 0
        assert result.efficiency_score >= 0


class TestExplanationScoring:
    def test_empty_explanation(self):
        case = _make_case()
        score = _score_explanation("", case, ["quantities match"])
        assert score == 0.0

    def test_good_explanation(self):
        case = _make_case(
            key_findings=["quantities match within tolerance", "price variance 3%"],
        )
        score = _score_explanation(
            "All quantities match within the 5% tolerance threshold per company policy. "
            "Price variance of 3% is acceptable.",
            case,
            case.ground_truth.key_findings,
        )
        assert score >= 0.6

    def test_explanation_with_numbers(self):
        case = _make_case()
        score = _score_explanation(
            "The invoice total of $55.00 matches the PO within tolerance.",
            case,
            [],
        )
        assert score > 0.0

    def test_vague_explanation_scores_low(self):
        case = _make_case()
        score = _score_explanation("ok", case, ["quantities match"])
        assert score <= 0.4
