"""
Data models for the InvoiceGuard environment.

Three-Way Invoice Matching Exception Resolution Environment.
All enums, business entities, and OpenEnv interface types.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


# -- Enums --------------------------------------------------------------------


class ActionType(str, Enum):
    """Available actions the agent can take."""

    inspect_invoice_line_items = "inspect_invoice_line_items"
    inspect_purchase_order = "inspect_purchase_order"
    inspect_goods_receipt_note = "inspect_goods_receipt_note"
    inspect_vendor_profile = "inspect_vendor_profile"
    inspect_policy_rules = "inspect_policy_rules"
    check_for_duplicate_invoice = "check_for_duplicate_invoice"
    compare_quantity = "compare_quantity"
    compare_price = "compare_price"
    compare_totals = "compare_totals"
    summarize_findings = "summarize_findings"
    propose_exception_type = "propose_exception_type"
    submit_final_resolution = "submit_final_resolution"


class DecisionType(str, Enum):
    """Final case resolution decisions."""

    approve_for_payment = "approve_for_payment"
    place_on_hold = "place_on_hold"
    reject_invoice = "reject_invoice"
    escalate_for_supervisor_review = "escalate_for_supervisor_review"


class ExceptionType(str, Enum):
    """Primary exception types for invoice cases."""

    clean_match = "clean_match"
    quantity_mismatch = "quantity_mismatch"
    price_mismatch = "price_mismatch"
    total_amount_mismatch = "total_amount_mismatch"
    partial_receipt = "partial_receipt"
    missing_receipt = "missing_receipt"
    duplicate_invoice = "duplicate_invoice"
    tax_variance = "tax_variance"
    policy_violation = "policy_violation"
    mixed_discrepancy = "mixed_discrepancy"


class TaskID(str, Enum):
    """Identifiers for evaluation tasks (4 canonical + variants)."""

    task_1_clean_match = "task_1_clean_match"
    task_1b_multi_line_clean = "task_1b_multi_line_clean"
    task_1c_preferred_vendor_clean = "task_1c_preferred_vendor_clean"
    task_2_partial_receipt = "task_2_partial_receipt"
    task_2b_missing_receipt = "task_2b_missing_receipt"
    task_2c_over_receipt = "task_2c_over_receipt"
    task_3_price_variance = "task_3_price_variance"
    task_3b_within_tolerance = "task_3b_within_tolerance"
    task_3c_total_mismatch = "task_3c_total_mismatch"
    task_4_duplicate_invoice = "task_4_duplicate_invoice"
    task_4b_corrected_invoice_trap = "task_4b_corrected_invoice_trap"
    task_4c_policy_violation = "task_4c_policy_violation"
    task_5_mixed_discrepancy = "task_5_mixed_discrepancy"
    task_6_false_positive_duplicate = "task_6_false_positive_duplicate"
    task_7_retroactive_price = "task_7_retroactive_price"
    task_8_split_invoice_pattern = "task_8_split_invoice_pattern"
    task_9_clean_from_risky_vendor = "task_9_clean_from_risky_vendor"
    task_10_rounding_false_alarm = "task_10_rounding_false_alarm"
    task_11_authorized_overship = "task_11_authorized_overship"
    task_12_corrected_resubmission = "task_12_corrected_resubmission"


class Difficulty(str, Enum):
    """Task difficulty levels."""

    easy = "easy"
    moderate = "moderate"
    hard = "hard"


# -- Business Entities --------------------------------------------------------


class InvoiceLineItem(BaseModel):
    item_code: str
    description: str
    quantity_billed: float
    unit_price_billed: float
    line_total_billed: float


class Invoice(BaseModel):
    invoice_number: str
    supplier_name: str
    supplier_id: str
    invoice_date: str
    currency: str = "USD"
    po_reference: str
    line_items: List[InvoiceLineItem]
    subtotal: float
    tax: float
    total_amount: float
    note: str = ""


class POLineItem(BaseModel):
    item_code: str
    description: str
    ordered_quantity: float
    unit_price_ordered: float
    line_total_ordered: float


class PurchaseOrder(BaseModel):
    po_number: str
    supplier_id: str
    order_date: str
    line_items: List[POLineItem]
    approved: bool = True
    payment_terms: str = "Net 30"


class GRNLineItem(BaseModel):
    item_code: str
    quantity_received: float
    accepted_quantity: float
    rejected_quantity: float = 0.0


class GoodsReceiptNote(BaseModel):
    grn_number: str
    po_reference: str
    receipt_date: str
    line_items: List[GRNLineItem]
    warehouse_note: str = ""


class VendorProfile(BaseModel):
    supplier_id: str
    supplier_name: str
    risk_tier: str
    duplicate_risk_count: int = 0
    preferred_vendor: bool = True
    tolerance_override: Optional[float] = None
    escalation_threshold: Optional[float] = None


class CompanyPolicy(BaseModel):
    quantity_tolerance_pct: float = 5.0
    price_tolerance_pct: float = 5.0
    total_tolerance_amt: float = 100.0
    duplicate_check_enabled: bool = True
    high_value_threshold: float = 50000.0
    mandatory_escalation_above: float = 100000.0


class CaseHistory(BaseModel):
    prior_comments: List[str] = Field(default_factory=list)
    similar_invoice_refs: List[str] = Field(default_factory=list)
    processed_invoice_numbers: List[str] = Field(default_factory=list)
    pending_invoice_numbers: List[str] = Field(default_factory=list)


# -- Ground Truth & Case Data ------------------------------------------------


class GroundTruth(BaseModel):
    correct_decision: DecisionType
    correct_exception_type: ExceptionType
    acceptable_evidence: List[str]
    key_findings: List[str]


class CaseData(BaseModel):
    case_id: str
    task_id: TaskID
    difficulty: Difficulty
    max_steps: int
    invoice: Invoice
    purchase_order: PurchaseOrder
    goods_receipt_note: GoodsReceiptNote
    vendor_profile: VendorProfile
    company_policy: CompanyPolicy
    case_history: CaseHistory
    ground_truth: GroundTruth


# -- OpenEnv Action ----------------------------------------------------------


class InvoiceGuardAction(Action):
    """Agent action for the InvoiceGuard environment."""

    action_type: ActionType
    final_decision: Optional[DecisionType] = None
    exception_type: Optional[ExceptionType] = None
    evidence_references: List[str] = Field(default_factory=list)
    explanation: str = ""
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# -- OpenEnv Observation -----------------------------------------------------


class InvoiceGuardObservation(Observation):
    """What the agent sees after each step."""

    case_id: str = ""
    task_id: str = ""
    difficulty: str = ""
    invoice_summary: str = ""
    goal: str = ""
    available_actions: List[str] = Field(default_factory=list)
    suggested_next_actions: List[str] = Field(default_factory=list)
    revealed_documents: List[str] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)
    remaining_steps: int = 0
    last_action_result: str = ""
    last_action_error: bool = False
    warnings: List[str] = Field(default_factory=list)
    grader_result: Dict[str, Any] = Field(default_factory=dict)


# -- OpenEnv State -----------------------------------------------------------


class InvoiceGuardState(State):
    """Internal state tracking for the environment."""

    task_id: str = ""
    difficulty: str = ""
    case_id: str = ""
    max_steps: int = 0
    actions_taken: List[str] = Field(default_factory=list)
    documents_revealed: List[str] = Field(default_factory=list)
    findings_collected: List[str] = Field(default_factory=list)
    proposed_exception: Optional[str] = None
    is_finalized: bool = False
    final_decision: Optional[str] = None
    final_exception_type: Optional[str] = None
    final_evidence: List[str] = Field(default_factory=list)
    final_explanation: str = ""
    final_confidence: Optional[float] = None
    cumulative_reward: float = 0.0
    repeated_action_counts: Dict[str, int] = Field(default_factory=dict)


# -- Grader Result -----------------------------------------------------------


class GraderResult(BaseModel):
    """Deterministic grading result for a completed episode."""

    score: float = Field(ge=0.0, le=1.0)
    decision_score: float = 0.0
    exception_type_score: float = 0.0
    evidence_score: float = 0.0
    investigation_score: float = 0.0
    explanation_score: float = 0.0
    efficiency_score: float = 0.0
    breakdown: Dict[str, Any] = Field(default_factory=dict)
