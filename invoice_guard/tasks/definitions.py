"""
Task definitions and synthetic case templates for InvoiceGuard.

Four tasks with increasing difficulty, each with fully synthetic
business documents and deterministic ground truth.
"""

from typing import Dict, List

try:
    from ..models import (
        CaseData, CaseHistory, CompanyPolicy, DecisionType, Difficulty,
        ExceptionType, GRNLineItem, GoodsReceiptNote, GroundTruth,
        Invoice, InvoiceLineItem, POLineItem, PurchaseOrder,
        TaskID, VendorProfile,
    )
except ImportError:
    from models import (
        CaseData, CaseHistory, CompanyPolicy, DecisionType, Difficulty,
        ExceptionType, GRNLineItem, GoodsReceiptNote, GroundTruth,
        Invoice, InvoiceLineItem, POLineItem, PurchaseOrder,
        TaskID, VendorProfile,
    )


TASK_LIST: List[TaskID] = [
    TaskID.task_1_clean_match,
    TaskID.task_2_partial_receipt,
    TaskID.task_3_price_variance,
    TaskID.task_4_duplicate_invoice,
]


def get_task_case(task_id: TaskID) -> CaseData:
    """Return the canonical case data for a given task."""
    builders: Dict[TaskID, callable] = {
        TaskID.task_1_clean_match: _build_task_1,
        TaskID.task_2_partial_receipt: _build_task_2,
        TaskID.task_3_price_variance: _build_task_3,
        TaskID.task_4_duplicate_invoice: _build_task_4,
    }
    return builders[task_id]()


# ── Task 1: Clean Match Approval ────────────────────────────────────────────
# Difficulty: easy | Decision: approve_for_payment | Exception: clean_match
# All three documents align within tolerance. Tests whether the agent can
# confirm a clean case without inventing problems.


def _build_task_1() -> CaseData:
    return CaseData(
        case_id="CASE-1001",
        task_id=TaskID.task_1_clean_match,
        difficulty=Difficulty.easy,
        max_steps=8,
        invoice=Invoice(
            invoice_number="INV-2024-0042",
            supplier_name="Apex Office Supplies",
            supplier_id="SUP-1001",
            invoice_date="2024-11-15",
            po_reference="PO-5500",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-A01",
                    description="Premium Copy Paper A4 (500 sheets)",
                    quantity_billed=100,
                    unit_price_billed=25.00,
                    line_total_billed=2500.00,
                ),
            ],
            subtotal=2500.00,
            tax=200.00,
            total_amount=2700.00,
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-5500",
            supplier_id="SUP-1001",
            order_date="2024-10-20",
            line_items=[
                POLineItem(
                    item_code="ITEM-A01",
                    description="Premium Copy Paper A4 (500 sheets)",
                    ordered_quantity=100,
                    unit_price_ordered=25.00,
                    line_total_ordered=2500.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-7801",
            po_reference="PO-5500",
            receipt_date="2024-11-10",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-A01",
                    quantity_received=100,
                    accepted_quantity=100,
                ),
            ],
            warehouse_note="All items received in good condition.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-1001",
            supplier_name="Apex Office Supplies",
            risk_tier="low",
            preferred_vendor=True,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=100.0,
        ),
        case_history=CaseHistory(),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.approve_for_payment,
            correct_exception_type=ExceptionType.clean_match,
            acceptable_evidence=[
                "inspect_purchase_order",
                "inspect_goods_receipt_note",
            ],
            key_findings=[
                "Quantities match across invoice, PO, and GRN",
                "Prices match between invoice and PO",
                "No discrepancies detected",
            ],
        ),
    )


# ── Task 2: Quantity & Partial Receipt Hold ──────────────────────────────────
# Difficulty: moderate | Decision: place_on_hold | Exception: partial_receipt
# Invoice bills 100 units, PO ordered 100, but GRN only received 60.
# Agent must recognize billed qty exceeds received qty.


def _build_task_2() -> CaseData:
    return CaseData(
        case_id="CASE-2002",
        task_id=TaskID.task_2_partial_receipt,
        difficulty=Difficulty.moderate,
        max_steps=10,
        invoice=Invoice(
            invoice_number="INV-2024-0187",
            supplier_name="Delta Industrial Parts",
            supplier_id="SUP-2050",
            invoice_date="2024-12-01",
            po_reference="PO-6200",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-D10",
                    description="Stainless Steel Bearings (Box of 50)",
                    quantity_billed=100,
                    unit_price_billed=50.00,
                    line_total_billed=5000.00,
                ),
            ],
            subtotal=5000.00,
            tax=400.00,
            total_amount=5400.00,
            note="Please process payment promptly.",
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-6200",
            supplier_id="SUP-2050",
            order_date="2024-11-01",
            line_items=[
                POLineItem(
                    item_code="ITEM-D10",
                    description="Stainless Steel Bearings (Box of 50)",
                    ordered_quantity=100,
                    unit_price_ordered=50.00,
                    line_total_ordered=5000.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-8830",
            po_reference="PO-6200",
            receipt_date="2024-11-25",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-D10",
                    quantity_received=60,
                    accepted_quantity=60,
                    rejected_quantity=0,
                ),
            ],
            warehouse_note="Partial shipment. Remaining 40 units expected next week.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-2050",
            supplier_name="Delta Industrial Parts",
            risk_tier="medium",
            preferred_vendor=True,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=200.0,
        ),
        case_history=CaseHistory(
            prior_comments=["Supplier has had partial shipments before."],
        ),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.place_on_hold,
            correct_exception_type=ExceptionType.partial_receipt,
            acceptable_evidence=[
                "inspect_purchase_order",
                "inspect_goods_receipt_note",
                "compare_quantity",
            ],
            key_findings=[
                "Invoice billed 100 units but GRN shows only 60 received",
                "Billed quantity exceeds received quantity by 40 units (40%)",
                "Variance exceeds quantity tolerance of 5%",
            ],
        ),
    )


# ── Task 3: Price Variance with Policy Tolerance ────────────────────────────
# Difficulty: moderate | Decision: escalate_for_supervisor_review
# Exception: price_mismatch
# Invoice unit price is $110 vs PO price of $100 (10% variance).
# Company tolerance is 5%, so this must be escalated.


def _build_task_3() -> CaseData:
    return CaseData(
        case_id="CASE-3003",
        task_id=TaskID.task_3_price_variance,
        difficulty=Difficulty.moderate,
        max_steps=10,
        invoice=Invoice(
            invoice_number="INV-2024-0315",
            supplier_name="Quantum Tech Solutions",
            supplier_id="SUP-3100",
            invoice_date="2024-12-10",
            po_reference="PO-7100",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-Q22",
                    description="Enterprise SSD 1TB NVMe",
                    quantity_billed=50,
                    unit_price_billed=110.00,
                    line_total_billed=5500.00,
                ),
            ],
            subtotal=5500.00,
            tax=440.00,
            total_amount=5940.00,
            note="Price adjusted due to component shortage surcharge.",
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-7100",
            supplier_id="SUP-3100",
            order_date="2024-11-15",
            line_items=[
                POLineItem(
                    item_code="ITEM-Q22",
                    description="Enterprise SSD 1TB NVMe",
                    ordered_quantity=50,
                    unit_price_ordered=100.00,
                    line_total_ordered=5000.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-9150",
            po_reference="PO-7100",
            receipt_date="2024-12-05",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-Q22",
                    quantity_received=50,
                    accepted_quantity=50,
                ),
            ],
            warehouse_note="All units received and tested OK.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-3100",
            supplier_name="Quantum Tech Solutions",
            risk_tier="medium",
            preferred_vendor=True,
            escalation_threshold=5000.00,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=200.0,
            high_value_threshold=50000.0,
        ),
        case_history=CaseHistory(
            prior_comments=[
                "Supplier occasionally applies surcharges without prior notice.",
            ],
        ),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.escalate_for_supervisor_review,
            correct_exception_type=ExceptionType.price_mismatch,
            acceptable_evidence=[
                "inspect_purchase_order",
                "inspect_policy_rules",
                "compare_price",
            ],
            key_findings=[
                "Invoice unit price $110.00 vs PO unit price $100.00",
                "Price variance of 10% exceeds company tolerance of 5%",
                "Requires supervisor review per escalation policy",
            ],
        ),
    )


# ── Task 4: Duplicate Invoice with Mixed Signals ────────────────────────────
# Difficulty: hard | Decision: reject_invoice | Exception: duplicate_invoice
# Quantities and prices match perfectly, but case history reveals a
# previously processed invoice with near-identical details. A misleading
# vendor note tries to distract.


def _build_task_4() -> CaseData:
    return CaseData(
        case_id="CASE-4004",
        task_id=TaskID.task_4_duplicate_invoice,
        difficulty=Difficulty.hard,
        max_steps=12,
        invoice=Invoice(
            invoice_number="INV-2024-0512",
            supplier_name="Redline Logistics Corp",
            supplier_id="SUP-4200",
            invoice_date="2024-12-18",
            po_reference="PO-8800",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-R05",
                    description="Heavy-Duty Shipping Pallets",
                    quantity_billed=200,
                    unit_price_billed=30.00,
                    line_total_billed=6000.00,
                ),
            ],
            subtotal=6000.00,
            tax=480.00,
            total_amount=6480.00,
            note="URGENT: Payment overdue. Please expedite. Ref: follow-up to INV-2024-0511.",
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-8800",
            supplier_id="SUP-4200",
            order_date="2024-11-20",
            line_items=[
                POLineItem(
                    item_code="ITEM-R05",
                    description="Heavy-Duty Shipping Pallets",
                    ordered_quantity=200,
                    unit_price_ordered=30.00,
                    line_total_ordered=6000.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-9900",
            po_reference="PO-8800",
            receipt_date="2024-12-12",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-R05",
                    quantity_received=200,
                    accepted_quantity=200,
                ),
            ],
            warehouse_note="Full shipment received.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-4200",
            supplier_name="Redline Logistics Corp",
            risk_tier="high",
            duplicate_risk_count=2,
            preferred_vendor=False,
            escalation_threshold=3000.00,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=200.0,
            duplicate_check_enabled=True,
        ),
        case_history=CaseHistory(
            prior_comments=[
                "Previous duplicate submission flagged for this supplier in Q3.",
                "Supplier has been warned about duplicate invoicing.",
            ],
            similar_invoice_refs=["INV-2024-0511"],
            processed_invoice_numbers=[
                "INV-2024-0511",
                "INV-2024-0388",
            ],
        ),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.reject_invoice,
            correct_exception_type=ExceptionType.duplicate_invoice,
            acceptable_evidence=[
                "check_for_duplicate_invoice",
                "inspect_vendor_profile",
            ],
            key_findings=[
                "Previously processed INV-2024-0511 has same supplier, same PO, same total",
                "Current invoice INV-2024-0512 is sequential to processed INV-2024-0511",
                "Supplier flagged as high risk with 2 prior duplicate incidents",
                "Invoice note references INV-2024-0511 as 'follow-up'",
            ],
        ),
    )
