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
    TaskID.task_5_mixed_discrepancy,
    TaskID.task_6_false_positive_duplicate,
    TaskID.task_7_retroactive_price,
    TaskID.task_8_split_invoice_pattern,
]

ALL_TASKS: List[TaskID] = [
    TaskID.task_1_clean_match,
    TaskID.task_1b_multi_line_clean,
    TaskID.task_1c_preferred_vendor_clean,
    TaskID.task_2_partial_receipt,
    TaskID.task_2b_missing_receipt,
    TaskID.task_2c_over_receipt,
    TaskID.task_3_price_variance,
    TaskID.task_3b_within_tolerance,
    TaskID.task_3c_total_mismatch,
    TaskID.task_4_duplicate_invoice,
    TaskID.task_4b_corrected_invoice_trap,
    TaskID.task_4c_policy_violation,
    TaskID.task_5_mixed_discrepancy,
    TaskID.task_6_false_positive_duplicate,
    TaskID.task_7_retroactive_price,
    TaskID.task_8_split_invoice_pattern,
]


def get_task_case(task_id: TaskID) -> CaseData:
    """Return the case data for a given task (canonical or variant)."""
    builders: Dict[TaskID, callable] = {
        TaskID.task_1_clean_match: _build_task_1,
        TaskID.task_1b_multi_line_clean: _build_task_1b,
        TaskID.task_1c_preferred_vendor_clean: _build_task_1c,
        TaskID.task_2_partial_receipt: _build_task_2,
        TaskID.task_2b_missing_receipt: _build_task_2b,
        TaskID.task_2c_over_receipt: _build_task_2c,
        TaskID.task_3_price_variance: _build_task_3,
        TaskID.task_3b_within_tolerance: _build_task_3b,
        TaskID.task_3c_total_mismatch: _build_task_3c,
        TaskID.task_4_duplicate_invoice: _build_task_4,
        TaskID.task_4b_corrected_invoice_trap: _build_task_4b,
        TaskID.task_4c_policy_violation: _build_task_4c,
        TaskID.task_5_mixed_discrepancy: _build_task_5,
        TaskID.task_6_false_positive_duplicate: _build_task_6,
        TaskID.task_7_retroactive_price: _build_task_7,
        TaskID.task_8_split_invoice_pattern: _build_task_8,
    }
    return builders[task_id]()


# -- Task 1: Clean Match Approval --------------------------------------------
# Difficulty: easy | Decision: approve_for_payment | Exception: clean_match
# All three documents align within tolerance. Tests whether the agent can
# confirm a clean case without inventing problems.


def _build_task_1() -> CaseData:
    return CaseData(
        case_id="CASE-1001",
        task_id=TaskID.task_1_clean_match,
        difficulty=Difficulty.easy,
        max_steps=10,
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


# -- Task 2: Quantity & Partial Receipt Hold ----------------------------------
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


# -- Task 3: Price Variance with Policy Tolerance ----------------------------
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


# -- Task 4: Duplicate Invoice with Mixed Signals ----------------------------
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


# ===========================================================================
# VARIANT TASKS
# ===========================================================================


# -- Task 1b: Multi-Line Clean Match --------------------------------------
# Three line items, all matching. Tests multi-item verification.

def _build_task_1b() -> CaseData:
    return CaseData(
        case_id="CASE-1002",
        task_id=TaskID.task_1b_multi_line_clean,
        difficulty=Difficulty.easy,
        max_steps=10,
        invoice=Invoice(
            invoice_number="INV-2024-0099",
            supplier_name="Summit Facility Services",
            supplier_id="SUP-1050",
            invoice_date="2024-11-20",
            po_reference="PO-5510",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-F01",
                    description="Industrial Floor Cleaner 5L",
                    quantity_billed=20,
                    unit_price_billed=35.00,
                    line_total_billed=700.00,
                ),
                InvoiceLineItem(
                    item_code="ITEM-F02",
                    description="Microfiber Mop Heads (Pack of 10)",
                    quantity_billed=15,
                    unit_price_billed=18.00,
                    line_total_billed=270.00,
                ),
                InvoiceLineItem(
                    item_code="ITEM-F03",
                    description="Nitrile Gloves Box (100ct)",
                    quantity_billed=50,
                    unit_price_billed=12.00,
                    line_total_billed=600.00,
                ),
            ],
            subtotal=1570.00,
            tax=125.60,
            total_amount=1695.60,
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-5510",
            supplier_id="SUP-1050",
            order_date="2024-10-25",
            line_items=[
                POLineItem(
                    item_code="ITEM-F01",
                    description="Industrial Floor Cleaner 5L",
                    ordered_quantity=20,
                    unit_price_ordered=35.00,
                    line_total_ordered=700.00,
                ),
                POLineItem(
                    item_code="ITEM-F02",
                    description="Microfiber Mop Heads (Pack of 10)",
                    ordered_quantity=15,
                    unit_price_ordered=18.00,
                    line_total_ordered=270.00,
                ),
                POLineItem(
                    item_code="ITEM-F03",
                    description="Nitrile Gloves Box (100ct)",
                    ordered_quantity=50,
                    unit_price_ordered=12.00,
                    line_total_ordered=600.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-7820",
            po_reference="PO-5510",
            receipt_date="2024-11-15",
            line_items=[
                GRNLineItem(item_code="ITEM-F01", quantity_received=20, accepted_quantity=20),
                GRNLineItem(item_code="ITEM-F02", quantity_received=15, accepted_quantity=15),
                GRNLineItem(item_code="ITEM-F03", quantity_received=50, accepted_quantity=50),
            ],
            warehouse_note="All items received, inspected, shelved.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-1050",
            supplier_name="Summit Facility Services",
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
                "All 3 line items match across invoice, PO, and GRN",
                "No price or quantity discrepancies",
            ],
        ),
    )


# -- Task 1c: Preferred Vendor Clean Match --------------------------------
# Single item, low-risk preferred vendor, very straightforward.

def _build_task_1c() -> CaseData:
    return CaseData(
        case_id="CASE-1003",
        task_id=TaskID.task_1c_preferred_vendor_clean,
        difficulty=Difficulty.easy,
        max_steps=10,
        invoice=Invoice(
            invoice_number="INV-2024-0155",
            supplier_name="BrightStar IT Solutions",
            supplier_id="SUP-1200",
            invoice_date="2024-12-02",
            po_reference="PO-5600",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-B10",
                    description="Wireless Keyboard & Mouse Combo",
                    quantity_billed=30,
                    unit_price_billed=45.00,
                    line_total_billed=1350.00,
                ),
            ],
            subtotal=1350.00,
            tax=108.00,
            total_amount=1458.00,
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-5600",
            supplier_id="SUP-1200",
            order_date="2024-11-10",
            line_items=[
                POLineItem(
                    item_code="ITEM-B10",
                    description="Wireless Keyboard & Mouse Combo",
                    ordered_quantity=30,
                    unit_price_ordered=45.00,
                    line_total_ordered=1350.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-7850",
            po_reference="PO-5600",
            receipt_date="2024-11-28",
            line_items=[
                GRNLineItem(item_code="ITEM-B10", quantity_received=30, accepted_quantity=30),
            ],
            warehouse_note="Full shipment. All items functional.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-1200",
            supplier_name="BrightStar IT Solutions",
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
                "Quantities match (30 ordered, 30 billed, 30 received)",
                "Price matches ($45.00 agreed and billed)",
            ],
        ),
    )


# -- Task 2b: Missing Receipt (GRN shows 0 received) ---------------------
# GRN was filed but zero units accepted. Decision: hold.

def _build_task_2b() -> CaseData:
    return CaseData(
        case_id="CASE-2003",
        task_id=TaskID.task_2b_missing_receipt,
        difficulty=Difficulty.moderate,
        max_steps=10,
        invoice=Invoice(
            invoice_number="INV-2024-0201",
            supplier_name="NorthWind Chemicals",
            supplier_id="SUP-2100",
            invoice_date="2024-12-05",
            po_reference="PO-6300",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-N05",
                    description="Laboratory Grade Ethanol 1L",
                    quantity_billed=200,
                    unit_price_billed=22.00,
                    line_total_billed=4400.00,
                ),
            ],
            subtotal=4400.00,
            tax=352.00,
            total_amount=4752.00,
            note="Shipment dispatched per schedule.",
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-6300",
            supplier_id="SUP-2100",
            order_date="2024-11-05",
            line_items=[
                POLineItem(
                    item_code="ITEM-N05",
                    description="Laboratory Grade Ethanol 1L",
                    ordered_quantity=200,
                    unit_price_ordered=22.00,
                    line_total_ordered=4400.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-8850",
            po_reference="PO-6300",
            receipt_date="2024-11-30",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-N05",
                    quantity_received=0,
                    accepted_quantity=0,
                    rejected_quantity=0,
                ),
            ],
            warehouse_note="Shipment not yet received. GRN opened pending delivery.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-2100",
            supplier_name="NorthWind Chemicals",
            risk_tier="medium",
            preferred_vendor=True,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=200.0,
        ),
        case_history=CaseHistory(),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.place_on_hold,
            correct_exception_type=ExceptionType.missing_receipt,
            acceptable_evidence=[
                "inspect_purchase_order",
                "inspect_goods_receipt_note",
                "compare_quantity",
            ],
            key_findings=[
                "GRN shows 0 units received against 200 billed",
                "Shipment has not been delivered or accepted",
                "Cannot pay before goods are received",
            ],
        ),
    )


# -- Task 2c: Over-Receipt (billed more than ordered, received matches ordered) -
# Billed 120 but PO was 100 and GRN accepted 100. Overbilling.

def _build_task_2c() -> CaseData:
    return CaseData(
        case_id="CASE-2004",
        task_id=TaskID.task_2c_over_receipt,
        difficulty=Difficulty.moderate,
        max_steps=10,
        invoice=Invoice(
            invoice_number="INV-2024-0220",
            supplier_name="Pacific Hardware Co",
            supplier_id="SUP-2200",
            invoice_date="2024-12-08",
            po_reference="PO-6400",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-P15",
                    description="Stainless Steel Bolts M10x50 (Box of 100)",
                    quantity_billed=120,
                    unit_price_billed=8.50,
                    line_total_billed=1020.00,
                ),
            ],
            subtotal=1020.00,
            tax=81.60,
            total_amount=1101.60,
            note="Additional 20 boxes included as courtesy stock.",
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-6400",
            supplier_id="SUP-2200",
            order_date="2024-11-10",
            line_items=[
                POLineItem(
                    item_code="ITEM-P15",
                    description="Stainless Steel Bolts M10x50 (Box of 100)",
                    ordered_quantity=100,
                    unit_price_ordered=8.50,
                    line_total_ordered=850.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-8870",
            po_reference="PO-6400",
            receipt_date="2024-12-03",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-P15",
                    quantity_received=100,
                    accepted_quantity=100,
                ),
            ],
            warehouse_note="Received 100 boxes per PO. No extra stock delivered.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-2200",
            supplier_name="Pacific Hardware Co",
            risk_tier="medium",
            preferred_vendor=True,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=100.0,
        ),
        case_history=CaseHistory(),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.place_on_hold,
            correct_exception_type=ExceptionType.quantity_mismatch,
            acceptable_evidence=[
                "inspect_purchase_order",
                "inspect_goods_receipt_note",
                "compare_quantity",
            ],
            key_findings=[
                "Invoice bills 120 units but only 100 were ordered and received",
                "Overbilling of 20 units (20% over received)",
                "Variance far exceeds 5% quantity tolerance",
            ],
        ),
    )


# -- Task 3b: Price Within Tolerance (should approve) ---------------------
# Price is 2% above PO, which is within the 5% tolerance. Clean approval.

def _build_task_3b() -> CaseData:
    return CaseData(
        case_id="CASE-3004",
        task_id=TaskID.task_3b_within_tolerance,
        difficulty=Difficulty.moderate,
        max_steps=10,
        invoice=Invoice(
            invoice_number="INV-2024-0330",
            supplier_name="EcoTech Green Supplies",
            supplier_id="SUP-3200",
            invoice_date="2024-12-12",
            po_reference="PO-7200",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-E08",
                    description="Recycled Paper Reams (500 sheets)",
                    quantity_billed=80,
                    unit_price_billed=20.40,
                    line_total_billed=1632.00,
                ),
            ],
            subtotal=1632.00,
            tax=130.56,
            total_amount=1762.56,
            note="Slight price increase due to recycled material costs.",
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-7200",
            supplier_id="SUP-3200",
            order_date="2024-11-18",
            line_items=[
                POLineItem(
                    item_code="ITEM-E08",
                    description="Recycled Paper Reams (500 sheets)",
                    ordered_quantity=80,
                    unit_price_ordered=20.00,
                    line_total_ordered=1600.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-9170",
            po_reference="PO-7200",
            receipt_date="2024-12-08",
            line_items=[
                GRNLineItem(item_code="ITEM-E08", quantity_received=80, accepted_quantity=80),
            ],
            warehouse_note="All reams received and stored.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-3200",
            supplier_name="EcoTech Green Supplies",
            risk_tier="low",
            preferred_vendor=True,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=200.0,
        ),
        case_history=CaseHistory(),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.approve_for_payment,
            correct_exception_type=ExceptionType.clean_match,
            acceptable_evidence=[
                "inspect_purchase_order",
                "inspect_goods_receipt_note",
                "compare_price",
            ],
            key_findings=[
                "Price variance of 2% is within 5% tolerance",
                "Quantities match (80 ordered, 80 billed, 80 received)",
                "Acceptable per company policy",
            ],
        ),
    )


# -- Task 3c: Total Amount Mismatch --------------------------------------
# Line-item prices match, but the invoice subtotal is inflated (arithmetic error).

def _build_task_3c() -> CaseData:
    return CaseData(
        case_id="CASE-3005",
        task_id=TaskID.task_3c_total_mismatch,
        difficulty=Difficulty.moderate,
        max_steps=10,
        invoice=Invoice(
            invoice_number="INV-2024-0345",
            supplier_name="Vanguard Medical Supplies",
            supplier_id="SUP-3300",
            invoice_date="2024-12-14",
            po_reference="PO-7300",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-V12",
                    description="Disposable Surgical Masks (Box of 50)",
                    quantity_billed=500,
                    unit_price_billed=6.00,
                    line_total_billed=3000.00,
                ),
            ],
            subtotal=3500.00,
            tax=280.00,
            total_amount=3780.00,
            note="Bulk order -- standard pricing applied.",
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-7300",
            supplier_id="SUP-3300",
            order_date="2024-11-20",
            line_items=[
                POLineItem(
                    item_code="ITEM-V12",
                    description="Disposable Surgical Masks (Box of 50)",
                    ordered_quantity=500,
                    unit_price_ordered=6.00,
                    line_total_ordered=3000.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-9200",
            po_reference="PO-7300",
            receipt_date="2024-12-10",
            line_items=[
                GRNLineItem(item_code="ITEM-V12", quantity_received=500, accepted_quantity=500),
            ],
            warehouse_note="Full shipment received.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-3300",
            supplier_name="Vanguard Medical Supplies",
            risk_tier="low",
            preferred_vendor=True,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=200.0,
        ),
        case_history=CaseHistory(),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.escalate_for_supervisor_review,
            correct_exception_type=ExceptionType.total_amount_mismatch,
            acceptable_evidence=[
                "inspect_purchase_order",
                "compare_totals",
            ],
            key_findings=[
                "Invoice subtotal $3,500 does not match line items sum $3,000",
                "Invoice subtotal exceeds PO total by $500 (beyond $200 tolerance)",
                "Possible arithmetic error or hidden charges",
            ],
        ),
    )


# -- Task 4b: Corrected Invoice Trap --------------------------------------
# Invoice claims to be a "corrected version" of a prior invoice, but the
# prior invoice was already paid. Still a duplicate.

def _build_task_4b() -> CaseData:
    return CaseData(
        case_id="CASE-4005",
        task_id=TaskID.task_4b_corrected_invoice_trap,
        difficulty=Difficulty.hard,
        max_steps=12,
        invoice=Invoice(
            invoice_number="INV-2024-0601",
            supplier_name="Atlas Freight Solutions",
            supplier_id="SUP-4300",
            invoice_date="2024-12-20",
            po_reference="PO-8900",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-A20",
                    description="Corrugated Shipping Boxes (Large)",
                    quantity_billed=300,
                    unit_price_billed=4.50,
                    line_total_billed=1350.00,
                ),
            ],
            subtotal=1350.00,
            tax=108.00,
            total_amount=1458.00,
            note="CORRECTED INVOICE -- replaces INV-2024-0600. Please disregard previous version.",
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-8900",
            supplier_id="SUP-4300",
            order_date="2024-11-22",
            line_items=[
                POLineItem(
                    item_code="ITEM-A20",
                    description="Corrugated Shipping Boxes (Large)",
                    ordered_quantity=300,
                    unit_price_ordered=4.50,
                    line_total_ordered=1350.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-9920",
            po_reference="PO-8900",
            receipt_date="2024-12-15",
            line_items=[
                GRNLineItem(item_code="ITEM-A20", quantity_received=300, accepted_quantity=300),
            ],
            warehouse_note="All boxes received per PO.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-4300",
            supplier_name="Atlas Freight Solutions",
            risk_tier="medium",
            duplicate_risk_count=1,
            preferred_vendor=True,
            escalation_threshold=5000.00,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=200.0,
            duplicate_check_enabled=True,
        ),
        case_history=CaseHistory(
            prior_comments=[
                "INV-2024-0600 was processed and paid on 2024-12-17.",
            ],
            similar_invoice_refs=["INV-2024-0600"],
            processed_invoice_numbers=["INV-2024-0600"],
        ),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.reject_invoice,
            correct_exception_type=ExceptionType.duplicate_invoice,
            acceptable_evidence=[
                "check_for_duplicate_invoice",
                "inspect_vendor_profile",
            ],
            key_findings=[
                "INV-2024-0600 was already processed and paid",
                "Current invoice claims to be a 'correction' but has identical amounts",
                "Supplier has 1 prior duplicate incident",
            ],
        ),
    )


# -- Task 4c: Policy Violation (high-value without authorization) ---------
# Invoice total exceeds high-value threshold. Everything else matches,
# but policy requires escalation for high-value invoices.

def _build_task_4c() -> CaseData:
    return CaseData(
        case_id="CASE-4006",
        task_id=TaskID.task_4c_policy_violation,
        difficulty=Difficulty.hard,
        max_steps=12,
        invoice=Invoice(
            invoice_number="INV-2024-0700",
            supplier_name="TitanWorks Engineering",
            supplier_id="SUP-4400",
            invoice_date="2024-12-22",
            po_reference="PO-9100",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-T30",
                    description="CNC Machined Aluminum Housing Unit",
                    quantity_billed=100,
                    unit_price_billed=520.00,
                    line_total_billed=52000.00,
                ),
            ],
            subtotal=52000.00,
            tax=4160.00,
            total_amount=56160.00,
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-9100",
            supplier_id="SUP-4400",
            order_date="2024-12-01",
            line_items=[
                POLineItem(
                    item_code="ITEM-T30",
                    description="CNC Machined Aluminum Housing Unit",
                    ordered_quantity=100,
                    unit_price_ordered=520.00,
                    line_total_ordered=52000.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-9950",
            po_reference="PO-9100",
            receipt_date="2024-12-18",
            line_items=[
                GRNLineItem(item_code="ITEM-T30", quantity_received=100, accepted_quantity=100),
            ],
            warehouse_note="All units received. Quality inspection passed.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-4400",
            supplier_name="TitanWorks Engineering",
            risk_tier="medium",
            preferred_vendor=True,
            escalation_threshold=40000.00,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=500.0,
            high_value_threshold=50000.0,
        ),
        case_history=CaseHistory(),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.escalate_for_supervisor_review,
            correct_exception_type=ExceptionType.policy_violation,
            acceptable_evidence=[
                "inspect_purchase_order",
                "inspect_policy_rules",
                "inspect_vendor_profile",
            ],
            key_findings=[
                "Invoice total $56,160 exceeds high-value threshold of $50,000",
                "All quantities and prices match, but policy requires supervisor review",
                "Invoice also exceeds vendor escalation threshold of $40,000",
            ],
        ),
    )


# ===========================================================================
# GENUINELY HARD TASKS
# ===========================================================================


# -- Task 5: Mixed Discrepancy (price over tolerance + partial receipt) ---
# Difficulty: hard | Decision: escalate_for_supervisor_review
# Exception: mixed_discrepancy
#
# This case has TWO simultaneous issues:
#   1. Price variance: $55 billed vs $50 PO (10%, exceeds 5% tolerance)
#   2. Partial receipt: billed 200, only 150 received
# Both "hold" and "escalate" seem reasonable. Per policy, price variance
# exceeding tolerance triggers escalation, which takes priority over hold.
# The agent must read the policy rules and recognize escalation outranks hold.

def _build_task_5() -> CaseData:
    return CaseData(
        case_id="CASE-5001",
        task_id=TaskID.task_5_mixed_discrepancy,
        difficulty=Difficulty.hard,
        max_steps=12,
        invoice=Invoice(
            invoice_number="INV-2024-0850",
            supplier_name="GlobalParts Manufacturing",
            supplier_id="SUP-5100",
            invoice_date="2024-12-20",
            po_reference="PO-9500",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-G40",
                    description="Precision Titanium Fasteners (Box of 25)",
                    quantity_billed=200,
                    unit_price_billed=55.00,
                    line_total_billed=11000.00,
                ),
            ],
            subtotal=11000.00,
            tax=880.00,
            total_amount=11880.00,
            note="Price reflects updated material costs. Partial shipment billed in full per contract terms.",
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-9500",
            supplier_id="SUP-5100",
            order_date="2024-11-25",
            line_items=[
                POLineItem(
                    item_code="ITEM-G40",
                    description="Precision Titanium Fasteners (Box of 25)",
                    ordered_quantity=200,
                    unit_price_ordered=50.00,
                    line_total_ordered=10000.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-10200",
            po_reference="PO-9500",
            receipt_date="2024-12-15",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-G40",
                    quantity_received=150,
                    accepted_quantity=150,
                    rejected_quantity=0,
                ),
            ],
            warehouse_note="Partial shipment. 50 boxes backordered, ETA 2 weeks.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-5100",
            supplier_name="GlobalParts Manufacturing",
            risk_tier="medium",
            preferred_vendor=True,
            escalation_threshold=8000.00,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=300.0,
            high_value_threshold=50000.0,
        ),
        case_history=CaseHistory(
            prior_comments=[
                "Supplier has a history of price adjustments on raw material orders.",
            ],
        ),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.escalate_for_supervisor_review,
            correct_exception_type=ExceptionType.mixed_discrepancy,
            acceptable_evidence=[
                "inspect_purchase_order",
                "inspect_goods_receipt_note",
                "compare_quantity",
                "compare_price",
                "inspect_policy_rules",
            ],
            key_findings=[
                "Price variance: $55 billed vs $50 PO (10%, exceeds 5% tolerance)",
                "Quantity discrepancy: billed 200 but only 150 received (25% short)",
                "Both price escalation and quantity hold conditions triggered",
                "Per policy, price variance exceeding tolerance requires escalation",
                "Escalation takes priority over hold when both apply",
            ],
        ),
    )


# -- Task 6: False Positive Duplicate (legitimate second order) -----------
# Difficulty: hard | Decision: approve_for_payment
# Exception: clean_match
#
# This case is designed to trap agents that rely on superficial pattern
# matching. The invoice is from the SAME supplier, for the SAME item,
# with a SIMILAR amount -- but references a DIFFERENT PO number. The case
# history has a processed invoice for PO-9600, but this invoice is for
# PO-9700 (a separate legitimate order). The vendor note mentioning
# "same as last month" is a red herring.
#
# The agent must carefully check PO references rather than assuming
# duplicate based on supplier + amount similarity.

def _build_task_6() -> CaseData:
    return CaseData(
        case_id="CASE-6001",
        task_id=TaskID.task_6_false_positive_duplicate,
        difficulty=Difficulty.hard,
        max_steps=12,
        invoice=Invoice(
            invoice_number="INV-2024-0920",
            supplier_name="Metro Cleaning Services",
            supplier_id="SUP-6100",
            invoice_date="2024-12-28",
            po_reference="PO-9700",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-M15",
                    description="Monthly Office Cleaning Service - January",
                    quantity_billed=1,
                    unit_price_billed=4500.00,
                    line_total_billed=4500.00,
                ),
            ],
            subtotal=4500.00,
            tax=360.00,
            total_amount=4860.00,
            note="Monthly service invoice -- same as last month's order.",
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-9700",
            supplier_id="SUP-6100",
            order_date="2024-12-15",
            line_items=[
                POLineItem(
                    item_code="ITEM-M15",
                    description="Monthly Office Cleaning Service - January",
                    ordered_quantity=1,
                    unit_price_ordered=4500.00,
                    line_total_ordered=4500.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-10500",
            po_reference="PO-9700",
            receipt_date="2024-12-26",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-M15",
                    quantity_received=1,
                    accepted_quantity=1,
                ),
            ],
            warehouse_note="Service confirmed completed for January cycle.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-6100",
            supplier_name="Metro Cleaning Services",
            risk_tier="low",
            duplicate_risk_count=0,
            preferred_vendor=True,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=200.0,
            duplicate_check_enabled=True,
        ),
        case_history=CaseHistory(
            prior_comments=[
                "Monthly recurring service contract. New PO issued each month.",
            ],
            processed_invoice_numbers=[
                "INV-2024-0880",
            ],
            similar_invoice_refs=[],
        ),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.approve_for_payment,
            correct_exception_type=ExceptionType.clean_match,
            acceptable_evidence=[
                "inspect_purchase_order",
                "inspect_goods_receipt_note",
                "check_for_duplicate_invoice",
            ],
            key_findings=[
                "Invoice references PO-9700, which is a separate order from prior PO-9600",
                "Prior invoice INV-2024-0880 was for a different PO (different month)",
                "Quantities and prices match PO exactly",
                "No actual duplicate -- this is a legitimate recurring service invoice",
                "Vendor has 0 duplicate risk incidents",
            ],
        ),
    )


# -- Task 7: Retroactive Price Adjustment -----------------------------------
# Difficulty: hard | Decision: escalate_for_supervisor_review | Exception: price_mismatch
# Vendor had a contract price change effective 2024-11-01, but the PO was issued
# at the old price on 2024-10-20. The invoice uses the NEW (higher) price.
# Agent must notice the temporal mismatch: PO predates the price change, so the
# PO price is the agreed-upon price. The invoice price exceeds tolerance.
# Tests temporal reasoning and policy awareness.


def _build_task_7() -> CaseData:
    return CaseData(
        case_id="CASE-7001",
        task_id=TaskID.task_7_retroactive_price,
        difficulty=Difficulty.hard,
        max_steps=12,
        invoice=Invoice(
            invoice_number="INV-2024-1150",
            supplier_name="Precision Parts Ltd",
            supplier_id="SUP-7100",
            invoice_date="2024-11-10",
            po_reference="PO-7500",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-PP01",
                    description="CNC Bearing Assembly Type-R",
                    quantity_billed=200,
                    unit_price_billed=47.50,
                    line_total_billed=9500.00,
                ),
                InvoiceLineItem(
                    item_code="ITEM-PP02",
                    description="Hydraulic Seal Kit Grade-A",
                    quantity_billed=500,
                    unit_price_billed=12.80,
                    line_total_billed=6400.00,
                ),
            ],
            subtotal=15900.00,
            tax=1272.00,
            total_amount=17172.00,
            note="Prices reflect updated catalog effective 2024-11-01.",
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-7500",
            supplier_id="SUP-7100",
            order_date="2024-10-20",
            line_items=[
                POLineItem(
                    item_code="ITEM-PP01",
                    description="CNC Bearing Assembly Type-R",
                    ordered_quantity=200,
                    unit_price_ordered=42.00,
                    line_total_ordered=8400.00,
                ),
                POLineItem(
                    item_code="ITEM-PP02",
                    description="Hydraulic Seal Kit Grade-A",
                    ordered_quantity=500,
                    unit_price_ordered=11.50,
                    line_total_ordered=5750.00,
                ),
            ],
            payment_terms="Net 45",
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-12500",
            po_reference="PO-7500",
            receipt_date="2024-11-05",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-PP01",
                    quantity_received=200,
                    accepted_quantity=200,
                ),
                GRNLineItem(
                    item_code="ITEM-PP02",
                    quantity_received=500,
                    accepted_quantity=500,
                ),
            ],
            warehouse_note="All items received in good condition. Full order.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-7100",
            supplier_name="Precision Parts Ltd",
            risk_tier="medium",
            duplicate_risk_count=0,
            preferred_vendor=True,
            escalation_threshold=15000.00,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=500.0,
            duplicate_check_enabled=True,
            high_value_threshold=50000.0,
        ),
        case_history=CaseHistory(
            prior_comments=[
                "Vendor announced price increase effective 2024-11-01.",
                "PO-7500 was placed on 2024-10-20 at the old catalog prices.",
                "Contract terms: PO price governs unless amended before shipment.",
            ],
        ),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.escalate_for_supervisor_review,
            correct_exception_type=ExceptionType.price_mismatch,
            acceptable_evidence=[
                "inspect_purchase_order",
                "inspect_goods_receipt_note",
                "compare_price",
                "inspect_policy_rules",
                "inspect_vendor_profile",
            ],
            key_findings=[
                "ITEM-PP01 billed at $47.50 vs PO price $42.00 (13.1% variance)",
                "ITEM-PP02 billed at $12.80 vs PO price $11.50 (11.3% variance)",
                "Both exceed 5% price tolerance",
                "PO was issued 2024-10-20, before price change effective 2024-11-01",
                "Per contract terms, PO price governs unless PO is amended",
                "Invoice total $17,172 exceeds vendor escalation threshold $15,000",
            ],
        ),
    )


# -- Task 8: Adversarial Split-Invoice Pattern --------------------------------
# Difficulty: hard | Decision: escalate_for_supervisor_review | Exception: policy_violation
# Supplier splits a large order into multiple smaller invoices to stay below
# the auto-approval threshold ($50,000). Each individual invoice looks clean,
# but case history reveals a pattern of split invoicing from this vendor.
# Agent must check vendor profile (high duplicate risk count) and case history
# to detect the pattern. Tests cross-case pattern recognition.


def _build_task_8() -> CaseData:
    return CaseData(
        case_id="CASE-8001",
        task_id=TaskID.task_8_split_invoice_pattern,
        difficulty=Difficulty.hard,
        max_steps=12,
        invoice=Invoice(
            invoice_number="INV-2024-2210",
            supplier_name="Global Tech Supplies",
            supplier_id="SUP-8200",
            invoice_date="2024-12-01",
            po_reference="PO-8800",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-GT10",
                    description="Enterprise Server Rack Unit 42U",
                    quantity_billed=3,
                    unit_price_billed=14500.00,
                    line_total_billed=43500.00,
                ),
            ],
            subtotal=43500.00,
            tax=3480.00,
            total_amount=46980.00,
            note="Shipment 3 of 3. Final batch for data center buildout.",
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-8800",
            supplier_id="SUP-8200",
            order_date="2024-11-15",
            line_items=[
                POLineItem(
                    item_code="ITEM-GT10",
                    description="Enterprise Server Rack Unit 42U",
                    ordered_quantity=3,
                    unit_price_ordered=14500.00,
                    line_total_ordered=43500.00,
                ),
            ],
            payment_terms="Net 30",
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-15200",
            po_reference="PO-8800",
            receipt_date="2024-11-28",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-GT10",
                    quantity_received=3,
                    accepted_quantity=3,
                ),
            ],
            warehouse_note="Final batch received. All 9 rack units now in inventory.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-8200",
            supplier_name="Global Tech Supplies",
            risk_tier="high",
            duplicate_risk_count=4,
            preferred_vendor=False,
            escalation_threshold=40000.00,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=500.0,
            duplicate_check_enabled=True,
            high_value_threshold=50000.00,
            mandatory_escalation_above=100000.00,
        ),
        case_history=CaseHistory(
            prior_comments=[
                "ALERT: Vendor SUP-8200 has submitted 3 invoices this month "
                "for similar items across 3 separate POs (PO-8600, PO-8700, PO-8800).",
                "Each invoice is just under the $50,000 auto-approval threshold.",
                "Combined value: $46,200 + $47,800 + $46,980 = $140,980.",
                "Finance team flagged potential invoice-splitting pattern.",
                "Vendor is non-preferred with 4 prior risk incidents.",
            ],
            similar_invoice_refs=["INV-2024-2180", "INV-2024-2195"],
            processed_invoice_numbers=["INV-2024-2180", "INV-2024-2195"],
        ),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.escalate_for_supervisor_review,
            correct_exception_type=ExceptionType.policy_violation,
            acceptable_evidence=[
                "inspect_purchase_order",
                "inspect_goods_receipt_note",
                "inspect_vendor_profile",
                "inspect_policy_rules",
                "check_for_duplicate_invoice",
            ],
            key_findings=[
                "Invoice $46,980 is just under auto-approval threshold of $50,000",
                "3 similar invoices from same vendor total $140,980",
                "Vendor is high-risk tier with 4 prior risk incidents",
                "Vendor is non-preferred",
                "Case history explicitly flags split-invoice pattern concern",
                "Combined value exceeds mandatory escalation threshold of $100,000",
                "Individual prices and quantities match PO -- no line-level discrepancy",
            ],
        ),
    )
