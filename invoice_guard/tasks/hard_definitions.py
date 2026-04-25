"""
Round 2 hard-mode task slice for InvoiceGuard.

These 10 tasks are designed for a *strong gap* baseline (~40-55%) so that
RL training improvement is visible. Each task targets one of the Round 2
themes: cross-case fraud, time-dependent policy, contradictory evidence,
partial observability, multi-party approvals.

All tasks reuse the existing CaseData entities so no schema changes are
required. Adversarial context is conveyed via:
  - invoice.note
  - po.payment_terms
  - grn.warehouse_note
  - vendor_profile fields
  - case_history.prior_comments  (surfaced by check_for_duplicate_invoice
    when similar_invoice_refs is non-empty)
"""

from typing import Dict

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


# Standardized step budget for hard-mode tasks. Slightly tighter than
# canonical hard tasks (12) to reward efficient investigation.
HARD_MAX_STEPS = 12


# -- H1: Phantom GRN from a prior delivery cycle ----------------------------
# Decision: place_on_hold | Exception: missing_receipt
# The GRN attached to this PO is from an OLD shipment cycle (months earlier).
# Vendor reuses old GRN numbers. compare_quantity will report all numbers
# matching, but no goods were actually received for THIS invoice cycle.
# Trap: agent runs compare_quantity, sees 100=100=100, and approves.

def _build_h1() -> CaseData:
    return CaseData(
        case_id="CASE-H1-001",
        task_id=TaskID.task_h1_phantom_grn_period,
        difficulty=Difficulty.hard,
        max_steps=HARD_MAX_STEPS,
        invoice=Invoice(
            invoice_number="INV-2024-9101",
            supplier_name="Helix Industrial Supply",
            supplier_id="SUP-H101",
            invoice_date="2024-12-22",
            po_reference="PO-H101-DEC",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-HX01",
                    description="Industrial Lubricant 20L Drum",
                    quantity_billed=100,
                    unit_price_billed=180.00,
                    line_total_billed=18000.00,
                ),
            ],
            subtotal=18000.00,
            tax=1440.00,
            total_amount=19440.00,
            note="Q4 restocking shipment. Please process per contract.",
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-H101-DEC",
            supplier_id="SUP-H101",
            order_date="2024-12-01",
            line_items=[
                POLineItem(
                    item_code="ITEM-HX01",
                    description="Industrial Lubricant 20L Drum",
                    ordered_quantity=100,
                    unit_price_ordered=180.00,
                    line_total_ordered=18000.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-H101-JUN",
            po_reference="PO-H101-DEC",
            receipt_date="2024-06-15",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-HX01",
                    quantity_received=100,
                    accepted_quantity=100,
                ),
            ],
            warehouse_note=(
                "WAREHOUSE FLAG: GRN-H101-JUN dated 2024-06-15 belongs to "
                "prior shipment cycle (PO-H101-JUN, already invoiced and paid). "
                "No fresh delivery has been logged against PO-H101-DEC. "
                "Linkage to PO-H101-DEC appears to be a clerical error by vendor."
            ),
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-H101",
            supplier_name="Helix Industrial Supply",
            risk_tier="medium",
            duplicate_risk_count=2,
            preferred_vendor=False,
            escalation_threshold=15000.00,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=200.0,
        ),
        case_history=CaseHistory(
            prior_comments=[
                "PO-H101-JUN was fully received (GRN-H101-JUN) and paid in July 2024.",
                "Vendor has previously attached old GRN numbers to new POs to "
                "accelerate payment before goods arrive.",
            ],
            similar_invoice_refs=["INV-2024-4500"],
            processed_invoice_numbers=["INV-2024-4500"],
        ),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.place_on_hold,
            correct_exception_type=ExceptionType.missing_receipt,
            acceptable_evidence=[
                "inspect_purchase_order",
                "inspect_goods_receipt_note",
                "check_for_duplicate_invoice",
                "inspect_vendor_profile",
            ],
            key_findings=[
                "GRN attached is dated 2024-06-15, predating PO-H101-DEC by 6 months",
                "GRN-H101-JUN was already consumed by prior invoice INV-2024-4500",
                "No fresh receipt exists for the December order",
                "Cannot pay before goods are actually received",
            ],
        ),
    )


# -- H2: Kickback / Inflated PO --------------------------------------------
# Decision: escalate_for_supervisor_review | Exception: policy_violation
# Invoice price = PO price exactly. Quantities match. But case history reveals
# internal audit found PO price ~30% above market and buyer recused. The
# correct call is escalate, not approve.

def _build_h2() -> CaseData:
    return CaseData(
        case_id="CASE-H2-001",
        task_id=TaskID.task_h2_kickback_inflated_po,
        difficulty=Difficulty.hard,
        max_steps=HARD_MAX_STEPS,
        invoice=Invoice(
            invoice_number="INV-2024-9202",
            supplier_name="Crestwave Engineering Pvt Ltd",
            supplier_id="SUP-H202",
            invoice_date="2024-12-19",
            po_reference="PO-H202",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-CW20",
                    description="Custom Aluminum Bracket Type-7",
                    quantity_billed=400,
                    unit_price_billed=85.00,
                    line_total_billed=34000.00,
                ),
            ],
            subtotal=34000.00,
            tax=2720.00,
            total_amount=36720.00,
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-H202",
            supplier_id="SUP-H202",
            order_date="2024-11-10",
            line_items=[
                POLineItem(
                    item_code="ITEM-CW20",
                    description="Custom Aluminum Bracket Type-7",
                    ordered_quantity=400,
                    unit_price_ordered=85.00,
                    line_total_ordered=34000.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-H202",
            po_reference="PO-H202",
            receipt_date="2024-12-12",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-CW20",
                    quantity_received=400,
                    accepted_quantity=400,
                ),
            ],
            warehouse_note="All units received and inspected.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-H202",
            supplier_name="Crestwave Engineering Pvt Ltd",
            risk_tier="medium",
            duplicate_risk_count=0,
            preferred_vendor=False,
            escalation_threshold=30000.00,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=300.0,
        ),
        case_history=CaseHistory(
            prior_comments=[
                "AUDIT FLAG (2024-12-15): PO-H202 unit price $85 is approx 30% "
                "above benchmarked market price of $65 for ITEM-CW20.",
                "Buyer who approved PO-H202 has been recused pending integrity review.",
                "Per policy, all invoices against PO-H202 require supervisor review "
                "before payment release.",
            ],
            similar_invoice_refs=["AUDIT-PO-H202"],
            processed_invoice_numbers=[],
        ),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.escalate_for_supervisor_review,
            correct_exception_type=ExceptionType.policy_violation,
            acceptable_evidence=[
                "inspect_purchase_order",
                "check_for_duplicate_invoice",
                "inspect_vendor_profile",
                "inspect_policy_rules",
            ],
            key_findings=[
                "Invoice and PO numerically match, but PO itself is under audit",
                "PO unit price is ~30% above market benchmark",
                "Approving buyer recused; mandatory supervisor review applies",
                "Escalation required regardless of line-level match",
            ],
        ),
    )


# -- H3: Currency Silent Swap ---------------------------------------------
# Decision: escalate_for_supervisor_review | Exception: total_amount_mismatch
# Invoice currency is EUR while PO and historical billing are USD. Numbers
# match by digit, but the invoiced amount in USD-equivalent is materially
# different. Trap: compare_price reports a numerical match.

def _build_h3() -> CaseData:
    return CaseData(
        case_id="CASE-H3-001",
        task_id=TaskID.task_h3_currency_silent_swap,
        difficulty=Difficulty.hard,
        max_steps=HARD_MAX_STEPS,
        invoice=Invoice(
            invoice_number="INV-2024-9303",
            supplier_name="Northbridge Components GmbH",
            supplier_id="SUP-H303",
            invoice_date="2024-12-17",
            currency="EUR",
            po_reference="PO-H303",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-NB15",
                    description="Precision Linear Actuator 200mm",
                    quantity_billed=80,
                    unit_price_billed=125.00,
                    line_total_billed=10000.00,
                ),
            ],
            subtotal=10000.00,
            tax=800.00,
            total_amount=10800.00,
            note="Standard monthly invoice. EUR billing per updated contract.",
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-H303",
            supplier_id="SUP-H303",
            order_date="2024-11-20",
            line_items=[
                POLineItem(
                    item_code="ITEM-NB15",
                    description="Precision Linear Actuator 200mm",
                    ordered_quantity=80,
                    unit_price_ordered=125.00,
                    line_total_ordered=10000.00,
                ),
            ],
            payment_terms="Net 30 USD",
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-H303",
            po_reference="PO-H303",
            receipt_date="2024-12-10",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-NB15",
                    quantity_received=80,
                    accepted_quantity=80,
                ),
            ],
            warehouse_note="All units received in good condition.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-H303",
            supplier_name="Northbridge Components GmbH",
            risk_tier="medium",
            preferred_vendor=True,
            escalation_threshold=12000.00,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=200.0,
        ),
        case_history=CaseHistory(
            prior_comments=[
                "Vendor SUP-H303 has historically invoiced in USD against this PO family.",
                "No contract amendment authorizing EUR billing is on file.",
                "Current EUR-to-USD rate would make invoiced amount ~10% higher than PO USD value.",
            ],
            similar_invoice_refs=["INV-2024-7800"],
            processed_invoice_numbers=["INV-2024-7800"],
        ),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.escalate_for_supervisor_review,
            correct_exception_type=ExceptionType.total_amount_mismatch,
            acceptable_evidence=[
                "inspect_invoice_line_items",
                "inspect_purchase_order",
                "check_for_duplicate_invoice",
                "compare_totals",
            ],
            key_findings=[
                "Invoice currency is EUR but PO is USD (no contract amendment)",
                "Numerical line totals match but currency differs",
                "USD-equivalent of EUR invoice is materially above PO total",
                "Currency change requires supervisor approval per contract",
            ],
        ),
    )


# -- H4: Threshold Dance ---------------------------------------------------
# Decision: escalate_for_supervisor_review | Exception: policy_violation
# Invoice $49,950 sits just under $50k auto-approval threshold. Vendor has
# 3 prior invoices in the same window all $49,9xx. Smaller cousin of task_8
# but with a tighter margin so the pattern is harder to spot.

def _build_h4() -> CaseData:
    return CaseData(
        case_id="CASE-H4-001",
        task_id=TaskID.task_h4_threshold_dance,
        difficulty=Difficulty.hard,
        max_steps=HARD_MAX_STEPS,
        invoice=Invoice(
            invoice_number="INV-2024-9404",
            supplier_name="Vector Distribution Co",
            supplier_id="SUP-H404",
            invoice_date="2024-12-21",
            po_reference="PO-H404-D",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-VD30",
                    description="Bulk Packaging Material Lot",
                    quantity_billed=1,
                    unit_price_billed=49950.00,
                    line_total_billed=49950.00,
                ),
            ],
            subtotal=49950.00,
            tax=3996.00,
            total_amount=53946.00,
            note="Final delivery for Q4 packaging contract.",
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-H404-D",
            supplier_id="SUP-H404",
            order_date="2024-12-05",
            line_items=[
                POLineItem(
                    item_code="ITEM-VD30",
                    description="Bulk Packaging Material Lot",
                    ordered_quantity=1,
                    unit_price_ordered=49950.00,
                    line_total_ordered=49950.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-H404-D",
            po_reference="PO-H404-D",
            receipt_date="2024-12-18",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-VD30",
                    quantity_received=1,
                    accepted_quantity=1,
                ),
            ],
            warehouse_note="Full lot received.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-H404",
            supplier_name="Vector Distribution Co",
            risk_tier="high",
            duplicate_risk_count=3,
            preferred_vendor=False,
            escalation_threshold=40000.00,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=300.0,
            high_value_threshold=50000.0,
            mandatory_escalation_above=100000.0,
        ),
        case_history=CaseHistory(
            prior_comments=[
                "ALERT: SUP-H404 submitted 3 invoices in past 14 days at "
                "$49,920 / $49,975 / $49,950 -- each just under $50k threshold.",
                "Combined invoiced value across 4 invoices: ~$199,795.",
                "PO numbering pattern (PO-H404-A/B/C/D) suggests deliberate split.",
                "Finance integrity team flagged for review.",
            ],
            similar_invoice_refs=["INV-2024-9401", "INV-2024-9402", "INV-2024-9403"],
            processed_invoice_numbers=["INV-2024-9401", "INV-2024-9402", "INV-2024-9403"],
        ),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.escalate_for_supervisor_review,
            correct_exception_type=ExceptionType.policy_violation,
            acceptable_evidence=[
                "inspect_purchase_order",
                "check_for_duplicate_invoice",
                "inspect_vendor_profile",
                "inspect_policy_rules",
            ],
            key_findings=[
                "Invoice $49,950 sits $50 under auto-approval threshold",
                "3 prior invoices from same vendor in past 14 days, all just under $50k",
                "Combined value ~$199,795 well above mandatory escalation",
                "High-risk vendor with 3 prior incidents",
                "Pattern consistent with deliberate threshold evasion",
            ],
        ),
    )


# -- H5: Retroactive PO Amendment ------------------------------------------
# Decision: escalate_for_supervisor_review | Exception: price_mismatch
# PO line items show $100/unit. Invoice billed $100/unit. compare_price
# reports a perfect match. But case_history reveals an amendment that
# retroactively reduced agreed price to $80/unit.

def _build_h5() -> CaseData:
    return CaseData(
        case_id="CASE-H5-001",
        task_id=TaskID.task_h5_retroactive_amendment,
        difficulty=Difficulty.hard,
        max_steps=HARD_MAX_STEPS,
        invoice=Invoice(
            invoice_number="INV-2024-9505",
            supplier_name="Solace Pharma Distribution",
            supplier_id="SUP-H505",
            invoice_date="2024-12-20",
            po_reference="PO-H505",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-SP40",
                    description="Diagnostic Reagent Kit Pro",
                    quantity_billed=300,
                    unit_price_billed=100.00,
                    line_total_billed=30000.00,
                ),
            ],
            subtotal=30000.00,
            tax=2400.00,
            total_amount=32400.00,
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-H505",
            supplier_id="SUP-H505",
            order_date="2024-11-05",
            line_items=[
                POLineItem(
                    item_code="ITEM-SP40",
                    description="Diagnostic Reagent Kit Pro",
                    ordered_quantity=300,
                    unit_price_ordered=100.00,
                    line_total_ordered=30000.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-H505",
            po_reference="PO-H505",
            receipt_date="2024-12-12",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-SP40",
                    quantity_received=300,
                    accepted_quantity=300,
                ),
            ],
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-H505",
            supplier_name="Solace Pharma Distribution",
            risk_tier="medium",
            preferred_vendor=True,
            escalation_threshold=25000.00,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=300.0,
        ),
        case_history=CaseHistory(
            prior_comments=[
                "PO-H505 Amendment 1 (2024-12-08): unit price retroactively reduced "
                "from $100 to $80 per kit as part of annual rebate true-up.",
                "Amendment is binding for all invoices dated after 2024-12-08.",
                "Vendor was notified in writing on 2024-12-09 and acknowledged.",
                "Effective PO total is now $24,000, not $30,000.",
            ],
            similar_invoice_refs=["AMENDMENT-PO-H505"],
            processed_invoice_numbers=[],
        ),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.escalate_for_supervisor_review,
            correct_exception_type=ExceptionType.price_mismatch,
            acceptable_evidence=[
                "inspect_purchase_order",
                "check_for_duplicate_invoice",
                "compare_price",
                "inspect_policy_rules",
            ],
            key_findings=[
                "PO line items show $100, invoice shows $100 -- looks aligned",
                "PO Amendment 1 retroactively set agreed price to $80",
                "Effective overcharge: $20 x 300 = $6,000",
                "Supervisor must reconcile invoice against amended PO",
            ],
        ),
    )


# -- H6: Returned-Then-Rebilled --------------------------------------------
# Decision: place_on_hold | Exception: quantity_mismatch
# GRN shows received=100, accepted=100. compare_quantity reports OK.
# But warehouse_note + case_history record that 30 units were returned
# post-receipt as defective. Effective net accepted is 70.

def _build_h6() -> CaseData:
    return CaseData(
        case_id="CASE-H6-001",
        task_id=TaskID.task_h6_returned_then_rebilled,
        difficulty=Difficulty.hard,
        max_steps=HARD_MAX_STEPS,
        invoice=Invoice(
            invoice_number="INV-2024-9606",
            supplier_name="Ironclad Hardware Co",
            supplier_id="SUP-H606",
            invoice_date="2024-12-22",
            po_reference="PO-H606",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-IH50",
                    description="Heavy-Duty Steel Coupling Set",
                    quantity_billed=100,
                    unit_price_billed=220.00,
                    line_total_billed=22000.00,
                ),
            ],
            subtotal=22000.00,
            tax=1760.00,
            total_amount=23760.00,
            note="Full shipment billing. Per delivery receipt.",
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-H606",
            supplier_id="SUP-H606",
            order_date="2024-11-15",
            line_items=[
                POLineItem(
                    item_code="ITEM-IH50",
                    description="Heavy-Duty Steel Coupling Set",
                    ordered_quantity=100,
                    unit_price_ordered=220.00,
                    line_total_ordered=22000.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-H606",
            po_reference="PO-H606",
            receipt_date="2024-12-05",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-IH50",
                    quantity_received=100,
                    accepted_quantity=100,
                ),
            ],
            warehouse_note=(
                "POST-RECEIPT NOTE (2024-12-15): 30 units returned to vendor "
                "due to QC failure (cracked welds). Net accepted in inventory: 70. "
                "Return Material Authorization RMA-H606-01 issued."
            ),
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-H606",
            supplier_name="Ironclad Hardware Co",
            risk_tier="medium",
            preferred_vendor=True,
            escalation_threshold=20000.00,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=200.0,
        ),
        case_history=CaseHistory(
            prior_comments=[
                "RMA-H606-01 (2024-12-15): 30 units of ITEM-IH50 returned for QC failure.",
                "Vendor acknowledged return and issued credit memo CM-H606-01.",
                "Net deliverable for this PO: 70 units at $220 = $15,400.",
                "Vendor invoiced full 100 units, ignoring credit memo.",
            ],
            similar_invoice_refs=["RMA-H606-01"],
            processed_invoice_numbers=[],
        ),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.place_on_hold,
            correct_exception_type=ExceptionType.quantity_mismatch,
            acceptable_evidence=[
                "inspect_goods_receipt_note",
                "check_for_duplicate_invoice",
                "compare_quantity",
            ],
            key_findings=[
                "GRN shows 100 accepted at receipt, but 30 were returned post-receipt",
                "Net inventory accepted is 70 units, not 100",
                "Vendor issued credit memo CM-H606-01 acknowledging return",
                "Invoice bills 100 units, ignoring the credit memo",
                "Hold required until vendor reissues with credit applied",
            ],
        ),
    )


# -- H7: Multi-Party Approval Pending --------------------------------------
# Decision: escalate_for_supervisor_review | Exception: policy_violation
# Invoice $85,000. Policy: high_value_threshold=50k requires supervisor;
# mandatory_escalation_above=80k requires CFO co-sign. CFO sign-off not on
# file. All line-level matches are clean. Trap: agent approves because
# everything matches.

def _build_h7() -> CaseData:
    return CaseData(
        case_id="CASE-H7-001",
        task_id=TaskID.task_h7_multi_party_pending,
        difficulty=Difficulty.hard,
        max_steps=HARD_MAX_STEPS,
        invoice=Invoice(
            invoice_number="INV-2024-9707",
            supplier_name="Skyline Capital Equipment",
            supplier_id="SUP-H707",
            invoice_date="2024-12-21",
            po_reference="PO-H707",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-SK60",
                    description="Industrial Compressor Unit Model X",
                    quantity_billed=1,
                    unit_price_billed=85000.00,
                    line_total_billed=85000.00,
                ),
            ],
            subtotal=85000.00,
            tax=6800.00,
            total_amount=91800.00,
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-H707",
            supplier_id="SUP-H707",
            order_date="2024-11-25",
            line_items=[
                POLineItem(
                    item_code="ITEM-SK60",
                    description="Industrial Compressor Unit Model X",
                    ordered_quantity=1,
                    unit_price_ordered=85000.00,
                    line_total_ordered=85000.00,
                ),
            ],
            payment_terms="Net 45",
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-H707",
            po_reference="PO-H707",
            receipt_date="2024-12-15",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-SK60",
                    quantity_received=1,
                    accepted_quantity=1,
                ),
            ],
            warehouse_note="Unit installed and commissioned successfully.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-H707",
            supplier_name="Skyline Capital Equipment",
            risk_tier="low",
            preferred_vendor=True,
            escalation_threshold=70000.00,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=500.0,
            high_value_threshold=50000.0,
            mandatory_escalation_above=80000.0,
        ),
        case_history=CaseHistory(
            prior_comments=[
                "Capex policy: invoices >= $80,000 require BOTH supervisor and CFO sign-off.",
                "PO-H707 has supervisor sign-off (S. Patel, 2024-11-25).",
                "CFO sign-off NOT on file; capex committee meets 2024-12-28.",
                "Payment must wait for CFO approval per controls policy.",
            ],
            similar_invoice_refs=["CAPEX-POLICY-2024"],
            processed_invoice_numbers=[],
        ),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.escalate_for_supervisor_review,
            correct_exception_type=ExceptionType.policy_violation,
            acceptable_evidence=[
                "inspect_purchase_order",
                "inspect_policy_rules",
                "check_for_duplicate_invoice",
                "inspect_vendor_profile",
            ],
            key_findings=[
                "Invoice total $91,800 exceeds mandatory escalation threshold $80,000",
                "Capex policy requires both supervisor AND CFO sign-off above $80k",
                "Only supervisor sign-off currently on file",
                "Payment cannot be released until CFO approval logged",
            ],
        ),
    )


# -- H8: Rush Premium Authorized -------------------------------------------
# Decision: approve_for_payment | Exception: clean_match
# Invoice price 8% over PO -- normally a price_mismatch escalation.
# But invoice.note + warehouse_note + case_history all confirm a buyer-
# approved 8% rush surcharge per a PO amendment. Trap: compare_price flags
# discrepancy and agent escalates.

def _build_h8() -> CaseData:
    return CaseData(
        case_id="CASE-H8-001",
        task_id=TaskID.task_h8_rush_premium_authorized,
        difficulty=Difficulty.hard,
        max_steps=HARD_MAX_STEPS,
        invoice=Invoice(
            invoice_number="INV-2024-9808",
            supplier_name="Falcon Express Components",
            supplier_id="SUP-H808",
            invoice_date="2024-12-20",
            po_reference="PO-H808",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-FE70",
                    description="Aviation-Grade Fastener Pack",
                    quantity_billed=200,
                    unit_price_billed=54.00,
                    line_total_billed=10800.00,
                ),
            ],
            subtotal=10800.00,
            tax=864.00,
            total_amount=11664.00,
            note=(
                "Includes 8% rush surcharge per PO-H808 Amendment 1 dated 2024-12-05 "
                "(buyer-approved expedited shipping). Base price unchanged at $50.00."
            ),
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-H808",
            supplier_id="SUP-H808",
            order_date="2024-12-01",
            line_items=[
                POLineItem(
                    item_code="ITEM-FE70",
                    description="Aviation-Grade Fastener Pack",
                    ordered_quantity=200,
                    unit_price_ordered=50.00,
                    line_total_ordered=10000.00,
                ),
            ],
            payment_terms="Net 30",
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-H808",
            po_reference="PO-H808",
            receipt_date="2024-12-12",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-FE70",
                    quantity_received=200,
                    accepted_quantity=200,
                ),
            ],
            warehouse_note=(
                "Expedited delivery received per buyer's rush request. "
                "8% surcharge authorized in PO-H808 Amendment 1."
            ),
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-H808",
            supplier_name="Falcon Express Components",
            risk_tier="low",
            preferred_vendor=True,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=200.0,
        ),
        case_history=CaseHistory(
            prior_comments=[
                "PO-H808 Amendment 1 (2024-12-05): buyer authorized up to 10% rush "
                "surcharge for expedited delivery on this order.",
                "Expedited shipping was specifically requested due to production line outage.",
                "Surcharge of 8% is within authorized range.",
            ],
            similar_invoice_refs=["PO-H808-AMD1"],
            processed_invoice_numbers=[],
        ),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.approve_for_payment,
            correct_exception_type=ExceptionType.clean_match,
            acceptable_evidence=[
                "inspect_invoice_line_items",
                "inspect_purchase_order",
                "inspect_goods_receipt_note",
                "check_for_duplicate_invoice",
            ],
            key_findings=[
                "Unit price $54 vs base PO $50 = 8% above PO line price",
                "PO Amendment 1 explicitly authorized up to 10% rush surcharge",
                "Surcharge applied is within authorized range",
                "Quantities and delivery match exactly",
                "Surface price variance is contractually authorized",
            ],
        ),
    )


# -- H9: Supplier ID Mismatch ----------------------------------------------
# Decision: reject_invoice | Exception: policy_violation
# Invoice supplier_id is similar but NOT identical to PO supplier_id.
# Vendor profile (looked up by invoice's id) shows it's a different entity
# with high-risk fraud history. Names look similar -- a phishing-style
# attempt to redirect payment.

def _build_h9() -> CaseData:
    return CaseData(
        case_id="CASE-H9-001",
        task_id=TaskID.task_h9_supplier_id_mismatch,
        difficulty=Difficulty.hard,
        max_steps=HARD_MAX_STEPS,
        invoice=Invoice(
            invoice_number="INV-2024-9909",
            supplier_name="Acme Industrial Solutions",
            supplier_id="SUP-H909-X",
            invoice_date="2024-12-22",
            po_reference="PO-H909",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-AI80",
                    description="Industrial Pump Assembly",
                    quantity_billed=10,
                    unit_price_billed=2200.00,
                    line_total_billed=22000.00,
                ),
            ],
            subtotal=22000.00,
            tax=1760.00,
            total_amount=23760.00,
            note="Please remit to updated bank account. Supplier ID corrected per recent re-registration.",
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-H909",
            supplier_id="SUP-H909",
            order_date="2024-11-20",
            line_items=[
                POLineItem(
                    item_code="ITEM-AI80",
                    description="Industrial Pump Assembly",
                    ordered_quantity=10,
                    unit_price_ordered=2200.00,
                    line_total_ordered=22000.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-H909",
            po_reference="PO-H909",
            receipt_date="2024-12-12",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-AI80",
                    quantity_received=10,
                    accepted_quantity=10,
                ),
            ],
            warehouse_note="Goods received from original vendor (SUP-H909).",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-H909-X",
            supplier_name="Acme Industrial Solutions",
            risk_tier="high",
            duplicate_risk_count=4,
            preferred_vendor=False,
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
                "FRAUD ALERT: SUP-H909-X is a flagged shell supplier under investigation.",
                "Original PO-H909 was issued to legitimate supplier SUP-H909.",
                "Goods were received from SUP-H909 (legitimate). Invoice arrived from SUP-H909-X.",
                "Bank-redirect attempts have been observed for this supplier family in past 90 days.",
                "Reject and refer to fraud team. Do NOT pay SUP-H909-X.",
            ],
            similar_invoice_refs=["FRAUD-SUP-H909-X"],
            processed_invoice_numbers=[],
        ),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.reject_invoice,
            correct_exception_type=ExceptionType.policy_violation,
            acceptable_evidence=[
                "inspect_invoice_line_items",
                "inspect_purchase_order",
                "inspect_vendor_profile",
                "check_for_duplicate_invoice",
            ],
            key_findings=[
                "Invoice supplier_id SUP-H909-X does not match PO supplier_id SUP-H909",
                "Vendor profile flags SUP-H909-X as high-risk shell supplier under fraud investigation",
                "Goods were physically delivered by legitimate SUP-H909 (per GRN note)",
                "Invoice attempts to redirect payment to fraud-flagged entity",
                "Mandatory rejection and referral to fraud team",
            ],
        ),
    )


# -- H10: Cross-Case Contradiction -----------------------------------------
# Decision: escalate_for_supervisor_review | Exception: mixed_discrepancy
# Quantity discrepancy (partial receipt -> hold signal) AND duplicate
# suspicion (-> reject signal). Both contradict each other; per policy
# escalation supersedes both individual signals. Tests reconciliation under
# conflicting evidence.

def _build_h10() -> CaseData:
    return CaseData(
        case_id="CASE-H10-001",
        task_id=TaskID.task_h10_cross_case_contradiction,
        difficulty=Difficulty.hard,
        max_steps=HARD_MAX_STEPS,
        invoice=Invoice(
            invoice_number="INV-2024-9A10",
            supplier_name="Ridgepoint Industrial Corp",
            supplier_id="SUP-HA10",
            invoice_date="2024-12-22",
            po_reference="PO-HA10",
            line_items=[
                InvoiceLineItem(
                    item_code="ITEM-RP90",
                    description="Composite Reinforcement Panel",
                    quantity_billed=100,
                    unit_price_billed=160.00,
                    line_total_billed=16000.00,
                ),
            ],
            subtotal=16000.00,
            tax=1280.00,
            total_amount=17280.00,
            note="Final delivery for Q4 build-out. Please process per terms.",
        ),
        purchase_order=PurchaseOrder(
            po_number="PO-HA10",
            supplier_id="SUP-HA10",
            order_date="2024-11-18",
            line_items=[
                POLineItem(
                    item_code="ITEM-RP90",
                    description="Composite Reinforcement Panel",
                    ordered_quantity=100,
                    unit_price_ordered=160.00,
                    line_total_ordered=16000.00,
                ),
            ],
        ),
        goods_receipt_note=GoodsReceiptNote(
            grn_number="GRN-HA10",
            po_reference="PO-HA10",
            receipt_date="2024-12-10",
            line_items=[
                GRNLineItem(
                    item_code="ITEM-RP90",
                    quantity_received=80,
                    accepted_quantity=80,
                    rejected_quantity=0,
                ),
            ],
            warehouse_note="Partial shipment: 20 units short. Backorder pending.",
        ),
        vendor_profile=VendorProfile(
            supplier_id="SUP-HA10",
            supplier_name="Ridgepoint Industrial Corp",
            risk_tier="high",
            duplicate_risk_count=3,
            preferred_vendor=False,
            escalation_threshold=12000.00,
        ),
        company_policy=CompanyPolicy(
            quantity_tolerance_pct=5.0,
            price_tolerance_pct=5.0,
            total_tolerance_amt=200.0,
            duplicate_check_enabled=True,
        ),
        case_history=CaseHistory(
            prior_comments=[
                "INV-2024-9A05 (similar amount, same PO family) processed 2024-12-08.",
                "Vendor flagged for prior duplicate-resubmission incidents.",
                "Note: signals point in conflicting directions -- partial receipt suggests "
                "hold, but duplicate suspicion suggests reject. Policy escalation hierarchy: "
                "ESCALATE supersedes HOLD and REJECT when conflicting signals coexist.",
            ],
            similar_invoice_refs=["INV-2024-9A05"],
            processed_invoice_numbers=["INV-2024-9A05"],
        ),
        ground_truth=GroundTruth(
            correct_decision=DecisionType.escalate_for_supervisor_review,
            correct_exception_type=ExceptionType.mixed_discrepancy,
            acceptable_evidence=[
                "inspect_purchase_order",
                "inspect_goods_receipt_note",
                "check_for_duplicate_invoice",
                "compare_quantity",
                "inspect_policy_rules",
                "inspect_vendor_profile",
            ],
            key_findings=[
                "Quantity discrepancy: billed 100 vs received 80 (20% short)",
                "Possible duplicate: similar invoice INV-2024-9A05 already processed",
                "Vendor is high-risk with 3 prior incidents",
                "Conflicting signals: hold (partial) vs reject (duplicate)",
                "Per policy hierarchy, escalation supersedes both",
            ],
        ),
    )


# -- Dispatch table --------------------------------------------------------

HARD_TASK_LIST = [
    TaskID.task_h1_phantom_grn_period,
    TaskID.task_h2_kickback_inflated_po,
    TaskID.task_h3_currency_silent_swap,
    TaskID.task_h4_threshold_dance,
    TaskID.task_h5_retroactive_amendment,
    TaskID.task_h6_returned_then_rebilled,
    TaskID.task_h7_multi_party_pending,
    TaskID.task_h8_rush_premium_authorized,
    TaskID.task_h9_supplier_id_mismatch,
    TaskID.task_h10_cross_case_contradiction,
]


HARD_TASK_BUILDERS: Dict[TaskID, callable] = {
    TaskID.task_h1_phantom_grn_period: _build_h1,
    TaskID.task_h2_kickback_inflated_po: _build_h2,
    TaskID.task_h3_currency_silent_swap: _build_h3,
    TaskID.task_h4_threshold_dance: _build_h4,
    TaskID.task_h5_retroactive_amendment: _build_h5,
    TaskID.task_h6_returned_then_rebilled: _build_h6,
    TaskID.task_h7_multi_party_pending: _build_h7,
    TaskID.task_h8_rush_premium_authorized: _build_h8,
    TaskID.task_h9_supplier_id_mismatch: _build_h9,
    TaskID.task_h10_cross_case_contradiction: _build_h10,
}
