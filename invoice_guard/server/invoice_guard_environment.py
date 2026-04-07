"""
InvoiceGuard Environment -- Three-Way Invoice Matching Exception Resolution.

An agent reviews one supplier invoice case per episode by comparing the
invoice, purchase order, and goods receipt note, then resolves the case.
"""

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        ActionType, CaseData, DecisionType, Difficulty, ExceptionType,
        InvoiceGuardAction, InvoiceGuardObservation, InvoiceGuardState, TaskID,
    )
    from ..tasks import get_task_case, TASK_LIST
    from ..graders import grade_episode
except ImportError:
    from models import (
        ActionType, CaseData, DecisionType, Difficulty, ExceptionType,
        InvoiceGuardAction, InvoiceGuardObservation, InvoiceGuardState, TaskID,
    )
    from tasks import get_task_case, TASK_LIST
    from graders import grade_episode


GOAL_TEXT = (
    "You are an accounts payable analyst. Review this supplier invoice by "
    "comparing it against the purchase order (PO) and goods receipt note (GRN).\n"
    "\n"
    "INVESTIGATION ACTIONS (reveal information):\n"
    "  inspect_purchase_order -- see what was ordered and at what price\n"
    "  inspect_goods_receipt_note -- see what was received at the warehouse\n"
    "  inspect_invoice_line_items -- see detailed invoice line items\n"
    "  inspect_vendor_profile -- see vendor risk tier, duplicate history, escalation thresholds\n"
    "  inspect_policy_rules -- see company tolerance thresholds and escalation rules\n"
    "  check_for_duplicate_invoice -- search for previously processed invoices\n"
    "  compare_quantity -- compare billed vs ordered vs received quantities\n"
    "  compare_price -- compare billed price vs PO-agreed price\n"
    "  compare_totals -- verify subtotal and total consistency\n"
    "  summarize_findings -- list all findings collected so far\n"
    "  propose_exception_type -- declare what type of exception you suspect\n"
    "\n"
    "RESOLUTION ACTION (ends the episode):\n"
    "  submit_final_resolution -- requires: final_decision, exception_type, "
    "evidence_references, explanation\n"
    "\n"
    "DECISIONS:\n"
    "  reject_invoice -- duplicate invoice or fraudulent submission detected\n"
    "  escalate_for_supervisor_review -- price/total variance exceeds tolerance, "
    "or invoice exceeds high-value threshold\n"
    "  place_on_hold -- billed quantity exceeds received quantity\n"
    "  approve_for_payment -- all documents match within tolerance\n"
    "\n"
    "Investigate thoroughly, then submit your resolution with evidence."
)

ALL_ACTIONS = [a.value for a in ActionType]

# Reward constants
R_NEW_DOCUMENT = 0.05
R_USEFUL_COMPARISON = 0.08
R_CONFIRM_NO_ISSUE = 0.03
R_CORRECT_PROPOSAL = 0.10
R_SUMMARIZE = 0.02
R_REPEAT_PENALTY = -0.02
R_INVALID_ACTION = -0.05
R_TIMEOUT_PENALTY = -0.10


class InvoiceGuardEnvironment(Environment):
    """
    Three-Way Invoice Matching Exception Resolution Environment.

    Each episode presents one synthetic invoice case. The agent investigates
    by inspecting documents and comparing fields, then submits a resolution.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._case: Optional[CaseData] = None
        self._env_state = InvoiceGuardState()
        self._task_index = 0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> InvoiceGuardObservation:
        task_id_str = kwargs.get("task_id")
        if task_id_str:
            task_id = TaskID(task_id_str)
        else:
            task_id = TASK_LIST[self._task_index % len(TASK_LIST)]
            self._task_index += 1

        self._case = get_task_case(task_id)
        c = self._case

        self._env_state = InvoiceGuardState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=c.task_id.value,
            difficulty=c.difficulty.value,
            case_id=c.case_id,
            max_steps=c.max_steps,
        )

        inv = c.invoice
        summary = (
            f"Supplier: {inv.supplier_name} ({inv.supplier_id}), "
            f"Invoice: {inv.invoice_number}, Date: {inv.invoice_date}, "
            f"PO Ref: {inv.po_reference}, Total: ${inv.total_amount:,.2f}"
        )
        if inv.note:
            summary += f"\nInvoice note: {inv.note}"

        initial_suggestions = [
            "inspect_purchase_order",
            "inspect_goods_receipt_note",
            "inspect_invoice_line_items",
        ]

        return InvoiceGuardObservation(
            case_id=c.case_id,
            task_id=c.task_id.value,
            difficulty=c.difficulty.value,
            invoice_summary=summary,
            goal=GOAL_TEXT,
            available_actions=ALL_ACTIONS,
            suggested_next_actions=initial_suggestions,
            revealed_documents=[],
            findings=[],
            remaining_steps=c.max_steps,
            last_action_result="Case loaded. Begin your investigation.",
            last_action_error=False,
            warnings=[],
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: InvoiceGuardAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> InvoiceGuardObservation:
        if self._case is None:
            return self._error_obs("Environment not initialized. Call reset() first.")

        s = self._env_state
        c = self._case

        if s.is_finalized:
            return self._error_obs("Episode already finalized.")

        s.step_count += 1
        action_name = action.action_type.value
        s.actions_taken.append(action_name)

        s.repeated_action_counts[action_name] = (
            s.repeated_action_counts.get(action_name, 0) + 1
        )

        # Check step budget
        remaining = c.max_steps - s.step_count
        if remaining < 0:
            return self._finalize_timeout()

        # Route action
        reward = 0.0
        is_repeat = s.repeated_action_counts[action_name] > 1
        terminal = False

        if action_name == ActionType.submit_final_resolution.value:
            return self._handle_submit(action, remaining)
        elif action_name == ActionType.propose_exception_type.value:
            result, reward = self._handle_propose_exception(action, is_repeat)
        elif action_name in _INVESTIGATION_HANDLERS:
            result, reward = _INVESTIGATION_HANDLERS[action_name](self, is_repeat)
        else:
            result = f"Unknown action: {action_name}"
            reward = R_INVALID_ACTION

        if is_repeat and action_name != ActionType.submit_final_resolution.value:
            reward = min(reward, R_REPEAT_PENALTY)

        s.cumulative_reward += reward
        remaining = c.max_steps - s.step_count

        if remaining <= 0 and not s.is_finalized:
            return self._finalize_timeout()

        return InvoiceGuardObservation(
            case_id=c.case_id,
            task_id=c.task_id.value,
            difficulty=c.difficulty.value,
            invoice_summary=self._invoice_summary(),
            goal=GOAL_TEXT,
            available_actions=ALL_ACTIONS,
            suggested_next_actions=self._suggest_next_actions(remaining),
            revealed_documents=list(s.documents_revealed),
            findings=list(s.findings_collected),
            remaining_steps=remaining,
            last_action_result=result,
            last_action_error=False,
            warnings=self._build_warnings(remaining),
            done=False,
            reward=reward,
        )

    @property
    def state(self) -> InvoiceGuardState:
        return self._env_state

    # -- Terminal handlers ------------------------------------------------

    def _handle_submit(
        self, action: InvoiceGuardAction, remaining: int
    ) -> InvoiceGuardObservation:
        s = self._env_state
        c = self._case

        if not action.final_decision:
            s.cumulative_reward += R_INVALID_ACTION
            error_msg = (
                "ERROR: submit_final_resolution requires all of these fields:\n"
                '  "final_decision": one of "approve_for_payment", '
                '"place_on_hold", "reject_invoice", "escalate_for_supervisor_review"\n'
                '  "exception_type": one of "clean_match", "quantity_mismatch", '
                '"price_mismatch", "partial_receipt", "duplicate_invoice", '
                '"total_amount_mismatch", "missing_receipt", "tax_variance", '
                '"policy_violation", "mixed_discrepancy"\n'
                '  "evidence_references": list of action names you performed\n'
                '  "explanation": one-sentence justification\n'
                "Example: {\"action_type\": \"submit_final_resolution\", "
                "\"final_decision\": \"approve_for_payment\", "
                "\"exception_type\": \"clean_match\", "
                "\"evidence_references\": [\"inspect_purchase_order\", "
                "\"compare_quantity\"], "
                "\"explanation\": \"All documents match within tolerance.\"}"
            )
            return InvoiceGuardObservation(
                case_id=c.case_id,
                task_id=c.task_id.value,
                difficulty=c.difficulty.value,
                invoice_summary=self._invoice_summary(),
                goal=GOAL_TEXT,
                available_actions=ALL_ACTIONS,
                revealed_documents=list(s.documents_revealed),
                findings=list(s.findings_collected),
                remaining_steps=remaining,
                last_action_result=error_msg,
                last_action_error=True,
                warnings=self._build_warnings(remaining),
                done=False,
                reward=R_INVALID_ACTION,
            )

        s.is_finalized = True
        s.final_decision = action.final_decision.value
        s.final_exception_type = (
            action.exception_type.value if action.exception_type else None
        )
        s.final_evidence = list(action.evidence_references)
        s.final_explanation = action.explanation
        s.final_confidence = action.confidence

        grader_result = grade_episode(c, s)

        confidence_bonus = 0.0
        if action.confidence is not None:
            is_correct = (s.final_decision == c.ground_truth.correct_decision.value)
            if is_correct and action.confidence >= 0.8:
                confidence_bonus = 0.05
            elif not is_correct and action.confidence >= 0.8:
                confidence_bonus = -0.05

        s.cumulative_reward += grader_result.score + confidence_bonus

        return InvoiceGuardObservation(
            case_id=c.case_id,
            task_id=c.task_id.value,
            difficulty=c.difficulty.value,
            invoice_summary=self._invoice_summary(),
            goal=GOAL_TEXT,
            available_actions=[],
            revealed_documents=list(s.documents_revealed),
            findings=list(s.findings_collected),
            remaining_steps=0,
            last_action_result=(
                f"Case resolved. Decision: {s.final_decision}. "
                f"Grader score: {grader_result.score:.4f}"
            ),
            last_action_error=False,
            warnings=[],
            done=True,
            reward=grader_result.score,
            grader_result=grader_result.model_dump(),
            metadata={
                "grader_result": grader_result.model_dump(),
                "cumulative_reward": s.cumulative_reward,
            },
        )

    def _finalize_timeout(self) -> InvoiceGuardObservation:
        s = self._env_state
        c = self._case
        s.is_finalized = True

        grader_result = grade_episode(c, s)
        s.cumulative_reward += grader_result.score + R_TIMEOUT_PENALTY

        return InvoiceGuardObservation(
            case_id=c.case_id,
            task_id=c.task_id.value,
            difficulty=c.difficulty.value,
            invoice_summary=self._invoice_summary(),
            goal=GOAL_TEXT,
            available_actions=[],
            revealed_documents=list(s.documents_revealed),
            findings=list(s.findings_collected),
            remaining_steps=0,
            last_action_result="Step budget exhausted. Episode ended without resolution.",
            last_action_error=True,
            warnings=[],
            done=True,
            reward=max(grader_result.score + R_TIMEOUT_PENALTY, 0.0),
            grader_result=grader_result.model_dump(),
            metadata={
                "grader_result": grader_result.model_dump(),
                "cumulative_reward": s.cumulative_reward,
                "timeout": True,
            },
        )

    def _handle_propose_exception(
        self, action: InvoiceGuardAction, is_repeat: bool
    ) -> tuple:
        s = self._env_state
        c = self._case
        if not action.exception_type:
            return "ERROR: propose_exception_type requires exception_type.", R_INVALID_ACTION

        s.proposed_exception = action.exception_type.value
        correct = c.ground_truth.correct_exception_type.value

        if s.proposed_exception == correct:
            finding = f"Proposed exception type: {s.proposed_exception} (noted)."
            return finding, R_CORRECT_PROPOSAL if not is_repeat else R_REPEAT_PENALTY
        else:
            finding = f"Proposed exception type: {s.proposed_exception} (noted)."
            return finding, 0.0

    # -- Investigation handlers -------------------------------------------

    def _inspect_invoice_line_items(self, is_repeat: bool) -> tuple:
        c = self._case
        s = self._env_state
        doc = "invoice_line_items"

        lines = []
        for li in c.invoice.line_items:
            lines.append(
                f"  {li.item_code}: {li.description} | "
                f"Qty: {li.quantity_billed} | "
                f"Unit Price: ${li.unit_price_billed:,.2f} | "
                f"Line Total: ${li.line_total_billed:,.2f}"
            )
        detail = (
            f"Invoice {c.invoice.invoice_number} line items:\n"
            + "\n".join(lines)
            + f"\nSubtotal: ${c.invoice.subtotal:,.2f} | "
            f"Tax: ${c.invoice.tax:,.2f} | "
            f"Total: ${c.invoice.total_amount:,.2f}"
        )

        if c.invoice.note:
            detail += f"\nInvoice note: {c.invoice.note}"

        reward = self._reveal_doc(doc, detail)
        return detail, reward

    def _inspect_purchase_order(self, is_repeat: bool) -> tuple:
        c = self._case
        s = self._env_state
        doc = "purchase_order"

        lines = []
        for li in c.purchase_order.line_items:
            lines.append(
                f"  {li.item_code}: {li.description} | "
                f"Ordered Qty: {li.ordered_quantity} | "
                f"Unit Price: ${li.unit_price_ordered:,.2f} | "
                f"Line Total: ${li.line_total_ordered:,.2f}"
            )
        detail = (
            f"PO {c.purchase_order.po_number} "
            f"(Supplier: {c.purchase_order.supplier_id}, "
            f"Date: {c.purchase_order.order_date}):\n"
            + "\n".join(lines)
            + f"\nApproved: {c.purchase_order.approved} | "
            f"Payment Terms: {c.purchase_order.payment_terms}"
        )

        reward = self._reveal_doc(doc, detail)
        return detail, reward

    def _inspect_goods_receipt_note(self, is_repeat: bool) -> tuple:
        c = self._case
        doc = "goods_receipt_note"

        lines = []
        for li in c.goods_receipt_note.line_items:
            lines.append(
                f"  {li.item_code}: "
                f"Received: {li.quantity_received} | "
                f"Accepted: {li.accepted_quantity} | "
                f"Rejected: {li.rejected_quantity}"
            )
        detail = (
            f"GRN {c.goods_receipt_note.grn_number} "
            f"(PO Ref: {c.goods_receipt_note.po_reference}, "
            f"Date: {c.goods_receipt_note.receipt_date}):\n"
            + "\n".join(lines)
        )
        if c.goods_receipt_note.warehouse_note:
            detail += f"\nWarehouse note: {c.goods_receipt_note.warehouse_note}"

        reward = self._reveal_doc(doc, detail)
        return detail, reward

    def _inspect_vendor_profile(self, is_repeat: bool) -> tuple:
        c = self._case
        doc = "vendor_profile"
        vp = c.vendor_profile

        detail = (
            f"Vendor Profile -- {vp.supplier_name} ({vp.supplier_id}):\n"
            f"  Risk Tier: {vp.risk_tier}\n"
            f"  Preferred Vendor: {vp.preferred_vendor}\n"
            f"  Duplicate Risk History: {vp.duplicate_risk_count} incidents"
        )
        if vp.tolerance_override is not None:
            detail += f"\n  Tolerance Override: {vp.tolerance_override}%"
        if vp.escalation_threshold is not None:
            detail += f"\n  Escalation Threshold: ${vp.escalation_threshold:,.2f}"
            inv_total = c.invoice.total_amount
            if inv_total > vp.escalation_threshold:
                detail += (
                    f"\n  NOTE: Invoice total ${inv_total:,.2f} exceeds vendor "
                    f"escalation threshold ${vp.escalation_threshold:,.2f}."
                )

        reward = self._reveal_doc(doc, detail)
        return detail, reward

    def _inspect_policy_rules(self, is_repeat: bool) -> tuple:
        c = self._case
        doc = "policy_rules"
        p = c.company_policy

        detail = (
            f"Company Matching Policy:\n"
            f"  Quantity Tolerance: {p.quantity_tolerance_pct}%\n"
            f"  Price Tolerance: {p.price_tolerance_pct}%\n"
            f"  Total Amount Tolerance: ${p.total_tolerance_amt:,.2f}\n"
            f"  Duplicate Check: {'Enabled' if p.duplicate_check_enabled else 'Disabled'}\n"
            f"  High-Value Review Threshold: ${p.high_value_threshold:,.2f}\n"
            f"  Mandatory Escalation Above: ${p.mandatory_escalation_above:,.2f}\n"
            f"\n"
            f"  Resolution Rules:\n"
            f"    - APPROVE: All matches within tolerance, no duplicates, no policy violations.\n"
            f"    - HOLD: Billed quantity exceeds received quantity (partial/missing receipt).\n"
            f"    - ESCALATE: Price or total variance exceeds tolerance. Invoice above high-value threshold.\n"
            f"    - REJECT: Duplicate invoice detected. Fraudulent or invalid submission."
        )

        inv_total = c.invoice.total_amount
        if inv_total >= p.high_value_threshold:
            detail += (
                f"\n\n  NOTE: Invoice total ${inv_total:,.2f} exceeds "
                f"high-value threshold ${p.high_value_threshold:,.2f}. "
                f"Supervisor review required."
            )
            self._add_finding(
                f"High-value invoice: ${inv_total:,.2f} exceeds "
                f"review threshold ${p.high_value_threshold:,.2f}."
            )

        reward = self._reveal_doc(doc, detail)
        return detail, reward

    def _check_for_duplicate_invoice(self, is_repeat: bool) -> tuple:
        c = self._case
        doc = "duplicate_check"
        ch = c.case_history

        if ch.processed_invoice_numbers or ch.similar_invoice_refs:
            refs = ch.similar_invoice_refs or ch.processed_invoice_numbers
            detail = (
                f"Duplicate check for {c.invoice.invoice_number}:\n"
                f"  Similar/processed invoices found: {', '.join(refs)}\n"
                f"  Prior comments: {'; '.join(ch.prior_comments) if ch.prior_comments else 'None'}\n"
                f"  Pending invoices: {', '.join(ch.pending_invoice_numbers) if ch.pending_invoice_numbers else 'None'}"
            )
            finding = (
                f"ALERT: Potential duplicate detected. "
                f"Previously processed invoices: {', '.join(refs)}."
            )
            self._add_finding(finding)
            reward = self._reveal_doc(doc, detail)
            return detail, max(reward, R_USEFUL_COMPARISON)
        else:
            detail = (
                f"Duplicate check for {c.invoice.invoice_number}: "
                f"No similar or duplicate invoices found in case history."
            )
            reward = self._reveal_doc(doc, detail)
            return detail, reward

    def _compare_quantity(self, is_repeat: bool) -> tuple:
        c = self._case
        results = []
        has_discrepancy = False

        for inv_li in c.invoice.line_items:
            po_qty = None
            grn_qty = None
            for po_li in c.purchase_order.line_items:
                if po_li.item_code == inv_li.item_code:
                    po_qty = po_li.ordered_quantity
                    break
            for grn_li in c.goods_receipt_note.line_items:
                if grn_li.item_code == inv_li.item_code:
                    grn_qty = grn_li.accepted_quantity
                    break

            line = (
                f"  {inv_li.item_code}: "
                f"Billed={inv_li.quantity_billed}, "
                f"Ordered={po_qty}, "
                f"Received={grn_qty}"
            )

            if grn_qty is not None and inv_li.quantity_billed > grn_qty:
                diff = inv_li.quantity_billed - grn_qty
                pct = (diff / inv_li.quantity_billed) * 100
                line += f" -> DISCREPANCY: Billed exceeds received by {diff:.0f} units ({pct:.1f}%)"
                has_discrepancy = True
                self._add_finding(
                    f"Quantity discrepancy on {inv_li.item_code}: "
                    f"billed {inv_li.quantity_billed} but only {grn_qty} received "
                    f"(difference: {diff:.0f} units, {pct:.1f}%)."
                )
            elif po_qty is not None and inv_li.quantity_billed > po_qty:
                diff = inv_li.quantity_billed - po_qty
                line += f" -> DISCREPANCY: Billed exceeds ordered by {diff:.0f} units"
                has_discrepancy = True
                self._add_finding(
                    f"Quantity discrepancy on {inv_li.item_code}: "
                    f"billed {inv_li.quantity_billed} but only {po_qty} ordered."
                )
            else:
                line += " -> OK"

            results.append(line)

        detail = "Quantity comparison:\n" + "\n".join(results)
        reward = R_USEFUL_COMPARISON if has_discrepancy else R_CONFIRM_NO_ISSUE
        return detail, reward

    def _compare_price(self, is_repeat: bool) -> tuple:
        c = self._case
        results = []
        has_discrepancy = False
        tolerance = c.company_policy.price_tolerance_pct

        for inv_li in c.invoice.line_items:
            po_price = None
            for po_li in c.purchase_order.line_items:
                if po_li.item_code == inv_li.item_code:
                    po_price = po_li.unit_price_ordered
                    break

            if po_price is not None:
                line = (
                    f"  {inv_li.item_code}: "
                    f"Billed=${inv_li.unit_price_billed:,.2f}, "
                    f"PO Price=${po_price:,.2f}"
                )
            else:
                line = (
                    f"  {inv_li.item_code}: "
                    f"Billed=${inv_li.unit_price_billed:,.2f}, "
                    f"PO Price=N/A"
                )

            if po_price is not None and po_price > 0:
                variance_pct = (
                    (inv_li.unit_price_billed - po_price) / po_price
                ) * 100
                if abs(variance_pct) > tolerance:
                    line += (
                        f" -> DISCREPANCY: {variance_pct:+.1f}% variance "
                        f"(exceeds {tolerance}% tolerance)"
                    )
                    has_discrepancy = True
                    self._add_finding(
                        f"Price discrepancy on {inv_li.item_code}: "
                        f"billed ${inv_li.unit_price_billed:,.2f} vs "
                        f"PO ${po_price:,.2f} "
                        f"({variance_pct:+.1f}% variance, "
                        f"tolerance is {tolerance}%)."
                    )
                    self._add_finding(
                        f"POLICY: Price variance exceeding {tolerance}% tolerance "
                        f"requires escalation for supervisor review."
                    )
                else:
                    line += f" -> Within tolerance ({variance_pct:+.1f}%)"

            results.append(line)

        detail = "Price comparison:\n" + "\n".join(results)
        reward = R_USEFUL_COMPARISON if has_discrepancy else R_CONFIRM_NO_ISSUE
        return detail, reward

    def _compare_totals(self, is_repeat: bool) -> tuple:
        c = self._case
        inv = c.invoice

        expected_subtotal = sum(li.line_total_billed for li in inv.line_items)
        po_total = sum(li.line_total_ordered for li in c.purchase_order.line_items)
        tolerance_amt = c.company_policy.total_tolerance_amt

        subtotal_ok = abs(inv.subtotal - expected_subtotal) < 0.01
        total_diff = inv.subtotal - po_total
        within_tolerance = abs(total_diff) <= tolerance_amt

        lines = [
            f"  Invoice subtotal: ${inv.subtotal:,.2f} (line items sum: ${expected_subtotal:,.2f}) "
            f"-> {'Consistent' if subtotal_ok else 'INCONSISTENT'}",
            f"  Invoice subtotal vs PO total: ${inv.subtotal:,.2f} vs ${po_total:,.2f} "
            f"(diff: ${total_diff:+,.2f}) -> "
            f"{'Within' if within_tolerance else 'EXCEEDS'} tolerance (${tolerance_amt:,.2f})",
            f"  Tax: ${inv.tax:,.2f} | Grand Total: ${inv.total_amount:,.2f}",
        ]

        has_discrepancy = not subtotal_ok or not within_tolerance
        if has_discrepancy:
            if not subtotal_ok:
                self._add_finding(
                    f"Total inconsistency: invoice subtotal ${inv.subtotal:,.2f} "
                    f"does not match line item sum ${expected_subtotal:,.2f}."
                )
            if not within_tolerance:
                self._add_finding(
                    f"Total discrepancy: invoice subtotal ${inv.subtotal:,.2f} vs "
                    f"PO total ${po_total:,.2f} (difference ${total_diff:+,.2f} "
                    f"exceeds tolerance ${tolerance_amt:,.2f})."
                )
                self._add_finding(
                    "POLICY: Total amount variance exceeding tolerance "
                    "requires escalation for supervisor review."
                )

        detail = "Totals comparison:\n" + "\n".join(lines)
        reward = R_USEFUL_COMPARISON if has_discrepancy else R_CONFIRM_NO_ISSUE
        return detail, reward

    def _summarize_findings(self, is_repeat: bool) -> tuple:
        s = self._env_state
        if not s.findings_collected:
            detail = "No findings collected yet. Investigate further before summarizing."
            return detail, 0.0

        summary_lines = [f"  {i+1}. {f}" for i, f in enumerate(s.findings_collected)]
        detail = (
            f"Summary of findings ({len(s.findings_collected)} items):\n"
            + "\n".join(summary_lines)
            + f"\nDocuments reviewed: {', '.join(s.documents_revealed) if s.documents_revealed else 'None'}"
        )
        return detail, R_SUMMARIZE

    # -- Helpers ----------------------------------------------------------

    def _reveal_doc(self, doc_name: str, detail: str) -> float:
        s = self._env_state
        if doc_name not in s.documents_revealed:
            s.documents_revealed.append(doc_name)
            s.findings_collected.append(detail)
            return R_NEW_DOCUMENT
        return R_REPEAT_PENALTY

    def _add_finding(self, finding: str) -> None:
        s = self._env_state
        if finding not in s.findings_collected:
            s.findings_collected.append(finding)

    def _invoice_summary(self) -> str:
        if self._case is None:
            return ""
        inv = self._case.invoice
        return (
            f"Supplier: {inv.supplier_name} ({inv.supplier_id}), "
            f"Invoice: {inv.invoice_number}, Date: {inv.invoice_date}, "
            f"PO Ref: {inv.po_reference}, Total: ${inv.total_amount:,.2f}"
        )

    def _suggest_next_actions(self, remaining: int) -> list:
        """Suggest the most useful next actions based on investigation progress."""
        s = self._env_state
        taken = set(s.actions_taken)
        suggestions = []

        if remaining <= 2:
            suggestions.append("submit_final_resolution")
            return suggestions

        core_docs = [
            "inspect_purchase_order",
            "inspect_goods_receipt_note",
            "inspect_invoice_line_items",
        ]
        for a in core_docs:
            if a not in taken:
                suggestions.append(a)

        comparisons = ["compare_quantity", "compare_price", "compare_totals"]
        docs_seen = set(s.documents_revealed)
        has_po = "purchase_order" in docs_seen
        has_grn = "goods_receipt_note" in docs_seen
        if has_po and has_grn:
            for a in comparisons:
                if a not in taken:
                    suggestions.append(a)

        secondary = [
            "inspect_policy_rules",
            "check_for_duplicate_invoice",
            "inspect_vendor_profile",
        ]
        for a in secondary:
            if a not in taken:
                suggestions.append(a)

        if not suggestions:
            if "summarize_findings" not in taken:
                suggestions.append("summarize_findings")
            suggestions.append("submit_final_resolution")

        return suggestions

    def _build_warnings(self, remaining: int) -> list:
        warnings = []
        s = self._env_state
        if remaining <= 2:
            warnings.append(
                f"CRITICAL: Only {remaining} step(s) remaining! "
                f"You MUST submit_final_resolution NOW."
            )
        elif remaining <= 4 and not s.is_finalized:
            warnings.append(
                f"WARNING: {remaining} steps remaining. "
                f"Submit your final resolution soon."
            )
        return warnings

    def _error_obs(self, message: str) -> InvoiceGuardObservation:
        return InvoiceGuardObservation(
            last_action_result=f"ERROR: {message}",
            last_action_error=True,
            done=True,
            reward=0.0,
        )


# Handler dispatch table -- maps action type strings to bound methods.
# Built outside the class to keep the routing clean.
_INVESTIGATION_HANDLERS = {
    ActionType.inspect_invoice_line_items.value: InvoiceGuardEnvironment._inspect_invoice_line_items,
    ActionType.inspect_purchase_order.value: InvoiceGuardEnvironment._inspect_purchase_order,
    ActionType.inspect_goods_receipt_note.value: InvoiceGuardEnvironment._inspect_goods_receipt_note,
    ActionType.inspect_vendor_profile.value: InvoiceGuardEnvironment._inspect_vendor_profile,
    ActionType.inspect_policy_rules.value: InvoiceGuardEnvironment._inspect_policy_rules,
    ActionType.check_for_duplicate_invoice.value: InvoiceGuardEnvironment._check_for_duplicate_invoice,
    ActionType.compare_quantity.value: InvoiceGuardEnvironment._compare_quantity,
    ActionType.compare_price.value: InvoiceGuardEnvironment._compare_price,
    ActionType.compare_totals.value: InvoiceGuardEnvironment._compare_totals,
    ActionType.summarize_findings.value: InvoiceGuardEnvironment._summarize_findings,
}
