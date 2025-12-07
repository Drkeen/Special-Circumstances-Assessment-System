# app/financial_report.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Tuple

from .models import WorkbookUnitSummary, UnitSummaryRow

CATEGORY_PRE_EASD = "PreEASD EWID"
CATEGORY_POST_EASD = "PostEASD EWID"
CATEGORY_FEE_WAIVER = "Fee Waiver"


@dataclass
class CategorisedUnit:
    unit_code: str
    category: str
    unit_price: Decimal
    start_date: date


@dataclass
class FinancialReportData:
    date_requested: date
    earliest_start_date: Optional[date]
    impacted_units: List[CategorisedUnit]
    total_units: int
    total_amount: Decimal
    start_account_balance: Decimal
    end_account_balance: Decimal


# -----------------------------
# Helpers
# -----------------------------


def _is_vpc_unit(unit: UnitSummaryRow) -> bool:
    """Return True if the unit should be ignored due to being a VPC unit."""
    return unit.unit_code.upper().startswith("VPC")


def _categorise_unit(u: UnitSummaryRow, date_requested: date) -> Optional[str]:
    """
    Determine which financial category (if any) a unit belongs to.

    Rules:
      - PreEASD EWID:
          date_requested <= (unit_start_date + 14 days)
          (recorded hours do not matter)
      - PostEASD EWID:
          date_requested  > (unit_start_date + 14 days)
          AND no recorded hours
      - Fee Waiver:
          date_requested  > (unit_start_date + 14 days)
          AND recorded hours present
    """
    start = u.start_date
    boundary = start + timedelta(days=14)

    # PreEASD EWID: up to and including 14 days after start
    if date_requested <= boundary:
        return CATEGORY_PRE_EASD

    # After 2 weeks:
    hours = u.recorded_hours
    has_hours = hours is not None and hours > 0

    if not has_hours:
        return CATEGORY_POST_EASD

    return CATEGORY_FEE_WAIVER


def _fmt_date(d: Optional[date]) -> str:
    if not d:
        return "N/A"
    return d.strftime("%d/%m/%Y")


def _fmt_money(amount: Decimal) -> str:
    """
    Format money without thousands separators,
    with negatives shown as -$123.45 instead of $-123.45.
    """
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):.2f}"


def _group_by_price(units: List[CategorisedUnit]) -> List[Tuple[int, Decimal]]:
    """
    Group units by unit_price.

    Returns a list of (count, price) pairs, sorted by price descending.
    """
    groups: Dict[Decimal, int] = {}
    for u in units:
        groups[u.unit_price] = groups.get(u.unit_price, 0) + 1

    # Sort by price (high to low just so it's deterministic)
    return sorted(
        ((count, price) for price, count in groups.items()),
        key=lambda x: x[1],
        reverse=True,
    )


def _category_line(name: str, units: List[CategorisedUnit]) -> str:
    """
    Build a line like:
      PreEASD EWID units: (2 units at $150.00 + 1 unit at $100.00) = $400.00

    If there are no units in the category, we still show 0 and $0.00.
    """
    if not units:
        return (
            f"{name} units: "
            f"(0 units at {_fmt_money(Decimal('0'))}) = {_fmt_money(Decimal('0'))}"
        )

    total = sum((u.unit_price for u in units), Decimal("0"))
    groups = _group_by_price(units)

    parts: List[str] = []
    for count, price in groups:
        unit_word = "unit" if count == 1 else "units"
        parts.append(f"{count} {unit_word} at {_fmt_money(price)}")

    breakdown = " + ".join(parts)
    return f"{name} units: ({breakdown}) = {_fmt_money(total)}"


# -----------------------------
# Core report builder
# -----------------------------


def build_financial_report_data(
    summary: WorkbookUnitSummary,
    date_requested: date,
) -> FinancialReportData:
    """
    Construct structured financial report data from a WorkbookUnitSummary and a Date Requested.

    - Ignores units whose code starts with 'VPC'.
    - Only includes units with a non-null unit_price in financial impact.
    - Uses summary.account_balance as the starting balance (0 if missing).
    """

    # Filter out VPC units for financial purposes
    relevant_units: List[UnitSummaryRow] = [
        u for u in summary.units if not _is_vpc_unit(u)
    ]

    # Earliest start date among relevant units (or None)
    earliest_start_date: Optional[date] = None
    if relevant_units:
        earliest_start_date = min(u.start_date for u in relevant_units)

    impacted_units: List[CategorisedUnit] = []

    for u in relevant_units:
        # Need a price to have financial impact
        if u.unit_price is None:
            continue

        category = _categorise_unit(u, date_requested)
        if category is None:
            continue

        impacted_units.append(
            CategorisedUnit(
                unit_code=u.unit_code,
                category=category,
                unit_price=u.unit_price,
                start_date=u.start_date,
            )
        )

    total_units = len(impacted_units)
    total_amount = sum((cu.unit_price for cu in impacted_units), Decimal("0"))

    # Account balance: if None, treat as 0 for now
    start_balance = (
        summary.account_balance if summary.account_balance is not None else Decimal("0")
    )
    end_balance = start_balance - total_amount

    return FinancialReportData(
        date_requested=date_requested,
        earliest_start_date=earliest_start_date,
        impacted_units=impacted_units,
        total_units=total_units,
        total_amount=total_amount,
        start_account_balance=start_balance,
        end_account_balance=end_balance,
    )


def generate_financial_report_text(
    summary: WorkbookUnitSummary,
    date_requested: date,
) -> str:
    """
    Build the FinancialReportData and render it as text.

    Section 1: Full picture (all categories).
    Section 2: Adjusted view assuming PreEASD EWID reversals already applied.
    """
    data = build_financial_report_data(summary, date_requested)

    lines: List[str] = []

    # -----------------------------
    # SECTION 1 – FULL VIEW
    # -----------------------------
    lines.append("SECTION 1 – Full Financial Impact")
    lines.append("")
    # Header dates
    lines.append(f"Date Requested: {_fmt_date(data.date_requested)}")
    lines.append(f"Enrolment Activity Start Date: {_fmt_date(data.earliest_start_date)}")
    lines.append("")

    # Impacted units (all categories)
    lines.append("Impacted Units:")
    if not data.impacted_units:
        lines.append("(No financially impacted units)")
    else:
        for cu in data.impacted_units:
            lines.append(
                f"{cu.unit_code} [{cu.category}] {_fmt_money(cu.unit_price)}"
            )
    lines.append("")

    # Split units by category for the summary lines
    pre_units = [cu for cu in data.impacted_units if cu.category == CATEGORY_PRE_EASD]
    post_units = [cu for cu in data.impacted_units if cu.category == CATEGORY_POST_EASD]
    fee_units = [cu for cu in data.impacted_units if cu.category == CATEGORY_FEE_WAIVER]

    pre_line = _category_line(CATEGORY_PRE_EASD, pre_units)
    post_line = _category_line(CATEGORY_POST_EASD, post_units)
    fee_line = _category_line(CATEGORY_FEE_WAIVER, fee_units)

    lines.append(pre_line)
    lines.append(post_line)
    lines.append(fee_line)
    lines.append("")

    # Total financial impact – using same grouped breakdown style
    if not data.impacted_units:
        total_line = (
            f"Total Financial Impact: "
            f"(0 units at {_fmt_money(Decimal('0'))}) = {_fmt_money(Decimal('0'))}"
        )
    else:
        from decimal import Decimal as _D  # avoid shadowing

        groups = _group_by_price(data.impacted_units)
        parts: List[str] = []
        for count, price in groups:
            unit_word = "unit" if count == 1 else "units"
            parts.append(f"{count} {unit_word} at {_fmt_money(price)}")
        breakdown = " + ".join(parts)
        total_line = (
            f"Total Financial Impact: "
            f"({breakdown}) = {_fmt_money(data.total_amount)}"
        )

    lines.append(total_line)
    lines.append("")

    # Account balances
    start_str = _fmt_money(data.start_account_balance)
    impact_str = _fmt_money(data.total_amount)
    end_str = _fmt_money(data.end_account_balance)

    lines.append(f"Start Account Balance: {start_str}")
    lines.append(f"End Account Balance: ({start_str} - {impact_str}) = {end_str}")
    lines.append("")
    lines.append("")

    # -----------------------------
    # SECTION 2 – AFTER PREEASD REVERSALS
    # -----------------------------
    lines.append("SECTION 2 – After PreEASD EWID Fee Reversals")
    lines.append("")
    lines.append(
        "This section assumes all PreEASD EWID units have already had their fees reversed."
    )
    lines.append("Only PostEASD EWID and Fee Waiver units are shown below.")
    lines.append("")

    # Re-use same dates
    lines.append(f"Date Requested: {_fmt_date(data.date_requested)}")
    lines.append(f"Enrolment Activity Start Date: {_fmt_date(data.earliest_start_date)}")
    lines.append("")

    # Adjusted impacted units: only PostEASD + Fee Waiver
    adjusted_units = post_units + fee_units

    lines.append("Impacted Units:")
    if not adjusted_units:
        lines.append("(No financially impacted units)")
    else:
        for cu in adjusted_units:
            lines.append(
                f"{cu.unit_code} [{cu.category}] {_fmt_money(cu.unit_price)}"
            )
    lines.append("")

    # Category summaries for adjusted view (no PreEASD line here)
    post_line_adj = _category_line(CATEGORY_POST_EASD, post_units)
    fee_line_adj = _category_line(CATEGORY_FEE_WAIVER, fee_units)

    lines.append(post_line_adj)
    lines.append(fee_line_adj)
    lines.append("")

    # Totals for adjusted view
    total_pre_amount = sum((u.unit_price for u in pre_units), Decimal("0"))
    total_postfee_amount = sum((u.unit_price for u in adjusted_units), Decimal("0"))

    if not adjusted_units:
        total_line_adj = (
            f"Total Financial Impact: "
            f"(0 units at {_fmt_money(Decimal('0'))}) = {_fmt_money(Decimal('0'))}"
        )
    else:
        groups_adj = _group_by_price(adjusted_units)
        parts_adj: List[str] = []
        for count, price in groups_adj:
            unit_word = "unit" if count == 1 else "units"
            parts_adj.append(f"{count} {unit_word} at {_fmt_money(price)}")
        breakdown_adj = " + ".join(parts_adj)
        total_line_adj = (
            f"Total Financial Impact: "
            f"({breakdown_adj}) = {_fmt_money(total_postfee_amount)}"
        )

    lines.append(total_line_adj)
    lines.append("")

    # Adjusted account balances
    # Start balance after PreEASD reversals applied
    adjusted_start_balance = data.start_account_balance - total_pre_amount
    adjusted_end_balance = adjusted_start_balance - total_postfee_amount

    adj_start_str = _fmt_money(adjusted_start_balance)
    adj_impact_str = _fmt_money(total_postfee_amount)
    adj_end_str = _fmt_money(adjusted_end_balance)

    lines.append(f"Adjusted Start Account Balance: {adj_start_str}")
    lines.append(
        f"Adjusted End Account Balance: ({adj_start_str} - {adj_impact_str}) = {adj_end_str}"
    )

    return "\n".join(lines)
