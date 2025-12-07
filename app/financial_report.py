# app/financial_report.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from .models import WorkbookUnitSummary, UnitSummaryRow

CATEGORY_PRE = "PreEASD EWID"
CATEGORY_POST = "PostEASD EWID"
CATEGORY_FEE = "Fee Waiver"

ADMIN_FEE = Decimal("100.00")


@dataclass
class CategorySummary:
    name: str
    units: List[UnitSummaryRow]
    total: Decimal
    price_counts: Dict[Decimal, int]  # price -> number of units at that price


@dataclass
class FinancialBreakdown:
    date_requested: date
    earliest_start: Optional[date]

    # (unit, category_name)
    impacted_units: List[Tuple[UnitSummaryRow, str]]

    # Per-category summaries
    categories: Dict[str, CategorySummary]

    # Overall totals
    total_impact: Decimal
    start_balance: Optional[Decimal]
    end_balance: Optional[Decimal]

    # “Section 2” (after PreEASD reversals)
    adjusted_start_balance: Optional[Decimal]
    adjusted_end_balance: Optional[Decimal]

    admin_fee: Decimal = ADMIN_FEE


# -----------------------------
# Helpers
# -----------------------------


def _format_date(d: Optional[date]) -> str:
    if not d:
        return "N/A"
    return d.strftime("%d/%m/%Y")


def _format_currency(amount: Optional[Decimal]) -> str:
    """
    Format Decimal as per your preferences:
      - negative sign in front of '$'  -> -$123.45
      - no thousands separator
      - 2 decimal places
    """
    if amount is None:
        return "N/A"

    q = amount.quantize(Decimal("0.01"))
    sign = "-" if q < 0 else ""
    return f"{sign}${abs(q):.2f}"


def _make_category_summary(name: str, units: List[UnitSummaryRow]) -> CategorySummary:
    total = Decimal("0")
    price_counts: Dict[Decimal, int] = {}
    for u in units:
        if u.unit_price is None:
            continue
        total += u.unit_price
        price_counts[u.unit_price] = price_counts.get(u.unit_price, 0) + 1

    return CategorySummary(
        name=name,
        units=units,
        total=total,
        price_counts=price_counts,
    )


def _merge_categories(name: str, cats: List[CategorySummary]) -> CategorySummary:
    all_units: List[UnitSummaryRow] = []
    total = Decimal("0")
    price_counts: Dict[Decimal, int] = {}

    for c in cats:
        all_units.extend(c.units)
        total += c.total
        for price, count in c.price_counts.items():
            price_counts[price] = price_counts.get(price, 0) + count

    return CategorySummary(name=name, units=all_units, total=total, price_counts=price_counts)


def _format_category_line(label: str, cat: CategorySummary) -> str:
    """
    Build lines like:
      PreEASD EWID units: (8 units at $527.00) = $4216.00
      Fee Waiver units: (2 at $150.00 + 1 at $100.00) = $400.00
    """
    num_units = len(cat.units)
    if num_units == 0:
        return f"{label}: (0 units at $0.00) = $0.00"

    # Build price expression
    parts: List[str] = []
    # Sort by price just for determinism
    for price in sorted(cat.price_counts.keys()):
        count = cat.price_counts[price]
        parts.append(f"{count} at {_format_currency(price)}")

    price_expr = " + ".join(parts)
    return f"{label}: ({price_expr}) = {_format_currency(cat.total)}"


def _get_category(bd: FinancialBreakdown, name: str) -> CategorySummary:
    if name in bd.categories:
        return bd.categories[name]
    return CategorySummary(name=name, units=[], total=Decimal("0"), price_counts={})


# -----------------------------
# Core computation
# -----------------------------


def compute_financial_breakdown(
    summary: WorkbookUnitSummary,
    date_requested: date,
) -> FinancialBreakdown:
    """
    Classify units and compute all financial pieces needed for the report
    AND for downstream logic (e.g. the Recommendation in Section 12).
    """

    # Earliest enrolment start across ALL units (not just impacted)
    earliest_start = (
        min((u.start_date for u in summary.units), default=None)
        if summary.units
        else None
    )

    impacted_units: List[Tuple[UnitSummaryRow, str]] = []
    category_units: Dict[str, List[UnitSummaryRow]] = {
        CATEGORY_PRE: [],
        CATEGORY_POST: [],
        CATEGORY_FEE: [],
    }

    for u in summary.units:
        # Skip non-financial units
        if u.unit_price is None:
            continue
        if u.unit_code.upper().startswith("VPC"):
            continue

        # Classify based on date requested and engagement
        days_diff = (date_requested - u.start_date).days

        if days_diff <= 14:
            cat_name = CATEGORY_PRE
        else:
            # > 2 weeks after start
            if u.recorded_hours is None or u.recorded_hours == 0:
                cat_name = CATEGORY_POST
            else:
                cat_name = CATEGORY_FEE

        impacted_units.append((u, cat_name))
        category_units[cat_name].append(u)

    # Build per-category summaries
    categories: Dict[str, CategorySummary] = {}
    for name, units in category_units.items():
        categories[name] = _make_category_summary(name, units)

    # Totals
    pre = categories[CATEGORY_PRE]
    post = categories[CATEGORY_POST]
    fee = categories[CATEGORY_FEE]

    total_impact = pre.total + post.total + fee.total

    start_balance = summary.account_balance
    end_balance: Optional[Decimal] = None
    if start_balance is not None:
        end_balance = start_balance - total_impact

    # “Section 2” – after applying PreEASD reversals
    adjusted_start_balance: Optional[Decimal] = None
    adjusted_end_balance: Optional[Decimal] = None
    if start_balance is not None:
        adjusted_start_balance = start_balance - pre.total
        adjusted_end_balance = adjusted_start_balance - (post.total + fee.total)

    return FinancialBreakdown(
        date_requested=date_requested,
        earliest_start=earliest_start,
        impacted_units=impacted_units,
        categories=categories,
        total_impact=total_impact,
        start_balance=start_balance,
        end_balance=end_balance,
        adjusted_start_balance=adjusted_start_balance,
        adjusted_end_balance=adjusted_end_balance,
    )


# -----------------------------
# Formatting
# -----------------------------


def _format_financial_report_text(bd: FinancialBreakdown) -> str:
    lines: List[str] = []

    # Section 1 – overall impact
    lines.append(f"Date Requested: {_format_date(bd.date_requested)}")
    lines.append(f"Enrolment Activity Start Date: {_format_date(bd.earliest_start)}")
    lines.append("")

    lines.append("Impacted Units:")
    if not bd.impacted_units:
        lines.append("None")
    else:
        for u, cat in bd.impacted_units:
            price_str = _format_currency(u.unit_price)
            lines.append(f"{u.unit_code} [{cat}] {price_str}")
    lines.append("")

    pre = _get_category(bd, CATEGORY_PRE)
    post = _get_category(bd, CATEGORY_POST)
    fee = _get_category(bd, CATEGORY_FEE)

    lines.append(_format_category_line("PreEASD EWID units", pre))
    lines.append(_format_category_line("PostEASD EWID units", post))
    lines.append(_format_category_line("Fee Waiver units", fee))
    lines.append("")

    total_cat = _merge_categories("Total Financial Impact", [pre, post, fee])
    lines.append(_format_category_line("Total Financial Impact", total_cat))
    lines.append("")

    lines.append(f"Start Account Balance: {_format_currency(bd.start_balance)}")
    if bd.start_balance is not None and bd.end_balance is not None:
        lines.append(
            f"End Account Balance: "
            f"({_format_currency(bd.start_balance)} - {_format_currency(bd.total_impact)}) "
            f"= {_format_currency(bd.end_balance)}"
        )

    # Section 2 – after applying PreEASD reversals
    lines.append("")
    lines.append("Section 2 – After applying PreEASD EWID fee reversals")
    lines.append("")

    lines.append(_format_category_line("PostEASD EWID units", post))
    lines.append(_format_category_line("Fee Waiver units", fee))
    lines.append("")

    total_pf_cat = _merge_categories("Total Financial Impact", [post, fee])
    lines.append(_format_category_line("Total Financial Impact", total_pf_cat))
    lines.append("")

    lines.append(f"Adjusted Start Account Balance: {_format_currency(bd.adjusted_start_balance)}")
    if bd.adjusted_start_balance is not None and bd.adjusted_end_balance is not None:
        post_fee_total = post.total + fee.total
        lines.append(
            f"Adjusted End Account Balance: "
            f"({_format_currency(bd.adjusted_start_balance)} - {_format_currency(post_fee_total)}) "
            f"= {_format_currency(bd.adjusted_end_balance)}"
        )

    return "\n".join(lines).strip()


# -----------------------------
# Public API
# -----------------------------


def generate_financial_report(
    summary: WorkbookUnitSummary,
    date_requested: date,
) -> tuple[str, FinancialBreakdown]:
    """
    Main entry point for the app.

    Returns:
      (report_text, FinancialBreakdown)
    """
    bd = compute_financial_breakdown(summary, date_requested)
    text = _format_financial_report_text(bd)
    return text, bd


def generate_financial_report_text(
    summary: WorkbookUnitSummary,
    date_requested: date,
) -> str:
    """
    Convenience wrapper if you only want the text.
    """
    text, _ = generate_financial_report(summary, date_requested)
    return text
