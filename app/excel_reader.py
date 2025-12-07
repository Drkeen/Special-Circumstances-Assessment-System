# app/excel_reader.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import re

import pandas as pd

from .models import (
    StudyPlanRow,
    UnitEngagementRow,
    StudentAccountRow,
    UnitSummaryRow,
    WorkbookUnitSummary,
)

# -----------------------------
# Helpers
# -----------------------------


def _to_date(value) -> Optional[date]:
    """
    Convert common date-like values to a `date` object, or return None.

    Handles:
      - pandas Timestamp
      - datetime/date
      - strings
    """
    if pd.isna(value) or value == "":
        return None

    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()

    # Let pandas try to parse; be forgiving
    dt = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return None

    if isinstance(dt, pd.Timestamp):
        return dt.date()
    if isinstance(dt, datetime):
        return dt.date()
    return None


def _parse_recorded_hours(raw) -> Optional[float]:
    """
    '50.87 hours' -> 50.87
    '40'          -> 40.0
    None / NaN    -> None
    """
    if pd.isna(raw):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)

    text = str(raw).strip()
    if not text:
        return None

    # Extract first numeric segment, allowing optional sign & decimal
    match = re.search(r"([-+]?\d*\.?\d+)", text)
    return float(match.group(1)) if match else None


def _to_decimal(value) -> Optional[Decimal]:
    """
    Parse currency/amount values into Decimal.

    Handles:
      - int/float
      - strings with '$' and/or commas
      - returns None for empty/NaN/invalid
    """
    if pd.isna(value):
        return None

    if isinstance(value, Decimal):
        return value

    if isinstance(value, (int, float)):
        return Decimal(str(value))

    s = str(value).strip()
    if not s:
        return None

    # Strip common currency formatting
    s = s.replace("$", "").replace(",", "").strip()
    try:
        return Decimal(s)
    except (InvalidOperation, ValueError):
        return None


# -----------------------------
# 1) Read workbook (optional helper)
# -----------------------------


def read_workbook(path: Path) -> Dict[str, pd.DataFrame]:
    """
    Read an Excel workbook and return {sheet_name: DataFrame}.

    Not used by the Streamlit UI directly, but handy if you ever have
    a single multi-sheet workbook instead of three separate files.
    """
    excel = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    return {name: df for name, df in excel.items()}


# -----------------------------
# 2) Sheet parsers
# -----------------------------


def parse_study_plan(df: pd.DataFrame) -> List[StudyPlanRow]:
    """
    Study Plan: filter to 'Enrolled' and keep only the relevant columns.
    Expected columns:
      - 'Spk Cd'
      - 'SSP Status'
      - 'Enrolment Activity Start Date'
      - 'Liability Category'

    Also:
      - Skips any units where unit_code starts with 'CLS' (they are clusters).
    """
    required_cols = [
        "Spk Cd",
        "SSP Status",
        "Enrolment Activity Start Date",
        "Liability Category",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Study Plan sheet missing columns: {missing}")

    # Filter to Enrolled
    df = df[df["SSP Status"] == "Enrolled"].copy()

    rows: List[StudyPlanRow] = []
    for _, row in df.iterrows():
        start = _to_date(row["Enrolment Activity Start Date"])
        if start is None:
            # Skip if no start date; keeps join logic clean
            continue

        unit_code = str(row["Spk Cd"]).strip()

        # Skip cluster codes like 'CLS-ACM-0011'
        if unit_code.upper().startswith("CLS"):
            continue

        rows.append(
            StudyPlanRow(
                unit_code=unit_code,
                status=str(row["SSP Status"]).strip(),
                start_date=start,
                liability_category=str(row["Liability Category"]).strip(),
            )
        )

    return rows


def parse_unit_engagement(df: pd.DataFrame) -> List[UnitEngagementRow]:
    """
    Unit Engagement:
      - Curriculum Item              -> unit code
      - Unit Start Date              -> start date
      - Recorded hours / Recorded Hours -> float
      - Latest Engagement Date (optional) -> date
    """

    def find_col(normalised_targets):
        """
        Given a list of target names like ["recorded hours"],
        find the first column whose stripped, lowercased name matches.
        """
        targets = {t.strip().lower() for t in normalised_targets}
        for c in df.columns:
            if str(c).strip().lower() in targets:
                return c
        return None

    # Required core columns
    curriculum_col = find_col(["curriculum item"])
    start_col = find_col(["unit start date"])
    if curriculum_col is None or start_col is None:
        missing = []
        if curriculum_col is None:
            missing.append("Curriculum Item")
        if start_col is None:
            missing.append("Unit Start Date")
        raise ValueError(f"Unit Engagement sheet missing columns: {missing}")

    # Recorded hours (we expect it, but match flexibly)
    hours_col = find_col(["recorded hours"])

    # Latest Engagement Date (optional)
    latest_engagement_col = find_col(["latest engagement date"])

    rows: List[UnitEngagementRow] = []
    for _, row in df.iterrows():
        start = _to_date(row[start_col])
        if start is None:
            continue

        recorded_hours = (
            _parse_recorded_hours(row[hours_col]) if hours_col is not None else None
        )

        latest_engagement = (
            _to_date(row[latest_engagement_col])
            if latest_engagement_col is not None
            else None
        )

        rows.append(
            UnitEngagementRow(
                unit_code=str(row[curriculum_col]).strip(),
                start_date=start,
                recorded_hours=recorded_hours,
                latest_engagement_date=latest_engagement,
            )
        )

    return rows


def parse_student_account(df: pd.DataFrame) -> List[StudentAccountRow]:
    """
    Student Account:
      - SSP Spk Cd -> unit code
      - Txn Amt / Txb Amt -> txn_amount
      - Txn Date / Acct Txn Date -> txn_date
      - Unalloc Amt -> unalloc_amt (per-transaction outstanding amount)

    If multiple rows exist for a unit_code, the merge logic later will select:
      - the one with the most recent txn_date, if any dates exist
      - otherwise, the last occurrence seen for that unit_code.
    """
    # Handle minor header variation for amount column
    amount_col_candidates = ["Txn Amt", "Txb Amt"]
    amount_col = next((c for c in amount_col_candidates if c in df.columns), None)
    if amount_col is None:
        raise ValueError("Student Account sheet missing 'Txn Amt'/'Txb Amt' column")

    required_cols = ["SSP Spk Cd", amount_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Student Account sheet missing columns: {missing}")

    # Txn date column can be named a couple of ways
    txn_date_candidates = ["Txn Date", "Acct Txn Date"]
    txn_date_col = next((c for c in txn_date_candidates if c in df.columns), None)

    # Unallocated amount column (for account balance)
    unalloc_col = "Unalloc Amt" if "Unalloc Amt" in df.columns else None

    rows: List[StudentAccountRow] = []
    for _, row in df.iterrows():
        unit_code = str(row["SSP Spk Cd"]).strip()
        amount = _to_decimal(row[amount_col])
        if amount is None:
            # ignore zero/blank/invalid amounts
            continue

        if txn_date_col:
            txn_date = _to_date(row[txn_date_col])
        else:
            txn_date = None

        unalloc_amt = _to_decimal(row[unalloc_col]) if unalloc_col else None

        rows.append(
            StudentAccountRow(
                unit_code=unit_code,
                txn_amount=amount,
                txn_date=txn_date,
                unalloc_amt=unalloc_amt,
            )
        )

    return rows


# -----------------------------
# 3) Merge into UnitSummaryRow
# -----------------------------


def build_unit_summary(
    study_plan: List[StudyPlanRow],
    unit_engagement: List[UnitEngagementRow],
    student_account: List[StudentAccountRow],
) -> WorkbookUnitSummary:
    """
    Join the three sources into a list of UnitSummaryRow, driven by Study Plan.

    Rules:
      - Only units from Study Plan with status 'Enrolled' are used.
      - Unit Engagement matched on (unit_code, start_date).
      - Student Account matched on unit_code; if multiple rows, we pick:
          * the one with the most recent txn_date, if available
          * otherwise, the last occurrence seen.

    Also computes:
      - account_balance: sum of Unalloc Amt across all StudentAccountRow entries
        (or None if no Unalloc Amt is present).
    """

    # Map (unit_code, start_date) -> UnitEngagementRow
    engagement_map: Dict[Tuple[str, date], UnitEngagementRow] = {}
    for r in unit_engagement:
        engagement_map[(r.unit_code, r.start_date)] = r

    # Map unit_code -> most relevant StudentAccountRow for pricing
    price_map: Dict[str, StudentAccountRow] = {}
    for r in student_account:
        existing = price_map.get(r.unit_code)
        if existing is None:
            price_map[r.unit_code] = r
        else:
            # Both have txn_date: keep the latest; if equal, keep last seen (>=)
            if r.txn_date and existing.txn_date:
                if r.txn_date >= existing.txn_date:
                    price_map[r.unit_code] = r
            # New has date, old doesn't -> prefer new
            elif r.txn_date and not existing.txn_date:
                price_map[r.unit_code] = r
            # Neither have dates -> prefer last seen
            elif not r.txn_date and not existing.txn_date:
                price_map[r.unit_code] = r
            # else: existing has date, new doesn't -> keep existing

    # Compute overall account balance: sum of Unalloc Amt
    account_balance: Optional[Decimal] = None
    total_unalloc = Decimal("0")
    has_unalloc = False
    for sa in student_account:
        if getattr(sa, "unalloc_amt", None) is not None:
            total_unalloc += sa.unalloc_amt  # type: ignore[arg-type]
            has_unalloc = True
    if has_unalloc:
        account_balance = total_unalloc

    # Build the summary rows (Study Plan is the driver)
    # Build the summary rows (Study Plan is the driver)
    summary_rows: List[UnitSummaryRow] = []
    for sp in study_plan:
        key = (sp.unit_code, sp.start_date)
        engagement = engagement_map.get(key)
        account = price_map.get(sp.unit_code)

        recorded_hours = engagement.recorded_hours if engagement else None

        # Take the txn_amount as the "unit price", but ensure it is never negative.
        unit_price = None
        if account is not None and account.txn_amount is not None:
            unit_price = account.txn_amount
            if unit_price < 0:
                unit_price = -unit_price  # normalise to positive

        summary_rows.append(
            UnitSummaryRow(
                unit_code=sp.unit_code,
                status=sp.status,
                start_date=sp.start_date,
                recorded_hours=recorded_hours,
                unit_price=unit_price,
                liability_category=sp.liability_category,
            )
        )

    return WorkbookUnitSummary(units=summary_rows, account_balance=account_balance)



# -----------------------------
# 4) High-level convenience
# -----------------------------


def load_unit_summary_from_files(
    study_plan_path: Path,
    unit_engagement_path: Path,
    student_account_path: Path,
) -> WorkbookUnitSummary:
    """
    Given three separate Excel files (study_plan, unit_engagement, student_account),
    parse and merge them into a WorkbookUnitSummary.

    This version automatically skips title/header rows such as
    'Study Plan TQ (Maintain)' or 'Student Account Mgt (Enquiry)'.
    """

    def _read_with_auto_header(path: Path, header_markers: List[str]) -> pd.DataFrame:
        """
        Detect the header row by scanning the first few rows for any of the
        expected header marker strings, then read again with that row as header.
        """
        preview = pd.read_excel(path, nrows=10, header=None, engine="openpyxl")
        header_row: Optional[int] = None

        for i, row in preview.iterrows():
            values = [str(v).strip() for v in row.tolist() if not pd.isna(v)]
            if not values:
                continue
            # If any marker appears in any cell in this row, treat as header
            if any(
                any(marker in cell for marker in header_markers)
                for cell in values
            ):
                header_row = i
                break

        if header_row is None:
            header_row = 0  # fallback if nothing matched

        return pd.read_excel(path, engine="openpyxl", header=header_row)

    # Tell each sheet what kind of headers we expect (names, not positions)
    sp_df = _read_with_auto_header(
        study_plan_path,
        header_markers=["Spk Cd", "SSP Status", "Enrolment Activity Start Date"],
    )
    ue_df = _read_with_auto_header(
        unit_engagement_path,
        header_markers=["Curriculum Item", "Unit Start Date", "Recorded"],
    )
    sa_df = _read_with_auto_header(
        student_account_path,
        header_markers=["SSP Spk Cd", "Txn Amt", "Txb Amt", "Txn Date", "Acct Txn Date"],
    )

    sp_rows = parse_study_plan(sp_df)
    ue_rows = parse_unit_engagement(ue_df)
    sa_rows = parse_student_account(sa_df)

    return build_unit_summary(sp_rows, ue_rows, sa_rows)
