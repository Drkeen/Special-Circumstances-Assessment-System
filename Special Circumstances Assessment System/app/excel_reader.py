# app/excel_reader.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import date
from decimal import Decimal
import re

from pathlib import Path

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
    if pd.isna(value):
        return None
    return pd.to_datetime(value).date()


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
    text = str(raw)
    match = re.search(r"([\d.]+)", text)
    return float(match.group(1)) if match else None


def _to_decimal(value) -> Optional[Decimal]:
    if pd.isna(value):
        return None
    return Decimal(str(value))


# -----------------------------
# 1) Read workbook
# -----------------------------


def read_workbook(path: Path) -> Dict[str, pd.DataFrame]:
    """
    Read an Excel workbook and return {sheet_name: DataFrame}.
    """
    excel = pd.read_excel(path, sheet_name=None)
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
            # You can choose to skip or handle differently
            continue

        rows.append(
            StudyPlanRow(
                unit_code=str(row["Spk Cd"]).strip(),
                status=str(row["SSP Status"]).strip(),
                start_date=start,
                liability_category=str(row["Liability Category"]).strip(),
            )
        )

    return rows


def parse_unit_engagement(df: pd.DataFrame) -> List[UnitEngagementRow]:
    """
    Unit Engagement:
      - Curriculum Item     -> unit code
      - Unit Start Date     -> start date
      - Recorded hours      -> float
    """
    required_cols = [
        "Curriculum Item",
        "Unit Start Date",
        "Recorded hours",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Unit Engagement sheet missing columns: {missing}")

    rows: List[UnitEngagementRow] = []
    for _, row in df.iterrows():
        start = _to_date(row["Unit Start Date"])
        if start is None:
            continue

        rows.append(
            UnitEngagementRow(
                unit_code=str(row["Curriculum Item"]).strip(),
                start_date=start,
                recorded_hours=_parse_recorded_hours(row["Recorded hours"]),
            )
        )

    return rows


def parse_student_account(df: pd.DataFrame) -> List[StudentAccountRow]:
    """
    Student Account:
      - SSP Spk Cd -> unit code
      - Txn Amt / Txb Amt -> txn_amount
      - Txn Date (if available) -> txn_date (for picking the most recent)
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

    has_txn_date = "Txn Date" in df.columns

    rows: List[StudentAccountRow] = []
    for _, row in df.iterrows():
        unit_code = str(row["SSP Spk Cd"]).strip()
        amount = _to_decimal(row[amount_col])
        if amount is None:
            continue

        txn_date = _to_date(row["Txn Date"]) if has_txn_date else None

        rows.append(
            StudentAccountRow(
                unit_code=unit_code,
                txn_amount=amount,
                txn_date=txn_date,
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

    - Only units from Study Plan with status 'Enrolled' are used.
    - Unit Engagement matched on (unit_code, start_date).
    - Student Account matched on unit_code; if multiple rows, we pick:
        - the one with the most recent txn_date, if available
        - otherwise, the last occurrence seen.
    """

    # Map (unit_code, start_date) -> UnitEngagementRow
    engagement_map: Dict[Tuple[str, date], UnitEngagementRow] = {}
    for r in unit_engagement:
        engagement_map[(r.unit_code, r.start_date)] = r

    # Map unit_code -> most relevant StudentAccountRow
    price_map: Dict[str, StudentAccountRow] = {}
    for r in student_account:
        existing = price_map.get(r.unit_code)
        if existing is None:
            price_map[r.unit_code] = r
        else:
            # If both have txn_date, keep the latest
            if r.txn_date and existing.txn_date:
                if r.txn_date > existing.txn_date:
                    price_map[r.unit_code] = r
            # If new has date but old doesn't, prefer new
            elif r.txn_date and not existing.txn_date:
                price_map[r.unit_code] = r
            # If neither have dates, keep existing (or you could override)

    # Build the summary rows
    summary_rows: List[UnitSummaryRow] = []
    for sp in study_plan:
        key = (sp.unit_code, sp.start_date)
        engagement = engagement_map.get(key)
        account = price_map.get(sp.unit_code)

        summary_rows.append(
            UnitSummaryRow(
                unit_code=sp.unit_code,
                status=sp.status,
                start_date=sp.start_date,
                recorded_hours=engagement.recorded_hours if engagement else None,
                unit_price=account.txn_amount if account else None,
                liability_category=sp.liability_category,
            )
        )

    return WorkbookUnitSummary(units=summary_rows)

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
    """
    sp_df = pd.read_excel(study_plan_path)
    ue_df = pd.read_excel(unit_engagement_path)
    sa_df = pd.read_excel(student_account_path)

    sp_rows = parse_study_plan(sp_df)
    ue_rows = parse_unit_engagement(ue_df)
    sa_rows = parse_student_account(sa_df)

    return build_unit_summary(sp_rows, ue_rows, sa_rows)
