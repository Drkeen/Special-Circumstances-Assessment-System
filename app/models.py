# app/models.py
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Optional, List


# -----------------------------
# Raw sheet row models
# -----------------------------

@dataclass
class StudyPlanRow:
    """
    Represents a single row from the Study Plan sheet.

    Fields:
      - unit_code: Spk Cd
      - status: SSP Status
      - start_date: Enrolment Activity Start Date (date only)
      - liability_category: Liability Category (for later reporting)
    """
    unit_code: str                  # Spk Cd
    status: str                     # SSP Status
    start_date: date                # Enrolment Activity Start Date (date only)
    liability_category: str         # Liability Category (for later reporting)


@dataclass
class UnitEngagementRow:
    """
    Represents a single row from the Unit Engagement sheet.

    Fields:
      - unit_code: Curriculum Item (matches Spk Cd)
      - start_date: Unit Start Date (matches Enrolment Activity Start Date)
      - recorded_hours: Parsed from strings like "50.87 hours" → 50.87
    """
    unit_code: str                  # Curriculum Item (matches Spk Cd)
    start_date: date                # Unit Start Date (matches Enrolment Activity Start Date)
    recorded_hours: Optional[float] # Parsed from "50.87 hours" → 50.87


@dataclass
class StudentAccountRow:
    """
    Represents a single row from the Student Account sheet.

    Fields:
      - unit_code: SSP Spk Cd (matches Spk Cd)
      - txn_amount: Txn Amt / Txb Amt (Decimal)
      - txn_date: Used to pick most recent price (if present)

    When multiple StudentAccountRow entries exist for a unit_code,
    the most relevant one is selected using:
      - Most recent txn_date (if any dates exist)
      - Otherwise, the last row seen for that unit_code.
    """
    unit_code: str                  # SSP Spk Cd (matches Spk Cd)
    txn_amount: Decimal             # Txn Amt / Txb Amt
    txn_date: Optional[date] = None # Used to pick most recent price (if present)


# -----------------------------
# Final merged output model
# -----------------------------

@dataclass
class UnitSummaryRow:
    """
    Final merged representation of a unit across all three sheets.

    Fields:
      - unit_code
      - status          (from Study Plan – SSP Status)
      - start_date      (from Study Plan – Enrolment Activity Start Date)
      - recorded_hours  (from Unit Engagement, if matched)
      - unit_price      (from Student Account, most relevant txn_amount)
      - liability_category (from Study Plan)
    """
    unit_code: str
    status: str
    start_date: date
    recorded_hours: Optional[float]
    unit_price: Optional[Decimal]
    liability_category: str


@dataclass
class WorkbookUnitSummary:
    """Container for all final unit rows (for one workbook / student)."""
    units: List[UnitSummaryRow]
