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
    unit_code: str                  # Spk Cd
    status: str                     # SSP Status
    start_date: date                # Enrolment Activity Start Date (date only)
    liability_category: str         # Liability Category (for later reporting)


@dataclass
class UnitEngagementRow:
    unit_code: str                  # Curriculum Item (matches Spk Cd)
    start_date: date                # Unit Start Date (matches Enrolment Activity Start Date)
    recorded_hours: Optional[float] # Parsed from "50.87 hours" → 50.87


@dataclass
class StudentAccountRow:
    unit_code: str                  # SSP Spk Cd (matches Spk Cd)
    txn_amount: Decimal             # Txn Amt / Txb Amt
    txn_date: Optional[date] = None # Used to pick most recent price (if present)


# -----------------------------
# Final merged output model
# -----------------------------

@dataclass
class UnitSummaryRow:
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
