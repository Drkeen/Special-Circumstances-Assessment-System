# ui/streamlit_app.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime, date
from typing import Dict, Optional

import os

import pandas as pd
import streamlit as st

from app.excel_reader import load_unit_summary_from_files
from app.financial_report import generate_financial_report, FinancialBreakdown
from app.special_circumstances import (
    SpecialCircumstancesInputs,
    generate_special_circumstances_report,
)


# -------------------------------------------------------------------
# Paths & setup
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# Where we store uploaded spreadsheets and supporting docs
SPREADSHEET_UPLOAD_DIR = BASE_DIR / "uploaded_spreadsheets"
SUPPORTING_UPLOAD_DIR = BASE_DIR / "uploaded_supporting_docs"

SPREADSHEET_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SUPPORTING_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Special Circumstances Assessment System (SCAS)",
    layout="wide",
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _parse_ddmmyyyy(value: str) -> Optional[date]:
    value = value.strip()
    if not value:
        return None
    try:
        dt = datetime.strptime(value, "%d/%m/%Y")
        return dt.date()
    except ValueError:
        return None


def _save_uploaded_file(upload_dir: Path, uploaded_file) -> Path:
    """Save an uploaded file to a target directory and return the path."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / uploaded_file.name
    with dest.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def classify_excel_file(temp_path: Path) -> Optional[str]:
    """
    Look inside an Excel file and classify it as one of:
      - 'study_plan'
      - 'unit_engagement'
      - 'student_account'
    using header markers, NOT the filename.
    """
    try:
        preview = pd.read_excel(temp_path, nrows=10, header=None, engine="openpyxl")
    except Exception:
        return None

    text_cells = [str(v) for v in preview.values.ravel() if not pd.isna(v)]
    lower_cells = [s.lower() for s in text_cells]

    def score_for(markers):
        score = 0
        for m in markers:
            m_low = m.lower()
            if any(m_low in cell for cell in lower_cells):
                score += 1
        return score

    scores: Dict[str, int] = {
        "study_plan": score_for(
            ["Spk Cd", "SSP Status", "Enrolment Activity Start Date"]
        ),
        "unit_engagement": score_for(
            ["Curriculum Item", "Unit Start Date", "Recorded"]
        ),
        "student_account": score_for(
            ["SSP Spk Cd", "Txn Amt", "Txb Amt", "Unalloc Amt"]
        ),
    }

    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        return None
    return best_type


def style_selected_rows(df: pd.DataFrame, selected_unit_codes: set) -> pd.io.formats.style.Styler:
    """Apply a soft red border + light red background to selected rows."""
    def _row_style(row):
        code = row.get("Unit Code")
        if code in selected_unit_codes:
            return [
                "border: 2px solid red; background-color: rgba(255, 0, 0, 0.05);"
            ] * len(row)
        return [""] * len(row)

    return df.style.apply(_row_style, axis=1)


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------

st.title("Special Circumstances Assessment System (SCAS)")

st.markdown(
    """
This tool helps generate:
- A **financial impact report** based on TechOne exports, and  
- A **Special Circumstances Investigation Report** (COE/refund recommendation).
"""
)

# -------------------------------------------------------------------
# Step 1 – Upload spreadsheets
# -------------------------------------------------------------------

st.header("Step 1 – Upload spreadsheets")

uploaded_spreadsheets = st.file_uploader(
    "Upload the three TechOne spreadsheets (any order): Study Plan, Unit Engagement, Student Account",
    type=["xlsx"],
    accept_multiple_files=True,
    key="spreadsheet_uploader",
)

spreadsheet_paths: Dict[str, Path] = {}  # type -> path

if uploaded_spreadsheets:
    st.subheader("Detected file types")
    for f in uploaded_spreadsheets:
        saved_path = _save_uploaded_file(SPREADSHEET_UPLOAD_DIR, f)
        file_type = classify_excel_file(saved_path)
        label = file_type if file_type is not None else "UNKNOWN"
        st.write(f"- `{f.name}` → **{label}**")
        if file_type:
            if file_type in spreadsheet_paths:
                st.warning(
                    f"Multiple files detected for type '{file_type}'. "
                    f"Currently using: {spreadsheet_paths[file_type].name}. "
                    f"Latest uploaded is: {f.name}"
                )
            spreadsheet_paths[file_type] = saved_path

# Validate presence of all three
required_types = ["study_plan", "unit_engagement", "student_account"]
missing_types = [t for t in required_types if t not in spreadsheet_paths]

if not uploaded_spreadsheets:
    st.info("Upload your spreadsheets above to continue.")
    st.stop()

if missing_types:
    st.error(
        "Missing the following spreadsheet types (based on internal headers): "
        + ", ".join(missing_types)
    )
    st.stop()

study_plan_path = spreadsheet_paths["study_plan"]
unit_engagement_path = spreadsheet_paths["unit_engagement"]
student_account_path = spreadsheet_paths["student_account"]

# -------------------------------------------------------------------
# Step 2 – Load and display Unit Summary
# -------------------------------------------------------------------

st.header("Step 2 – Unit Summary & scope selection")

try:
    summary = load_unit_summary_from_files(
        study_plan_path=study_plan_path,
        unit_engagement_path=unit_engagement_path,
        student_account_path=student_account_path,
    )
except Exception as e:
    st.error(f"Error loading/merging spreadsheets: {e}")
    st.stop()

# Build main summary table
summary_rows = [
    {
        "Unit Code": u.unit_code,
        "Unit Status": u.status,
        "Unit Start Date": u.start_date,
        "Unit Recorded Hours": u.recorded_hours,
        "Unit Price": float(u.unit_price) if u.unit_price is not None else None,
        "Liability Category": u.liability_category,
    }
    for u in summary.units
]

df_summary = pd.DataFrame(summary_rows)

scope = st.radio(
    "Scope of withdrawal/refund calculation",
    ["Full course", "Specific units"],
    index=0,
    horizontal=True,
)

# Initialise selection state
if "selected_unit_codes" not in st.session_state:
    st.session_state["selected_unit_codes"] = set()

if scope == "Specific units":
    st.markdown(
        "Select the units the student is **withdrawing from**. "
        "Selected rows will be highlighted in light red."
    )

    # Multiselect based on unit codes for now (simple + robust).
    available_codes = list(df_summary["Unit Code"].unique())
    default_selection = list(st.session_state["selected_unit_codes"]) or []

    selected_codes = st.multiselect(
        "Units selected for withdrawal:",
        options=available_codes,
        default=default_selection,
    )

    st.session_state["selected_unit_codes"] = set(selected_codes)
    selected_code_set = st.session_state["selected_unit_codes"]
else:
    # Full course – no specific selection
    selected_code_set = set()

# Display summary with styling for selected rows
styled_summary = style_selected_rows(df_summary, selected_code_set)
st.dataframe(styled_summary, use_container_width=True)

# Build the subset summary for reporting
if scope == "Full course":
    report_units = summary.units
else:
    selected_codes = st.session_state["selected_unit_codes"]
    report_units = [u for u in summary.units if u.unit_code in selected_codes]

from app.models import WorkbookUnitSummary  # import here to avoid circular issues

report_summary = WorkbookUnitSummary(
    units=report_units,
    account_balance=summary.account_balance,
)

if scope == "Specific units" and not report_units:
    st.warning("No units selected. The financial report will be empty until you select at least one unit.")

# -------------------------------------------------------------------
# Step 3 – Date Requested & Financial Report
# -------------------------------------------------------------------

st.header("Step 3 – Financial Report")

date_str = st.text_input(
    "Date requested (student's withdrawal request date) – format dd/mm/yyyy:",
    value="",
    placeholder="dd/mm/yyyy",
)

date_requested: Optional[date] = None
if date_str:
    parsed = _parse_ddmmyyyy(date_str)
    if parsed is None:
        st.error("Please enter the date in dd/mm/yyyy format (e.g. 11/12/2025).")
    else:
        date_requested = parsed

report_text: Optional[str] = None
financial_breakdown: Optional[FinancialBreakdown] = None

if date_requested is not None:
    if scope == "Specific units" and not report_units:
        st.info("Select at least one unit above to calculate the financial impact.")
    else:
        try:
            report_text, financial_breakdown = generate_financial_report(
                report_summary,
                date_requested,
            )
            st.subheader("Financial Report")
            st.code(report_text, language="text")
        except Exception as e:
            st.error(f"Error generating financial report: {e}")

# -------------------------------------------------------------------
# Step 4 – Special Circumstances Investigation Report
# -------------------------------------------------------------------

st.header("Step 4 – Special Circumstances Investigation Report")

use_special_circumstances = st.checkbox(
    "This is a Special Circumstances case (generate investigation report)",
    value=False,
)

special_report_text: Optional[str] = None

if use_special_circumstances:
    st.markdown(
        "Upload the student's supporting documents (screenshots, letters, medical certificates, etc.)."
    )
    uploaded_supporting = st.file_uploader(
        "Supporting documents",
        type=["png", "jpg", "jpeg", "pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="supporting_docs_uploader",
    )

    support_paths = []
    if uploaded_supporting:
        for f in uploaded_supporting:
            saved_path = _save_uploaded_file(SUPPORTING_UPLOAD_DIR, f)
            support_paths.append(saved_path)

    notes = st.text_area(
        "Additional case officer notes (optional)",
        value="",
        placeholder="Any context you want the model to consider.",
        height=120,
    )

    can_generate_sc = (
        date_requested is not None
        and report_text is not None
    )

    if not can_generate_sc:
        st.info(
            "To generate the Special Circumstances Investigation Report, please ensure "
            "you have entered a valid Date Requested and the Financial Report has been generated."
        )

    if st.button(
        "Generate Investigation Report",
        type="primary",
        disabled=not can_generate_sc,
    ):
        try:
            sc_inputs = SpecialCircumstancesInputs(
                date_requested=date_requested,  # type: ignore[arg-type]
                workbook_summary=report_summary,
                financial_report_text=report_text or "",
                supporting_document_paths=support_paths,
                case_officer_notes=notes or None,
                student_name=None,
                student_number=None,
                program_name=None,
                campus=None,
                study_period_description=None,
                financial_breakdown=financial_breakdown,
            )

            result = generate_special_circumstances_report(sc_inputs)
            special_report_text = result.report_text

            st.subheader("Special Circumstances Investigation Report")
            st.code(special_report_text, language="text")

        except Exception as e:
            st.error(
                "There was an error generating the Special Circumstances report. "
                f"Details: {e}"
            )

# End of file
