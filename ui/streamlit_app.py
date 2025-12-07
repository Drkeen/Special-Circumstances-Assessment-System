# ui/streamlit_app.py
from __future__ import annotations

from pathlib import Path
import sys
from datetime import datetime, date
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# Optional: load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# -------------------------------------------------
# Make sure the project root is on sys.path so that
# `import app...` works when Streamlit runs from ui/
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.excel_reader import load_unit_summary_from_files
from app.financial_report import generate_financial_report, FinancialBreakdown
from app.models import WorkbookUnitSummary
from app.special_circumstances import (
    SpecialCircumstancesInputs,
    generate_special_circumstances_report,
)

SAMPLE_DIR = BASE_DIR / "sample_data"
UPLOAD_SPREADSHEET_DIR = BASE_DIR / "uploaded_spreadsheets"
UPLOAD_SUPPORT_DIR = BASE_DIR / "uploaded_supporting_docs"

UPLOAD_SPREADSHEET_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="SCAS – Special Circumstances Assessment", layout="wide")

st.title("Special Circumstances Assessment System (SCAS)")

# -------------------------------------------------
# Sidebar – Date Requested + data source choice
# -------------------------------------------------

st.sidebar.subheader("Application Date")

date_str = st.sidebar.text_input(
    "Date requested (dd/mm/yyyy)",
    placeholder="e.g. 11/12/2025",
)

date_requested: Optional[date] = None
if date_str.strip():
    try:
        date_requested = datetime.strptime(date_str.strip(), "%d/%m/%Y").date()
    except ValueError:
        st.sidebar.error("Please enter a valid date in dd/mm/yyyy format.")

st.sidebar.markdown("---")
data_source = st.sidebar.radio(
    "Data source",
    ["Sample data (demo)", "Upload spreadsheets"],
    index=0,
)

# -------------------------------------------------
# Determine file paths based on data source
# -------------------------------------------------

study_plan_path: Optional[Path] = None
unit_engagement_path: Optional[Path] = None
student_account_path: Optional[Path] = None

def classify_excel_file(temp_path: Path) -> Optional[str]:
    """
    Classify an Excel file as 'study_plan', 'unit_engagement', 'student_account', or None
    based on header markers found in the first few rows.
    """
    try:
        preview = pd.read_excel(temp_path, nrows=10, header=None, engine="openpyxl")
    except Exception:
        return None

    text_cells: List[str] = [
        str(v) for v in preview.values.ravel() if not pd.isna(v)
    ]
    lower_cells = [s.lower() for s in text_cells]

    def score_for(markers: List[str]) -> int:
        score = 0
        for m in markers:
            m_low = m.lower()
            if any(m_low in cell for cell in lower_cells):
                score += 1
        return score

    scores: Dict[str, int] = {
        "study_plan": score_for(["Spk Cd", "SSP Status", "Enrolment Activity Start Date"]),
        "unit_engagement": score_for(["Curriculum Item", "Unit Start Date", "Recorded"]),
        "student_account": score_for(["SSP Spk Cd", "Txn Amt", "Txb Amt", "Unalloc Amt"]),
    }

    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        return None
    return best_type


if data_source == "Sample data (demo)":
    # Choose which sample dataset
    student_choice = st.sidebar.selectbox(
        "Select sample dataset",
        ["student_a", "student_b", "student_c"],
        index=0,
    )

    data_dir = SAMPLE_DIR / student_choice / "data"
    st.sidebar.write("Data folder:", f"`{data_dir}`")

    study_plan_path = data_dir / "study_plan.xlsx"
    unit_engagement_path = data_dir / "unit_engagement.xlsx"
    student_account_path = data_dir / "student_account.xlsx"

    missing_files = [
        p
        for p in [study_plan_path, unit_engagement_path, student_account_path]
        if not p.exists()
    ]
    if missing_files:
        st.error(
            "Missing sample files: "
            + ", ".join(str(p.name) for p in missing_files)
        )
        st.stop()

else:
    st.markdown("### Step 1 – Upload TechOne Spreadsheets")
    st.write(
        "Upload the three Excel exports for this student:\n"
        "- Study Plan\n"
        "- Unit Engagement\n"
        "- Student Account\n\n"
        "**Tip:** You can drag-and-drop all three files at once."
    )

    uploaded_sheets = st.file_uploader(
        "Upload the 3 spreadsheets (.xlsx)",
        type=["xlsx"],
        accept_multiple_files=True,
    )

    if not uploaded_sheets:
        st.info("Please upload the three Excel files to proceed.")
        st.stop()

    # Save and classify uploaded spreadsheets
    detected_paths: Dict[str, Path] = {}
    classification_messages: List[str] = []

    for f in uploaded_sheets:
        # Save to disk
        target_path = UPLOAD_SPREADSHEET_DIR / f.name
        target_path.write_bytes(f.getbuffer())

        sheet_type = classify_excel_file(target_path)
        classification_messages.append(
            f"- `{f.name}` → {sheet_type or 'UNKNOWN'}"
        )

        if sheet_type is None:
            continue

        if sheet_type in detected_paths:
            # Duplicate type – that's a problem
            st.error(
                f"Detected more than one file for '{sheet_type}'.\n\n"
                "Detected so far:\n"
                + "\n".join(classification_messages)
            )
            st.stop()

        detected_paths[sheet_type] = target_path

    st.markdown("**Detected file types:**")
    st.markdown("\n".join(classification_messages))

    # Ensure we have each required type exactly once
    required_types = ["study_plan", "unit_engagement", "student_account"]
    missing_types = [t for t in required_types if t not in detected_paths]

    if missing_types:
        st.error(
            "The following required sheet types were not detected: "
            + ", ".join(missing_types)
            + ".\n\nPlease check that you've uploaded the correct TechOne exports."
        )
        st.stop()

    study_plan_path = detected_paths["study_plan"]
    unit_engagement_path = detected_paths["unit_engagement"]
    student_account_path = detected_paths["student_account"]

# At this point, we have valid paths for all three spreadsheets
assert study_plan_path is not None
assert unit_engagement_path is not None
assert student_account_path is not None

# -------------------------------------------------
# Load & merge into WorkbookUnitSummary
# -------------------------------------------------

summary = load_unit_summary_from_files(
    study_plan_path=study_plan_path,
    unit_engagement_path=unit_engagement_path,
    student_account_path=student_account_path,
)

# -------------------------------------------------
# Scope selection + Unit Summary table
# -------------------------------------------------

st.markdown("### Step 2 – Scope of Withdrawal")
scope = st.radio(
    "What is the student withdrawing from?",
    ["Full Course", "Specific Units"],
    index=0,
    horizontal=True,
)

# Build full Unit Summary DataFrame (all units)
rows_for_table_all = []
for u in summary.units:
    if date_requested is not None:
        days_diff = (date_requested - u.start_date).days
    else:
        days_diff = None

    rows_for_table_all.append(
        {
            "Unit Code": u.unit_code,
            "Unit Status": u.status,
            "Unit Start Date": u.start_date,
            "Date Requested": date_requested,
            "Days Between Request and Start": days_diff,
            "Unit Recorded Hours": u.recorded_hours,
            "Unit Price": float(u.unit_price) if u.unit_price is not None else None,
            "Liability Category": u.liability_category,
            "Latest Engagement Date": u.latest_engagement_date,
        }
    )

df_full = pd.DataFrame(rows_for_table_all)

if data_source == "Sample data (demo)":
    current_case_label = f"Sample dataset – {student_choice}"
else:
    current_case_label = "Uploaded spreadsheets"

st.subheader(f"Unit Summary ({current_case_label})")

if scope == "Full Course":
    st.caption("All units are included in this financial assessment.")
    # Read-only table, no selection needed
    st.dataframe(df_full, use_container_width=True)
    report_units = summary.units
else:
    st.caption(
        "Select one or more rows in the table below. "
        "Selected units will be used to calculate the financial impact."
    )

    event = st.dataframe(
        df_full,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
    )

    selected_rows: List[int] = []
    if hasattr(event, "selection") and hasattr(event.selection, "rows"):
        selected_rows = event.selection.rows

    if selected_rows:
        selected_codes = df_full.iloc[selected_rows]["Unit Code"].tolist()
        report_units = [u for u in summary.units if u.unit_code in selected_codes]
    else:
        report_units = []

# Create a filtered WorkbookUnitSummary for the report logic
report_summary = WorkbookUnitSummary(
    units=report_units,
    account_balance=summary.account_balance,
)

# -------------------------------------------------
# Financial report
# -------------------------------------------------

st.markdown("---")
st.subheader("Step 3 – Financial Report")

report_text: Optional[str] = None
financial_breakdown: Optional[FinancialBreakdown] = None

if date_requested is None:
    st.info("Enter a valid Date Requested in the sidebar to generate the financial report.")
elif scope == "Specific Units" and not report_units:
    st.info("Select at least one row in the Unit Summary to generate the financial report.")
else:
    report_text, financial_breakdown = generate_financial_report(
        report_summary,
        date_requested,
    )
    st.code(report_text, language="text")

# -------------------------------------------------
# Special Circumstances Investigation Report
# -------------------------------------------------

st.markdown("---")
sc_enabled = st.checkbox("Generate Special Circumstances Investigation Report")

if sc_enabled:
    st.subheader("Step 4 – Special Circumstances Investigation Report")

    if date_requested is None:
        st.warning("Please enter a valid Date Requested first.")
    elif report_text is None:
        st.warning("Please generate the Financial Report first (see section above).")
    else:
        st.markdown(
            "**Upload supporting documents** (e.g. medical certificates, employer letters, "
            "QTAC offers, etc.). These can be images, PDFs, Word documents, or text files."
        )

        uploaded_supporting = st.file_uploader(
            "Supporting documents",
            type=["png", "jpg", "jpeg", "pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

        support_paths: List[Path] = []
        if uploaded_supporting:
            for f in uploaded_supporting:
                target_path = UPLOAD_SUPPORT_DIR / f.name
                target_path.write_bytes(f.getbuffer())
                support_paths.append(target_path)

            st.markdown("**Files received:**")
            for p in support_paths:
                st.write(f"- {p.name}")
        else:
            st.info("No supporting documents uploaded yet.")

        notes = st.text_area(
            "Case officer notes (optional)",
            placeholder="E.g. summary of phone conversation, clarifications from student, etc.",
        )

        generate_clicked = st.button("Generate Investigation Report")

        if generate_clicked:
            if not support_paths:
                st.warning(
                    "It's strongly recommended to upload at least one supporting document before generating the report."
                )

            with st.spinner("Generating Special Circumstances Investigation Report..."):
                inputs = SpecialCircumstancesInputs(
                    date_requested=date_requested,
                    workbook_summary=report_summary,
                    financial_report_text=report_text,
                    supporting_document_paths=support_paths,
                    case_officer_notes=notes or None,
                    student_name=None,
                    student_number=None,
                    program_name=None,
                    campus=None,
                    study_period_description=None,
                    financial_breakdown=financial_breakdown,
                )


                try:
                    result = generate_special_circumstances_report(inputs)
                    st.markdown("**Investigation Report (AI-generated):**")
                    st.code(result.report_text, language="text")
                except Exception as e:
                    st.error(
                        "There was an error generating the Special Circumstances report. "
                        f"Details: {e}"
                    )

# -------------------------------------------------
# Debug: raw sheets (always show raw, full sheets)
# -------------------------------------------------

with st.expander("Show raw Study Plan"):
    raw_sp = pd.read_excel(study_plan_path, engine="openpyxl", dtype=str)
    st.dataframe(raw_sp, use_container_width=True)

with st.expander("Show raw Unit Engagement"):
    raw_ue = pd.read_excel(unit_engagement_path, engine="openpyxl", dtype=str)
    st.dataframe(raw_ue, use_container_width=True)

with st.expander("Show raw Student Account"):
    raw_sa = pd.read_excel(student_account_path, engine="openpyxl", dtype=str)
    st.dataframe(raw_sa, use_container_width=True)

# Optional: download as CSV (full unit summary)
csv = df_full.to_csv(index=False)
st.download_button(
    label="Download Unit Summary as CSV",
    data=csv,
    file_name="unit_summary.csv",
    mime="text/csv",
)
