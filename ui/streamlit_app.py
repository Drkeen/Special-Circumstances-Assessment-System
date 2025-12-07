# ui/streamlit_app.py
from __future__ import annotations

from pathlib import Path
import sys
from datetime import datetime, date

import pandas as pd
import streamlit as st

# -------------------------------------------------
# Make sure the project root is on sys.path so that
# `import app...` works when Streamlit runs from ui/
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.excel_reader import load_unit_summary_from_files
from app.financial_report import generate_financial_report_text
from app.models import WorkbookUnitSummary
from app.special_circumstances import (
    SpecialCircumstancesInputs,
    generate_special_circumstances_report,
)

SAMPLE_DIR = BASE_DIR / "sample_data"

st.set_page_config(page_title="SCAS Sample Data Viewer", layout="wide")

st.title("Special Circumstances Assessment System – Sample Data Test")

# Sidebar: choose which sample student
student_choice = st.sidebar.selectbox(
    "Select sample dataset",
    ["student_a", "student_b", "student_c"],
    index=0,
)

data_dir = SAMPLE_DIR / student_choice / "data"

study_plan_path = data_dir / "study_plan.xlsx"
unit_engagement_path = data_dir / "unit_engagement.xlsx"
student_account_path = data_dir / "student_account.xlsx"

st.sidebar.write("Data folder:", f"`{data_dir}`")

# Sidebar: Date Requested (application date)
st.sidebar.markdown("---")
st.sidebar.subheader("Application Date")

date_str = st.sidebar.text_input(
    "Date requested (dd/mm/yyyy)",
    placeholder="e.g. 11/12/2025",
)

date_requested: date | None = None
if date_str.strip():
    try:
        date_requested = datetime.strptime(date_str.strip(), "%d/%m/%Y").date()
    except ValueError:
        st.sidebar.error("Please enter a valid date in dd/mm/yyyy format.")

# Check files exist
missing_files = [
    p for p in [study_plan_path, unit_engagement_path, student_account_path] if not p.exists()
]

if missing_files:
    st.error(f"Missing files: {', '.join(str(p.name) for p in missing_files)}")
    st.stop()

# Load & merge (full workbook summary)
summary = load_unit_summary_from_files(
    study_plan_path=study_plan_path,
    unit_engagement_path=unit_engagement_path,
    student_account_path=student_account_path,
)

# ---------------------------------
# Scope selection (main area)
# ---------------------------------
st.markdown("### Scope")
scope = st.radio(
    "What is the student withdrawing from?",
    ["Full Course", "Specific Units"],
    index=0,
    horizontal=True,
)

# ---------------------------------
# Build full Unit Summary DataFrame (all units)
# ---------------------------------
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
        }
    )

df_full = pd.DataFrame(rows_for_table_all)

st.subheader(f"Unit Summary – {student_choice}")

# ---------------------------------
# Show table & capture selection
# ---------------------------------
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

    # st.dataframe with row selection and built-in highlight
    event = st.dataframe(
        df_full,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
    )

    # Get selected row indices (if any)
    selected_rows = []
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

# -----------------------------
# Financial report section
# -----------------------------
st.markdown("---")
st.subheader("Financial Report")

report_text: str | None = None

if date_requested is None:
    st.info("Enter a valid Date Requested in the sidebar to generate the financial report.")
elif scope == "Specific Units" and not report_units:
    st.info("Select at least one row in the Unit Summary to generate the financial report.")
else:
    report_text = generate_financial_report_text(report_summary, date_requested)
    st.code(report_text, language="text")

# -----------------------------
# Special Circumstances section
# -----------------------------
st.markdown("---")
sc_enabled = st.checkbox("Generate Special Circumstances Investigation Report")

if sc_enabled:
    st.subheader("Special Circumstances Investigation Report")

    if date_requested is None:
        st.warning("Please enter a valid Date Requested first.")
    elif report_text is None:
        st.warning("Please generate the Financial Report first (see section above).")
    else:
        # Locate supporting_documents for this student
        support_dir = SAMPLE_DIR / student_choice / "supporting_documents"
        support_files: list[Path] = []
        if support_dir.exists() and support_dir.is_dir():
            for p in support_dir.iterdir():
                if p.is_file() and p.suffix.lower() in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".pdf",
                    ".txt",
                    ".docx",
                }:
                    support_files.append(p)


        if not support_files:
            st.info(
                "No supporting documents found in "
                f"`{support_dir}`.\n\n"
                "For testing, place PNG/JPEG/PDF files in that folder."
            )
        else:
            st.markdown("**Supporting documents found:**")
            for p in support_files:
                st.write(f"- {p.name}")

        notes = st.text_area(
            "Case officer notes (optional)",
            placeholder="E.g. summary of phone conversation, clarifications from student, etc.",
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            generate_clicked = st.button("Generate Investigation Report")

        if generate_clicked:
            with st.spinner("Generating Special Circumstances Investigation Report..."):
                inputs = SpecialCircumstancesInputs(
                    date_requested=date_requested,
                    workbook_summary=report_summary,
                    financial_report_text=report_text,
                    supporting_document_paths=support_files,
                    case_officer_notes=notes or None,
                    # You can wire these up from other UI fields later if you like
                    student_name=None,
                    student_number=None,
                    program_name=None,
                    campus=None,
                    study_period_description=None,
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

# -----------------------------
# Debug: raw sheets (always show raw, full sheets)
# -----------------------------
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
    file_name=f"{student_choice}_unit_summary.csv",
    mime="text/csv",
)
