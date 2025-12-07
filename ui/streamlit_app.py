# ui/streamlit_app.py
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
    placeholder="e.g. 11/02/2026",
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

# Load & merge
summary = load_unit_summary_from_files(
    study_plan_path=study_plan_path,
    unit_engagement_path=unit_engagement_path,
    student_account_path=student_account_path,
)

# Convert to DataFrame for display
rows_for_table = []
for u in summary.units:
    if date_requested:
        days_diff = (date_requested - u.start_date).days
    else:
        days_diff = None

    rows_for_table.append(
        {
            "Unit Code": u.unit_code,
            "Unit Status": u.status,
            "Unit Start Date": u.start_date,                # date
            "Date Requested": date_requested,               # same for all rows, user-entered
            "Days Between Request and Start": days_diff,    # int (or None)
            "Unit Recorded Hours": u.recorded_hours,
            "Unit Price": float(u.unit_price) if u.unit_price is not None else None,
            "Liability Category": u.liability_category,
        }
    )

df = pd.DataFrame(rows_for_table)

st.subheader(f"Unit Summary – {student_choice}")
st.dataframe(df, use_container_width=True)

# Optional: show raw sheets in expanders for debugging
with st.expander("Show raw Study Plan"):
    raw_sp = pd.read_excel(study_plan_path, engine="openpyxl", dtype=str)
    st.dataframe(raw_sp, use_container_width=True)

with st.expander("Show raw Unit Engagement"):
    raw_ue = pd.read_excel(unit_engagement_path, engine="openpyxl", dtype=str)
    st.dataframe(raw_ue, use_container_width=True)

with st.expander("Show raw Student Account"):
    raw_sa = pd.read_excel(student_account_path, engine="openpyxl", dtype=str)
    st.dataframe(raw_sa, use_container_width=True)

# Optional: download as CSV
csv = df.to_csv(index=False)
st.download_button(
    label="Download as CSV",
    data=csv,
    file_name=f"{student_choice}_unit_summary.csv",
    mime="text/csv",
)
