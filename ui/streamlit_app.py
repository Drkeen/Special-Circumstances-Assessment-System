# ui/streamlit_app.py
from pathlib import Path

import pandas as pd
import streamlit as st

from app.excel_reader import load_unit_summary_from_files

# Resolve repo root: ui/ is one level below root
BASE_DIR = Path(__file__).resolve().parent.parent
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

st.sidebar.write("Data folder:", data_dir)

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
rows_for_table = [
    {
        "Unit Code": u.unit_code,
        "Unit Status": u.status,
        "Unit Start Date": u.start_date,           # will display as date
        "Unit Recorded Hours": u.recorded_hours,
        "Unit Price": float(u.unit_price) if u.unit_price is not None else None,
        "Liability Category": u.liability_category,
    }
    for u in summary.units
]

df = pd.DataFrame(rows_for_table)

st.subheader(f"Unit Summary – {student_choice}")
st.dataframe(df, use_container_width=True)

# Optional: download as CSV
csv = df.to_csv(index=False)
st.download_button(
    label="Download as CSV",
    data=csv,
    file_name=f"{student_choice}_unit_summary.csv",
    mime="text/csv",
)
