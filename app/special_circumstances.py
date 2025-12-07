# app/special_circumstances.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional
from base64 import b64encode
from decimal import Decimal

from openai import OpenAI

from .models import WorkbookUnitSummary, UnitSummaryRow

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Default model – change this to whatever you want in your account
# e.g. "gpt-4.1", "gpt-4o", "gpt-5.1", etc.
DEFAULT_MODEL = os.environ.get("SCAS_LLM_MODEL", "gpt-4.1")

# Where we expect to find instructions.txt by default
DEFAULT_INSTRUCTIONS_PATH = (
    Path(__file__).resolve().parent.parent / "instructions.txt"
)

# Supported doc extensions for vision
SUPPORTED_DOC_EXTS = {".png", ".jpg", ".jpeg", ".pdf"}


@dataclass
class SpecialCircumstancesInputs:
    """
    Container for all data needed to generate a Special Circumstances
    investigation report for ONE case.
    """

    date_requested: date
    workbook_summary: WorkbookUnitSummary
    financial_report_text: str

    # Local file paths to supporting docs (images/PDFs)
    supporting_document_paths: List[Path]

    # Optional extra notes from the case officer (free text)
    case_officer_notes: Optional[str] = None

    # Optional student/case metadata
    student_name: Optional[str] = None
    student_number: Optional[str] = None
    program_name: Optional[str] = None
    campus: Optional[str] = None
    study_period_description: Optional[str] = None  # e.g. "Semester 2 2025"


@dataclass
class SpecialCircumstancesResult:
    """Output of the generator – currently just the report text."""
    report_text: str


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _load_instructions_text(
    instructions_path: Optional[Path] = None,
) -> str:
    """
    Load the contents of instructions.txt.

    You can keep editing instructions.txt and the app will always use
    the latest version next time it runs.
    """
    path = instructions_path or DEFAULT_INSTRUCTIONS_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Instructions file not found at {path}. "
            f"Update DEFAULT_INSTRUCTIONS_PATH in special_circumstances.py "
            f"or pass instructions_path explicitly."
        )

    return path.read_text(encoding="utf-8")


def _format_date(d: Optional[date]) -> str:
    if not d:
        return "N/A"
    return d.strftime("%d/%m/%Y")


def _get_easd(summary: WorkbookUnitSummary) -> Optional[date]:
    """Earliest enrolment activity start date (EASD) from the workbook summary."""
    if not summary.units:
        return None
    return min(u.start_date for u in summary.units)


def _format_units_block(units: List[UnitSummaryRow]) -> str:
    """
    Format the units into a simple text block for the LLM.

    Each line looks like:
      ACMGEN301 | Status: Enrolled | Start: 14/07/2025 | Hours: 50.0 | Price: $527.00 | Liability: VFH
    """
    if not units:
        return "No units in scope for this assessment."

    lines = []
    for u in units:
        price: Optional[Decimal] = u.unit_price
        price_str = f"${price:.2f}" if price is not None else "N/A"
        hours_str = f"{u.recorded_hours:.2f}" if u.recorded_hours is not None else "N/A"
        lines.append(
            f"{u.unit_code} | Status: {u.status} | Start: {_format_date(u.start_date)} | "
            f"Recorded hours: {hours_str} | Price: {price_str} | Liability: {u.liability_category}"
        )
    return "\n".join(lines)


def _encode_file_to_data_url(path: Path) -> Optional[str]:
    """
    Encode an image/PDF file as a data URL for use with OpenAI vision.

    Returns a string like:
      data:image/png;base64,....

    If the extension is not supported, returns None.
    """
    ext = path.suffix.lower()
    if ext not in SUPPORTED_DOC_EXTS:
        return None

    mime = None
    if ext in {".png"}:
        mime = "image/png"
    elif ext in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif ext == ".pdf":
        # Many student PDFs are screenshots in a PDF wrapper.
        # OpenAI's vision models can usually handle this as a document.
        mime = "application/pdf"

    if mime is None:
        return None

    data = path.read_bytes()
    b64 = b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_case_text_payload(inputs: SpecialCircumstancesInputs) -> str:
    """
    Build the textual part of the 'user' message content describing the case.
    This does NOT include the images themselves – those are attached separately.
    """

    easd = _get_easd(inputs.workbook_summary)

    # Basic student / case metadata
    meta_lines = [
        "CASE CONTEXT",
        "-------------",
        f"Date Requested (withdrawal request date): {_format_date(inputs.date_requested)}",
        f"Earliest Enrolment Activity Start Date (EASD): {_format_date(easd)}",
    ]

    if inputs.student_name:
        meta_lines.append(f"Student name: {inputs.student_name}")
    if inputs.student_number:
        meta_lines.append(f"Student number: {inputs.student_number}")
    if inputs.program_name:
        meta_lines.append(f"Program: {inputs.program_name}")
    if inputs.campus:
        meta_lines.append(f"Campus: {inputs.campus}")
    if inputs.study_period_description:
        meta_lines.append(
            f"Relevant study period: {inputs.study_period_description}"
        )

    meta_block = "\n".join(meta_lines)

    # Units in scope
    units_block = _format_units_block(inputs.workbook_summary.units)

    # Supporting documentation – filename view
    if inputs.supporting_document_paths:
        docs_lines = [
            f"- {p.name} (loaded from {p})"
            for p in inputs.supporting_document_paths
        ]
        docs_block = "\n".join(docs_lines)
    else:
        docs_block = "No supporting documents were found for this case."

    # Case officer notes
    notes_block = (
        inputs.case_officer_notes.strip()
        if inputs.case_officer_notes and inputs.case_officer_notes.strip()
        else "No additional notes provided by the case officer."
    )

    # Financial report already generated
    financial_block = inputs.financial_report_text.strip()

    return f"""
{meta_block}

UNITS IN SCOPE
--------------
{units_block}

SUPPORTING DOCUMENTS (FILENAMES)
--------------------------------
{docs_block}

CASE OFFICER NOTES
------------------
{notes_block}

FINANCIAL IMPACT SUMMARY (PRE-COMPUTED)
---------------------------------------
This section is the output of the financial engine. Use it as factual context
for dates, unit categories (PreEASD EWID / PostEASD EWID / Fee Waiver), and
account balances, but do NOT recalculate the dollar amounts.

{financial_block}
""".strip()


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------


def generate_special_circumstances_report(
    inputs: SpecialCircumstancesInputs,
    *,
    instructions_path: Optional[Path] = None,
    model: Optional[str] = None,
) -> SpecialCircumstancesResult:
    """
    Generate a Special Circumstances Investigation Report by calling OpenAI
    with both text and supporting documents (images/PDFs).

    - Reads instructions.txt (which you can keep updating).
    - Uses the workbook summary + financial report + supporting docs.
    - Asks the model to:
        * Infer the Reason for COE from the provided options.
        * Assess special circumstances against TAFE Queensland criteria.
        * Build a timeline of events starting at the EASD.
        * Provide a short impact summary.

    Returns:
      SpecialCircumstancesResult(report_text=...)
    """

    instructions_text = _load_instructions_text(instructions_path)
    case_text = _build_case_text_payload(inputs)

    # Build the multimodal content list for the 'user' message
    user_content: List[dict] = [
        {
            "type": "text",
            "text": (
                "Using the case data and supporting documents below, please:\n"
                "1. Examine the documents and infer the most appropriate Reason for COE "
                "   from this list (choose exactly ONE and name it explicitly):\n"
                "   - Medical reasons\n"
                "   - Family/personal reasons\n"
                "   - Employment related reasons\n"
                "   - Course related reasons\n"
                "   - QTAC or higher education\n"
                "2. Assess whether the student's circumstances meet TAFE Queensland's "
                "   special circumstances criteria (beyond control; timing after start of study "
                "   or census/EASD; and impact on ability to complete study requirements).\n"
                "3. Create a clear TIMELINE OF EVENTS starting from the EASD and including "
                "   key dates: onset/worsening of circumstances, supporting document dates, "
                "   and withdrawal request date.\n"
                "4. Provide a short IMPACT SUMMARY focusing on the relevant study period.\n"
                "5. Do NOT make an explicit approval/refusal decision – your role is analysis only.\n\n"
                "CASE DATA (TEXTUAL SUMMARY):\n\n"
                f"{case_text}"
            ),
        }
    ]

    # Attach each supporting document as an image/PDF for the vision model
    for path in inputs.supporting_document_paths:
        data_url = _encode_file_to_data_url(path)
        if not data_url:
            continue
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                },
            }
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an experienced TAFE Queensland case officer. "
                "You write clear, professional, and concise Special Circumstances "
                "Investigation Reports that will be pasted into an internal system. "
                "Use Australian English and date format dd/mm/yyyy."
            ),
        },
        {
            "role": "developer",
            "content": (
                "Follow these policy rules and drafting instructions exactly. "
                "You must assess special circumstances against the criteria, "
                "and you must not contradict the definitions.\n\n"
                + instructions_text
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    client = OpenAI()  # Uses OPENAI_API_KEY from environment by default
    model_id = model or DEFAULT_MODEL

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.2,  # keep it precise and policy-aligned
    )

    report_text = response.choices[0].message.content.strip()

    return SpecialCircumstancesResult(report_text=report_text)
