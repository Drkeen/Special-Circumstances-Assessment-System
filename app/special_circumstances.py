# app/special_circumstances.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional
from base64 import b64encode
from decimal import Decimal

from openai import OpenAI

from .models import WorkbookUnitSummary, UnitSummaryRow
from .pdf_to_images import convert_pdf_to_images  # PDF → image converter

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

# Filetype groups
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
TEXT_EXTS = {".txt"}
DOCX_EXTS = {".docx"}
PDF_EXTS = {".pdf"}


@dataclass
class SpecialCircumstancesInputs:
    """
    Container for all data needed to generate a Special Circumstances
    investigation report for ONE case.
    """

    date_requested: date
    workbook_summary: WorkbookUnitSummary
    financial_report_text: str

    # Local file paths to supporting docs (any of the supported types)
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
# Helpers – instructions, dates, units
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
      ACMGEN301 | Status: Enrolled | Start: 14/07/2025 | Latest engagement: 20/08/2025 | Hours: 50.0 | Price: $527.00 | Liability: VFH
    """
    if not units:
        return "No units in scope for this assessment."

    lines = []
    for u in units:
        price: Optional[Decimal] = u.unit_price
        price_str = f"${price:.2f}" if price is not None else "N/A"
        hours_str = f"{u.recorded_hours:.2f}" if u.recorded_hours is not None else "N/A"

        latest_eng = getattr(u, "latest_engagement_date", None)
        latest_str = _format_date(latest_eng) if latest_eng else "N/A"

        lines.append(
            f"{u.unit_code} | Status: {u.status} | Start: {_format_date(u.start_date)} | "
            f"Latest engagement: {latest_str} | Recorded hours: {hours_str} | "
            f"Price: {price_str} | Liability: {u.liability_category}"
        )
    return "\n".join(lines)


# -------------------------------------------------------------------
# Helpers – file handling & text extraction
# -------------------------------------------------------------------


def _encode_image_to_data_url(path: Path) -> Optional[str]:
    """
    Encode an image file as a data URL for use with OpenAI vision.

    Returns a string like:
      data:image/png;base64,....

    If the extension is not an image, returns None.
    """
    ext = path.suffix.lower()
    if ext not in IMAGE_EXTS:
        return None

    if ext == ".png":
        mime = "image/png"
    else:
        mime = "image/jpeg"

    data = path.read_bytes()
    b64 = b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _extract_text_from_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_text_from_docx(path: Path) -> str:
    """
    Extract text from a .docx file using python-docx.
    """
    try:
        from docx import Document  # type: ignore
    except ImportError:
        # Library not installed – return empty text
        return ""

    try:
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception:
        return ""


def _extract_text_from_pdf(path: Path) -> str:
    """
    Extract text from a PDF using PyPDF2.

    NOTE:
    - This works well for text-based PDFs.
    - For screenshot-only PDFs, text may be empty.
    """
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except ImportError:
        return ""

    try:
        reader = PdfReader(str(path))
        texts: List[str] = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                texts.append(t)
        return "\n\n".join(texts)
    except Exception:
        return ""


def _split_docs_by_type(
    paths: List[Path],
) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    """
    Split document paths into (images, txt, docx, pdf).
    """
    images: list[Path] = []
    txts: list[Path] = []
    docxs: list[Path] = []
    pdfs: list[Path] = []

    for p in paths:
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            images.append(p)
        elif ext in TEXT_EXTS:
            txts.append(p)
        elif ext in DOCX_EXTS:
            docxs.append(p)
        elif ext in PDF_EXTS:
            pdfs.append(p)
        # other extensions are ignored for now

    return images, txts, docxs, pdfs


def _build_supporting_docs_text(
    txt_paths: List[Path],
    docx_paths: List[Path],
    pdf_paths: List[Path],
    max_chars_per_doc: int = 2000,
) -> str:
    """
    Build a block of text containing extracted contents of txt/docx/pdf files.

    We truncate each document's text to avoid blowing out the context window.
    """
    sections: List[str] = []

    def add_doc_section(label: str, path: Path, content: str) -> None:
        if not content.strip():
            sections.append(
                f"----\nDocument: {path.name} ({label})\n[No machine-readable text could be extracted.]"
            )
        else:
            truncated = content.strip()
            if len(truncated) > max_chars_per_doc:
                truncated = truncated[:max_chars_per_doc] + "\n[Text truncated...]"
            sections.append(
                f"----\nDocument: {path.name} ({label})\nExtracted text:\n{truncated}"
            )

    # TXT files
    for p in txt_paths:
        content = _extract_text_from_txt(p)
        add_doc_section("TXT", p, content)

    # DOCX files
    for p in docx_paths:
        content = _extract_text_from_docx(p)
        add_doc_section("DOCX", p, content)

    # PDFs
    for p in pdf_paths:
        content = _extract_text_from_pdf(p)
        add_doc_section("PDF", p, content)

    if not sections:
        return "No machine-readable text could be extracted from supporting documents."

    return "\n\n".join(sections)


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

    # Split docs by type
    images, txts, docxs, pdfs = _split_docs_by_type(inputs.supporting_document_paths)

    # Supporting documentation – filename view
    if inputs.supporting_document_paths:
        docs_lines = []
        for p in inputs.supporting_document_paths:
            ext = p.suffix.lower()
            if ext in IMAGE_EXTS:
                kind = "image"
            elif ext in TEXT_EXTS:
                kind = "txt"
            elif ext in DOCX_EXTS:
                kind = "docx"
            elif ext in PDF_EXTS:
                kind = "pdf"
            else:
                kind = "other"
            docs_lines.append(f"- {p.name} ({kind})")
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

    # Extracted text from txt/docx/pdf
    docs_text_block = _build_supporting_docs_text(txts, docxs, pdfs)

    return f"""
{meta_block}

UNITS IN SCOPE
--------------
{units_block}

SUPPORTING DOCUMENTS (FILENAMES & TYPES)
----------------------------------------
{docs_block}

SUPPORTING DOCUMENT CONTENT (MACHINE-EXTRACTED TEXT)
----------------------------------------------------
{docs_text_block}

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
# Helpers – financial report post-processing
# -------------------------------------------------------------------


def _extract_section2_from_financial_report(
    financial_report_text: str,
) -> tuple[str, Optional[str]]:
    """
    Best-effort extraction of the 'Section 2' view (after PreEASD EWID reversals)
    and its Adjusted Start Account Balance from the financial report text.

    Current heuristic:
      - Look for a line starting with 'Adjusted Start Account Balance:'.
      - Take the block from the previous blank line up to the end as Section 2.
      - Extract the amount that follows 'Adjusted Start Account Balance:'.

    If the markers aren't found, we fall back to using the full financial_report_text
    and no separate adjusted_start_balance.
    """
    lines = financial_report_text.splitlines()

    adj_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Adjusted Start Account Balance:"):
            adj_idx = i
            break

    if adj_idx is None:
        # No clear Section 2 marker – just return the whole thing.
        section2_text = financial_report_text.strip()
        return section2_text, None

    # Extract adjusted start balance from that line
    adj_line = lines[adj_idx]
    m = re.search(r"Adjusted Start Account Balance:\s*(.+)", adj_line)
    adjusted_start_balance = m.group(1).strip() if m else None

    # Find the previous blank line to mark the start of Section 2
    start_idx = 0
    for j in range(adj_idx - 1, -1, -1):
        if not lines[j].strip():
            start_idx = j + 1
            break
        if j == 0:
            start_idx = 0

    section2_lines = lines[start_idx:]
    section2_text = "\n".join(section2_lines).strip()
    if not section2_text:
        section2_text = financial_report_text.strip()

    return section2_text, adjusted_start_balance

def _format_financial_impact_for_document(section2_text: str) -> str:
    """
    From the 'Section 2' text of the financial report, build the block for
    item 11 (Financial Impact):

    - Include lines that describe units (contain 'units:' case-insensitive).
    - Optionally include 'Total Financial Impact:'.
    - Include 'Adjusted End Account Balance:'.
    - Exclude 'Adjusted Start Account Balance:'.

    If we can't find anything meaningful, fall back to the original section2_text.
    """
    lines = section2_text.splitlines()
    impact_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Skip Adjusted Start
        if "adjusted start account balance" in stripped.lower():
            continue

        lower = stripped.lower()

        # Unit lines (e.g. 'Fee Waiver units:', 'PostEASD EWID units:')
        if "units:" in lower:
            impact_lines.append(stripped)
            continue

        # Total Financial Impact line – keep it, it's still useful
        if stripped.startswith("Total Financial Impact:"):
            impact_lines.append(stripped)
            continue

        # Adjusted End Account Balance – keep this
        if stripped.startswith("Adjusted End Account Balance:"):
            impact_lines.append(stripped)
            continue

    if not impact_lines:
        # Fallback: if for some reason the patterns don't match, keep original
        return section2_text.strip()

    return "\n".join(impact_lines)


def _format_liability_category(summary: WorkbookUnitSummary) -> str:
    """
    Combine liability categories from units in scope.
    If there's exactly one, use it. If multiple, show them comma-separated.
    """
    categories = sorted(
        {u.liability_category for u in summary.units if u.liability_category}
    )
    if not categories:
        return "N/A"
    if len(categories) == 1:
        return categories[0]
    return ", ".join(categories)


def _build_investigation_document(
    inputs: SpecialCircumstancesInputs,
    ai_analysis_text: str,
) -> str:
    """
    Build the final investigation document in the strict format requested:

    Hi [leave blank],

    Please find below a COE/refund recommendation for approval:
    ...
    """

    easd = _get_easd(inputs.workbook_summary)
    application_date_str = _format_date(inputs.date_requested)
    easd_str = _format_date(easd)

    # Liability category derived from the units in scope
    liability_category_str = _format_liability_category(inputs.workbook_summary)

    # Has evidence been supplied:
    # In the current UI, this function is only called when Special Circumstances
    # has been checked. So:
    #   - Yes if any docs uploaded
    #   - No  if none uploaded
    has_docs = bool(inputs.supporting_document_paths)
    has_evidence_str = "Yes" if has_docs else "No"

    # Extract "Section 2" view and Adjusted Start Account Balance from the
    # financial report text, where possible.
    section2_text, adjusted_start_balance = _extract_section2_from_financial_report(
        inputs.financial_report_text
    )

    # Account Balance field – prefer Adjusted Start Account Balance from Section 2,
    # fall back to overall account_balance if available, otherwise N/A.
    if adjusted_start_balance:
        account_balance_str = adjusted_start_balance
    else:
        if inputs.workbook_summary.account_balance is not None:
            bal = inputs.workbook_summary.account_balance
            # Format similar to the financial report (no thousands commas)
            sign = "-" if bal < 0 else ""
            account_balance_str = f"{sign}${abs(bal):.2f}"
        else:
            account_balance_str = "N/A"

    # Build the strict template
    lines: List[str] = []

    lines.append("Hi ,")
    lines.append("")
    lines.append("Please find below a COE/refund recommendation for approval:")
    lines.append("")
    lines.append("1. Course Code and Name: ")
    lines.append(f"")
    lines.append(f"2. Liability Category: {liability_category_str}")
    lines.append("")
    lines.append("3. Citizenship: ")
    lines.append("")
    lines.append(f"4. Application date: {application_date_str}")
    lines.append("")
    lines.append(f"5. EASD: {easd_str}")
    lines.append("")
    lines.append("6. Census Date: ")
    lines.append("")
    lines.append("7. Reason for COE:")
    lines.append("")
    lines.append(ai_analysis_text.strip())
    lines.append("")
    lines.append(
        f"8. Has evidence been supplied: {has_evidence_str} "
        "(If Special Circumstances was not selected, this should be recorded as 'NA'.)"
    )
    lines.append("")
    lines.append("9. AHPRA Verified?: ")
    lines.append("")
    lines.append(f"10. Account Balance: {account_balance_str}")
    lines.append("")
    impact_block = _format_financial_impact_for_document(section2_text)

    lines.append("11. Financial Impact:")
    lines.append("")
    lines.append(impact_block)
    lines.append("")
    lines.append("12. Recommendation: ")
    lines.append("")
    lines.append(
        "By sending through this request, I acknowledge that I am aware of "
        "TAFE Queensland Procedure 650 Student Fees, the TAFE Queensland Financial "
        "Management Delegations and will be processing the above transaction in SMS "
        "as per the FACT SHEET - Fee Waivers, Reversals and Overrides"
    )
    lines.append(" ")
    lines.append("Kind regards,")
    lines.append("")
    lines.append("")

    return "\n".join(lines).rstrip()


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
    with both text and supporting documents (images + extracted text).

    - Reads instructions.txt (which you can keep updating).
    - Uses the workbook summary + financial report + supporting docs.
    - Asks the model to:
        * Infer the Reason for COE from the provided options.
        * Assess special circumstances against TAFE Queensland criteria.
        * Build a timeline of events starting at the EASD.
        * Provide a short impact summary.

    The AI output is then embedded into a strict, TAFE-style template to form
    the final document that you paste into your internal system.
    """

    instructions_text = _load_instructions_text(instructions_path)
    case_text = _build_case_text_payload(inputs)

    # Split documents for images and text
    images, txts, docxs, pdfs = _split_docs_by_type(inputs.supporting_document_paths)

    # Convert each PDF to one or more page images for the vision model
    pdf_image_paths: List[Path] = []
    for pdf in pdfs:
        try:
            page_images = convert_pdf_to_images(pdf)
            pdf_image_paths.extend(page_images)
        except Exception:
            # If conversion fails, we just skip images for that PDF
            continue

    # Combine original images + PDF-derived images
    all_image_paths: List[Path] = list(images) + list(pdf_image_paths)

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
                "Return your analysis as a concise but complete narrative, suitable to be placed "
                "under the heading 'Reason for COE' in an internal recommendation document.\n\n"
                "CASE DATA (TEXTUAL SUMMARY):\n\n"
                f"{case_text}"
            ),
        }
    ]

    # Attach each image document for the vision model (PNG/JPEG + PDF pages)
    for path in all_image_paths:
        data_url = _encode_image_to_data_url(path)
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
                "Investigation analyses that will be inserted into a structured "
                "internal template. Use Australian English and date format dd/mm/yyyy."
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

    ai_analysis_text = response.choices[0].message.content.strip()

    # Wrap the AI analysis into the strict 1–12 template
    final_doc = _build_investigation_document(
        inputs=inputs,
        ai_analysis_text=ai_analysis_text,
    )

    return SpecialCircumstancesResult(report_text=final_doc)
