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
from .pdf_to_images import convert_pdf_to_images  # NEW: PDF → image converter

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

    Returns:
      SpecialCircumstancesResult(report_text=...)
    """

    instructions_text = _load_instructions_text(instructions_path)
    case_text = _build_case_text_payload(inputs)

    # Split documents for images and text
    images, txts, docxs, pdfs = _split_docs_by_type(inputs.supporting_document_paths)

    # NEW: convert each PDF to one or more page images for the vision model
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
