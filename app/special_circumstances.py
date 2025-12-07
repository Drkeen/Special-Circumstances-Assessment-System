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
from .financial_report import FinancialBreakdown

# Category names must match financial_report
CATEGORY_PRE = "PreEASD EWID"
CATEGORY_POST = "PostEASD EWID"
CATEGORY_FEE = "Fee Waiver"
ADMIN_FEE = Decimal("100.00")

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

DEFAULT_MODEL = os.environ.get("SCAS_LLM_MODEL", "gpt-4.1")

DEFAULT_INSTRUCTIONS_PATH = (
    Path(__file__).resolve().parent.parent / "instructions.txt"
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
TEXT_EXTS = {".txt"}
DOCX_EXTS = {".docx"}
PDF_EXTS = {".pdf"}


@dataclass
class SpecialCircumstancesInputs:
    """
    All inputs needed to generate the Special Circumstances Investigation Report.
    """
    date_requested: date
    workbook_summary: WorkbookUnitSummary
    financial_report_text: str
    supporting_document_paths: List[Path]

    case_officer_notes: Optional[str] = None
    student_name: Optional[str] = None
    student_number: Optional[str] = None
    program_name: Optional[str] = None
    campus: Optional[str] = None
    study_period_description: Optional[str] = None

    # Structured financials from financial_report.compute_financial_breakdown
    financial_breakdown: Optional[FinancialBreakdown] = None


@dataclass
class SpecialCircumstancesResult:
    report_text: str


# -------------------------------------------------------------------
# Basic helpers
# -------------------------------------------------------------------


def _load_instructions_text(
    instructions_path: Optional[Path] = None,
) -> str:
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


def _format_currency(amount: Optional[Decimal]) -> str:
    """
    Shared currency formatting:
      - sign in front of $
      - no thousands comma
      - 2 decimal places
    """
    if amount is None:
        return "N/A"
    q = amount.quantize(Decimal("0.01"))
    sign = "-" if q < 0 else ""
    return f"{sign}${abs(q):.2f}"


def _get_easd(summary: WorkbookUnitSummary) -> Optional[date]:
    if not summary.units:
        return None
    return min(u.start_date for u in summary.units)


def _format_units_block(units: List[UnitSummaryRow]) -> str:
    if not units:
        return "No units in scope for this assessment."

    lines = []
    for u in units:
        price_str = _format_currency(u.unit_price) if u.unit_price is not None else "N/A"
        hours_str = (
            f"{u.recorded_hours:.2f}" if u.recorded_hours is not None else "N/A"
        )
        latest_eng = getattr(u, "latest_engagement_date", None)
        latest_str = _format_date(latest_eng) if latest_eng else "N/A"

        lines.append(
            f"{u.unit_code} | Status: {u.status} | Start: {_format_date(u.start_date)} | "
            f"Latest engagement: {latest_str} | Recorded hours: {hours_str} | "
            f"Price: {price_str} | Liability: {u.liability_category}"
        )
    return "\n".join(lines)


# -------------------------------------------------------------------
# Supporting docs: encoding / text extraction
# -------------------------------------------------------------------


def _encode_image_to_data_url(path: Path) -> Optional[str]:
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
    try:
        from docx import Document  # type: ignore
    except ImportError:
        return ""

    try:
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception:
        return ""


def _extract_text_from_pdf(path: Path) -> str:
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

    return images, txts, docxs, pdfs


def _build_supporting_docs_text(
    txt_paths: List[Path],
    docx_paths: List[Path],
    pdf_paths: List[Path],
    max_chars_per_doc: int = 2000,
) -> str:
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

    for p in txt_paths:
        content = _extract_text_from_txt(p)
        add_doc_section("TXT", p, content)

    for p in docx_paths:
        content = _extract_text_from_docx(p)
        add_doc_section("DOCX", p, content)

    for p in pdf_paths:
        content = _extract_text_from_pdf(p)
        add_doc_section("PDF", p, content)

    if not sections:
        return "No machine-readable text could be extracted from supporting documents."

    return "\n\n".join(sections)


# -------------------------------------------------------------------
# Case text payload sent to the LLM
# -------------------------------------------------------------------


def _build_case_text_payload(inputs: SpecialCircumstancesInputs) -> str:
    easd = _get_easd(inputs.workbook_summary)

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
    units_block = _format_units_block(inputs.workbook_summary.units)

    images, txts, docxs, pdfs = _split_docs_by_type(inputs.supporting_document_paths)

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

    notes_block = (
        inputs.case_officer_notes.strip()
        if inputs.case_officer_notes and inputs.case_officer_notes.strip()
        else "No additional notes provided by the case officer."
    )

    financial_block = inputs.financial_report_text.strip()
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
# Financial report text parsing for Section 10–11
# -------------------------------------------------------------------


def _extract_section2_from_financial_report(
    financial_report_text: str,
) -> tuple[str, Optional[str]]:
    """
    Find the 'Section 2' chunk and the Adjusted Start Account Balance line.
    """
    lines = financial_report_text.splitlines()

    adj_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Adjusted Start Account Balance:"):
            adj_idx = i
            break

    if adj_idx is None:
        section2_text = financial_report_text.strip()
        return section2_text, None

    adj_line = lines[adj_idx]
    m = re.search(r"Adjusted Start Account Balance:\s*(.+)", adj_line)
    adjusted_start_balance = m.group(1).strip() if m else None

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
    Reduce Section 2 to:
      - the units lines (PostEASD, Fee Waiver, Total)
      - the Adjusted End Account Balance
    """
    lines = section2_text.splitlines()
    impact_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()

        # Exclude Adjusted Start
        if "adjusted start account balance" in lower:
            continue

        # Keep unit lines / total / adjusted end
        if "units:" in lower:
            impact_lines.append(stripped)
            continue

        if stripped.startswith("Total Financial Impact:"):
            impact_lines.append(stripped)
            continue

        if stripped.startswith("Adjusted End Account Balance:"):
            impact_lines.append(stripped)
            continue

    if not impact_lines:
        return section2_text.strip()

    return "\n".join(impact_lines)


def _format_liability_category(summary: WorkbookUnitSummary) -> str:
    categories = sorted(
        {u.liability_category for u in summary.units if u.liability_category}
    )
    if not categories:
        return "N/A"
    if len(categories) == 1:
        return categories[0]
    return ", ".join(categories)


# -------------------------------------------------------------------
# Section 12 – positive recommendation logic
# -------------------------------------------------------------------


def _build_positive_recommendation_from_breakdown(
    bd: FinancialBreakdown,
) -> str:
    """
    Build the positive recommendation string based on:
      - Fee Waiver total
      - PostEASD EWID total
      - Adjusted Start Account Balance
      - Admin fee

    Rules (your description, implemented):

    - Never add PreEASD EWID (already reversed).
    - If Fee Waiver AND no PostEASD EWID:
        Fee Waiver only path.
    - If PostEASD EWID AND no Fee Waiver:
        PostEASD only path.
    - If both Fee Waiver and PostEASD EWID:
        Combined total path.
    """

    pre = bd.categories.get(CATEGORY_PRE)
    post = bd.categories.get(CATEGORY_POST)
    fee = bd.categories.get(CATEGORY_FEE)

    if pre is None or post is None or fee is None:
        return (
            "A positive recommendation cannot be automatically generated because "
            "the financial breakdown is incomplete. Please review manually."
        )

    if bd.adjusted_start_balance is None:
        return (
            "A positive recommendation cannot be automatically generated because "
            "the adjusted account balance is unavailable. Please calculate manually."
        )

    adj = bd.adjusted_start_balance
    adj_str = _format_currency(adj)
    if adj > 0:
        balance_label = "Debt"
    elif adj < 0:
        balance_label = "Credit"
    else:
        balance_label = "Nil"

    def price_expr(cat) -> str:
        if not cat.price_counts:
            return "0 at $0.00"
        parts: List[str] = []
        for price in sorted(cat.price_counts.keys()):
            count = cat.price_counts[price]
            parts.append(f"{count} at {_format_currency(price)}")
        return " + ".join(parts)

    fee_units = len(fee.units)
    post_units = len(post.units)

    fee_total = fee.total
    post_total = post.total

    admin_str = _format_currency(ADMIN_FEE)

    # Case 1: Fee Waiver only
    if fee_total > 0 and post_total == 0:
        base_total = fee_total
        result = base_total - adj - ADMIN_FEE
        result_str = _format_currency(result)
        fee_total_str = _format_currency(base_total)
        expr = price_expr(fee)

        return (
            f"Apply Fee Waiver for {fee_units} unit(s) totalling {fee_total_str} "
            f"({expr}). {fee_total_str} - {adj_str} {balance_label} "
            f"- {admin_str} Administration Fee = {result_str}."
        )

    # Case 2: PostEASD EWID only
    if post_total > 0 and fee_total == 0:
        base_total = post_total
        result = base_total - adj - ADMIN_FEE
        result_str = _format_currency(result)
        post_total_str = _format_currency(base_total)
        expr = price_expr(post)

        return (
            f"Apply PostEASD EWID for {post_units} unit(s) totalling {post_total_str} "
            f"({expr}). {post_total_str} - {adj_str} {balance_label} "
            f"- {admin_str} Administration Fee = {result_str}."
        )

    # Case 3: Both Fee Waiver and PostEASD EWID
    if fee_total > 0 and post_total > 0:
        combined = fee_total + post_total
        combined_str = _format_currency(combined)
        result = combined - adj - ADMIN_FEE
        result_str = _format_currency(result)
        fee_expr = price_expr(fee)
        post_expr = price_expr(post)
        fee_total_str = _format_currency(fee_total)
        post_total_str = _format_currency(post_total)

        return (
            f"Apply Fee Waiver for {fee_units} unit(s) totalling {fee_total_str} "
            f"({fee_expr}) and PostEASD EWID for {post_units} unit(s) totalling "
            f"{post_total_str} ({post_expr}) for a combined total of {combined_str}. "
            f"{combined_str} - {adj_str} {balance_label} - {admin_str} Administration "
            f"Fee = {result_str}."
        )

    # No Fee/Post units
    return (
        "No PostEASD EWID or Fee Waiver units are in scope for this recommendation. "
        "Please review the financial breakdown before proceeding."
    )


def _build_recommendation_text(inputs: SpecialCircumstancesInputs) -> str:
    """
    Wrapper for Section 12 – for now we only implement the positive path.
    Later we can add a negative / no-refund branch when needed.
    """
    if inputs.financial_breakdown is None:
        return (
            "Recommendation to be completed by the case officer. "
            "Structured financial data was not available."
        )
    return _build_positive_recommendation_from_breakdown(inputs.financial_breakdown)


# -------------------------------------------------------------------
# Build the final investigation document
# -------------------------------------------------------------------


def _build_investigation_document(
    inputs: SpecialCircumstancesInputs,
    ai_analysis_text: str,
) -> str:
    easd = _get_easd(inputs.workbook_summary)
    application_date_str = _format_date(inputs.date_requested)
    easd_str = _format_date(easd)
    liability_category_str = _format_liability_category(inputs.workbook_summary)

    has_docs = bool(inputs.supporting_document_paths)
    has_evidence_str = "Yes" if has_docs else "No"

    # Use the financial report text to derive Section 2 + adjusted start
    section2_text, adjusted_start_balance_text = _extract_section2_from_financial_report(
        inputs.financial_report_text
    )

    if adjusted_start_balance_text:
        account_balance_str = adjusted_start_balance_text
    else:
        if inputs.workbook_summary.account_balance is not None:
            bal = inputs.workbook_summary.account_balance
            account_balance_str = _format_currency(bal)
        else:
            account_balance_str = "N/A"

    impact_block = _format_financial_impact_for_document(section2_text)
    recommendation_text = _build_recommendation_text(inputs)

    lines: List[str] = []

    lines.append("Hi ,")
    lines.append("")
    lines.append("Please find below a COE/refund recommendation for approval:")
    lines.append("")
    lines.append("1. Course Code and Name: ")
    lines.append("")
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
    lines.append("11. Financial Impact:")
    lines.append("")
    lines.append(impact_block)
    lines.append("")
    lines.append("12. Recommendation:")
    lines.append("")
    lines.append(recommendation_text)
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
    instructions_text = _load_instructions_text(instructions_path)
    case_text = _build_case_text_payload(inputs)

    images, txts, docxs, pdfs = _split_docs_by_type(inputs.supporting_document_paths)

    pdf_image_paths: List[Path] = []
    for pdf in pdfs:
        try:
            page_images = convert_pdf_to_images(pdf)
            pdf_image_paths.extend(page_images)
        except Exception:
            continue

    all_image_paths: List[Path] = list(images) + list(pdf_image_paths)

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

    client = OpenAI()
    model_id = model or DEFAULT_MODEL

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.2,
    )

    ai_analysis_text = response.choices[0].message.content.strip()
    final_doc = _build_investigation_document(
        inputs=inputs,
        ai_analysis_text=ai_analysis_text,
    )

    return SpecialCircumstancesResult(report_text=final_doc)
