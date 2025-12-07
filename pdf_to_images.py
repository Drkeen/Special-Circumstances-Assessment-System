# app/pdf_to_images.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF


def convert_pdf_to_images(
    pdf_path: Path,
    output_dir: Optional[Path] = None,
    *,
    dpi: int = 200,
    prefix: Optional[str] = None,
) -> List[Path]:
    """
    Convert all pages of a PDF into PNG images.

    Args:
        pdf_path: Path to the source PDF file.
        output_dir: Directory to save generated PNG files.
                    - If None, uses pdf_path.parent / "pdf_images".
        dpi: Render resolution (higher = sharper images, bigger files).
             200–300 is usually plenty for documents.
        prefix: Optional prefix for output filenames. If None, uses pdf_path.stem.

    Returns:
        List[Path]: A list of paths to the generated PNG files, in page order.

    Example:
        from pathlib import Path
        from app.pdf_to_images import convert_pdf_to_images

        pdf = Path("sample_data/student_a/supporting_documents/doctor_letter.pdf")
        images = convert_pdf_to_images(pdf)
        for img in images:
            print("Created:", img)
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")

    if output_dir is None:
        output_dir = pdf_path.parent / "pdf_images"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use file stem as default prefix if not provided
    if prefix is None:
        prefix = pdf_path.stem

    doc = fitz.open(pdf_path)
    image_paths: List[Path] = []

    # PyMuPDF uses 72 dpi as base; scale factor = dpi / 72
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix)

        out_name = f"{prefix}_page{page_index + 1}.png"
        out_path = output_dir / out_name

        pix.save(out_path.as_posix())
        image_paths.append(out_path)

    doc.close()
    return image_paths
