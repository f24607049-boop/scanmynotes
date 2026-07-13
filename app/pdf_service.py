"""
PDF-to-images conversion using PyMuPDF.
Isolated from OCR logic — this module's only job is turning validated PDF bytes into PIL Images.
"""

import io
import fitz  # PyMuPDF
from PIL import Image
from app.config import settings
from app.validators import ValidationError, validate_pdf_bytes


def get_pdf_page_count(file_bytes: bytes, filename: str) -> int:
    """
    Cheaply gets a PDF's page count without rendering any pages.
    Used to enforce usage limits BEFORE spending OCR API calls on pages we'd reject anyway.
    """
    validate_pdf_bytes(file_bytes, filename)

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise ValidationError(f"Failed to open '{filename}' as PDF: {e}")

    if doc.is_encrypted:
        doc.close()
        raise ValidationError(f"'{filename}' is password-protected — cannot process.")

    count = doc.page_count
    doc.close()

    if count == 0:
        raise ValidationError(f"'{filename}' has no pages.")

    return count


def pdf_to_images(file_bytes: bytes, filename: str) -> list:

    """
    Converts a validated PDF's pages into a list of PIL Images.
    Raises ValidationError on corrupt PDFs, password-protection, or page-count overflow.
    """
    validate_pdf_bytes(file_bytes, filename)

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise ValidationError(f"Failed to open '{filename}' as PDF: {e}")

    if doc.is_encrypted:
        doc.close()
        raise ValidationError(f"'{filename}' is password-protected — cannot process.")

    page_count = doc.page_count
    if page_count == 0:
        doc.close()
        raise ValidationError(f"'{filename}' has no pages.")

    if page_count > settings.MAX_PDF_PAGES:
        doc.close()
        raise ValidationError(
            f"'{filename}' has {page_count} pages — exceeds max allowed ({settings.MAX_PDF_PAGES})."
        )

    zoom = settings.PDF_RENDER_DPI / 72  # PDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)
    images = []

    try:
        for page_index in range(page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images.append(image)
    except Exception as e:
        doc.close()
        raise ValidationError(f"Failed while rendering pages of '{filename}': {e}")

    doc.close()
    return images
