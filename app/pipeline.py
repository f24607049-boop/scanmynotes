"""
Full processing pipeline: file validation -> OCR extraction -> text structuring.
This is the only module that knows about ALL the steps — each step itself stays decoupled,
so any single stage (OCR engine, structuring model, PDF renderer) can be swapped independently.
"""

from app.config import settings
from app.validators import (
    ValidationError,
    check_supported_extension,
    is_pdf,
    load_and_validate_image,
)
from app.pdf_service import pdf_to_images
from app.ocr_service import run_ocr
from app.structuring_service import structure_notes


def process_file(file_bytes: bytes, filename: str) -> dict:
    """
    Runs the full pipeline on one uploaded file (image or PDF).

    Returns a dict:
    {
        "filename": str,
        "is_pdf": bool,
        "pages": [{"label": str, "success": bool, "text": str, "error": str|None, "time_sec": float}, ...],
        "raw_combined_text": str,
        "structured_text": str,
        "structuring_error": str|None,
    }

    Never raises — all failure paths are captured in the returned dict so callers
    (API routes) can always respond with a 200 + structured error, rather than a crash.
    """
    try:
        check_supported_extension(filename, file_bytes)
    except ValidationError as e:
        return _empty_result(filename, is_pdf=False, error=str(e))

    if is_pdf(filename, file_bytes):
        try:
            page_images = pdf_to_images(file_bytes, filename)
        except ValidationError as e:
            return _empty_result(filename, is_pdf=True, error=str(e))

        pages = [
            {**run_ocr(img), "label": f"Page {i}"}
            for i, img in enumerate(page_images, start=1)
        ]
        result = {"filename": filename, "is_pdf": True, "pages": pages}

    else:
        try:
            image = load_and_validate_image(file_bytes, filename)
        except ValidationError as e:
            return _empty_result(filename, is_pdf=False, error=str(e))

        page = {**run_ocr(image), "label": "Image"}
        result = {"filename": filename, "is_pdf": False, "pages": [page]}

    raw_combined = "\n\n".join(p["text"] for p in result["pages"] if p["success"] and p["text"])
    result["raw_combined_text"] = raw_combined

    if not raw_combined:
        result["structured_text"] = ""
        result["structuring_error"] = None
        return result

    structuring_result = structure_notes(raw_combined)
    result["structured_text"] = structuring_result["text"]
    result["structuring_error"] = structuring_result["error"] if not structuring_result["success"] else None

    return result


def _empty_result(filename: str, is_pdf: bool, error: str) -> dict:
    """Builds a consistent failure result shape for early-exit validation errors."""
    return {
        "filename": filename,
        "is_pdf": is_pdf,
        "pages": [{"label": "N/A", "success": False, "text": "", "error": error, "time_sec": 0}],
        "raw_combined_text": "",
        "structured_text": "",
        "structuring_error": None,
    }


def count_pages_in_result(result: dict) -> int:
    """Helper for usage-limit tracking — counts pages actually processed."""
    return len(result.get("pages", []))
