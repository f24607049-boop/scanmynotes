"""
File validation logic — shared by both image and PDF processing paths.
All validation raises ValidationError with a clear, user-facing message.
Never trust file extensions alone — magic bytes are checked where it matters.
"""

import io
from PIL import Image, UnidentifiedImageError
from app.config import settings


class ValidationError(Exception):
    """Raised when an uploaded file fails validation. Message is safe to show to the end user."""
    pass


def check_supported_extension(filename: str, file_bytes: bytes) -> None:
    """
    Rejects unsupported file types explicitly and early.
    Checks both extension and PDF magic bytes (a renamed file shouldn't slip through as the wrong type).
    """
    is_pdf_by_ext = filename.lower().endswith(".pdf")
    is_pdf_by_magic = file_bytes.startswith(b"%PDF-")
    is_image_by_ext = filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))

    if not (is_pdf_by_ext or is_pdf_by_magic or is_image_by_ext):
        ext = filename.split(".")[-1].upper() if "." in filename else "UNKNOWN"
        raise ValidationError(
            f"Unsupported file type '.{ext}'. Only PDF and image files (JPG, PNG, WEBP) are supported."
        )


def is_pdf(filename: str, file_bytes: bytes) -> bool:
    """Detects PDF by extension OR magic bytes — either signal is sufficient."""
    return filename.lower().endswith(".pdf") or file_bytes.startswith(b"%PDF-")


def load_and_validate_image(file_bytes: bytes, filename: str) -> Image.Image:
    """Validates a single image file. Raises ValidationError with a clear reason on failure."""
    if not file_bytes:
        raise ValidationError(f"'{filename}' is empty — no data received.")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_IMAGE_SIZE_MB:
        raise ValidationError(f"'{filename}' too large ({size_mb:.1f}MB). Max: {settings.MAX_IMAGE_SIZE_MB}MB.")

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()
    except UnidentifiedImageError:
        raise ValidationError(f"'{filename}' is not a valid/readable image file.")
    except Exception as e:
        raise ValidationError(f"Validation failed for '{filename}': {e}")

    image = Image.open(io.BytesIO(file_bytes))  # reopen — verify() invalidates the file pointer

    if image.format not in settings.ALLOWED_IMAGE_FORMATS:
        raise ValidationError(
            f"Unsupported format '{image.format}' for '{filename}'. Allowed: {settings.ALLOWED_IMAGE_FORMATS}"
        )

    if image.width < 10 or image.height < 10:
        raise ValidationError(f"'{filename}' dimensions too small ({image.width}x{image.height}) — likely corrupt.")

    return image.convert("RGB")


def validate_pdf_bytes(file_bytes: bytes, filename: str) -> None:
    """Validates raw PDF bytes before attempting to open with PyMuPDF."""
    if not file_bytes:
        raise ValidationError(f"'{filename}' is empty — no data received.")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_PDF_SIZE_MB:
        raise ValidationError(f"'{filename}' too large ({size_mb:.1f}MB). Max: {settings.MAX_PDF_SIZE_MB}MB.")

    if not file_bytes.startswith(b"%PDF-"):
        raise ValidationError(f"'{filename}' does not look like a valid PDF file (missing PDF header).")
