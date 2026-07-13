"""
Centralized configuration for ScanMyNotes backend.
All settings load from environment variables — never hardcode secrets or magic values here.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file in local dev; in production, env vars are set by the host


def _get_required_env(key: str) -> str:
    """Fetches a required env var, fails fast at startup instead of failing later mid-request."""
    value = os.getenv(key)
    if not value or not value.strip():
        raise RuntimeError(
            f"Missing required environment variable: '{key}'. "
            f"Set it in your .env file (local) or host's environment settings (production)."
        )
    return value.strip()


class Settings:
    # --- Secrets (never hardcode; always from env) ---
    GROQ_API_KEY: str = _get_required_env("GROQ_API_KEY")

    # --- Model config ---
    VISION_MODEL_NAME: str = os.getenv("VISION_MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")
    STRUCTURING_MODEL_NAME: str = os.getenv("STRUCTURING_MODEL_NAME", "llama-3.3-70b-versatile")

    # --- File limits ---
    MAX_IMAGE_SIZE_MB: float = float(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
    MAX_PDF_SIZE_MB: float = float(os.getenv("MAX_PDF_SIZE_MB", "25"))
    MAX_PDF_PAGES: int = int(os.getenv("MAX_PDF_PAGES", "30"))
    PDF_RENDER_DPI: int = int(os.getenv("PDF_RENDER_DPI", "200"))

    ALLOWED_IMAGE_FORMATS: tuple = ("JPEG", "PNG", "JPG", "WEBP")
    SUPPORTED_EXTENSIONS: tuple = (".pdf", ".jpg", ".jpeg", ".png", ".webp")

    # --- Reliability config ---
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))
    RETRY_BACKOFF_SEC: float = float(os.getenv("RETRY_BACKOFF_SEC", "2"))
    REQUEST_TIMEOUT_SEC: int = int(os.getenv("REQUEST_TIMEOUT_SEC", "60"))

    # --- Free tier usage limits (basic cost control, enforced in routes) ---
    FREE_DAILY_PAGE_LIMIT: int = int(os.getenv("FREE_DAILY_PAGE_LIMIT", "5"))

    # --- CORS (restrict to your actual frontend domain(s) in production) ---
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")


settings = Settings()
