"""
OCR extraction via Groq's highly-capable production model.
Isolated from structuring logic — this module's only job is image -> raw text.
"""

import io
import base64
import time
import logging
from PIL import Image
from groq import Groq
from app.config import settings

# Setup standard logger for production environment
logger = logging.getLogger("scanmynotes")

# Groq client initialize
_client = Groq(api_key=settings.GROQ_API_KEY)

EXTRACTION_PROMPT = (
    "Extract ALL handwritten text from this image exactly as written. "
    "The text may be in English, Roman Urdu, Urdu script, or a mix of these. "
    "Do not translate, correct spelling, or interpret meaning — "
    "output the raw text exactly as it appears, preserving line breaks. "
    "If no readable text is present, respond with exactly: NO_TEXT_FOUND"
)

def _classify_error(err_str: str) -> str:
    """Buckets an exception message into 'auth', 'rate_limit', or 'other' for retry decisions."""
    err_lower = err_str.lower()
    if any(k in err_lower for k in ("api key", "invalid", "401", "403", "permission")):
        return "auth"
    if any(k in err_lower for k in ("quota", "rate", "429")):
        return "rate_limit"
    return "other"

def run_ocr(image: Image.Image) -> dict:
    """
    Calls Groq production model for text processing and handwriting extraction extraction.
    Returns a consistent dict shape: {success, text, error, time_sec} regardless of outcome.
    """
    last_error = None

    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Optimize image dimension for fast encoding
        max_size = 1024
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    except Exception as e:
        logger.warning(f"[OCR] Image preprocessing warning: {e}")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    
    # Clean base64 string
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8").replace("\n", "").replace("\r", "")

    for attempt in range(1, settings.MAX_RETRIES + 2):
        start = time.time()
        try:
            logger.info(f"[OCR] Attempt {attempt}: Sending payload to llama-3.3-70b-versatile...")
            
            # Using your active production model with robust structural messages
            response = _client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": f"{EXTRACTION_PROMPT}\n\n[Attached Image Data (Base64 Enriched Chunk)]:\ndata:image/jpeg;base64,{base64_image}"
                    }
                ]
            )
        except Exception as e:
            # Render logger par guaranteed display hoga
            logger.error(f"[OCR] CRITICAL API ERROR on attempt {attempt}: {str(e)}")
            
            error_type = _classify_error(str(e))
            if error_type == "auth":
                return {"success": False, "text": "", "error": f"Auth error: {e}", "time_sec": 0}
            last_error = f"{error_type}: {e}"
        else:
            elapsed = round(time.time() - start, 2)

            if not response.choices:
                logger.error("[OCR] Error: Groq returned response with empty choices.")
                return {"success": False, "text": "", "error": "No choices returned in response.", "time_sec": elapsed}

            text = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
            logger.info(f"[OCR] Success! Raw text length parsed: {len(text)} characters.")

            # Strict validation checking
            if text == "NO_TEXT_FOUND" or not text:
                logger.warning("[OCR] Warning: Model returned empty content framework.")
                return {"success": False, "text": "", "error": "Model returned empty text or NO_TEXT_FOUND", "time_sec": elapsed}

            return {"success": True, "text": text, "error": None, "time_sec": elapsed}

        if attempt <= settings.MAX_RETRIES:
            time.sleep(settings.RETRY_BACKOFF_SEC * attempt)

    return {"success": False, "text": "", "error": f"Failed after retries: {last_error}", "time_sec": 0}
