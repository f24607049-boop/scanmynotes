"""
OCR extraction via Groq's vision-capable model.
Isolated from structuring logic — this module's only job is image -> raw text.
"""

import io
import base64
import time
from PIL import Image
from groq import Groq
from app.config import settings

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
    Calls Groq vision model for handwriting extraction on a single image.
    Returns a consistent dict shape: {success, text, error, time_sec} regardless of outcome.
    """
    last_error = None

    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Force a safe resolution for the Vision model to avoid token limit 400s
        max_size = 512
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Image preprocessing warning: {e}")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=75)
    
    # Clean base64 string strictly to avoid payload formatting errors
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8").replace("\n", "").replace("\r", "")

    for attempt in range(1, settings.MAX_RETRIES + 2):
        start = time.time()
        try:
            # 1. Switched to 90B model which is more stable
            # 2. Removed all optional arguments (temperature, timeouts) for strict compliance
            response = _client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": EXTRACTION_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ]
            )
        except Exception as e:
            # --- CRITICAL LOGGING ---
            # Ye line Render logs mein asal masla print karegi (agar API fail hoti hai)
            print(f"GROQ API ERROR on attempt {attempt}: {e}")
            
            error_type = _classify_error(str(e))
            if error_type == "auth":
                return {"success": False, "text": "", "error": f"Auth/permission error: {e}", "time_sec": 0}
            last_error = f"{error_type}: {e}"
        else:
            elapsed = round(time.time() - start, 2)

            if not response.choices:
                return {"success": False, "text": "", "error": "No choices returned in response.", "time_sec": elapsed}

            text = response.choices[0].message.content.strip() if response.choices[0].message.content else ""

            if text == "NO_TEXT_FOUND" or not text:
                return {"success": True, "text": "", "error": None, "time_sec": elapsed}

            return {"success": True, "text": text, "error": None, "time_sec": elapsed}

        if attempt <= settings.MAX_RETRIES:
            time.sleep(settings.RETRY_BACKOFF_SEC * attempt)

    return {"success": False, "text": "", "error": f"Failed after retries: {last_error}", "time_sec": 0}
