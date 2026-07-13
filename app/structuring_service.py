"""
Note structuring via Groq's text LLM.
Isolated from OCR logic — this module's only job is raw text -> clean, organized text.
"""

import time
from groq import Groq
from app.config import settings
from app.ocr_service import _classify_error  # reuse the same error classification logic

_client = Groq(api_key=settings.GROQ_API_KEY)

STRUCTURING_PROMPT = """You are a note-formatting assistant. You will receive raw OCR-extracted text from a student's handwritten notes. The text may be in English, Roman Urdu, Urdu script, or a mix of these.

Your task:
1. Organize the text into clear paragraphs and bullet points where appropriate.
2. Detect and format obvious headings/titles (if the note structure suggests one).
3. Fix obvious OCR artifacts (e.g. broken words, stray characters) ONLY when the correction is clearly evident from context.
4. Preserve the original language exactly as written — do NOT translate Roman Urdu to English or vice versa.
5. Do NOT add any new information, explanations, or content that wasn't in the original text.
6. Do NOT add commentary, greetings, or notes about what you did — output ONLY the cleaned/structured notes.

If the input text is empty or meaningless, respond with exactly: NO_CONTENT_TO_STRUCTURE

Raw OCR text to structure:
---
{raw_text}
---"""


def structure_notes(raw_text: str) -> dict:
    """
    Sends raw OCR text to Groq LLM to organize into clean, structured notes.
    Returns a consistent dict shape: {success, text, error, time_sec} regardless of outcome.
    """
    if not raw_text or not raw_text.strip():
        return {"success": True, "text": "", "error": None, "time_sec": 0}

    last_error = None
    prompt = STRUCTURING_PROMPT.format(raw_text=raw_text)

    for attempt in range(1, settings.MAX_RETRIES + 2):
        start = time.time()
        try:
            response = _client.chat.completions.create(
                model=settings.STRUCTURING_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # low temperature — faithful cleanup, not creative rewriting
                timeout=settings.REQUEST_TIMEOUT_SEC,
            )
        except Exception as e:
            error_type = _classify_error(str(e))
            if error_type == "auth":
                return {"success": False, "text": "", "error": f"Auth/permission error: {e}", "time_sec": 0}
            last_error = f"{error_type}: {e}"
        else:
            elapsed = round(time.time() - start, 2)

            if not response.choices:
                return {"success": False, "text": "", "error": "No choices returned in response.", "time_sec": elapsed}

            text = response.choices[0].message.content.strip() if response.choices[0].message.content else ""

            if text == "NO_CONTENT_TO_STRUCTURE" or not text:
                return {"success": True, "text": "", "error": None, "time_sec": elapsed}

            return {"success": True, "text": text, "error": None, "time_sec": elapsed}

        if attempt <= settings.MAX_RETRIES:
            time.sleep(settings.RETRY_BACKOFF_SEC * attempt)

    return {"success": False, "text": "", "error": f"Failed after retries: {last_error}", "time_sec": 0}
