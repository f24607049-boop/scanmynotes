"""
ScanMyNotes API — entry point.

Endpoints:
  GET  /health              -> liveness check
  POST /api/process         -> upload an image or PDF, get back OCR + structured notes
  GET  /api/usage           -> check remaining free-tier pages for the caller today
  POST /api/explain         -> explain notes using Groq API
  POST /api/glossary        -> generate bilingual glossary using Groq API
  POST /api/flashcards       -> generate flashcards using Groq API
"""

import logging
import os
import json
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# Groq client import
try:
    from groq import Groq
except ImportError:
    Groq = None

from app.config import settings
from app.pipeline import process_file, count_pages_in_result
from app.pdf_service import get_pdf_page_count
from app.validators import is_pdf, ValidationError
from app.rate_limiter import page_limiter, RateLimitExceeded

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scanmynotes")

app = FastAPI(title="ScanMyNotes API", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client using environment variable
groq_key = os.getenv("GROQ_API_KEY") or getattr(settings, "GROQ_API_KEY", None)
groq_client = Groq(api_key=groq_key) if (Groq and groq_key) else None

# --- Request Models ---
class ExplainRequest(BaseModel):
    text: str
    query: Optional[str] = None

class TextRequest(BaseModel):
    text: str

def _get_client_id(request: Request) -> str:
    return request.client.host if request.client else "unknown"

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/usage")
def get_usage(request: Request):
    client_id = _get_client_id(request)
    remaining = page_limiter.get_remaining(client_id)
    return {"remaining_pages_today": remaining, "daily_limit": settings.FREE_DAILY_PAGE_LIMIT}

@app.post("/api/process")
async def process_upload(request: Request, file: UploadFile = File(...)):
    client_id = _get_client_id(request)
    file_bytes = await file.read()
    filename = file.filename or "unknown"

    if not file_bytes:
        raise HTTPException(status_code=400, detail=f"'{filename}' is empty — no data received.")

    try:
        if is_pdf(filename, file_bytes):
            page_count = get_pdf_page_count(file_bytes, filename)
        else:
            page_count = 1
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        page_limiter.check_and_record(client_id, page_count)
    except RateLimitExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))

    logger.info(f"Processing '{filename}' ({page_count} page(s)) for client={client_id}")
    result = process_file(file_bytes, filename)
    return result

# ==========================================
# 🆕 AI Endpoints (Groq Integration Fixed)
# ==========================================

@app.post("/api/explain")
async def explain_notes(req: ExplainRequest):
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API Key is not configured on the backend.")
    
    prompt = (
        "You are an expert study assistant. Explain the following study notes clearly. "
    )
    if req.query:
        prompt += f"Focus specifically on answering this: '{req.query}'. "
    
    prompt += (
        "\n\nCRITICAL FORMATTING RULES:\n"
        "1. DO NOT use single or double dollar signs (like $CO_2$ or $$E=mc^2$$) under any circumstances. "
        "Write math equations, chemical formulas, and scientific terms in clean, standard plain text (e.g., CO2, H2O, E = mc^2).\n"
        "2. For bolding key terms, ALWAYS use standard double asterisks (e.g., **Key Term**).\n"
        "3. If presenting comparisons or datasets, DO NOT generate complex markdown tables. Instead, represent them using a clean, well-spaced vertical bullet-point structure (e.g., - **Input**: Description) so it is perfectly readable without table styling.\n\n"
        f"Notes:\n{req.text}"
    )

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # FIXED: Updated to active versatile model
            messages=[{"role": "user", "content": prompt}]
        )
        explanation = completion.choices[0].message.content
        return {"explanation": explanation}
    except Exception as e:
        logger.error(f"Groq Explain Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/glossary")
async def generate_glossary(req: TextRequest):
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API Key is not configured on the backend.")

    prompt = (
        "You are a helpful assistant. Extract key terms, definitions, and simple translations from these notes.\n"
        "CRITICAL: Do not use single or double dollar signs ($) for chemical or mathematical symbols. Write them in plain readable text.\n"
        "You MUST return the output as a valid JSON object with a single key 'glossary' which is a list of objects.\n"
        "Each object in the list must have exactly these keys: 'term', 'definition', and 'translation'.\n"
        "Do not include markdown code blocks, just raw JSON text.\n\n"
        f"Notes:\n{req.text}"
    )

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # FIXED: Updated to active versatile model
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        raw_content = completion.choices[0].message.content.strip()
        data = json.loads(raw_content)
        
        if "glossary" not in data or not isinstance(data["glossary"], list):
            data = {"glossary": []}
        return data
    except Exception as e:
        logger.error(f"Groq Glossary Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/flashcards")
async def generate_flashcards(req: TextRequest):
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API Key is not configured on the backend.")

    prompt = (
        "You are a helpful assistant. Create study flashcards from the following notes.\n"
        "CRITICAL: Do not use single or double dollar signs ($) for chemical or mathematical symbols. Write them in plain readable text.\n"
        "You MUST return the output as a valid JSON object with a single key 'flashcards' which is a list of objects.\n"
        "Each object in the list MUST have exactly these two keys:\n"
        "1. 'front': (contains the question or term)\n"
        "2. 'back': (contains the answer or explanation)\n"
        "Do not use keys like 'question' or 'answer'. Only use 'front' and 'back'.\n"
        "Do not include markdown formatting, just raw JSON.\n\n"
        f"Notes:\n{req.text}"
    )

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # FIXED: Updated to active versatile model
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        raw_content = completion.choices[0].message.content.strip()
        data = json.loads(raw_content)
        
        if "flashcards" not in data or not isinstance(data["flashcards"], list):
            data = {"flashcards": []}
        else:
            cleaned_cards = []
            for card in data["flashcards"]:
                front_val = card.get("front") or card.get("question") or card.get("term") or ""
                back_val = card.get("back") or card.get("answer") or card.get("definition") or ""
                if front_val and back_val:
                    cleaned_cards.append({"front": front_val, "back": back_val})
            data["flashcards"] = cleaned_cards
            
        return data
    except Exception as e:
        logger.error(f"Groq Flashcards Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )
