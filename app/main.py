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
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# Groq client import (Make sure groq is installed: pip install groq)
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

# --- Request Models for new endpoints ---
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
# 🆕 ADDED ENDPOINTS (Groq AI Integration)
# ==========================================

@app.post("/api/explain")
async def explain_notes(req: ExplainRequest):
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API Key is not configured on the backend.")
    
    prompt = f"Explain the following study notes clearly. "
    if req.query:
        prompt += f"Focus specifically on answering this: '{req.query}'. "
    prompt += f"\n\nNotes:\n{req.text}"

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
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
        "You are a helpful assistant. Extract key terms, definitions, and simple translations from these notes. "
        "You MUST return the output as a valid JSON object with a single key 'glossary' which is a list of objects. "
        "Each object must have 'term', 'definition', and 'translation' keys.\n"
        "Do not include any explanation, introductory text, or markdown blocks (like ```json). Just return the raw JSON object.\n\n"
        f"Notes:\n{req.text}"
    )

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            # response_format hamesha json_object return karega
            response_format={"type": "json_object"}
        )
        import json
        raw_content = completion.choices[0].message.content.strip()
        data = json.loads(raw_content)
        
        # Safe-check: Agar structure sahi nahi hai toh empty list return ho jaye crash na kare
        if "glossary" not in data:
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
        "You are a helpful assistant. Create study flashcards from the following notes. "
        "You MUST return the output as a valid JSON object with a single key 'flashcards' which is a list of objects. "
        "Each object must have 'front' (the question or term) and 'back' (the answer or explanation).\n"
        "Do not include any explanation, introductory text, or markdown blocks (like ```json). Just return the raw JSON object.\n\n"
        f"Notes:\n{req.text}"
    )

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        import json
        raw_content = completion.choices[0].message.content.strip()
        data = json.loads(raw_content)
        
        # Safe-check: structure validation
        if "flashcards" not in data:
            data = {"flashcards": []}
        return data
    except Exception as e:
        logger.error(f"Groq Flashcards Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
