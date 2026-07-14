"""
ScanMyNotes API — entry point.

Endpoints:
  GET  /health              -> liveness check
  POST /api/process         -> upload an image or PDF, get back OCR + structured notes
  GET  /api/usage           -> check remaining free-tier pages for the caller today

Run locally:
  uvicorn app.main:app --reload --port 8000
"""

import logging

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.pipeline import process_file, count_pages_in_result
from app.pdf_service import get_pdf_page_count
from app.validators import is_pdf, ValidationError
from app.rate_limiter import page_limiter, RateLimitExceeded

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scanmynotes")

app = FastAPI(title="ScanMyNotes API", version="0.1.0")

# CORS — only allow your actual frontend origin(s) in production, never "*" once real users are on it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_client_id(request: Request) -> str:
    """
    Identifies the caller for rate-limiting purposes.
    Uses the client's IP address — simple and sufficient for MVP.
    NOTE: if you deploy behind a reverse proxy/load balancer, this will need to read
    X-Forwarded-For instead, otherwise every request will appear to come from the proxy's IP.
    """
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

    # Pre-check page count for PDFs so we reject over-limit requests BEFORE spending any OCR calls.
    # For a single image, that's always exactly 1 page.
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


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Catch-all safety net — ensures unexpected errors never leak internal details
    (stack traces, file paths, etc.) to the client, while still being logged for debugging.
    """
    logger.exception(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )
