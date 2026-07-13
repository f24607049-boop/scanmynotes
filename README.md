# ScanMyNotes Backend

FastAPI backend: handwritten notes (image or PDF) -> OCR (Groq vision) -> structured, cleaned text (Groq LLM).

## Project Structure

```
scanmynotes-backend/
  app/
    config.py               # centralized settings, loaded from environment variables
    validators.py           # file validation (size, format, magic bytes)
    pdf_service.py          # PDF -> images (PyMuPDF)
    ocr_service.py          # image -> raw text (Groq vision model)
    structuring_service.py  # raw text -> clean, organized text (Groq LLM)
    rate_limiter.py         # in-memory daily free-tier page cap
    pipeline.py             # orchestrates validation -> OCR -> structuring
    main.py                 # FastAPI app + routes
  requirements.txt
  .env.example              # copy to .env and fill in your key
  .gitignore
```

## Local Setup

1. Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment file:
   ```
   cp .env.example .env
   ```
   Then open `.env` and paste your real Groq API key. **Never commit `.env` to git.**

4. Run the server:
   ```
   uvicorn app.main:app --reload --port 8000
   ```

5. Check it's alive:
   ```
   curl http://localhost:8000/health
   ```

## API Endpoints

### `POST /api/process`
Upload a file (image or PDF) and get back OCR + structured notes.

```
curl -X POST http://localhost:8000/api/process \
  -F "file=@/path/to/your/note.jpg"
```

Response shape:
```json
{
  "filename": "note.jpg",
  "is_pdf": false,
  "pages": [
    {"label": "Image", "success": true, "text": "...", "error": null, "time_sec": 1.2}
  ],
  "raw_combined_text": "...",
  "structured_text": "...",
  "structuring_error": null
}
```

### `GET /api/usage`
Check your remaining free-tier pages for today (resets daily, tracked per IP).

### `GET /health`
Basic liveness check.

## Security Notes

- API key is loaded from environment variables only — never hardcoded, never logged.
- CORS is restricted to `ALLOWED_ORIGINS` in `.env` — update this to your real frontend domain before deploying.
- File uploads are validated on size, format, and magic bytes before being processed.
- A daily per-IP page limit protects against runaway API costs on the free tier (`FREE_DAILY_PAGE_LIMIT` in `.env`).

## Known Limitation (by design, for MVP)

The rate limiter is in-memory — it resets on server restart and won't work correctly across
multiple server instances. This is fine for a single-instance MVP deployment (e.g. one Render
or Hugging Face Spaces instance). If you scale to multiple instances, swap `rate_limiter.py`'s
storage for Redis or a database table — the function signatures are small on purpose to make
that swap easy later.

## Next Steps

- Deploy to Render or Hugging Face Spaces (free tier)
- Build the Next.js frontend to call `/api/process`
- Add persistent usage tracking (Supabase) once you move past IP-based limiting
