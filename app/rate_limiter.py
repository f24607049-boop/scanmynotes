"""
Basic in-memory rate limiter — enforces a daily page-processing cap per IP address.

IMPORTANT LIMITATION (be aware of this):
This is in-memory, so it resets if the server restarts, and does NOT work correctly
if you ever run multiple server instances/workers (each would have its own counter).
This is intentional for MVP simplicity. For real production scale, replace this with
a shared store (e.g. Redis) — the interface below is small on purpose so that swap is easy.
"""

import threading
from datetime import date
from collections import defaultdict

from app.config import settings


class RateLimitExceeded(Exception):
    """Raised when a client has exceeded their daily free-tier page limit."""
    pass


class DailyPageLimiter:
    def __init__(self, daily_limit: int):
        self._daily_limit = daily_limit
        self._usage = defaultdict(int)   # {(date, client_id): pages_used}
        self._lock = threading.Lock()    # avoids race conditions under concurrent requests

    def check_and_record(self, client_id: str, pages_to_add: int) -> None:
        """
        Checks if adding `pages_to_add` would exceed the client's daily limit.
        Raises RateLimitExceeded if so; otherwise records the usage.
        """
        today = date.today().isoformat()
        key = (today, client_id)

        with self._lock:
            current = self._usage[key]
            if current + pages_to_add > self._daily_limit:
                remaining = max(0, self._daily_limit - current)
                raise RateLimitExceeded(
                    f"Daily free limit reached ({self._daily_limit} pages/day). "
                    f"You have {remaining} page(s) remaining today."
                )
            self._usage[key] += pages_to_add

    def get_remaining(self, client_id: str) -> int:
        today = date.today().isoformat()
        key = (today, client_id)
        with self._lock:
            return max(0, self._daily_limit - self._usage[key])


page_limiter = DailyPageLimiter(daily_limit=settings.FREE_DAILY_PAGE_LIMIT)
