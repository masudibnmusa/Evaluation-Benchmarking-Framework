from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from app.utils.logger import get_logger

log = get_logger(__name__)
T = TypeVar("T")


def is_retryable(exc: Exception) -> bool:
    """Retry on rate limits, timeouts, connection errors and 5xx. Never on 4xx client errors."""
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in (408, 409, 429) or status >= 500
    return isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__name__ in {
        "APIConnectionError", "APITimeoutError",
    }


def call_with_retries(fn: Callable[..., T], *args, max_retries: int = 3,
                      base_delay: float = 1.0, max_delay: float = 30.0, **kwargs) -> T:
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries or not is_retryable(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** attempt)) * (0.5 + random.random() / 2)
            attempt += 1
            log.warning("Retry %d/%d in %.1fs after %s", attempt, max_retries, delay, type(exc).__name__)
            time.sleep(delay)