from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class RateLimiter:
    """Thread-safe limiter that spaces requests at least 1/rps seconds apart."""

    def __init__(self, requests_per_second: float):
        self.interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if delay:
            time.sleep(delay)


def run_batch(items: Iterable[T], fn: Callable[[T], R], concurrency: int = 8,
              requests_per_second: float = 0.0) -> list[R]:
    """Run fn over items in parallel. Results come back in input order."""
    limiter = RateLimiter(requests_per_second)

    def task(item: T) -> R:
        limiter.wait()
        return fn(item)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(task, items))