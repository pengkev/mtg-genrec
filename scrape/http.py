"""Http scraping support."""

from __future__ import annotations

import hashlib
import logging
import requests
import time
from collections.abc import Iterable, Mapping
from email.utils import parsedate_to_datetime
from typing import Any


def retry_after_seconds(response: requests.Response, fallback: float) -> float:
    value = response.headers.get("Retry-After")
    if not value:
        return fallback
    try:
        return max(float(value), 0.0)
    except ValueError:
        try:
            return max((parsedate_to_datetime(value).timestamp() - time.time()), 0.0)
        except (TypeError, ValueError, OverflowError):
            return fallback


class ScrapeHTTPError(RuntimeError):
    def __init__(self, status: int, url: str):
        self.status = status
        super().__init__(f"HTTP {status} from {url}")


class TransientRequestError(RuntimeError):
    """A retryable server or transport failure exhausted its local retries."""

    def __init__(
        self,
        url: str,
        cause: Exception,
        status: int | None = None,
    ):
        self.url = url
        self.status = status
        self.cause = cause
        label = f"HTTP {status}" if status is not None else type(cause).__name__
        super().__init__(f"{label} persisted after retries for {url}: {cause}")


class PaginationError(RuntimeError):
    """A response did not honor the requested search or page."""


def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: float,
    retries: int,
    **kwargs: Any,
) -> requests.Response:
    last_error: Exception | None = None
    last_status: int | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
            retryable_status = (
                response.status_code in (408, 425, 429)
                or 500 <= response.status_code < 600
            )
            if retryable_status:
                last_status = response.status_code
                last_error = ScrapeHTTPError(response.status_code, url)
                if attempt == retries:
                    break
                wait = retry_after_seconds(response, min(60.0, 2.0**attempt))
                logging.warning("HTTP %s from %s; retrying in %.1fs", response.status_code, url, wait)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else None
            last_status = status
            retryable_status = status in (408, 425, 429) if status is not None else False
            if status is not None and 400 <= status < 500 and not retryable_status:
                raise ScrapeHTTPError(status, url) from exc
            if attempt < retries:
                wait = min(60.0, 2.0**attempt)
                logging.warning("Request failed (%s/%s); retrying in %.1fs: %s", attempt, retries, wait, exc)
                time.sleep(wait)
    assert last_error is not None
    raise TransientRequestError(url, last_error, status=last_status) from last_error


def page_fingerprint(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def validate_page_fingerprint(job: Mapping[str, Any], ids: Iterable[str]) -> str:
    fingerprint = page_fingerprint(ids)
    if job.get("last_fingerprint") == fingerprint:
        raise PaginationError(f"Search {job['key']} returned the preceding page again")
    return fingerprint
