"""HTTP client + retry policy shared by scrapers and downloaders.

Two layers of retry exist in the system:
  * here: short, in-process retries for blips (dropped connection, 5xx, 429),
    a few seconds to a couple of minutes total;
  * the job queue (db.py): long backoff across minutes/hours for outages.
"""
from __future__ import annotations

import logging
import random
import ssl
import threading
import time

import httpx
import truststore

log = logging.getLogger(__name__)

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class TransientHTTPError(Exception):
    """Retryable failure that outlived the in-process retries."""


class PermanentHTTPError(Exception):
    """4xx (other than 408/429): retrying the same request won't help."""

    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status} for {url}")
        self.status = status


class SourceChanged(Exception):
    """The remote file changed (ETag mismatch) while we were downloading it."""


def make_client(user_agent: str, timeout: float = 60.0) -> httpx.Client:
    # house.mi.gov omits an intermediate certificate. The OS trust store
    # (via truststore) fetches it like a browser does; certifi alone fails.
    # We never disable verification.
    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT) # object containing rules and trusted certificate information used when establishing TLS connections
    return httpx.Client(
        headers={"User-Agent": user_agent},
        timeout=httpx.Timeout(timeout, connect=15.0),
        verify=ctx,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
    )


def request(client: httpx.Client, method: str, url: str, *, tries: int = 6,
            base_delay: float = 1.0, stop: threading.Event | None = None,
            **kw) -> httpx.Response:
    """Send a request, retrying transient failures with jittered backoff.

    Returns the response for 2xx/3xx. Raises PermanentHTTPError for
    non-retryable 4xx and TransientHTTPError once retries are exhausted.
    """
    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            resp = client.request(method, url, **kw)
            if resp.status_code < 400:
                return resp
            if resp.status_code not in RETRY_STATUS:
                raise PermanentHTTPError(resp.status_code, url)
            last = TransientHTTPError(f"HTTP {resp.status_code} for {url}")
            retry_after = resp.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else None
        except httpx.TransportError as e:  # connect/read timeouts, resets, DNS...
            last, delay = e, None
        if attempt == tries:
            break
        delay = delay if delay is not None else base_delay * 2 ** (attempt - 1)
        delay = min(delay, 60.0) * random.uniform(0.5, 1.5)
        log.warning("retry %d/%d in %.1fs: %s %s: %s", attempt, tries - 1, delay,
                    method, url, last)
        if stop is not None:
            if stop.wait(delay):
                raise InterruptedError("shutdown requested")
        else:
            time.sleep(delay)
    raise TransientHTTPError(f"{method} {url} failed after {tries} tries: {last}") from last


def get_range(client: httpx.Client, url: str, offset: int, length: int,
              etag: str | None = None, **kw) -> bytes:
    """Fetch exactly `length` bytes at `offset`, verifying the server honoured it.

    With an ETag we send If-Range: if the file changed, the server answers 200
    with the whole new file instead of 206, and we raise SourceChanged rather
    than stitching bytes from two different versions together.
    """
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
    if etag:
        headers["If-Range"] = etag
    for _ in range(3):
        resp = request(client, "GET", url, headers=headers, **kw)
        if resp.status_code == 200:
            raise SourceChanged(f"server ignored range (file changed?) for {url}")
        if resp.status_code != 206:
            raise TransientHTTPError(f"unexpected HTTP {resp.status_code} for range")
        data = resp.content
        if len(data) == length:
            return data
        log.warning("short range read %d/%d at %d for %s", len(data), length, offset, url)
    raise TransientHTTPError(f"range {offset}+{length} kept coming back short")
