"""
nba_api_utils.py — shared NBA API retry / backoff helpers.

All three predictor modules (newnbapredictor, playerlinepredictor,
ucl_predictor) previously duplicated exponential-backoff retry loops
with slight variations.  This module provides a single canonical
implementation imported by all of them.

It also owns how nba_api talks to stats.nba.com: configure_stats_http() sets
the live request headers, and importing this module installs ArchivingSession
so every stats.nba.com response is saved to the raw archive (raw_archive.py).
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar
from urllib.parse import urlsplit

import requests

import raw_archive

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Live stats.nba.com request settings (moved from newnbapredictor.py, which used
# them first). Certificate verification stays on; see the TLS note there.
STATS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
    "DNT": "1",
}
STATS_TIMEOUT = 90


def configure_stats_http() -> None:
    """Apply the live headers and timeout to nba_api's stats.nba.com client."""
    import nba_api.stats.library.http as nba_http  # resolved at call time so test stubs apply

    nba_http.STATS_HEADERS.update(STATS_HEADERS)
    nba_http.STATS_TIMEOUT = STATS_TIMEOUT


class ArchivingSession(requests.Session):
    """The session nba_api sends every stats.nba.com request through, archiving each one.

    One session covers every endpoint and call site. The endpoint name and params
    come from the request itself, and the payload is the raw JSON body (what
    Endpoint.get_dict() returns), saved before nba_api parses it. Requests and
    responses are otherwise untouched.
    """

    def request(self, method, url, params=None, **kwargs):
        endpoint = urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]
        try:
            response = super().request(method, url, params=params, **kwargs)
        except Exception as exc:
            raw_archive.save("nba_api", endpoint, params, None, status="fetch_failed",
                             error=type(exc).__name__)
            raise
        if raw_archive.enabled():
            status = raw_archive.http_status_of(response)
            try:
                payload = response.json() if status is not None and 200 <= status < 300 else None
            except ValueError:
                payload = None
            raw_archive.save("nba_api", endpoint, params, payload, http_status=status,
                             status="ok" if payload is not None else "fetch_failed",
                             error=None if payload is not None else f"HTTP {status}, no JSON body")
        return response


def install_raw_archive() -> bool:
    """Route nba_api's stats.nba.com requests through ArchivingSession (idempotent)."""
    try:
        from nba_api.stats.library.http import NBAStatsHTTP
    except ImportError:  # tests stub nba_api without its HTTP layer
        return False
    if not isinstance(NBAStatsHTTP.get_session(), ArchivingSession):
        NBAStatsHTTP.set_session(ArchivingSession())
    return True


install_raw_archive()


@dataclass
class DataFrameCallResult:
    """Outcome of a DataFrame API call without conflating failure and no rows."""

    status: str  # "ok" | "ok_empty" | "failed"
    frame: object
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "ok_empty"}


def api_call_with_retry(
    fn: Callable[[], T],
    retries: int = 3,
    base_delay: float = 5.0,
    post_success_sleep: float = 2.0,
    label: str = "",
) -> T:
    """Call fn() with exponential-backoff retries.

    Args:
        fn: Zero-argument callable to attempt.
        retries: Maximum number of attempts.
        base_delay: Seconds to wait before the first retry; doubles each time.
        post_success_sleep: Seconds to sleep after a successful call (rate-limit courtesy).
        label: Optional description for log messages.

    Returns:
        The return value of fn() on success.

    Raises:
        RuntimeError: If all attempts fail.
    """
    prefix = f"[{label}] " if label else ""
    last_exc: Optional[Exception] = None

    for attempt in range(retries):
        try:
            result = fn()
            if post_success_sleep > 0:
                time.sleep(post_success_sleep)
            return result
        except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
            last_exc = exc
            wait = base_delay * (2**attempt)
            logger.warning(
                "%sAttempt %d/%d failed (%s: %s) — retrying in %.0fs",
                prefix,
                attempt + 1,
                retries,
                type(exc).__name__,
                exc,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError(f"{prefix}NBA API call failed after {retries} attempts: {last_exc}")


def safe_dataframe_call(
    fn: Callable[[], T],
    retries: int = 3,
    label: str = "",
):
    """Like api_call_with_retry but returns an empty pandas DataFrame on total failure."""
    import pandas as pd

    try:
        return api_call_with_retry(fn, retries=retries, label=label)
    except RuntimeError as exc:
        logger.warning("Giving up after retries — %s", exc)
        return pd.DataFrame()


def dataframe_call_result(
    fn: Callable[[], T],
    retries: int = 3,
    label: str = "",
) -> DataFrameCallResult:
    """Run a DataFrame call and preserve whether an empty result was valid.

    The older :func:`safe_dataframe_call` intentionally retains its legacy
    empty-on-failure behavior for callers that have not migrated yet. New
    model data paths should use this function so failures are never cached or
    interpreted as a player having zero games.
    """
    import pandas as pd

    try:
        frame = api_call_with_retry(fn, retries=retries, label=label)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("NBA API call did not return a DataFrame")
        return DataFrameCallResult("ok_empty" if frame.empty else "ok", frame)
    except (RuntimeError, TypeError) as exc:
        logger.warning("Giving up after retries — %s", exc)
        return DataFrameCallResult("failed", None, error=f"{type(exc).__name__}: {exc}")
