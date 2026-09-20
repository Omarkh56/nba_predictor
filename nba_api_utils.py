"""
nba_api_utils.py — shared NBA API retry / backoff helpers.

All three predictor modules (newnbapredictor, playerlinepredictor,
ucl_predictor) previously duplicated exponential-backoff retry loops
with slight variations.  This module provides a single canonical
implementation imported by all of them.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

import requests

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
