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
from typing import Callable, Optional, TypeVar

import requests

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
