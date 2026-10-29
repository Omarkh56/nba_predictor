"""Verified Odds API GETs, including a standard-library TLS fallback on macOS.

Every request is saved to the raw archive (raw_archive.py) with the apiKey
removed: successful responses before the caller uses them, failures with
status="fetch_failed" and no payload.
"""

import copy
import http.client
import json
import os
import ssl
import time
from typing import Union
from urllib.parse import urlencode, urlsplit

import requests

import raw_archive
import team_codes


class MissingApiKey(requests.exceptions.RequestException):
    """ODDS_API_KEY is unset, so the request was skipped rather than sent. Every caller
    already handles RequestException, so each one skips odds instead of crashing."""


# Within one process, an identical request answered in the last CACHE_SECONDS is served from
# memory: the odds snapshot and the props model asked for the same props per game minutes
# apart, so every run paid for each game twice (96 of a 216-credit run on Oct 23). Sixty
# minutes matches nba_combined's own snapshot freshness rule (NBA_ODDS_MAX_AGE_MINUTES).
# Each scheduled or closing run is a new process, so it always starts with fresh prices.
CACHE_SECONDS = 3600
_cache = {}

# Waits before retrying a request whose connection dropped or timed out (seconds). On
# Oct 23 a Wi-Fi drop failed the last 8 games' props in one second. An HTTP answer
# (401 out of credits, 422, ...) and a substituted certificate are never retried.
RETRY_DELAYS = (5, 15, 40)
# Set once a request has used every retry on a dropped connection, cleared by any answer.
# While set, requests try once: in a long outage each one would otherwise wait out the
# full backoff, and a 12-game snapshot would take about half an hour to fail.
_network_down = False

# A 401 (out of credits, or a bad key) holds for the rest of the run: later paid requests
# fail at once, without a call, so callers move straight to ESPN (espn_odds.py).
_refused = False

# Each event's odds are paid for once per slate day. A rerun the same day (or a closing
# fallback) that would buy the same request again is refused with AlreadyBought, and the
# caller uses ESPN's free DraftKings feed instead. Spot it in the raw archive, where every
# paid answer is kept. NBA_ODDS_ONCE_PER_DAY=0 pays again.
ONCE_PER_DAY = os.environ.get("NBA_ODDS_ONCE_PER_DAY", "1").strip().lower() not in ("0", "false", "off", "no")


class AlreadyBought(requests.exceptions.RequestException):
    """This exact request was already answered (and paid for) today; see ONCE_PER_DAY."""


def _bought_today(endpoint: str, archive_params: dict) -> bool:
    digest = raw_archive.params_hash(raw_archive.strip_secrets(archive_params))
    folder = (raw_archive.root() / "odds_api" / endpoint
              / f"dt={team_codes.today_slate_date().isoformat()}")
    for path in sorted(folder.glob(f"*_{digest}*.json.gz"), reverse=True):
        try:
            record = raw_archive.load(path)
        except (OSError, ValueError):
            continue
        payload = record.get("payload")
        if record.get("status") == "ok" and isinstance(payload, dict) and payload.get("bookmakers"):
            return True
    return False


def _failure(message: str, http_status=None, retryable=False) -> requests.exceptions.RequestException:
    """A credential-free RequestException that still tells the archive the HTTP status."""
    error = requests.exceptions.RequestException(message)
    error.http_status = http_status
    error.retryable = retryable
    return error


def _archive_target(path: str, params: dict) -> tuple:
    """Stable archive endpoint for an Odds API path, with path ids moved into params:
    /v4/sports/{sport}/events -> "events"; .../events/{id}/odds -> "event_odds"."""
    parts = [p for p in path.split("/") if p]
    if parts[:1] == ["v4"]:
        parts = parts[1:]
    extra = {}
    if len(parts) >= 2 and parts[0] == "sports":
        extra["sport"], parts = parts[1], parts[2:]
    if len(parts) >= 3 and parts[0] == "events":
        extra["event_id"], parts = parts[1], ["event"] + parts[2:]
    return "_".join(parts) or "root", {**params, **extra}


def get_json(url: str, params: dict, timeout: int = 25) -> Union[dict, list]:
    """Keep certificate checks enabled and credentials out of error messages."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "api.the-odds-api.com":
        raise ValueError("Odds API requests require the official HTTPS host")
    endpoint, archive_params = _archive_target(parsed.path, params)
    cache_key = (url, tuple(sorted((str(k), str(v)) for k, v in params.items())))
    hit = _cache.get(cache_key)
    if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
        return copy.deepcopy(hit[1])  # already paid for (and archived) in this run
    global _network_down, _refused
    if _refused and endpoint != "events":  # the events list is free: it answers without credits
        raise _failure("Odds API refused this key earlier in the run (401: out of credits or bad key)", 401)
    if ONCE_PER_DAY and endpoint == "event_odds" and _bought_today(endpoint, archive_params):
        raise AlreadyBought("Odds API: this game's odds were already bought today")
    for delay in ((None,) if _network_down else (*RETRY_DELAYS, None)):
        try:
            payload, http_status = _fetch(parsed, url, params, timeout)
            break
        except (requests.exceptions.RequestException, ValueError) as exc:
            raw_archive.save("odds_api", endpoint, archive_params, None, status="fetch_failed",
                             http_status=getattr(exc, "http_status", None), error=str(exc))
            retryable = getattr(exc, "retryable", False)
            if delay is None or not retryable:
                _network_down = retryable  # out of retries, or an answer (even 401): reachable
                _refused = _refused or getattr(exc, "http_status", None) == 401
                raise
            time.sleep(delay)
    _network_down = False
    raw_archive.save("odds_api", endpoint, archive_params, payload, http_status=http_status)
    _cache[cache_key] = (time.monotonic(), copy.deepcopy(payload))
    return payload


def _fetch(parsed, url: str, params: dict, timeout: int) -> tuple:
    """Return (payload, http_status) for one verified GET."""
    if not params.get("apiKey"):
        raise MissingApiKey("ODDS_API_KEY not set — Odds API request skipped")
    try:
        response = requests.get(url, params=params, timeout=timeout,
                                verify=True, allow_redirects=False)
        response.raise_for_status()
        if isinstance(response.status_code, int) and response.status_code >= 300:
            raise requests.exceptions.HTTPError("Odds API redirects are not supported")
        return response.json(), raw_archive.http_status_of(response)
    except requests.exceptions.SSLError:
        # LibreSSL/urllib3 can fail while a standard SSLContext validates the
        # same host. This retries through a verified context, never verify=False.
        pass
    except requests.exceptions.RequestException as exc:
        dropped = isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))
        raise _failure(f"Odds API request failed ({type(exc).__name__})",
                       raw_archive.http_status_of(exc), retryable=dropped) from None

    bundle = (os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
              or requests.utils.DEFAULT_CA_BUNDLE_PATH)
    connection = http.client.HTTPSConnection(
        parsed.hostname, timeout=timeout, context=ssl.create_default_context(cafile=bundle))
    try:
        query = urlencode(params)
        path = parsed.path + ("?" + query if query else "")
        connection.request("GET", path, headers={"Accept": "application/json"})
        response = connection.getresponse()
        if response.status >= 300:
            raise _failure(f"Odds API HTTP status {response.status}", response.status)
        return json.loads(response.read()), response.status
    except requests.exceptions.RequestException:
        raise
    except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
        # A dropped socket is worth retrying; a certificate failure (an intercepting network) is not.
        raise _failure(f"Odds API verified HTTPS failed ({type(exc).__name__})",
                       retryable=isinstance(exc, OSError) and not isinstance(exc, ssl.SSLError)) from None
    finally:
        connection.close()
