"""Verified Odds API GETs, including a standard-library TLS fallback on macOS.

Every request is saved to the raw archive (raw_archive.py) with the apiKey
removed: successful responses before the caller uses them, failures with
status="fetch_failed" and no payload.
"""

import http.client
import json
import os
import ssl
from typing import Union
from urllib.parse import urlencode, urlsplit

import requests

import raw_archive


class MissingApiKey(requests.exceptions.RequestException):
    """ODDS_API_KEY is unset, so the request was skipped rather than sent. Every caller
    already handles RequestException, so each one skips odds instead of crashing."""


def _failure(message: str, http_status=None) -> requests.exceptions.RequestException:
    """A credential-free RequestException that still tells the archive the HTTP status."""
    error = requests.exceptions.RequestException(message)
    error.http_status = http_status
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
    try:
        payload, http_status = _fetch(parsed, url, params, timeout)
    except (requests.exceptions.RequestException, ValueError) as exc:
        raw_archive.save("odds_api", endpoint, archive_params, None, status="fetch_failed",
                         http_status=getattr(exc, "http_status", None), error=str(exc))
        raise
    raw_archive.save("odds_api", endpoint, archive_params, payload, http_status=http_status)
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
        raise _failure(f"Odds API request failed ({type(exc).__name__})",
                       raw_archive.http_status_of(exc)) from None

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
        raise requests.exceptions.RequestException(
            f"Odds API verified HTTPS failed ({type(exc).__name__})") from None
    finally:
        connection.close()
