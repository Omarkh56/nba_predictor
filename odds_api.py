"""Verified Odds API GETs, including a standard-library TLS fallback on macOS."""

import http.client
import json
import os
import ssl
from typing import Union
from urllib.parse import urlencode, urlsplit

import requests


def get_json(url: str, params: dict, timeout: int = 25) -> Union[dict, list]:
    """Keep certificate checks enabled and credentials out of error messages."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "api.the-odds-api.com":
        raise ValueError("Odds API requests require the official HTTPS host")
    try:
        response = requests.get(url, params=params, timeout=timeout,
                                verify=True, allow_redirects=False)
        response.raise_for_status()
        if isinstance(response.status_code, int) and response.status_code >= 300:
            raise requests.exceptions.HTTPError("Odds API redirects are not supported")
        return response.json()
    except requests.exceptions.SSLError:
        # LibreSSL/urllib3 can fail while a standard SSLContext validates the
        # same host. This retries through a verified context, never verify=False.
        pass
    except requests.exceptions.RequestException as exc:
        raise requests.exceptions.RequestException(
            f"Odds API request failed ({type(exc).__name__})") from None

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
            raise requests.exceptions.RequestException(f"Odds API HTTP status {response.status}")
        return json.loads(response.read())
    except requests.exceptions.RequestException:
        raise
    except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
        raise requests.exceptions.RequestException(
            f"Odds API verified HTTPS failed ({type(exc).__name__})") from None
    finally:
        connection.close()
