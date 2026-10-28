"""TLS fallback must authenticate the host and redact credential-bearing URLs."""

import ssl
from unittest import mock

import pytest
import requests

import odds_api

URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
PARAMS = {"apiKey": "private-test-key"}


def test_primary_client_keeps_verification_and_disables_redirects(monkeypatch):
    response = mock.Mock(status_code=200, json=lambda: [{"id": "event"}])
    get = mock.Mock(return_value=response)
    monkeypatch.setattr(odds_api.requests, "get", get)
    assert odds_api.get_json(URL, PARAMS, timeout=20) == [{"id": "event"}]
    assert get.call_args.kwargs["verify"] is True
    assert get.call_args.kwargs["allow_redirects"] is False


@pytest.fixture
def fallback(monkeypatch):
    monkeypatch.setattr(odds_api.requests, "get", mock.Mock(side_effect=requests.exceptions.SSLError("bad transport")))
    connection = mock.Mock()
    connection.getresponse.return_value = mock.Mock(status=200, read=lambda: b'{"bookmakers":[]}')
    constructor = mock.Mock(return_value=connection)
    monkeypatch.setattr(odds_api.http.client, "HTTPSConnection", constructor)
    return constructor, connection


def test_standard_fallback_uses_verified_context_and_closes_connection(fallback):
    constructor, connection = fallback
    assert odds_api.get_json(URL, PARAMS) == {"bookmakers": []}
    context = constructor.call_args.kwargs["context"]
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    assert constructor.call_args.args[0] == "api.the-odds-api.com"
    assert connection.request.call_args.args[0] == "GET"
    connection.close.assert_called_once()


@pytest.mark.parametrize("status", [301, 401, 429, 500])
def test_fallback_http_errors_and_redirects_are_sanitized(status, fallback):
    _, connection = fallback
    connection.getresponse.return_value.status = status
    with pytest.raises(requests.exceptions.RequestException) as exc:
        odds_api.get_json(URL, PARAMS)
    assert str(status) in str(exc.value)
    assert "private-test-key" not in str(exc.value)
    connection.close.assert_called_once()


def test_invalid_fallback_certificate_is_rejected_and_redacted(fallback):
    _, connection = fallback
    connection.request.side_effect = ssl.SSLCertVerificationError(1, "private-test-key")
    with pytest.raises(requests.exceptions.RequestException) as exc:
        odds_api.get_json(URL, PARAMS)
    assert "private-test-key" not in str(exc.value)
    connection.close.assert_called_once()


def test_primary_error_url_is_not_exposed(monkeypatch):
    monkeypatch.setattr(odds_api.requests, "get", mock.Mock(side_effect=requests.ConnectionError(
        URL + "?apiKey=private-test-key")))
    with pytest.raises(requests.exceptions.RequestException) as exc:
        odds_api.get_json(URL, PARAMS)
    assert "private-test-key" not in str(exc.value)


def test_identical_request_in_one_run_is_paid_for_once(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(odds_api.time, "monotonic", lambda: clock[0])
    get = mock.Mock(side_effect=lambda *a, **k: mock.Mock(status_code=200, json=lambda: {"bookmakers": [1]}))
    monkeypatch.setattr(odds_api.requests, "get", get)
    first = odds_api.get_json(URL, PARAMS)
    first["bookmakers"].append("mutated by a caller")
    assert odds_api.get_json(URL, PARAMS) == {"bookmakers": [1]} and get.call_count == 1
    odds_api.get_json(URL, {**PARAMS, "regions": "us,uk"})  # a different request is sent
    clock[0] += odds_api.CACHE_SECONDS + 1  # stale after the freshness window
    odds_api.get_json(URL, PARAMS)
    assert get.call_count == 3


def test_dropped_connection_is_retried_but_http_errors_are_not(monkeypatch):
    monkeypatch.setattr(odds_api, "RETRY_DELAYS", (0, 0))
    monkeypatch.setattr(odds_api.time, "sleep", lambda s: None)
    ok = mock.Mock(status_code=200, json=lambda: {"bookmakers": []})
    flaky = mock.Mock(side_effect=[requests.exceptions.ConnectionError("drop"), ok])
    monkeypatch.setattr(odds_api.requests, "get", flaky)
    assert odds_api.get_json(URL, PARAMS) == {"bookmakers": []} and flaky.call_count == 2

    odds_api._cache.clear()  # later phases reuse the URL; they must reach the network
    out_of_credits = requests.exceptions.HTTPError("401", response=mock.Mock(status_code=401))
    rejected = mock.Mock(side_effect=out_of_credits)
    monkeypatch.setattr(odds_api.requests, "get", rejected)
    with pytest.raises(requests.exceptions.RequestException):
        odds_api.get_json(URL, PARAMS)
    assert rejected.call_count == 1  # an HTTP answer is final; retrying would only waste credits

    down = mock.Mock(side_effect=requests.exceptions.ConnectionError("still down"))
    monkeypatch.setattr(odds_api.requests, "get", down)
    with pytest.raises(requests.exceptions.RequestException):
        odds_api.get_json(URL, PARAMS)
    assert down.call_count == 3  # first try plus one per delay


def test_substituted_certificate_is_not_retried(fallback, monkeypatch):
    monkeypatch.setattr(odds_api, "RETRY_DELAYS", (0, 0))
    monkeypatch.setattr(odds_api.time, "sleep", lambda s: None)
    constructor, connection = fallback
    connection.request.side_effect = ssl.SSLCertVerificationError(1, "self signed certificate in chain")
    with pytest.raises(requests.exceptions.RequestException):
        odds_api.get_json(URL, PARAMS)
    assert constructor.call_count == 1


@pytest.mark.parametrize("key", [None, ""])
def test_missing_key_skips_the_request(monkeypatch, key):
    get = mock.Mock()
    monkeypatch.setattr(odds_api.requests, "get", get)
    with pytest.raises(requests.exceptions.RequestException, match="ODDS_API_KEY not set"):
        odds_api.get_json(URL, {"apiKey": key})
    get.assert_not_called()


@pytest.mark.parametrize("url", ["http://api.the-odds-api.com/v4", "https://unknown.example/v4"])
def test_unauthenticated_or_wrong_host_is_rejected(url):
    with pytest.raises(ValueError):
        odds_api.get_json(url, PARAMS)


def test_after_one_request_exhausts_its_retries_the_rest_try_once(monkeypatch):
    monkeypatch.setattr(odds_api, "RETRY_DELAYS", (0, 0))
    monkeypatch.setattr(odds_api.time, "sleep", lambda s: None)
    down = mock.Mock(side_effect=requests.exceptions.ConnectionError("no route"))
    monkeypatch.setattr(odds_api.requests, "get", down)
    for _ in range(2):
        with pytest.raises(requests.exceptions.RequestException):
            odds_api.get_json(URL, PARAMS)
    assert down.call_count == 3 + 1  # full backoff once, then a single attempt

    back = mock.Mock(return_value=mock.Mock(status_code=200, json=lambda: {"bookmakers": []}))
    monkeypatch.setattr(odds_api.requests, "get", back)
    odds_api.get_json(URL, PARAMS)
    assert odds_api._network_down is False  # an answer: later drops get the full backoff again
