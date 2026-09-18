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


@pytest.mark.parametrize("url", ["http://api.the-odds-api.com/v4", "https://unknown.example/v4"])
def test_unauthenticated_or_wrong_host_is_rejected(url):
    with pytest.raises(ValueError):
        odds_api.get_json(url, PARAMS)
