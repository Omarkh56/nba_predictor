"""Dummy API keys so unit tests can import modules without a local .env."""

import os

import pytest

os.environ.setdefault("ODDS_API_KEY", "test-odds-api-key")
os.environ.setdefault("API_FOOTBALL_KEY", "test-api-football-key")


@pytest.fixture(scope="session")
def _raw_archive_session_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("raw_archive")


@pytest.fixture(autouse=True)
def _raw_archive_outside_repo(monkeypatch, _raw_archive_session_dir):
    """Every test archives into a temp dir, never data/raw. Archiving stays on so the
    whole suite exercises it (a replay worker test switches it off process-wide)."""
    monkeypatch.setenv("NBA_RAW_ARCHIVE", "1")
    monkeypatch.setenv("NBA_RAW_ARCHIVE_DIR", str(_raw_archive_session_dir))


@pytest.fixture(autouse=True)
def _no_odds_retry_waits(monkeypatch):
    """Outage tests expect one attempt; tests of the retry itself set their own delays."""
    import odds_api
    monkeypatch.setattr(odds_api, "RETRY_DELAYS", ())
    monkeypatch.setattr(odds_api, "_cache", {})  # each test sees its own mocked responses
    monkeypatch.setattr(odds_api, "_network_down", False)
    monkeypatch.setattr(odds_api, "_refused", False)
    # The archive dir is shared by the whole session, so "already bought today" would leak
    # between tests; the tests of it switch it back on.
    monkeypatch.setattr(odds_api, "ONCE_PER_DAY", False)


@pytest.fixture(autouse=True)
def _no_espn_odds(monkeypatch):
    """Tests never reach ESPN: without the fallback, odds failures behave as they always
    have. tests/test_espn_odds.py switches it on with mocked responses."""
    import espn_odds
    monkeypatch.setenv("NBA_ESPN_ODDS", "0")
    monkeypatch.setattr(espn_odds, "_cache", {})
