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
