"""Dummy API keys so unit tests can import modules without a local .env."""

import os

os.environ.setdefault("ODDS_API_KEY", "test-odds-api-key")
os.environ.setdefault("API_FOOTBALL_KEY", "test-api-football-key")
