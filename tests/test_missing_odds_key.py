"""A missing ODDS_API_KEY must warn and skip odds fetching, not exit at import."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Runs in a fresh interpreter so the import is real. load_dotenv is disabled so a
# developer's local .env can't supply the key, and any network call fails the test.
SCRIPT = """
import dotenv
from unittest import mock
dotenv.load_dotenv = lambda *a, **k: False
with mock.patch("requests.sessions.Session.request",
                side_effect=AssertionError("Network forbidden")):
    import config, oddstracker, playerlinepredictor
    assert config.ODDS_API_KEY is None
    assert playerlinepredictor.ODDS_API_KEY is None and oddstracker.ODDS_API_KEY is None
    assert playerlinepredictor._fetch_slate() == (None, None)
print("IMPORT_OK")
"""


def test_missing_odds_api_key_does_not_raise_at_import():
    env = {k: v for k, v in os.environ.items() if k != "ODDS_API_KEY"}
    result = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "IMPORT_OK" in result.stdout
    assert "ODDS_API_KEY not set" in result.stderr
