"""Season rollover: the suite and the live artifacts agree in both seasons.

On Oct 1, 2026 current_season() moved from 2025-26 to 2026-27. That switched on
the Platt probability calibration (training cutoff 2026-06-13) and broke 12
tests that assumed raw probabilities. These tests pin the gate itself, so the
suite passes under NBA_SEASON=2025-26 and NBA_SEASON=2026-27 alike.
"""

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from unittest import mock

import pytest

import config

ROOT = Path(__file__).resolve().parent.parent
PLATT = {
    "version": "v9-probability-1",
    "method": "platt",
    "training_cutoff": "2026-06-13",
    "markets": {"PTS": {"intercept": 0.0, "slope": 0.5}},
}


@pytest.fixture
def predictor():
    with mock.patch(
        "requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")
    ):
        import playerlinepredictor

        yield playerlinepredictor


@pytest.mark.parametrize(
    "season,active", [("2025-26", False), ("2026-27", True), ("2027-28", True)]
)
def test_artifact_is_active_only_after_its_training_cutoff(predictor, season, active):
    result = predictor._artifact_for_season(PLATT, season)
    assert (result == PLATT) is active
    assert (result == {}) is (not active)


@pytest.mark.parametrize(
    "artifact", [{}, None, {"method": "platt"}, {"training_cutoff": "not-a-date"}]
)
def test_artifact_without_valid_cutoff_stays_off(predictor, artifact):
    assert predictor._artifact_for_season(artifact, "2026-27") == {}


def test_platt_changes_probability_only_when_active(predictor, monkeypatch):
    monkeypatch.setattr(predictor, "PROBABILITY_CALIBRATION", {})
    raw = predictor.evaluate_prop(20.5, 25.0, "player_points", sd=5.0)
    monkeypatch.setattr(predictor, "PROBABILITY_CALIBRATION", PLATT)
    calibrated = predictor.evaluate_prop(20.5, 25.0, "player_points", sd=5.0)
    assert raw["pick"] == calibrated["pick"] == "OVER"
    # A slope of 0.5 halves the logit: closer to 50%, same side.
    assert 50.0 < calibrated["over_prob"] < raw["over_prob"]


@pytest.mark.parametrize(
    "today,season,validation",
    [(date(2026, 9, 30), "2025-26", "2024-25"), (date(2026, 10, 1), "2026-27", "2025-26")],
)
def test_rollover_moves_season_and_validation_window(monkeypatch, today, season, validation):
    monkeypatch.delenv("NBA_SEASON", raising=False)
    assert config.current_season(today) == season
    assert config.validation_season(today) == validation
    assert config.training_seasons(5, today)[-1] == validation


def _cutoff(name):
    path = ROOT / name
    if not path.is_file():
        pytest.skip(f"{name} not present")
    return date.fromisoformat(json.loads(path.read_text())["training_cutoff"])


@pytest.mark.parametrize("season", ["2025-26", "2026-27"])
def test_live_artifacts_follow_the_season_gate(season):
    """Import the real module under each season, the way the CLI does."""
    prob_cutoff = _cutoff("probability_calibration.json")
    resid_cutoff = _cutoff("residual_corrections.json")
    start = date(int(season[:4]), config.SEASON_START_MONTH, 1)
    env = {**os.environ, "NBA_SEASON": season}
    code = (
        "import json, playerlinepredictor as p; "
        "print(json.dumps([bool(p.PROBABILITY_CALIBRATION), bool(p.RESIDUAL_CORRECTIONS)]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
        timeout=180,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    prob_on, resid_on = json.loads(out.stdout.strip().splitlines()[-1])
    assert prob_on is (start > prob_cutoff)
    assert resid_on is (start > resid_cutoff)
