"""Grading fails closed: no partial scores, no fetch failures logged as voids."""

import csv
from datetime import date
from unittest import mock

import pandas as pd
import pytest
import requests


@pytest.fixture
def cr():
    import checkresults

    return checkresults


def _finder(rows):
    finder = mock.Mock()
    finder.get_data_frames = lambda: [pd.DataFrame(rows)]
    return finder


def test_game_list_failure_is_none_not_empty(cr, monkeypatch):
    def down(**kwargs):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(cr.leaguegamefinder, "LeagueGameFinder", down)
    assert cr.get_games_on_date(date(2026, 10, 21)) is None
    with pytest.raises(cr.GradingFetchError):
        cr.build_actual_stats(date(2026, 10, 21))


def test_unfinished_games_block_grading(cr, monkeypatch):
    rows = [  # the May 7 CLE/DET row was logged at 36-25 mid-game
        {"GAME_ID": "1", "MATCHUP": "DET vs. CLE", "TEAM_ABBREVIATION": "DET", "PTS": 36, "WL": None},
        {"GAME_ID": "1", "MATCHUP": "CLE @ DET", "TEAM_ABBREVIATION": "CLE", "PTS": 25, "WL": None},
    ]
    monkeypatch.setattr(cr.leaguegamefinder, "LeagueGameFinder", lambda **k: _finder(rows))
    with pytest.raises(cr.GradingFetchError):
        cr.get_games_on_date(date(2026, 5, 7))
    assert cr.build_game_results(date(2026, 5, 7)) == {}


def test_one_failed_box_score_fails_the_run(cr, monkeypatch):
    monkeypatch.setattr(cr, "get_games_on_date", lambda d: ["g1", "g2"])

    def box(game_id):
        if game_id == "g2":
            raise requests.exceptions.ReadTimeout("slow")
        return pd.DataFrame([{"PLAYER_ID": 1, "PLAYER_NAME": "A", "PTS": 10, "REB": 2,
                              "AST": 3, "FG3M": 1, "MIN": "30:00"}])

    monkeypatch.setattr(cr, "get_box_score", box)
    with pytest.raises(cr.GradingFetchError):
        cr.build_actual_stats(date(2026, 10, 21))


def test_main_writes_nothing_when_the_fetch_fails(cr, monkeypatch, tmp_path):
    monkeypatch.setenv("NBA_RUN_LOG", str(tmp_path / "runs.jsonl"))

    def fail(check_date):
        raise cr.GradingFetchError("stats.nba.com down")

    appended = mock.Mock()
    monkeypatch.setattr(cr, "build_actual_stats", fail)
    monkeypatch.setattr(cr, "append_to_log", appended)
    monkeypatch.setattr(cr, "_load_game_predictions_db", lambda d: [])
    picks = [{"player": "A", "market": "PTS", "line": 10.5, "pick": "OVER", "projection": 12.0}]
    assert cr.main(date(2026, 10, 21), predictions=picks) == 2
    appended.assert_not_called()


def test_game_grading_normalizes_codes_and_fixes_the_legacy_sign(cr):
    results = {("SAS", "NYK"): {"game_id": "g", "home": "SAS", "away": "NYK", "home_score": 100,
                                "away_score": 105, "margin": -5, "winner": "NYK"}}
    preds = [{"home_abbr": "SA", "away_abbr": "NY", "winner_pick": "NY", "spread": -6.1,
              "spread_pick": "NY"}]
    graded = cr.check_game_predictions(preds, results)[0]
    assert graded["winner_hit"].startswith("HIT")
    # NY was favoured, so the home line is +6.1 and a 5-point win doesn't cover.
    assert graded["spread"] == 6.1
    assert graded["spread_hit"].startswith("MISS")


def test_game_log_upsert_is_idempotent(cr, monkeypatch, tmp_path):
    path = tmp_path / "game_log.csv"
    monkeypatch.setattr(cr, "GAME_LOG_FILE", str(path))
    graded = [{"home": "SAS", "away": "NYK", "home_score": 100, "away_score": 105,
               "winner": "NYK", "winner_pick": "NYK", "winner_hit": "HIT ✓", "spread": 6.1,
               "spread_pick": "NYK", "home_margin": -5, "spread_hit": "MISS ✗"}]
    cr.append_game_to_log(graded, date(2026, 6, 5))
    cr.append_game_to_log(graded, date(2026, 6, 5))
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["spread_hit"] == "MISS"
