"""
Round-trip tests for predictions_db.py.

All tests use a tmp_path SQLite file — the real predictions.db is never
touched.  Pass path= explicitly to every function call.
"""
import sys
import os
from datetime import date

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from predictions_db import (
    upsert_prop_predictions,
    upsert_game_predictions,
    load_prop_predictions,
    load_game_predictions,
    available_dates,
)


_D1 = date(2024, 1, 15)
_D2 = date(2024, 1, 16)

_PROP_ROW = {
    "game":       "LAL @ BOS",
    "player":     "LeBron James",
    "market":     "PTS",
    "line":       27.5,
    "pick":       "OVER",
    "projection": 29.1,
    "confidence": 0.72,
    "edge_pct":   0.06,
    "flags":      "",
    "po_count":   3,
    "avg_min":    35.0,
    "book_count": 5,
    "usg_pct":    0.30,
    "game_spread": -4.5,
    "game_total":  228.0,
    "game_pace":   100.5,
    "fav_win_pct": 0.62,
}

_GAME_ROW = {
    "home_abbr":    "BOS",
    "away_abbr":    "LAL",
    "winner_pick":  "BOS",
    "spread":       -4.5,
    "spread_pick":  "BOS -4.5",
    "win_pct":      0.62,
    "total":        228.0,
}


# ===========================================================================
# upsert_prop_predictions / load_prop_predictions
# ===========================================================================
class TestPropPredictions:
    def test_empty_rows_returns_zero(self, tmp_path):
        db = tmp_path / "test.db"
        count = upsert_prop_predictions([], _D1, path=db)
        assert count == 0

    def test_single_row_round_trip(self, tmp_path):
        db = tmp_path / "test.db"
        upsert_prop_predictions([_PROP_ROW], _D1, path=db)
        rows = load_prop_predictions(_D1, path=db)
        assert len(rows) == 1
        r = rows[0]
        assert r["player"] == "LeBron James"
        assert r["market"] == "PTS"
        assert r["pick"]   == "OVER"
        assert r["line"]   == pytest.approx(27.5)
        assert r["projection"] == pytest.approx(29.1)

    def test_multiple_rows(self, tmp_path):
        db = tmp_path / "test.db"
        rows_in = [
            {**_PROP_ROW, "player": "Jayson Tatum",  "market": "PTS", "confidence": 0.80},
            {**_PROP_ROW, "player": "LeBron James",  "market": "REB", "confidence": 0.65},
            {**_PROP_ROW, "player": "LeBron James",  "market": "AST", "confidence": 0.55},
        ]
        upsert_prop_predictions(rows_in, _D1, path=db)
        rows_out = load_prop_predictions(_D1, path=db)
        assert len(rows_out) == 3

    def test_upsert_overwrites_on_same_primary_key(self, tmp_path):
        # Primary key = (date, player, market); second write should replace
        db = tmp_path / "test.db"
        upsert_prop_predictions([_PROP_ROW], _D1, path=db)
        updated = {**_PROP_ROW, "projection": 31.0}
        upsert_prop_predictions([updated], _D1, path=db)
        rows = load_prop_predictions(_D1, path=db)
        assert len(rows) == 1                        # still one row
        assert rows[0]["projection"] == pytest.approx(31.0)

    def test_different_dates_isolated(self, tmp_path):
        db = tmp_path / "test.db"
        upsert_prop_predictions([_PROP_ROW], _D1, path=db)
        upsert_prop_predictions([{**_PROP_ROW, "player": "Anthony Davis", "market": "PTS"}],
                                _D2, path=db)
        # D1 should have only LeBron, D2 only Anthony Davis
        d1 = load_prop_predictions(_D1, path=db)
        d2 = load_prop_predictions(_D2, path=db)
        assert len(d1) == 1 and d1[0]["player"] == "LeBron James"
        assert len(d2) == 1 and d2[0]["player"] == "Anthony Davis"

    def test_missing_db_returns_empty_list(self, tmp_path):
        db = tmp_path / "does_not_exist.db"
        rows = load_prop_predictions(_D1, path=db)
        assert rows == []

    def test_no_rows_for_missing_date(self, tmp_path):
        db = tmp_path / "test.db"
        upsert_prop_predictions([_PROP_ROW], _D1, path=db)
        rows = load_prop_predictions(_D2, path=db)
        assert rows == []

    def test_return_count_matches_input(self, tmp_path):
        db = tmp_path / "test.db"
        rows_in = [
            {**_PROP_ROW, "player": "P1", "market": "PTS"},
            {**_PROP_ROW, "player": "P2", "market": "PTS"},
        ]
        count = upsert_prop_predictions(rows_in, _D1, path=db)
        assert count == 2


# ===========================================================================
# upsert_game_predictions / load_game_predictions
# ===========================================================================
class TestGamePredictions:
    def test_empty_rows_returns_zero(self, tmp_path):
        db = tmp_path / "test.db"
        assert upsert_game_predictions([], _D1, path=db) == 0

    def test_single_game_round_trip(self, tmp_path):
        db = tmp_path / "test.db"
        upsert_game_predictions([_GAME_ROW], _D1, path=db)
        rows = load_game_predictions(_D1, path=db)
        assert len(rows) == 1
        r = rows[0]
        assert r["home_abbr"]   == "BOS"
        assert r["away_abbr"]   == "LAL"
        assert r["winner_pick"] == "BOS"
        assert r["spread"]      == pytest.approx(-4.5)
        assert r["spread_pick"] == "BOS -4.5"

    def test_multiple_games_same_date(self, tmp_path):
        db = tmp_path / "test.db"
        rows_in = [
            {**_GAME_ROW, "home_abbr": "BOS", "away_abbr": "LAL"},
            {**_GAME_ROW, "home_abbr": "GSW", "away_abbr": "MIL"},
            {**_GAME_ROW, "home_abbr": "DEN", "away_abbr": "PHX"},
        ]
        upsert_game_predictions(rows_in, _D1, path=db)
        rows = load_game_predictions(_D1, path=db)
        assert len(rows) == 3

    def test_upsert_overwrites_on_same_primary_key(self, tmp_path):
        db = tmp_path / "test.db"
        upsert_game_predictions([_GAME_ROW], _D1, path=db)
        updated = {**_GAME_ROW, "win_pct": 0.75}
        upsert_game_predictions([updated], _D1, path=db)
        rows = load_game_predictions(_D1, path=db)
        assert len(rows) == 1
        assert rows[0]["spread"] == pytest.approx(-4.5)   # other fields unchanged

    def test_different_dates_isolated(self, tmp_path):
        db = tmp_path / "test.db"
        upsert_game_predictions([_GAME_ROW], _D1, path=db)
        upsert_game_predictions([{**_GAME_ROW, "home_abbr": "MIA", "away_abbr": "NYK"}],
                                _D2, path=db)
        d1 = load_game_predictions(_D1, path=db)
        d2 = load_game_predictions(_D2, path=db)
        assert len(d1) == 1 and d1[0]["home_abbr"] == "BOS"
        assert len(d2) == 1 and d2[0]["home_abbr"] == "MIA"

    def test_missing_db_returns_empty(self, tmp_path):
        db = tmp_path / "no_db.db"
        assert load_game_predictions(_D1, path=db) == []

    def test_none_spread_handled(self, tmp_path):
        # spread=None should not crash and round-trips as None
        db = tmp_path / "test.db"
        row = {**_GAME_ROW, "spread": None, "spread_pick": None}
        upsert_game_predictions([row], _D1, path=db)
        rows = load_game_predictions(_D1, path=db)
        assert len(rows) == 1
        assert rows[0]["spread"] is None


# ===========================================================================
# available_dates
# ===========================================================================
class TestAvailableDates:
    def test_empty_db_returns_empty_list(self, tmp_path):
        db = tmp_path / "test.db"
        # Initialise the schema by writing 0 rows
        upsert_prop_predictions([], _D1, path=db)
        assert available_dates(path=db) == []

    def test_single_date_present(self, tmp_path):
        db = tmp_path / "test.db"
        upsert_prop_predictions([_PROP_ROW], _D1, path=db)
        dates = available_dates(path=db)
        assert dates == [_D1.isoformat()]

    def test_multiple_dates_sorted(self, tmp_path):
        db = tmp_path / "test.db"
        upsert_prop_predictions([_PROP_ROW], _D2, path=db)
        upsert_prop_predictions([{**_PROP_ROW, "market": "REB"}], _D1, path=db)
        dates = available_dates(path=db)
        assert dates == sorted([_D1.isoformat(), _D2.isoformat()])

    def test_missing_file_returns_empty(self, tmp_path):
        assert available_dates(path=tmp_path / "ghost.db") == []
