import sqlite3

import pytest

import predictions_db
from finance import (
    FINANCE_FIELDS,
    american_to_decimal,
    decimal_to_implied,
    ensure_finance_fields,
    make_bet_id,
)


@pytest.mark.parametrize("american,expected", [(-110, 1.9090909), (100, 2.0), (150, 2.5)])
def test_american_to_decimal(american, expected):
    assert american_to_decimal(american) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [None, 0, -99, 99, float("nan")])
def test_invalid_american_odds_are_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        american_to_decimal(bad)


def test_decimal_implied_probability():
    assert decimal_to_implied(2.5) == pytest.approx(0.4)


def test_bet_id_is_stable_across_display_formatting():
    first = make_bet_id("2026-05-01", "Nikola Jokić", "PTS", "25.500", "over")
    second = make_bet_id("2026-05-01T12:00:00Z", " Nikola Jokic ", "pts", 25.5, "OVER")
    assert first == second


def test_legacy_row_migrates_without_assuming_finance_values():
    row = ensure_finance_fields({"date": "2026-05-01", "player": "A", "market": "PTS",
                                 "line": 10.5, "pick": "OVER", "decimal_odds": 2.0})
    assert row["entry_decimal_odds"] == 2.0
    assert row["bet_id"].startswith("bet_")
    assert all(field in row for field in FINANCE_FIELDS)
    assert row["stake_amount"] == ""
    assert row["closing_decimal_odds"] == ""


def test_sqlite_additive_finance_schema_is_backward_compatible(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE prop_predictions (date TEXT, game TEXT, player TEXT, "
                     "market TEXT, PRIMARY KEY(date, player, market))")
    with predictions_db._connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(prop_predictions)")}
    assert set(predictions_db.PROP_FINANCE_FIELDS) <= columns
