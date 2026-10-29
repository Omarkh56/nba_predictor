"""Historical finance backfill only accepts stored exact-line quotes."""

from pathlib import Path

import pandas as pd
import pytest

from backfill_finance import backfill_rows, run


def bet(**updates):
    row = {
        "date": "2026-05-01", "player": "Nikola Jokić", "market": "PTS",
        "line": 25.5, "pick": "OVER", "result": "HIT", "decimal_odds": "",
        "stake_pct": "", "realized_profit": "",
    }
    row.update(updates)
    return row


def odds(**updates):
    row = {
        "date": "2026-05-01", "player": "Nikola Jokic", "market": "PTS",
        "line": 25.5, "avg_over_odds": -110, "avg_under_odds": -110,
        "devig_over": 0.5, "devig_under": 0.5, "vig_pct": 4.76,
    }
    row.update(updates)
    return row


def test_exact_line_recovers_real_price_without_inventing_stake_or_timestamp():
    rows, report = backfill_rows([bet()], pd.DataFrame([odds()]))
    row = rows[0]
    assert row["entry_american_odds"] == -110
    assert row["entry_decimal_odds"] == pytest.approx(1.90909091)
    assert row["market_implied_probability"] == pytest.approx(0.52380952)
    assert row["stake_amount"] == ""
    assert row["realized_profit"] == ""
    assert row["odds_snapshot_at"] == ""
    assert report["rows_with_entry_odds_recovered"] == 1


def test_different_line_does_not_recover_or_validate_legacy_decimal():
    rows, report = backfill_rows(
        [bet(decimal_odds=1.91)], pd.DataFrame([odds(line=26.5)])
    )
    assert rows[0]["entry_decimal_odds"] == ""
    assert rows[0]["decimal_odds"] == 1.91
    assert report["legacy_decimal_odds_without_exact_line_match"] == 1
    assert report["unmatched_reasons"]["no_exact_line_match"] == 1


def test_conflicting_exact_quotes_are_ambiguous():
    rows, report = backfill_rows(
        [bet()], pd.DataFrame([odds(avg_over_odds=-110), odds(avg_over_odds=-125)])
    )
    assert rows[0]["entry_decimal_odds"] == ""
    assert report["ambiguous_matches"] == 1


def test_run_writes_new_file_and_preserves_source(tmp_path):
    source = tmp_path / "bet_log.csv"
    odds_path = tmp_path / "odds_tracker.csv"
    destination = tmp_path / "backfilled.csv"
    reports = tmp_path / "reports"
    pd.DataFrame([bet()]).to_csv(source, index=False)
    pd.DataFrame([odds()]).to_csv(odds_path, index=False)
    before = source.read_bytes()
    report = run(source, odds_path, destination, reports)
    assert source.read_bytes() == before
    assert destination.exists()
    assert report["rows_with_entry_odds_recovered"] == 1
    assert (reports / "finance_backfill_summary.json").exists()


def test_run_refuses_to_replace_source(tmp_path):
    source = tmp_path / "bet_log.csv"
    source.write_text("date,player,market,line,pick\n", encoding="utf-8")
    with pytest.raises(ValueError, match="destination must differ"):
        run(source, tmp_path / "odds.csv", source, tmp_path / "reports")
