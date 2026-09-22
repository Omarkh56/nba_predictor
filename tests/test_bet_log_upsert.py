"""Idempotent bet-log storage and deterministic duplicate migration."""
from datetime import date

import pandas as pd
import pytest

from bet_log_utils import BetLogConflict, canonical_bet_key, dedupe_bet_log, upsert_bet_log


def row(**over):
    out = {
        "date": "2026-05-01",
        "player": "Nikola Jokic",
        "market": "PTS",
        "line": "25.5",
        "pick": "OVER",
        "projection": "27.1",
        "actual": "30",
        "margin": "4.5",
        "result": "HIT",
        "flags": "",
        "p_model": "0.61",
        "p_market": "",
        "decimal_odds": "",
        "stake_pct": "",
        "realized_profit": "",
    }
    out.update(over)
    return out


def test_key_normalizes_date_accents_spacing_market_line_and_pick():
    a = row(player="Nikola Jokić", market=" fg3m ", line="2.500", pick="under", date=date(2026, 5, 1))
    b = row(player="  Nikola Jokic ", market="3PM", line=2.5, pick="UNDER")
    assert canonical_bet_key(a) == canonical_bet_key(b)


def test_regrading_same_bet_twice_is_idempotent(tmp_path):
    path = tmp_path / "bet_log.csv"
    first = upsert_bet_log(path, [row()])
    second = upsert_bet_log(path, [row()])
    assert first.inserted == 1 and first.after_rows == 1
    assert second.inserted == 0 and second.unchanged == 1 and second.after_rows == 1
    assert second.duplicates_removed == 0
    assert len(pd.read_csv(path)) == 1
    assert not any(c.endswith(".1") for c in pd.read_csv(path).columns)


def test_repeat_can_fill_missing_fields_without_duplicate(tmp_path):
    path = tmp_path / "bet_log.csv"
    upsert_bet_log(path, [row(projection="", p_market="")])
    stats = upsert_bet_log(path, [row(p_market="0.55")])
    saved = pd.read_csv(path)
    assert stats.updated == 1 and len(saved) == 1
    assert saved.loc[0, "projection"] == pytest.approx(27.1)
    assert saved.loc[0, "p_market"] == pytest.approx(0.55)


def test_distinct_line_or_pick_remains_distinct(tmp_path):
    path = tmp_path / "bet_log.csv"
    upsert_bet_log(path, [row(), row(line="26.5"), row(pick="UNDER", result="MISS")])
    assert len(pd.read_csv(path)) == 3


def test_conflicting_populated_values_abort_without_rewriting_file(tmp_path):
    path = tmp_path / "bet_log.csv"
    upsert_bet_log(path, [row()])
    before = path.read_bytes()
    with pytest.raises(BetLogConflict, match="conflicting actual"):
        upsert_bet_log(path, [row(actual="29")])
    assert path.read_bytes() == before


def test_dedupe_merges_complementary_rows_and_reports_counts(tmp_path):
    src, dst = tmp_path / "raw.csv", tmp_path / "clean.csv"
    pd.DataFrame([row(projection=""), row(p_model="")]).to_csv(src, index=False)
    stats = dedupe_bet_log(src, dst)
    saved = pd.read_csv(dst)
    assert stats.before_rows == 2
    assert stats.duplicate_groups == 1
    assert stats.duplicates_removed == 1
    assert len(saved) == 1
    assert saved.loc[0, "projection"] == pytest.approx(27.1)
    assert saved.loc[0, "p_model"] == pytest.approx(0.61)


def test_checkresults_append_path_is_idempotent(monkeypatch, tmp_path):
    import sys

    monkeypatch.setattr(sys, "argv", ["checkresults.py"])
    import checkresults

    path = tmp_path / "bet_log.csv"
    monkeypatch.setattr(checkresults, "LOG_FILE", str(path))
    graded = [{
        "player": "Nikola Jokic", "market": "PTS", "line": 25.5, "pick": "OVER",
        "projection": 27.1, "actual": 30, "margin": 4.5, "result": "HIT ✓",
        "flags": "", "p_model": 0.61, "p_market": None,
        "rs_raw_count": 70, "rs_usable_count": 20, "po_raw_count": 5,
        "po_usable_count": 4, "fetch_status": "rs=ok;po=ok;po_raw=ok",
        "source_last_game_date": "2026-04-30", "data_as_of_utc": "2026-05-01T12:00:00+00:00",
        "data_age_days": 1, "active_season": "2025-26", "season_mode": "playoffs",
        "model_version": "v-test", "book_spread": 4.5, "model_margin": 6.2,
    }]
    checkresults.append_to_log(graded, date(2026, 5, 1))
    checkresults.append_to_log(graded, date(2026, 5, 1))
    saved = pd.read_csv(path)
    assert len(saved) == 1
    assert saved.loc[0, "actual"] == 30
    assert saved.loc[0, "po_raw_count"] == 5
    assert saved.loc[0, "fetch_status"] == "rs=ok;po=ok;po_raw=ok"


def test_box_score_zero_minutes_is_dnp_and_not_logged(monkeypatch, tmp_path):
    import sys

    monkeypatch.setattr(sys, "argv", ["checkresults.py"])
    import checkresults

    assert checkresults._played_minutes(None) == 0
    assert checkresults._played_minutes("DNP") == 0
    assert checkresults._played_minutes("00:00") == 0
    assert checkresults._played_minutes("12:30") == 12.5
    monkeypatch.setattr(checkresults, "get_player_id", lambda _: 1)
    path = tmp_path / "bet_log.csv"
    monkeypatch.setattr(checkresults, "LOG_FILE", str(path))
    pred = {"player": "Known Player", "market": "PTS", "line": 10.5,
            "pick": "OVER", "projection": 15.0}
    graded = checkresults.check_predictions([pred], {1: {"PTS": 0, "MIN": "0:00"}})
    assert graded[0]["result"] == "DID NOT PLAY"
    checkresults.append_to_log(graded, date(2026, 5, 1))
    assert not path.exists()


def test_grading_preserves_calibrated_probability_over_ranking_confidence(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["checkresults.py"])
    import checkresults
    monkeypatch.setattr(checkresults, "get_player_id", lambda _: 1)
    pred = {"player": "Known Player", "market": "PTS", "line": 10.5,
            "pick": "OVER", "confidence": 53.0, "p_model": .72}
    graded = checkresults.check_predictions([pred], {1: {"PTS": 20, "MIN": "35:00"}})
    assert graded[0]["p_model"] == pytest.approx(.72)
