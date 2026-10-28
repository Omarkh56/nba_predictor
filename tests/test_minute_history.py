"""Expected-minutes history includes short games without contaminating rate samples."""

import pandas as pd
import pytest

import playerlinepredictor as predictor


def _logs(minutes):
    return pd.DataFrame({
        "GAME_DATE": pd.date_range("2026-01-01", periods=len(minutes), freq="D")[::-1],
        "MIN": minutes,
        "PTS": [10.0] * len(minutes),
    })


def test_log_history_separates_rate_and_minutes_samples(monkeypatch):
    po = _logs([32, 8, 24])
    rs = _logs([30, 5, 20, 14])
    calls = []

    def fetch(_pid, _season, season_type, minimum):
        calls.append((season_type, minimum))
        return po.copy() if season_type == predictor.SEASON_TYPE_PO else rs.copy()

    monkeypatch.setattr(predictor, "_fetch_logs", fetch)
    history = predictor.get_player_logs_blended(1)
    assert calls == [(predictor.SEASON_TYPE_PO, 0), (predictor.SEASON_TYPE_RS, 0)]
    assert history.po_rates.MIN.tolist() == [32, 24]
    assert history.rs_rates.MIN.tolist() == [30, 20]
    assert history.po_minutes.MIN.tolist() == [32, 8, 24]
    assert history.rs_minutes.MIN.tolist() == [30, 5, 20, 14]
    assert history.po_rate_count == 2 and history.po_raw_count == 3


def test_minute_candidates_bracket_short_appearance_effect(monkeypatch):
    rates = _logs([30, 30, 30])
    played = _logs([30, 5, 30])
    monkeypatch.setattr(predictor, "MINUTES_HISTORY_MODE", "filtered")
    filtered = predictor._minutes_average(rates, played)
    monkeypatch.setattr(predictor, "MINUTES_HISTORY_MODE", "all_played")
    all_played = predictor._minutes_average(rates, played)
    monkeypatch.setattr(predictor, "MINUTES_HISTORY_MODE", "robust")
    robust = predictor._minutes_average(rates, played)
    assert all_played < robust < filtered


def test_unknown_minute_history_mode_fails_closed(monkeypatch):
    logs = _logs([20])
    monkeypatch.setattr(predictor, "MINUTES_HISTORY_MODE", "typo")
    with pytest.raises(ValueError, match="unknown MINUTES_HISTORY_MODE"):
        predictor._minutes_average(logs, logs)


def test_regular_season_skips_the_always_empty_playoff_fetch(monkeypatch):
    """Once get_league_logs() has found no playoff games, a player's current-season playoff
    log can only be empty; on Oct 24 2026 that fetch timed out and cost three games."""
    rs = _logs([30, 5, 20])
    calls = []

    def fetch(_pid, _season, season_type, minimum):
        calls.append(season_type)
        return rs.copy()

    monkeypatch.setattr(predictor, "_fetch_logs", fetch)
    monkeypatch.setattr(predictor, "LEAGUE_GAMELOGS_CACHE",
                        {(predictor.SEASON, predictor.SEASON_TYPE_PO): pd.DataFrame()})
    monkeypatch.setattr(predictor, "IS_PLAYOFF_SEASON", False)
    monkeypatch.setattr(predictor, "PLAYER_LOGS_FETCH_META", {
        (7, predictor.SEASON, predictor.SEASON_TYPE_RS, 0): {"status": "ok", "raw_count": 3}})
    history = predictor.get_player_logs_blended(7)
    assert calls == [predictor.SEASON_TYPE_RS]
    assert history.po_minutes.empty and list(history.po_minutes.columns) == list(rs.columns)
    assert history.po_raw_count == 0 and not history.using_playoffs
    assert predictor._player_fetch_provenance(7, history)["fetch_status"] == "rs=ok;po=skipped"

    monkeypatch.setattr(predictor, "IS_PLAYOFF_SEASON", True)  # playoffs started: fetch them
    calls.clear()
    predictor.get_player_logs_blended(7)
    assert calls == [predictor.SEASON_TYPE_PO, predictor.SEASON_TYPE_RS]
