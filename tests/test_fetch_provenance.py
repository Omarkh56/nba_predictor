"""Failure-aware NBA fetches and per-pick history provenance."""
from datetime import date

import pandas as pd
import pytest

import nba_api_utils
import playerlinepredictor as predictor


def test_dataframe_result_distinguishes_success_empty_and_failure(monkeypatch):
    ok = nba_api_utils.dataframe_call_result(lambda: pd.DataFrame({"x": [1]}), retries=1)
    empty = nba_api_utils.dataframe_call_result(lambda: pd.DataFrame(), retries=1)
    monkeypatch.setattr(
        nba_api_utils,
        "api_call_with_retry",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    failed = nba_api_utils.dataframe_call_result(lambda: pd.DataFrame(), retries=1)
    assert ok.status == "ok" and ok.ok and len(ok.frame) == 1
    assert empty.status == "ok_empty" and empty.ok and empty.frame.empty
    assert failed.status == "failed" and not failed.ok and failed.frame is None


def test_failed_player_fetch_is_not_cached_then_successful_empty_is_cached(monkeypatch):
    predictor.PLAYER_LOGS_CACHE.clear()
    predictor.PLAYER_LOGS_FETCH_META.clear()
    calls = []
    outcomes = [
        nba_api_utils.DataFrameCallResult("failed", None, "timeout"),
        nba_api_utils.DataFrameCallResult("ok_empty", pd.DataFrame()),
    ]

    def fake_result(*_a, **_k):
        calls.append(1)
        return outcomes.pop(0)

    monkeypatch.setattr(predictor, "dataframe_call_result", fake_result)
    key = (99, "2025-26", predictor.SEASON_TYPE_PO, 0)
    with pytest.raises(predictor.NbaDataUnavailable):
        predictor._fetch_logs(*key)
    assert key not in predictor.PLAYER_LOGS_CACHE
    assert key not in predictor.PLAYER_LOGS_FETCH_META

    assert predictor._fetch_logs(*key).empty
    assert key in predictor.PLAYER_LOGS_CACHE
    assert predictor.PLAYER_LOGS_FETCH_META[key]["status"] == "ok_empty"
    assert predictor._fetch_logs(*key).empty
    assert len(calls) == 2


def test_failed_league_fetch_is_not_cached_or_reinterpreted_as_regular_season(monkeypatch):
    predictor.LEAGUE_GAMELOGS_CACHE.clear()
    monkeypatch.setattr(predictor, "SEASON", "2025-26")
    failed = nba_api_utils.DataFrameCallResult("failed", None, "league timeout")
    call = monkeypatch.setattr(predictor, "_safe_lgf_result", lambda **_k: failed)
    with pytest.raises(predictor.NbaDataUnavailable, match="playoff league logs unavailable"):
        predictor.get_league_logs()
    assert predictor.LEAGUE_GAMELOGS_CACHE == {}


def test_player_fetch_provenance_reports_counts_freshness_and_versions(monkeypatch):
    predictor.PLAYER_LOGS_FETCH_META.clear()
    monkeypatch.setattr(predictor, "SEASON", "2025-26")
    monkeypatch.setattr(predictor, "IS_PLAYOFF_SEASON", True)
    monkeypatch.setattr(predictor, "_LEARNED_PARAMS_VERSION", "v-test")
    pid = 10
    common = {"status": "ok", "fetched_at_utc": "2026-05-02T12:00:00+00:00", "source_last_game_date": "2026-05-01"}
    predictor.PLAYER_LOGS_FETCH_META[(pid, "2025-26", predictor.SEASON_TYPE_RS, predictor.MIN_MINUTES_RS)] = {
        **common, "raw_count": 70, "usable_count": 20,
    }
    predictor.PLAYER_LOGS_FETCH_META[(pid, "2025-26", predictor.SEASON_TYPE_PO, predictor.MIN_MINUTES_PO)] = {
        **common, "raw_count": 5, "usable_count": 4,
    }
    predictor.PLAYER_LOGS_FETCH_META[(pid, "2025-26", predictor.SEASON_TYPE_PO, 0)] = {
        **common, "raw_count": 5, "usable_count": 5,
    }
    po, rs = pd.DataFrame(index=range(4)), pd.DataFrame(index=range(20))
    out = predictor._player_fetch_provenance(pid, po, rs, po_raw_count=5)
    assert out["rs_raw_count"] == 70 and out["rs_usable_count"] == 20
    assert out["po_raw_count"] == 5 and out["po_usable_count"] == 4
    assert out["fetch_status"] == "rs=ok;po=ok;po_raw=ok"
    assert out["source_last_game_date"] == "2026-05-01"
    assert out["active_season"] == "2025-26" and out["season_mode"] == "playoffs"
    assert out["model_version"] == "v-test"
