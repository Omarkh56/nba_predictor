"""Live CLI status contracts, using fake feeds and temporary data only."""

from datetime import date
from unittest import mock

import pytest
import requests

import run_log

DAY = date(2026, 10, 4)
EVENT = {"id": "preseason", "home_team": "Denver Nuggets", "away_team": "Utah Jazz",
         "commence_time": "2026-10-04T23:00:00Z"}


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    with mock.patch("requests.sessions.Session.request", side_effect=AssertionError("No network")):
        import oddstracker
    for name in ("TRACKER_FILE", "SNAPSHOT_FILE", "BET_LOG_FILE", "CALIB_FILE"):
        monkeypatch.setattr(oddstracker, name, str(tmp_path / name))
    monkeypatch.setenv("NBA_RUN_LOG", str(tmp_path / "runs.jsonl"))
    monkeypatch.setattr(oddstracker.time, "sleep", lambda _: None)
    monkeypatch.setattr(oddstracker, "LAST_SNAPSHOT_STATUS", {})
    return oddstracker


@pytest.mark.parametrize("error", [requests.ConnectionError, requests.exceptions.SSLError])
def test_tracker_outage_logs_failure_exits_two_and_preserves_data(tracker, tmp_path, monkeypatch, error):
    from pathlib import Path
    for filename in (tracker.TRACKER_FILE, tracker.SNAPSHOT_FILE):
        Path(filename).write_bytes(b"existing data\n")
    calls = []
    def down():
        calls.append(1)
        raise error("unreachable")
    monkeypatch.setattr(tracker, "get_all_events", down)
    monkeypatch.setattr(tracker, "update_results", lambda: pytest.fail("Must not grade on failure"))
    monkeypatch.setattr(tracker, "show_today", lambda _: pytest.fail("Must not show stale odds"))
    assert tracker.main([str(DAY)]) == 2
    assert len(calls) == 1
    record = run_log.read_runs(tmp_path / "runs.jsonl")[0]
    assert record["command"] == "oddstracker"
    assert record["slate_date"] == str(DAY)
    assert record["status"] == "fetch_failed"
    for filename in (tracker.TRACKER_FILE, tracker.SNAPSHOT_FILE):
        assert Path(filename).read_bytes() == b"existing data\n"


def test_prop_transport_failure_is_not_empty_and_is_not_retried(tracker, monkeypatch):
    calls = []
    def down(*args, **kwargs):
        calls.append(kwargs["params"]["regions"])
        raise requests.ConnectionError("unreachable")
    monkeypatch.setattr(tracker, "_get_odds_json", down)
    with pytest.raises(requests.ConnectionError):
        tracker.get_event_props("preseason")
    assert calls == ["us"]


@pytest.mark.parametrize("failed", [False, True])
def test_preseason_empty_response_and_failed_response_are_distinct(tracker, tmp_path, monkeypatch, failed):
    monkeypatch.setattr(tracker, "get_all_events", lambda: [EVENT])
    def fetch(*args, **kwargs):
        if failed:
            raise requests.ConnectionError("unreachable")
        return {"bookmakers": []}
    monkeypatch.setattr(tracker, "_get_odds_json", fetch)
    monkeypatch.setattr(tracker, "update_results", lambda: None)
    monkeypatch.setattr(tracker, "show_today", lambda _: None)
    assert tracker.main([str(DAY)]) == (2 if failed else 0)
    record = run_log.read_runs(tmp_path / "runs.jsonl")[0]
    assert record["stages"][0]["status"] == ("fetch_failed" if failed else "market_unavailable")


def test_mixed_failed_and_empty_events_stay_partial(tracker, tmp_path, monkeypatch):
    monkeypatch.setattr(tracker, "get_all_events", lambda: [EVENT, {**EVENT, "id": "empty"}])
    def fetch(event_id):
        if event_id == "preseason":
            raise requests.ConnectionError("unreachable")
        return {"bookmakers": []}
    monkeypatch.setattr(tracker, "get_event_props", fetch)
    assert tracker.main([str(DAY)]) == 3
    record = run_log.read_runs(tmp_path / "runs.jsonl")[0]
    assert record["status"] == "partial"
    assert record["stages"][0]["failed_events"] == 1


@pytest.fixture
def combined(tracker, monkeypatch):
    with mock.patch("requests.sessions.Session.request", side_effect=AssertionError("No network")):
        import nba_combined
    monkeypatch.setattr(nba_combined, "_init_calibration", lambda: None)
    return nba_combined


def test_combined_espn_outage_exits_two(combined, tmp_path, monkeypatch):
    def down(*args, **kwargs):
        raise combined.team_model.ScheduleFetchError("unreachable")
    monkeypatch.setattr(combined, "_fetch_matchups_and_lean", down)
    assert combined.main(["--date", str(DAY)]) == 2
    assert run_log.read_runs(tmp_path / "runs.jsonl")[0]["stages"][0]["status"] == "fetch_failed"


def test_combined_snapshot_outage_stops_before_models_or_exports(combined, tmp_path, monkeypatch):
    def schedule(*args, **kwargs):
        combined.oddstracker._set_snapshot_status("fetch_failed", error="SSLError")
        return [{"away_abbr": "UTA", "home_abbr": "DEN", "game_date": DAY}], DAY
    monkeypatch.setattr(combined, "_fetch_matchups_and_lean", schedule)
    monkeypatch.setattr(combined, "_fetch_injuries", lambda: pytest.fail("Must stop before models"))
    monkeypatch.setattr(combined, "_print_global_summary", lambda *a: pytest.fail("Must not export"))
    assert combined.main(["--date", str(DAY)]) == 2
    record = run_log.read_runs(tmp_path / "runs.jsonl")[0]
    assert record["status"] == "fetch_failed"
    assert [s["stage"] for s in record["stages"]] == ["schedule", "odds_snapshot"]


def test_combined_odds_events_outage_is_required_and_stops_exports(combined, tmp_path, monkeypatch):
    matchup = {"away_abbr": "UTA", "home_abbr": "DEN", "game_date": DAY}
    monkeypatch.setattr(combined, "_fetch_matchups_and_lean", lambda *a, **k: ([matchup], DAY))
    monkeypatch.setattr(combined, "_fetch_injuries", lambda: ({}, {}))
    monkeypatch.setattr(combined, "_run_team_model", lambda *a: ([{}], {}))
    def down():
        raise requests.ConnectionError("unreachable")
    monkeypatch.setattr(combined.props_model, "get_all_odds_events", down)
    monkeypatch.setattr(combined, "_print_global_summary", lambda *a: pytest.fail("Must not export"))
    monkeypatch.setattr(combined.props_model, "get_team_season_stats", lambda: pytest.fail("Must stop"))
    assert combined.main(["--date", str(DAY)]) == 2
    record = run_log.read_runs(tmp_path / "runs.jsonl")[0]
    stage = next(s for s in record["stages"] if s["stage"] == "odds_events")
    assert stage["status"] == "fetch_failed"
    assert stage["required"] is True


@pytest.mark.parametrize("method,args", [
    ("get_all_odds_events", ()),
    ("get_event_player_props", ("preseason",)),
    ("_auto_fetch_spread", ("preseason", "Denver Nuggets")),
])
def test_prediction_odds_errors_do_not_expose_credentials(combined, monkeypatch, caplog, method, args):
    import odds_api
    monkeypatch.setattr(combined.props_model, "ODDS_API_KEY", "private-regression-key")
    def down(*args, **kwargs):
        raise requests.ConnectionError("https://api.the-odds-api.com/?apiKey=private-regression-key")
    monkeypatch.setattr(odds_api.requests, "get", down)
    fetch = getattr(combined.props_model, method)
    if method == "_auto_fetch_spread":
        assert fetch(*args) == 0.0
    else:
        with pytest.raises(requests.RequestException) as exc:
            fetch(*args)
        assert "private-regression-key" not in str(exc.value)
    assert "private-regression-key" not in caplog.text
