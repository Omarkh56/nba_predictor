"""snapshot_injuries.py archives the raw report, appends flat CSV rows and logs the run."""

import csv
from unittest import mock

import pytest
import requests

import raw_archive
import run_log
import snapshot_injuries

# Trimmed from the real ESPN injury report's structure (Oct 2026).
REPORT = {
    "timestamp": "2026-10-20T12:09:39Z", "status": "success",
    "injuries": [
        {"id": "18", "displayName": "New York Knicks", "injuries": [{
            "id": "-58751", "status": "Out", "date": "2026-10-19T19:38Z",
            "shortComment": "Underwent foot surgery.", "longComment": "Re-evaluated in 3-4 months.",
            "athlete": {"displayName": "Probe Guard",
                        "links": [{"href": "https://www.espn.com/nba/player/_/id/4712863/probe-guard"}],
                        "position": {"abbreviation": "G"},
                        "team": {"abbreviation": "NY", "displayName": "New York Knicks"}},
            "type": {"name": "INJURY_STATUS_OUT", "description": "out"},
            "details": {"fantasyStatus": {"abbreviation": "OUT"}, "type": "Foot", "location": "Leg",
                        "detail": "Fracture", "side": "Left", "returnDate": "2026-11-20"},
        }]},
        {"id": "26", "displayName": "Utah Jazz", "injuries": [
            {"id": "-1", "status": "Day-To-Day", "date": "2026-10-20T01:00Z",
             "athlete": {"displayName": "Probe Wing", "links": [],
                         "headshot": {"href": "https://a.espncdn.com/i/headshots/nba/players/full/123.png"},
                         "position": {"abbreviation": "F"}, "team": {"abbreviation": "UTAH"}},
             "type": {"description": "day-to-day"}, "details": {}},
            {"id": "-2", "status": "Out", "athlete": {"displayName": "Probe Center"}},
        ]},
    ],
}


@pytest.fixture
def snapshot_env(monkeypatch, tmp_path):
    monkeypatch.setenv("NBA_RUN_LOG", str(tmp_path / "run_log.jsonl"))
    monkeypatch.setenv("NBA_RAW_ARCHIVE_DIR", str(tmp_path / "raw"))
    return tmp_path


def _serve(monkeypatch, payload):
    response = mock.Mock(status_code=200, raise_for_status=lambda: None, json=lambda: payload)
    get = mock.Mock(return_value=response)
    monkeypatch.setattr(snapshot_injuries.requests, "get", get)
    return get


def test_snapshot_parses_report_into_csv_archive_and_run_log(snapshot_env, monkeypatch):
    get = _serve(monkeypatch, REPORT)
    csv_path = snapshot_env / "data" / "injury_snapshots.csv"
    assert snapshot_injuries.main(["2026-10-21"], csv_path=csv_path) == 0
    assert snapshot_injuries.main(["2026-10-21"], csv_path=csv_path) == 0  # appends, one header
    assert get.call_count == 2

    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6
    first, wing, center = rows[:3]
    assert {k: first[k] for k in ("slate_date", "team", "player", "espn_player_id", "status",
                                  "position", "injury_type", "return_date", "short_comment")} == {
        "slate_date": "2026-10-21", "team": "NYK", "player": "Probe Guard",
        "espn_player_id": "4712863", "status": "Out", "position": "G", "injury_type": "Foot",
        "return_date": "2026-11-20", "short_comment": "Underwent foot surgery."}
    assert (wing["team"], wing["espn_player_id"], wing["status_type"]) == ("UTA", "123", "day-to-day")
    assert (center["team"], center["espn_player_id"], center["position"]) == ("UTA", "", "")

    runs = run_log.read_runs(snapshot_env / "run_log.jsonl")
    assert [r["status"] for r in runs] == ["ok", "ok"]
    stages = {s["stage"]: s for s in runs[0]["stages"]}
    assert stages["espn_injuries"]["players"] == 3 and stages["injury_csv"]["rows"] == 3
    assert first["run_id"] == runs[0]["run_id"]

    archives = sorted((snapshot_env / "raw" / "espn" / "injuries").rglob("*.json.gz"))
    envelopes = [raw_archive.load(p) for p in archives]
    assert [e["run_id"] for e in envelopes] == [r["run_id"] for r in runs]
    assert envelopes[0]["payload"] == REPORT


def test_failed_fetch_exits_2_and_archives_the_gap(snapshot_env, monkeypatch):
    monkeypatch.setattr(snapshot_injuries.requests, "get",
                        mock.Mock(side_effect=requests.exceptions.ConnectionError("no route")))
    csv_path = snapshot_env / "injury_snapshots.csv"
    assert snapshot_injuries.main(["2026-10-21"], csv_path=csv_path) == 2
    assert not csv_path.exists()
    [record] = run_log.read_runs(snapshot_env / "run_log.jsonl")
    assert record["status"] == "fetch_failed"
    [archive] = (snapshot_env / "raw" / "espn" / "injuries").rglob("*.json.gz")
    envelope = raw_archive.load(archive)
    assert envelope["status"] == "fetch_failed" and envelope["payload"] is None
    assert envelope["run_id"] == record["run_id"]


def test_malformed_report_is_a_failure_not_an_empty_day(snapshot_env, monkeypatch):
    _serve(monkeypatch, {"status": "success"})
    csv_path = snapshot_env / "injury_snapshots.csv"
    assert snapshot_injuries.main([], csv_path=csv_path) == 2
    assert not csv_path.exists()
