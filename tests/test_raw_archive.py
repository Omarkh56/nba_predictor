"""Every external response is archived before use, and the archive never breaks a fetch."""

import gzip
import json
import logging
import os
from datetime import date, datetime, timezone
from unittest import mock

import pytest
import requests

import nba_api_utils
import odds_api
import raw_archive
import team_codes

EVENTS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
EVENT_ODDS_URL = EVENTS_URL + "/evt123/odds"


@pytest.fixture
def archive_root(monkeypatch, tmp_path):
    root = tmp_path / "raw"
    monkeypatch.setenv("NBA_RAW_ARCHIVE_DIR", str(root))
    return root


def archived(root, source=None, endpoint=None):
    """Envelopes under root (optionally one source/endpoint), oldest first."""
    base = root.joinpath(*[p for p in (source, endpoint) if p])
    return [raw_archive.load(p) for p in sorted(base.rglob("*.json.gz"))] if base.exists() else []


def odds_response(payload, status=200):
    return mock.Mock(status_code=status, json=lambda: payload)


def test_envelope_is_readable_and_api_key_is_removed(archive_root):
    path = raw_archive.save("odds_api", "events", {"apiKey": "private-key", "regions": "us"},
                            [{"id": "e1"}], run_id="run-1", http_status=200)
    envelope = raw_archive.load(path)
    assert envelope == {
        "source": "odds_api", "endpoint": "events", "params": {"regions": "us"},
        "params_hash": raw_archive.params_hash({"regions": "us"}),
        "fetched_at_utc": envelope["fetched_at_utc"], "run_id": "run-1", "status": "ok",
        "http_status": 200, "payload": [{"id": "e1"}],
    }
    assert b"private-key" not in gzip.decompress(path.read_bytes())
    day = team_codes.today_slate_date(datetime.fromisoformat(envelope["fetched_at_utc"]))
    assert path.parent == archive_root / "odds_api" / "events" / f"dt={day.isoformat()}"
    assert path.name.endswith(f"_{envelope['params_hash']}.json.gz")


def test_dt_is_the_us_eastern_slate_date(archive_root, monkeypatch):
    class LateTip(datetime):  # 10:30 pm ET on Oct 20 is 02:30 UTC on Oct 21
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 21, 2, 30, tzinfo=timezone.utc)

    monkeypatch.setattr(raw_archive, "datetime", LateTip)
    path = raw_archive.save("espn", "injuries", {}, {"injuries": []})
    assert path.parent.name == "dt=2026-10-20"


def test_failed_fetch_is_archived_without_payload(archive_root, monkeypatch):
    error = requests.exceptions.HTTPError("429", response=mock.Mock(status_code=429))
    monkeypatch.setattr(odds_api.requests, "get", mock.Mock(side_effect=error))
    with pytest.raises(requests.exceptions.RequestException):
        odds_api.get_json(EVENTS_URL, {"apiKey": "private-key"})
    [envelope] = archived(archive_root, "odds_api", "events")
    assert envelope["status"] == "fetch_failed" and envelope["payload"] is None
    assert envelope["http_status"] == 429
    assert "private-key" not in json.dumps(envelope)


@pytest.mark.parametrize("breakage", ["read_only_dir", "root_is_a_file"])
def test_archive_failure_does_not_change_the_callers_result(monkeypatch, tmp_path, caplog, breakage):
    root = tmp_path / "raw"
    if breakage == "read_only_dir":
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            pytest.skip("root ignores directory permissions")
        root.mkdir()
        root.chmod(0o500)
    else:
        root.write_text("not a directory")
    monkeypatch.setenv("NBA_RAW_ARCHIVE_DIR", str(root))
    monkeypatch.setattr(raw_archive, "_warned", set())
    payload = [{"id": "e1", "commence_time": "2099-01-01T00:00:00Z"}]
    monkeypatch.setattr(odds_api.requests, "get", mock.Mock(return_value=odds_response(payload)))
    try:
        with caplog.at_level(logging.WARNING, logger="raw_archive"):
            assert odds_api.get_json(EVENTS_URL, {"apiKey": "k"}) == payload
            assert raw_archive.save("espn", "injuries", {}, {"injuries": []}) is None
    finally:
        if root.is_dir():
            root.chmod(0o700)
    assert "raw archive: could not save" in caplog.text


def test_odds_writes_one_file_per_call(archive_root, monkeypatch):
    monkeypatch.setattr(odds_api.requests, "get", mock.Mock(side_effect=[
        odds_response([{"id": "evt123"}]), odds_response({"bookmakers": []})]))
    odds_api.get_json(EVENTS_URL, {"apiKey": "k"})
    odds_api.get_json(EVENT_ODDS_URL, {"apiKey": "k", "markets": "player_points"})
    [events] = archived(archive_root, "odds_api", "events")
    [event_odds] = archived(archive_root, "odds_api", "event_odds")
    assert events["params"] == {"sport": "basketball_nba"}
    assert event_odds["params"] == {"markets": "player_points", "sport": "basketball_nba",
                                    "event_id": "evt123"}
    assert event_odds["payload"] == {"bookmakers": []}
    assert len(archived(archive_root)) == 2


def _stats_response(url, payload, status=200):
    response = requests.Response()
    response.status_code, response.url, response.encoding = status, url, "utf-8"
    response._content = json.dumps(payload).encode()
    return response


def test_nba_api_writes_one_raw_file_per_call(archive_root, monkeypatch):
    from nba_api.stats.endpoints import leaguegamefinder

    assert nba_api_utils.install_raw_archive()
    payload = {"resource": "leaguegamefinderresults", "parameters": {},
               "resultSets": [{"name": "LeagueGameFinderResults",
                               "headers": ["GAME_ID", "TEAM_ABBREVIATION"],
                               "rowSet": [["0022500001", "BOS"]]}]}
    sent = []

    def transport(self, method, url, params=None, **kwargs):
        sent.append(url)
        return _stats_response(url, payload)

    monkeypatch.setattr(requests.Session, "request", transport)
    finder = leaguegamefinder.LeagueGameFinder(player_or_team_abbreviation="T",
                                               season_nullable="2025-26")
    assert finder.get_data_frames()[0]["TEAM_ABBREVIATION"].tolist() == ["BOS"]
    [envelope] = archived(archive_root, "nba_api", "leaguegamefinder")
    assert len(sent) == 1 and envelope["status"] == "ok" and envelope["http_status"] == 200
    assert envelope["payload"] == finder.get_dict()  # the raw body, not a DataFrame
    assert envelope["params"]["Season"] == "2025-26"  # the parameters as sent to stats.nba.com

    def down(self, method, url, params=None, **kwargs):
        raise requests.exceptions.ConnectionError("unreachable")

    monkeypatch.setattr(requests.Session, "request", down)
    with pytest.raises(requests.exceptions.ConnectionError):
        leaguegamefinder.LeagueGameFinder(player_or_team_abbreviation="T", season_nullable="2025-26")
    failed = archived(archive_root, "nba_api", "leaguegamefinder")[-1]
    assert failed["status"] == "fetch_failed" and failed["payload"] is None
    assert failed["error"] == "ConnectionError"


def test_espn_fetches_write_one_file_per_call(archive_root, monkeypatch):
    import newnbapredictor as nb
    import playerlinepredictor as plp

    scoreboard = {"events": []}
    injuries = {"injuries": [{"displayName": "Utah Jazz", "team": {"displayName": "Utah Jazz"},
                              "injuries": []}]}
    responses = iter([scoreboard, injuries, injuries])

    def get(*args, **kwargs):
        return mock.Mock(status_code=200, raise_for_status=lambda: None,
                         json=lambda payload=next(responses): payload)

    monkeypatch.setattr(nb.requests, "get", get)
    assert nb.get_games_for_date(date(2026, 10, 21)) == []
    assert nb.get_espn_injuries() == {}
    assert plp.get_espn_injuries() == {}
    [board] = archived(archive_root, "espn", "scoreboard")
    assert board["params"] == {"dates": "20261021"} and board["payload"] == scoreboard
    assert [e["payload"] for e in archived(archive_root, "espn", "injuries")] == [injuries] * 2

    def down(*args, **kwargs):
        raise requests.exceptions.ConnectionError("no route")

    monkeypatch.setattr(nb.requests, "get", down)
    with pytest.raises(nb.ScheduleFetchError):
        nb.get_games_for_date(date(2026, 10, 22))
    failed = archived(archive_root, "espn", "scoreboard")[-1]
    assert failed["status"] == "fetch_failed" and failed["params"] == {"dates": "20261022"}
    assert len(archived(archive_root)) == 4


def test_nba_raw_archive_0_writes_nothing(archive_root, monkeypatch):
    monkeypatch.setenv("NBA_RAW_ARCHIVE", "0")
    monkeypatch.setattr(odds_api.requests, "get", mock.Mock(return_value=odds_response([])))
    assert odds_api.get_json(EVENTS_URL, {"apiKey": "k"}) == []
    assert raw_archive.save("espn", "injuries", {}, {"injuries": []}) is None
    assert not archive_root.exists()
