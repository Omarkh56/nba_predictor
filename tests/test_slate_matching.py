"""Slate dates and team codes: odds must belong to the selected slate."""

from datetime import date, datetime, timezone
from unittest import mock

import pytest
import requests


@pytest.fixture
def combined():
    with mock.patch(
        "requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")
    ):
        import nba_combined

        yield nba_combined


def _event(eid, home, away, tip):
    return {"id": eid, "home_team": home, "away_team": away, "commence_time": tip}


def test_match_uses_the_event_on_the_slate_date(combined):
    events = [
        _event("later", "Boston Celtics", "New York Knicks", "2026-10-24T23:30:00Z"),
        _event("tonight", "Boston Celtics", "New York Knicks", "2026-10-22T23:30:00Z"),
    ]
    assert combined._match_odds_event(events, "BOS", "NY", date(2026, 10, 22))["id"] == "tonight"
    assert combined._match_odds_event(events, "BOS", "NY", date(2026, 10, 23)) is None


@pytest.mark.parametrize(
    "home,away,home_name,away_name",
    [
        ("DEN", "UTAH", "Denver Nuggets", "Utah Jazz"),
        ("SA", "NY", "San Antonio Spurs", "New York Knicks"),
        ("LAC", "WSH", "Los Angeles Clippers", "Washington Wizards"),
    ],
)
def test_espn_codes_match_odds_api_names(combined, home, away, home_name, away_name):
    events = [_event("e1", home_name, away_name, "2026-10-22T02:00:00Z")]
    assert combined._match_odds_event(events, home, away, date(2026, 10, 21))["id"] == "e1"


def test_setup_keeps_only_the_slates_events(combined, monkeypatch):
    events = [
        _event("tonight", "Boston Celtics", "New York Knicks", "2026-10-22T23:30:00Z"),
        _event("next-week", "Denver Nuggets", "Utah Jazz", "2026-10-29T02:00:00Z"),
    ]
    monkeypatch.setattr(combined.props_model, "get_all_odds_events", lambda: list(events))
    monkeypatch.setattr(combined.props_model, "_auto_fetch_spread", lambda *a, **k: -3.5)
    monkeypatch.setattr(combined.props_model, "GAME_SPREADS", {})
    matchups = [{
        "home_abbr": "BOS", "away_abbr": "NYK", "home_name": "Boston Celtics",
        "away_name": "New York Knicks", "game_date": date(2026, 10, 22),
    }]
    kept = combined._setup_odds_and_spreads(matchups, {}, date(2026, 10, 22))
    assert [ev["id"] for ev in kept] == ["tonight"]


def test_game_pick_stores_the_home_line_for_an_away_favourite(combined, monkeypatch):
    captured = {}

    def upsert(rows, game_date):
        captured["rows"] = rows
        return len(rows)

    monkeypatch.setattr(combined.pred_db, "upsert_game_predictions", upsert)
    pred = {
        "home_abbr": "LAL", "away_abbr": "OKC", "expected_margin": -6.1, "home_spread": 6.1,
        "away_spread": -6.1, "favorite": "OKC", "fav_prob": 0.70, "predicted_total": 220.0,
    }
    combined.export_game_picks_db([pred], date(2026, 5, 11))
    assert captured["rows"][0]["spread"] == 6.1
    assert captured["rows"][0]["spread_pick"] == "OKC"


def test_team_ids_resolve_espn_codes():
    with mock.patch(
        "requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")
    ):
        import playerlinepredictor as props

    for espn, nba in [("NY", "NYK"), ("GS", "GSW"), ("SA", "SAS"), ("NO", "NOP"),
                      ("UTAH", "UTA"), ("WSH", "WAS")]:
        assert props.get_team_id(espn) == props.get_team_id(nba) is not None


PROPS = {"bookmakers": [{"title": "BookA", "markets": [{"key": "player_points", "outcomes": [
    {"name": "Over", "description": "P1", "price": -110, "point": 20.5},
    {"name": "Under", "description": "P1", "price": -110, "point": 20.5},
]}]}]}


def test_tracker_snapshot_keeps_the_slate_and_uses_nba_codes(tmp_path, monkeypatch):
    import oddstracker

    events = [
        _event("e1", "Los Angeles Lakers", "New York Knicks", "2026-10-22T02:00:00Z"),  # Oct 21 ET
        _event("e2", "Los Angeles Clippers", "New Orleans Pelicans", "2026-10-25T02:00:00Z"),
    ]
    monkeypatch.setattr(oddstracker, "get_event_props", lambda event_id: PROPS)
    monkeypatch.setattr(oddstracker.time, "sleep", lambda seconds: None)
    rows = oddstracker.snapshot(
        date(2026, 10, 21), events=events, now=datetime(2026, 10, 21, 18, tzinfo=timezone.utc),
        tracker_path=str(tmp_path / "t.csv"), snapshot_path=str(tmp_path / "s.csv"),
    )
    assert {row["game"] for row in rows} == {"NYK@LAL"}
    assert {row["event_id"] for row in rows} == {"e1"}
    assert oddstracker.LAST_SNAPSHOT_STATUS["status"] == "ok"


def test_tracker_snapshot_reports_a_fetch_failure(tmp_path, monkeypatch):
    import oddstracker

    def down(event_id):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(oddstracker, "get_event_props", down)
    monkeypatch.setattr(oddstracker.time, "sleep", lambda seconds: None)
    events = [_event("e1", "Los Angeles Lakers", "New York Knicks", "2026-10-22T02:00:00Z")]
    rows = oddstracker.snapshot(
        date(2026, 10, 21), events=events,
        tracker_path=str(tmp_path / "t.csv"), snapshot_path=str(tmp_path / "s.csv"),
    )
    assert rows == []
    assert oddstracker.LAST_SNAPSHOT_STATUS["status"] == "fetch_failed"


def test_espn_schedule_normalizes_codes_and_raises_on_failure(monkeypatch):
    import newnbapredictor as nb

    payload = {"events": [{
        "status": {"type": {"state": "pre", "shortDetail": "7:00 PM"}},
        "competitions": [{"competitors": [
            {"homeAway": "home", "team": {"abbreviation": "UTAH", "displayName": "Utah Jazz"}},
            {"homeAway": "away", "team": {"abbreviation": "NY", "displayName": "New York Knicks"}},
        ]}],
    }]}
    response = mock.Mock()
    response.raise_for_status = lambda: None
    response.json = lambda: payload
    monkeypatch.setattr(nb.requests, "get", lambda *a, **k: response)
    games = nb.get_games_for_date(date(2026, 10, 21))
    assert [(g["away_abbr"], g["home_abbr"]) for g in games] == [("NYK", "UTA")]
    assert games[0]["home_name"] == "Utah Jazz"

    calls = []

    def down(*a, **k):
        calls.append(1)
        raise requests.exceptions.ConnectionError("no route")

    monkeypatch.setattr(nb.requests, "get", down)
    with pytest.raises(nb.ScheduleFetchError):
        nb.get_games_for_date(date(2026, 10, 21))
    calls.clear()
    with pytest.raises(nb.ScheduleFetchError):
        nb.get_todays_games()
    assert len(calls) == 3  # fails fast instead of 46 timeouts
