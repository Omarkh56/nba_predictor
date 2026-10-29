"""ESPN's free DraftKings feed: props, game lines, and the fallback from The Odds API."""

from datetime import date, datetime, timezone
from unittest import mock

import pytest
import requests

import espn_odds
import odds_api

DAY = date(2026, 10, 27)
EVENT = {"id": "ev1", "home_team": "Denver Nuggets", "away_team": "Utah Jazz",
         "commence_time": "2026-10-27T23:00:00Z"}  # 7 PM ET on Oct 27
ODDS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/events/ev1/odds"
CORE = "http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"

SCOREBOARD = {"events": [{"id": "401", "competitions": [{
    "competitors": [{"homeAway": "home", "team": {"id": "7", "abbreviation": "DEN"}},
                    {"homeAway": "away", "team": {"id": "26", "abbreviation": "UTAH"}}],
    "odds": [{"provider": {"id": "100", "name": "DraftKings"}, "spread": 4.5, "overUnder": 230.5,
              "homeTeamOdds": {"favorite": False}, "awayTeamOdds": {"favorite": True},
              "moneyline": {"home": {"close": {"odds": "+150"}}, "away": {"close": {"odds": "-180"}}}}],
}]}]}


def _prop(athlete, kind, line, price):
    return {"athlete": {"$ref": f"{CORE}/seasons/2027/athletes/{athlete}?lang=en"},
            "type": {"name": kind}, "current": {"target": {"value": line}},
            "odds": {"american": {"value": price}}}


PROP_BETS = {"pageCount": 1, "items": [
    _prop(1, "Total Points", 20.5, "-115"), _prop(1, "Total Points", 20.5, "-105"),
    _prop(2, "Total Rebounds", 8.5, "EVEN"), _prop(2, "Total Rebounds", 8.5, "-130"),
    _prop(1, "Points Milestones", 25, "+200"),                    # alternate line: not a market here
    _prop(3, "Total Assists", 4.5, "-110"),                       # one side only: dropped
    _prop(2, "Total Points", 15.5, "-300"), _prop(2, "Total Points", 15.5, "-300"),  # not an over/under
    _prop(4, "Total Assists", 3.5, "-110"), _prop(4, "Total Assists", 3.5, "-110"),  # traded, off both rosters
]}
ROSTERS = {"7": [{"id": "1", "displayName": "Home Guard"}],
           "26": [{"id": "2", "displayName": "Away Big"}, {"id": "3", "displayName": "Away Wing"}]}


class Web:
    """Fake ESPN, plus an Odds API that answers `odds_status` to paid requests."""

    def __init__(self, odds_status=200):
        self.odds_status, self.calls = odds_status, []

    def __call__(self, url, params=None, timeout=None, **kwargs):
        self.calls.append(url)
        if "the-odds-api.com" in url:
            if self.odds_status != 200:
                raise requests.exceptions.HTTPError(str(self.odds_status),
                                                    response=mock.Mock(status_code=self.odds_status))
            return self._json({"id": "ev1", "bookmakers": [{"key": "fanduel", "title": "FanDuel", "markets": []}]})
        if url.endswith("/scoreboard"):
            assert params == {"dates": "20261027"}
            return self._json(SCOREBOARD)
        if url.endswith("/propBets"):
            assert "/events/401/competitions/401/odds/100/" in url
            return self._json(PROP_BETS)
        if "/roster" in url:
            return self._json({"athletes": ROSTERS[url.split("/teams/")[1].split("/")[0]]})
        if url.endswith("/athletes/4"):
            return self._json({"displayName": "Traded Wing"})
        raise AssertionError(f"unexpected request {url}")

    @staticmethod
    def _json(body):
        return mock.Mock(status_code=200, json=lambda: body, raise_for_status=lambda: None)

    def odds_api_calls(self):
        return [u for u in self.calls if "the-odds-api.com" in u]


@pytest.fixture
def web(monkeypatch):
    monkeypatch.setenv("NBA_ESPN_ODDS", "1")
    fake = Web()
    monkeypatch.setattr(requests, "get", fake)
    return fake


def _outcomes(payload):
    return {(m["key"], o["description"], o["name"]): (o["point"], o["price"])
            for b in payload["bookmakers"] for m in b["markets"] for o in m["outcomes"]}


def test_props_pair_over_first_and_drop_what_is_not_a_clean_over_under(web):
    payload = espn_odds.props_for_event(EVENT)
    assert payload["source"] == "espn_draftkings"
    assert [b["key"] for b in payload["bookmakers"]] == ["draftkings"]
    assert _outcomes(payload) == {
        ("player_points", "Home Guard", "Over"): (20.5, -115.0),
        ("player_points", "Home Guard", "Under"): (20.5, -105.0),
        ("player_rebounds", "Away Big", "Over"): (8.5, 100.0),
        ("player_rebounds", "Away Big", "Under"): (8.5, -130.0),
        ("player_assists", "Traded Wing", "Over"): (3.5, -110.0),
        ("player_assists", "Traded Wing", "Under"): (3.5, -110.0),
    }


def test_the_existing_parsers_read_espn_props_and_label_the_source(web):
    import oddstracker

    payload = espn_odds.props_for_event(EVENT)
    rows = oddstracker.parse_props_to_rows(payload, "UTA@DEN", "2026-10-27")
    guard = next(r for r in rows if r["player"] == "Home Guard")
    assert (guard["market"], guard["line"], guard["books_count"]) == ("PTS", 20.5, 1)
    assert guard["odds_source"] == "espn_draftkings" and 0.5 < guard["devig_over"] < 0.55  # -115 over
    quotes = oddstracker.parse_props_to_snapshot_rows(payload, "UTA@DEN", "2026-10-27",
                                                      "2026-10-27T22:45:00+00:00", "ev1", EVENT["commence_time"])
    assert {q["book_key"] for q in quotes} == {"draftkings"}
    assert {q["odds_source"] for q in quotes} == {"espn_draftkings"}


def test_game_lines_take_the_spread_sign_from_the_favourite(web):
    lines = espn_odds.game_lines_for_event(EVENT)
    assert lines == {"home_spread": 4.5, "total": 230.5, "home_moneyline": 150.0, "away_moneyline": -180.0}
    import game_odds_tracker

    now = datetime(2026, 10, 27, 20, tzinfo=timezone.utc)
    row = game_odds_tracker.parse_game_odds_to_row(espn_odds.h2h_for_event(EVENT), EVENT, now)
    assert (row["game"], row["books_count"]) == ("UTA@DEN", 1)
    assert row["devig_away"] > 0.5 and game_odds_tracker.usable_snapshot(row, DAY, now)


def test_book_spread_is_free_from_espn_and_positive_when_home_is_favoured(web, monkeypatch):
    import nba_combined

    monkeypatch.setattr(nba_combined.props_model, "_auto_fetch_spread", lambda *a: pytest.fail("Must not pay"))
    assert nba_combined._book_spread(EVENT, "Denver Nuggets") == -4.5  # Utah favoured by 4.5
    assert web.odds_api_calls() == []


def test_odds_api_out_of_credits_falls_back_to_espn_and_stops_asking(web, monkeypatch):
    import oddstracker

    web.odds_status = 401
    first = oddstracker.get_event_props("ev1", EVENT)
    assert first["source"] == "espn_draftkings"
    assert len(web.odds_api_calls()) == 1
    other = {**EVENT, "id": "ev2"}
    oddstracker.get_event_props("ev2", other)
    assert len(web.odds_api_calls()) == 1  # the 401 holds for the run: no second refused call
    with pytest.raises(requests.exceptions.RequestException):
        oddstracker.get_event_props("ev3")  # no event to look up on ESPN: the failure stands


def test_props_already_bought_today_come_from_espn_on_a_rerun(web, monkeypatch, tmp_path):
    import oddstracker

    monkeypatch.setenv("NBA_RAW_ARCHIVE_DIR", str(tmp_path))
    monkeypatch.setattr(odds_api, "ONCE_PER_DAY", True)
    monkeypatch.setattr(odds_api.team_codes, "today_slate_date", lambda now=None: DAY)
    monkeypatch.setattr(odds_api.raw_archive.team_codes, "today_slate_date", lambda now=None: DAY)
    bought = oddstracker.get_event_props("ev1", EVENT)
    assert bought["bookmakers"][0]["key"] == "fanduel" and len(web.odds_api_calls()) == 1
    odds_api._cache.clear()  # a rerun is a new process
    again = oddstracker.get_event_props("ev1", EVENT)
    assert again["source"] == "espn_draftkings" and len(web.odds_api_calls()) == 1
    with pytest.raises(odds_api.AlreadyBought):
        oddstracker.get_event_props("ev1")


def test_closing_quotes_come_from_espn_without_spending_credits(web, monkeypatch, tmp_path):
    import oddstracker

    monkeypatch.setenv("NBA_RUN_LOG", str(tmp_path / "runs.jsonl"))
    pick = {"odds_event_id": "ev1", "game": "UTA@DEN", "pick": "OVER",
            "game_start_at": EVENT["commence_time"]}
    now = datetime(2026, 10, 27, 22, 45, tzinfo=timezone.utc)
    snapshots = tmp_path / "snapshots.csv"
    assert oddstracker.closing_snapshot(DAY, now=now, predictions=[pick], snapshot_path=str(snapshots)) == 0
    assert web.odds_api_calls() == []
    text = snapshots.read_text()
    assert "draftkings" in text and "closing" in text and "espn_draftkings" in text


def test_espn_down_is_a_request_failure_callers_already_handle(monkeypatch):
    monkeypatch.setenv("NBA_ESPN_ODDS", "1")

    def down(*args, **kwargs):
        raise requests.exceptions.ConnectionError("no route")

    monkeypatch.setattr(requests, "get", down)
    with pytest.raises(requests.exceptions.RequestException, match="ESPN scoreboard unavailable"):
        espn_odds.props_for_event(EVENT)


def test_espn_lines_are_not_dropped_as_thin_markets(monkeypatch):
    """ESPN's feed carries one book for every line. Without this, a run priced from ESPN
    left every pick out (book_count < 3) and saved nothing."""
    import pandas as pd

    import nba_combined

    monkeypatch.setattr(nba_combined.props_model, "CALIB", {})
    rows = pd.DataFrame([
        {"Player": "A", "Market": "PTS", "Pick": "OVER", "Edge%": 30.0, "book_count": 1,
         "odds_source": "espn_draftkings"},
        {"Player": "B", "Market": "PTS", "Pick": "OVER", "Edge%": 30.0, "book_count": 1,
         "odds_source": "the_odds_api"},  # one of six books quoting it: still thin
        {"Player": "C", "Market": "PTS", "Pick": "OVER", "Edge%": 30.0, "book_count": 5,
         "odds_source": "the_odds_api"},
    ])
    assert nba_combined._edge_floored(rows)["Player"].tolist() == ["A", "C"]


def test_one_book_penalty_skips_espn_lines():
    import playerlinepredictor as pl

    meta = {"conf_penalty": 0.0}
    penalty, flag = pl._compute_direction_penalty("player_points", "PTS", "OVER", 22.0, 20.5, None, meta)
    thin, thin_flag = pl._compute_direction_penalty("player_points", "PTS", "OVER", 22.0, 20.5, 1, meta)
    assert (flag, thin_flag) == ("", "1-BOOK") and thin - penalty == pytest.approx(4.0)
