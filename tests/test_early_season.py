"""v10: opening-week props. Before a player has played this season, the live model could
neither place him on a team nor project him. These tests cover the roster-based team
assignment and the regular-season, fewer-than-10-games prior-season path."""
from unittest import mock

import pandas as pd
import pytest
import requests

BOS, DET = 1610612738, 1610612765


@pytest.fixture
def predictor(monkeypatch):
    with mock.patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")):
        import playerlinepredictor

    monkeypatch.setattr(playerlinepredictor, "TEAM_ROSTER_CACHE", {})
    monkeypatch.setattr(playerlinepredictor, "TEAM_ROSTER_IDS", {})
    monkeypatch.setattr(playerlinepredictor, "api_sleep", lambda: None)
    return playerlinepredictor


def _history(pl, matchups=()):
    logs = pd.DataFrame({"MATCHUP": list(matchups), "MIN": [30] * len(matchups),
                         "GAME_DATE": pd.date_range("2026-10-20", periods=len(matchups))})
    empty = logs.iloc[0:0]
    return pl.PlayerLogHistory(po_rates=empty, rs_rates=logs, po_minutes=empty, rs_minutes=logs,
                               using_playoffs=False, po_rate_count=0, po_raw_count=0)


def _context(pl, monkeypatch, players, histories):
    monkeypatch.setattr(pl, "get_player_id", lambda name: players.get(name))
    monkeypatch.setattr(pl, "get_player_logs_blended", lambda pid: histories.get(pid, _history(pl)))
    cdf = pd.DataFrame({"player": list(players)})
    return pl._build_player_context(cdf, "Detroit Pistons", "Boston Celtics", "DET", "BOS", DET, BOS, {})


def test_zero_game_players_are_placed_from_the_current_roster(predictor, monkeypatch):
    predictor.TEAM_ROSTER_IDS.update({DET: {1, 2}, BOS: {3}})
    predictor.TEAM_ROSTER_CACHE.update({DET: {}, BOS: {}})
    ctx = _context(predictor, monkeypatch, {"Home Guy": 1, "Away Guy": 3, "Nobody": 9}, {})
    assert ctx["Home Guy"]["is_home"] is True and ctx["Home Guy"]["team_abbr"] == "DET"
    assert ctx["Home Guy"]["opp_tid"] == BOS
    assert ctx["Away Guy"]["is_home"] is False and ctx["Away Guy"]["team_abbr"] == "BOS"
    assert ctx["Away Guy"]["opp_tid"] == DET
    assert "Nobody" not in ctx  # on neither roster: unknown stays unknown


def test_a_player_on_both_rosters_is_ambiguous_and_skipped(predictor, monkeypatch):
    predictor.TEAM_ROSTER_IDS.update({DET: {1}, BOS: {1}})
    predictor.TEAM_ROSTER_CACHE.update({DET: {}, BOS: {}})
    assert _context(predictor, monkeypatch, {"Both": 1}, {}) == {}


def test_current_season_logs_win_over_the_roster(predictor, monkeypatch):
    # Logs say he plays for BOS (the away team tonight); a stale roster says DET.
    predictor.TEAM_ROSTER_IDS.update({DET: {1}, BOS: set()})
    predictor.TEAM_ROSTER_CACHE.update({DET: {}, BOS: {}})
    hist = {1: _history(predictor, ["BOS vs. NYK", "BOS @ MIA"])}
    ctx = _context(predictor, monkeypatch, {"Traded": 1}, hist)
    assert ctx["Traded"]["team_abbr"] == "BOS" and ctx["Traded"]["is_home"] is False


def test_roster_ids_include_blank_position_players(predictor, monkeypatch):
    frame = pd.DataFrame({"PLAYER": ["Starter", "Camp Invite"], "POSITION": ["G", ""], "PLAYER_ID": [11, 12]})
    fake = mock.Mock()
    fake.return_value.get_data_frames.return_value = [frame]
    monkeypatch.setattr(predictor.commonteamroster, "CommonTeamRoster", fake)
    predictor._fetch_team_roster(BOS)
    assert predictor.TEAM_ROSTER_IDS[BOS] == {11, 12}
    assert "Camp Invite" not in predictor.TEAM_ROSTER_CACHE[BOS]  # position cache unchanged


def test_a_failed_roster_fetch_is_not_cached(predictor, monkeypatch):
    fake = mock.Mock(side_effect=requests.exceptions.ConnectionError("down"))
    monkeypatch.setattr(predictor.commonteamroster, "CommonTeamRoster", fake)
    assert predictor._fetch_team_roster(BOS) == {}
    assert BOS not in predictor.TEAM_ROSTER_IDS and BOS not in predictor.TEAM_ROSTER_CACHE
    assert predictor._team_from_roster(1, BOS, DET, "BOS", "DET") is None
