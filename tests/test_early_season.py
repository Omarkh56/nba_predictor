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


# --------------------------------------------------------------------------- the early-season projection path
@pytest.mark.parametrize("full,early,games,playoffs,expected", [
    (False, False, 0, False, False),   # both off: old behaviour
    (False, True, 0, False, True),     # opening night, regular season
    (False, True, 9, False, True),
    (False, True, 10, False, False),   # 10+ games: the validated default path
    (False, True, 0, True, False),     # never in the playoffs (where the forecast failed its gate)
    (True, False, 25, True, True),     # the full experimental feature still forces it on
])
def test_use_forecast_path_regime(predictor, monkeypatch, full, early, games, playoffs, expected):
    monkeypatch.setattr(predictor, "MINUTES_FORECAST_ENABLED", full)
    monkeypatch.setattr(predictor, "EARLY_SEASON_FALLBACK_ENABLED", early)
    assert predictor._use_forecast_path(games, playoffs) is expected


@pytest.fixture
def neutral(predictor, monkeypatch):
    """Real project_prop with every external factor neutral and no network."""
    for name, value in {"get_series_h2h": 1.0, "get_series_def_factor": 1.0, "_ha_factor": 1.0,
                        "compute_usage_boost": 1.0, "get_team_stat_avg": 0.0}.items():
        monkeypatch.setattr(predictor, name, mock.Mock(return_value=value))
    monkeypatch.setattr(predictor.dvp_engine, "get_dvp_factor", mock.Mock(return_value=1.0))
    monkeypatch.setattr(predictor, "RESIDUAL_CORRECTIONS", {})
    monkeypatch.setattr(predictor, "USE_KALMAN_PLAYER_MODEL", False)
    monkeypatch.setattr(predictor, "_LEARNED_PLAYER_PARAMS", {})
    monkeypatch.setattr(predictor, "PLAYER_USG_CACHE", {})
    monkeypatch.setattr(predictor, "MINUTES_FORECAST_ENABLED", False)
    monkeypatch.setattr(predictor, "IS_PLAYOFF_SEASON", False)
    monkeypatch.setattr(predictor, "SEASON", "2026-27")
    monkeypatch.setattr(predictor, "get_player_id", lambda name: 7)
    previous = pd.DataFrame({"PTS": [20] * 12, "REB": [5] * 12, "AST": [4] * 12, "FG3M": [2] * 12,
                             "MIN": [32] * 12, "MATCHUP": ["DET vs. BOS"] * 12,
                             "GAME_DATE": pd.date_range("2026-02-01", periods=12)[::-1]})

    def run(current_games, flag):
        monkeypatch.setattr(predictor, "EARLY_SEASON_FALLBACK_ENABLED", flag)
        cur = pd.DataFrame({"PTS": [10] * current_games, "REB": [3] * current_games, "AST": [2] * current_games,
                            "FG3M": [1] * current_games, "MIN": [30] * current_games,
                            "MATCHUP": ["DET vs. BOS"] * current_games,
                            "GAME_DATE": pd.date_range("2026-10-21", periods=current_games)[::-1]})
        empty = cur.iloc[0:0]
        hist = predictor.PlayerLogHistory(po_rates=empty, rs_rates=cur, po_minutes=empty, rs_minutes=cur,
                                          using_playoffs=False, po_rate_count=0, po_raw_count=0)
        monkeypatch.setattr(predictor, "get_player_logs_blended", lambda pid: hist)
        fetched = []

        def fake_fetch(pid, season, season_type, min_minutes, retries=3):
            fetched.append(season)
            return previous.copy() if season == "2025-26" else empty

        monkeypatch.setattr(predictor, "_fetch_logs", fake_fetch)
        with mock.patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")):
            out = predictor.project_prop("Probe", "player_points", "BOS", BOS, [], "G", True, "DET",
                                         minutes_context={"cutoff": pd.Timestamp("2026-11-15").date()})
        return out, fetched

    return run


def test_zero_game_player_is_projected_from_last_season_only_when_enabled(neutral):
    (proj_off, *_), fetched_off = neutral(0, flag=False)
    assert proj_off is None and fetched_off == []  # today's behaviour: nothing to project from
    (proj_on, details, sd, meta), fetched_on = neutral(0, flag=True)
    assert proj_on is not None and proj_on > 10  # last season's 20 PTS, not this season's (none)
    assert fetched_on == ["2025-26"]


def test_ten_plus_games_never_take_the_forecast_path(neutral, monkeypatch):
    import minutes_forecast
    spy = mock.Mock(side_effect=AssertionError("forecast path used at 10+ games"))
    monkeypatch.setattr(minutes_forecast, "forecast_minutes", spy)
    (proj_off, *_), _ = neutral(12, flag=False)
    (proj_on, *_), fetched = neutral(12, flag=True)
    assert proj_on == proj_off and fetched == [] and not spy.called


def test_playoffs_never_take_the_early_path(neutral, monkeypatch, predictor):
    import minutes_forecast
    spy = mock.Mock(side_effect=AssertionError("forecast path used in the playoffs"))
    monkeypatch.setattr(minutes_forecast, "forecast_minutes", spy)
    monkeypatch.setattr(predictor, "IS_PLAYOFF_SEASON", True)
    (proj, *_), fetched = neutral(3, flag=True)
    assert proj is not None and fetched == [] and not spy.called


def test_replay_opening_night_with_the_early_path_makes_no_network_call(tmp_path):
    """Opening night has no league logs. dvp.py used to fall back to a live fetch there (blocked in
    replay workers); the engine now seeds the same empty table a live run gets."""
    import json
    import subprocess
    import sys
    from tests.test_replay_engine import ROOT, SCRIPT
    script = SCRIPT.split('a, act_a, counts =')[0] + '''
bundle, _ = build(20, 20, 20)
previous = bundle.players.copy()
previous.GAME_DATE = previous.GAME_DATE - pd.Timedelta(days=365)
bundle.previous_players = previous
first = str(sorted(bundle.players.GAME_DATE.unique())[0])[:10]
df, info = eng.run_replay(bundle, ["Regular Season"], 1, {"EARLY_SEASON_FALLBACK_ENABLED": True}, ROOT,
                          date_from=first, date_to=first, min_prior_games=0)
print(json.dumps({"rows": int(len(df)), "players": sorted(df.player.unique().tolist()) if len(df) else []}))
'''
    proc = subprocess.run([sys.executable, "-c", script, ROOT], cwd=tmp_path, capture_output=True,
                          text=True, timeout=240)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["rows"] > 0 and "Test Star" in out["players"]
