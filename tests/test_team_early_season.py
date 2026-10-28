"""Early-season team ratings shrink toward last season until a team has 10 games."""

import pandas as pd
import pytest

import newnbapredictor as nb


def ratings(rows):
    return pd.DataFrame(rows).set_index("TEAM_ID")


CURRENT = ratings([
    {"TEAM_ID": 1, "TEAM_NAME": "One-Game Team", "GP": 1, "NET_RATING": 40.0, "PACE": 110.0, "TOTAL_PTS": 140},
    {"TEAM_ID": 2, "TEAM_NAME": "Ten-Game Team", "GP": 10, "NET_RATING": -6.0, "PACE": 98.0, "TOTAL_PTS": 1100},
])
PRIOR = ratings([
    {"TEAM_ID": 1, "TEAM_NAME": "One-Game Team", "GP": 82, "NET_RATING": 0.0, "PACE": 100.0, "TOTAL_PTS": 9000},
    {"TEAM_ID": 2, "TEAM_NAME": "Ten-Game Team", "GP": 82, "NET_RATING": 4.0, "PACE": 100.0, "TOTAL_PTS": 9000},
    {"TEAM_ID": 3, "TEAM_NAME": "Unplayed Team", "GP": 82, "NET_RATING": 2.0, "PACE": 99.0, "TOTAL_PTS": 9000},
])


def test_rates_blend_by_games_played_and_counts_stay_current():
    blended = nb._blend_early_season(CURRENT, PRIOR)
    assert blended.loc[1, "NET_RATING"] == pytest.approx(0.1 * 40.0)  # 1 game: 10% this season
    assert blended.loc[1, "PACE"] == pytest.approx(0.1 * 110 + 0.9 * 100)
    assert blended.loc[2, "NET_RATING"] == pytest.approx(-6.0)  # 10 games: this season only
    assert blended.loc[1, "GP"] == 1 and blended.loc[1, "TOTAL_PTS"] == 140


def test_a_team_without_games_uses_last_season_so_the_matchup_resolves():
    blended = nb._blend_early_season(CURRENT, PRIOR)
    assert blended.loc[3, "NET_RATING"] == pytest.approx(2.0) and blended.loc[3, "GP"] == 0
    assert nb._resolve_team("Unplayed Team", blended)[0] == 3


def test_injury_ppg_lookup_giving_up_keeps_the_espn_defaults(monkeypatch):
    """Oct 24: stats.nba.com timed out 3 times here and the whole nba_combined run crashed."""
    def gave_up(*args, **kwargs):
        raise RuntimeError("NBA API call failed after 3 attempts: Read timed out")

    monkeypatch.setattr(nb, "_api_call", gave_up)
    injuries = {"Chicago Bulls": [{"player": "Probe", "status": "Out", "ppg": 0.0}]}
    assert nb._enrich_injuries_with_ppg(injuries, "2026-27") == injuries


def test_team_model_outage_leaves_the_run_to_props(monkeypatch):
    import nba_combined

    def gave_up(*args, **kwargs):
        raise RuntimeError("NBA API call failed after 3 attempts")

    monkeypatch.setattr(nba_combined.team_model, "predict_games_data", gave_up)
    assert nba_combined._run_team_model([{"away_abbr": "A", "home_abbr": "B"}], {}) == ([], {})


def test_predict_games_data_blends_only_while_a_team_is_short_of_ten_games(monkeypatch):
    fetched, seen = [], {}
    monkeypatch.setattr(nb, "_season_with_team_stats", lambda season: season)
    monkeypatch.setattr(nb, "_fetch_all_team_data", lambda season: fetched.append(season) or (
        CURRENT if season == "2026-27" else PRIOR).assign(OFF_RATING=110.0, DEF_RATING=110.0, EFG_PCT=0.5))
    def capture(matchup, ratings_table, *args):
        seen["ratings"] = ratings_table

    monkeypatch.setattr(nb, "_compute_prediction", capture)
    nb.predict_games_data("2026-27", [{"away_abbr": "A", "home_abbr": "B"}], {})
    assert fetched == ["2026-27", "2025-26"]
    assert seen["ratings"].loc[1, "NET_RATING"] == pytest.approx(4.0)

    fetched.clear()
    full = CURRENT.assign(GP=40, OFF_RATING=110.0, DEF_RATING=110.0, EFG_PCT=0.5)
    monkeypatch.setattr(nb, "_fetch_all_team_data", lambda season: fetched.append(season) or full)
    nb.predict_games_data("2026-27", [{"away_abbr": "A", "home_abbr": "B"}], {})
    assert fetched == ["2026-27"]  # every team past 10 games: no prior-season fetch
