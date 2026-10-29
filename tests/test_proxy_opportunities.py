"""Synthetic paper-bet candidates are selected before checking if they played."""

import pandas as pd
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "replay"))
import replay_engine as engine  # noqa: E402
from backtest_bets import join_proxy_history, online_corrections, probability_over, proxy_history  # noqa: E402
from replay_config import MARKETS  # noqa: E402
from playerlinepredictor import _nb_over_prob  # noqa: E402


def test_schedule_and_prior_team_select_void_opportunity_without_game_row(monkeypatch):
    dates = pd.to_datetime(["2026-04-01", "2026-04-03", "2026-04-05", "2026-04-07", "2026-04-09"])
    history = pd.DataFrame({"GAME_DATE": dates, "TEAM_ID": [1] * 5,
                            "PLAYER_NAME": ["Known Player"] * 5,
                            "USG_ASOF": [0.2] * 5})
    schedule = pd.DataFrame([{"GAME_DATE": pd.Timestamp("2026-04-11"),
                             "GAME_ID": "G6", "TEAM_ID": 1,
                             "MATCHUP": "ABC vs. DEF", "SEASON_TYPE": "Playoffs"}])
    monkeypatch.setitem(engine._W, "pgroups", {1: history})
    monkeypatch.setitem(engine._W, "teams_by_date", {
        pd.Timestamp("2026-04-11"): {1: next(schedule.itertuples(index=False))}
    })
    candidates, failures = engine._pregame_candidates(pd.Timestamp("2026-04-11"))
    assert failures["stale_prior_team"] == 0
    assert len(candidates) == 1
    pid, prior_count, row, played = candidates[0]
    assert (pid, prior_count, played) == (1, 5, False)
    assert row["GAME_ID"] == "G6"
    assert engine.eligibility(history.index.to_numpy(), min_prior_games=5)[0]
    assert not engine.eligibility(history.index.to_numpy(), min_prior_games=10)[0]


def test_changed_team_today_remains_pregame_candidate_but_is_void(monkeypatch):
    history = pd.DataFrame({
        "GAME_DATE": pd.to_datetime(["2026-04-09", "2026-04-11"]),
        "TEAM_ID": [1, 2], "PLAYER_NAME": ["Traded", "Traded"]})
    team = pd.DataFrame([{"GAME_ID": "G6", "TEAM_ID": 1, "MATCHUP": "ABC vs. DEF",
                          "SEASON_TYPE": "Playoffs"}])
    monkeypatch.setitem(engine._W, "pgroups", {1: history})
    monkeypatch.setitem(engine._W, "teams_by_date", {
        pd.Timestamp("2026-04-11"): {1: next(team.itertuples(index=False))}
    })
    candidates, failures = engine._pregame_candidates(pd.Timestamp("2026-04-11"))
    assert len(candidates) == 1 and failures["changed_team_unobservable"] == 1
    assert candidates[0][3] is False
    assert candidates[0][2]["TEAM_ID"] == 1


def test_proxy_uses_exactly_prior_played_games_and_half_point_floor():
    played = pd.DataFrame({
        "PLAYER_ID": [1] * 11,
        "GAME_DATE": pd.date_range("2026-01-01", periods=11),
        "MIN": [1.0] + [30.0] * 10,
        "PTS": list(range(1, 12)), "REB": [1] * 11,
        "AST": [2] * 11, "FG3M": [0] * 11,
    })
    obs = pd.DataFrame({"date": ["2026-01-11", "2026-01-12"], "player_id": [1, 1],
                        "market": ["PTS+REB", "PTS+REB"], "projection": [12.0, 12.0],
                        "sd": [4.0, 4.0]})
    merged = join_proxy_history(obs, proxy_history(played, 10)).sort_values("prediction_date")
    assert merged.rolling_average.tolist() == pytest.approx([6.5, 7.5])
    assert (np.floor(merged.rolling_average) + 0.5).tolist() == [6.5, 7.5]
    assert merged.previous_game_date.max() < pd.Timestamp("2026-01-12")
    assert merged.sample_count.tolist() == [10, 11]


def test_online_correction_uses_only_prior_dates():
    opps = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"],
                         "market": ["PTS", "PTS"], "projection": [20.0, 20.0], "sd": [5.0, 5.0]})
    source = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"],
                           "market": ["PTS", "PTS"], "projection": [20.0, 20.0], "actual": [10.0, 100.0]})
    out = online_corrections(opps, source, k=1).sort_values("date")
    assert out.final_corrected_projection.tolist() == pytest.approx([20, 15])
    assert out.correction_n_prior.tolist() == [0, 1]


def test_vectorized_nb_probability_matches_live_single_stat():
    p = probability_over([15, 4], [6, 3], [15.5, 4.5], np.array([True, True]))
    assert p[0] == pytest.approx(_nb_over_prob(15, 6, 15.5))
    assert p[1] == pytest.approx(_nb_over_prob(4, 3, 4.5))
