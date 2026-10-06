"""Offline feature, chronology, and low-sample tests; no runtime data touched."""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from minutes_forecast import forecast_minutes, shrink_rate


def logs(minutes, dates=None):
    return pd.DataFrame({"MIN": minutes, "GAME_DATE": dates or list(pd.date_range("2025-10-01", periods=len(minutes)))})


def test_future_minutes_and_schedule_cannot_change_features():
    history = logs([30, 32, 100], ["2025-10-01", "2025-10-03", "2025-10-05"])
    a = forecast_minutes(history, "2025-10-05", team_game_dates=["2025-10-01", "2025-10-03"])
    history.loc[2, "MIN"] = 0
    b = forecast_minutes(history, "2025-10-05", team_game_dates=["2025-10-01", "2025-10-03", "2025-10-07"])
    assert a == b


def test_starter_rest_and_blowout():
    history = logs([32, 32])
    rested, features = forecast_minutes(history, "2025-10-04", starter=True)
    back_to_back, _ = forecast_minutes(history, "2025-10-03", starter=True)
    favored, favored_features = forecast_minutes(history, "2025-10-04", spread=18, is_home=True, starter=True)
    underdog, _ = forecast_minutes(history, "2025-10-04", spread=18, is_home=False, starter=True)
    bench, _ = forecast_minutes(history, "2025-10-04", spread=18, is_home=True, starter=False)
    assert rested - back_to_back == pytest.approx(0.5)
    assert rested - favored == 5 and favored_features["blowout_reduction"] == 5
    assert underdog == rested == bench
    assert features["starter_source"] == "lineup"


def test_proxy_is_explicit_and_playoff_role_affects_minutes():
    starter, features = forecast_minutes(logs([30, 30]), "2025-10-04", playoffs=True)
    bench, _ = forecast_minutes(logs([30, 30]), "2025-10-04", playoffs=True, starter=False)
    assert starter == pytest.approx(31.8) and bench == 30
    assert features["starter_source"] == "minutes_proxy"


def test_calendar_gap_does_not_prove_absence():
    history = logs([32, 32], ["2025-10-01", "2025-10-12"])
    normal, unknown = forecast_minutes(history, "2025-10-14")
    returned, known = forecast_minutes(history, "2025-10-14", team_game_dates=["2025-10-03", "2025-10-06", "2025-10-09"])
    assert unknown["games_since_absence_return"] is None
    assert known["games_since_absence_return"] == 1 and returned < normal


@pytest.mark.parametrize("games,expected", [(0, 1.0), (1, 1.1), (9, 1.9), (10, 2.0), (20, 2.0)])
def test_prior_fades_at_ten_games(games, expected):
    assert shrink_rate(2.0, 1.0, games) == pytest.approx(expected)


def test_missing_prior_does_not_impute_rate():
    assert shrink_rate(2.0, None, 2) == 2
    assert shrink_rate(None, 1.0, 2) == 1
    assert shrink_rate(None, None, 2) is None


def test_no_minutes_fail_closed():
    assert forecast_minutes(pd.DataFrame(), "2025-10-01")[0] is None


def test_diagnostics_counts_player_game_once_and_uses_paired_rows(tmp_path, monkeypatch):
    root = os.path.dirname(os.path.dirname(__file__))
    monkeypatch.syspath_prepend(os.path.join(root, "scripts", "replay"))
    from minutes_diagnostics import diagnostics, write_diagnostics
    frame = pd.DataFrame([
        {"date": "2025-10-22", "game_id": "1", "player_id": 1, "market": "PTS", "season_type": "Regular Season",
         "actual": 10, "projection": 12, "actual_min": 30, "proj_min": 32, "n_prior": 2},
        {"date": "2025-10-22", "game_id": "1", "player_id": 1, "market": "REB", "season_type": "Regular Season",
         "actual": 5, "projection": 5, "actual_min": 30, "proj_min": 32, "n_prior": 2}])
    baseline = frame.copy()
    baseline.proj_min = 34
    result = write_diagnostics(frame, baseline, "test", "2025-26", tmp_path)
    values = result["regimes"]["Regular Season"]["candidate"]
    assert values["player_games"] == 1 and values["minutes_mae"] == 2
    assert values["macro_bias_pct"] == 10
    assert sum(bucket["rows"] for bucket in values["minutes_error_buckets"].values()) == 2
    assert json.loads((tmp_path / "test_2025-26_minutes.json").read_text())["paired_rows"] == 2
    with pytest.raises(ValueError, match="duplicate"):
        diagnostics(pd.concat([frame, frame]))


def test_candidate_replay_reads_only_prior_data(tmp_path):
    from tests.test_replay_engine import SCRIPT, ROOT
    script = SCRIPT.replace('eng.run_replay(bundle, ["Regular Season"], 1, {}, ROOT',
                            'eng.run_replay(bundle, ["Regular Season"], 1, {"MINUTES_FORECAST_ENABLED": True}, ROOT')
    proc = subprocess.run([sys.executable, "-c", script, ROOT], cwd=tmp_path, capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["same"] == result["other_future"]
    assert result["same"]["PTS"] != result["other_prior"]["PTS"]


def test_low_games_replay_uses_previous_season_only_before_ten(tmp_path):
    from tests.test_replay_engine import SCRIPT, ROOT
    script = SCRIPT.split('a, act_a, counts =')[0] + '''
def early(previous_pts, count):
    bundle, _ = build(20, 20, 20)
    previous = bundle.players.copy()
    previous.GAME_DATE = previous.GAME_DATE - pd.Timedelta(days=365)
    previous.PTS = previous_pts
    bundle.previous_players = previous
    cutoff = str(sorted(bundle.players.GAME_DATE.unique())[count])[:10]
    df, _ = eng.run_replay(bundle, ["Regular Season"], 1,
                          {"MINUTES_FORECAST_ENABLED": True}, ROOT,
                          date_from=cutoff, date_to=cutoff, min_prior_games=1)
    return float(df[(df.player == "Test Star") & (df.market == "PTS")].projection.iloc[0])
print(json.dumps({"early_a": early(10, 3), "early_b": early(30, 3),
                  "late_a": early(10, 15), "late_b": early(30, 15)}))
'''
    proc = subprocess.run([sys.executable, "-c", script, ROOT], cwd=tmp_path,
                          capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["early_a"] < result["early_b"]
    assert result["late_a"] == result["late_b"]


def test_gate_requires_all_metrics_and_both_bias_thresholds(monkeypatch):
    import copy
    root = os.path.dirname(os.path.dirname(__file__))
    monkeypatch.syspath_prepend(os.path.join(root, "scripts", "replay"))
    from minutes_diagnostics import evaluate_gate
    good = {"regimes": {regime: {"baseline": {"minutes_mae": 5, "projection_mae": 3},
                               "candidate": {"minutes_mae": 4, "projection_mae": 2, "macro_bias_pct": 1}}
                         for regime in ("Regular Season", "Playoffs")}}
    assert evaluate_gate(good, good)["passed"]
    for regime, metric, value in [("Regular Season", "macro_bias_pct", -2.3),
                                  ("Playoffs", "macro_bias_pct", 4.2),
                                  ("Regular Season", "minutes_mae", 5),
                                  ("Playoffs", "projection_mae", float("nan"))]:
        bad = copy.deepcopy(good)
        bad["regimes"][regime]["candidate"][metric] = value
        assert not evaluate_gate(good, bad)["passed"]
    assert not evaluate_gate({}, good)["passed"]


def test_saved_numeric_game_ids_keep_leading_zeroes(tmp_path, monkeypatch):
    root = os.path.dirname(os.path.dirname(__file__))
    monkeypatch.syspath_prepend(os.path.join(root, "scripts", "replay"))
    from replay_props import read_replay
    from minutes_diagnostics import diagnostics
    path = tmp_path / "rows.csv"
    path.write_text("date,game_id,player_id,market,season_type,actual,projection,actual_min,proj_min,n_prior\n"
                    "2025-10-22,0022500001,1,PTS,Regular Season,10,12,30,32,2\n")
    saved = read_replay(path)
    assert saved.game_id.iloc[0] == "0022500001"
    legacy = pd.read_csv(path)
    assert diagnostics(saved, legacy)["paired_rows"] == 1
