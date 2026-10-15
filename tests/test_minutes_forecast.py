"""Offline feature, chronology, and low-sample tests; no runtime data touched."""
import json
import os
import subprocess
import sys
from unittest import mock

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


def test_blowout_reduction_surfaces_as_min_reduced_under_the_forecast_path(monkeypatch):
    """v10 fix: forecast_minutes() applies its own blowout cut internally, but
    detail['min_reduced'] used to stay 0.0 on this path -- the legacy line that set
    it was gated `if not MINUTES_FORECAST_ENABLED`. _build_prop_flags()'s 'BLW-Xm'
    flag and the console annotation both read min_reduced, so a real blowout cut
    must still show up there, not just inside detail['minutes_forecast']."""
    with mock.patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")):
        import playerlinepredictor as predictor

    monkeypatch.setattr(predictor, "MINUTES_FORECAST_ENABLED", True)
    monkeypatch.setattr(predictor, "_blend_rate", mock.Mock(return_value=1.0))
    monkeypatch.setattr(predictor, "detect_injury_return", mock.Mock(return_value=100))
    monkeypatch.setattr(predictor, "injury_rust_discount", mock.Mock(return_value=0.0))
    monkeypatch.setattr(predictor, "get_series_h2h", mock.Mock(return_value=1.0))
    monkeypatch.setattr(predictor, "get_series_def_factor", mock.Mock(return_value=1.0))
    monkeypatch.setattr(predictor, "_ha_factor", mock.Mock(return_value=1.0))
    monkeypatch.setattr(predictor, "_player_season_avg", mock.Mock(return_value=20.0))
    monkeypatch.setattr(predictor, "get_team_stat_avg", mock.Mock(return_value=100.0))
    monkeypatch.setattr(predictor, "compute_usage_boost", mock.Mock(return_value=1.0))
    monkeypatch.setattr(predictor.dvp_engine, "get_dvp_factor", mock.Mock(return_value=1.0))
    monkeypatch.setattr(predictor, "RESIDUAL_CORRECTIONS", {})
    monkeypatch.setattr(predictor, "USE_KALMAN_PLAYER_MODEL", False)
    monkeypatch.setattr(predictor, "_LEARNED_PLAYER_PARAMS", {})

    import minutes_forecast
    features = {"ewm_minutes": 32.0, "starter": True, "starter_source": "lineup",
                "rest_days": 2, "spread": 20.0, "blowout_reduction": 5.0,
                "games_since_absence_return": None, "absence_source": "unavailable"}
    monkeypatch.setattr(minutes_forecast, "forecast_minutes", mock.Mock(return_value=(27.0, features)))

    with mock.patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")):
        _proj, detail = predictor.project_stat(
            "Probe", "PTS", "BOS", 2, [], "PG",
            pd.DataFrame({"PTS": [20], "MIN": [27]}), pd.DataFrame({"PTS": [20], "MIN": [27]}),
            True, "NYK", False, 0, spread=20.0,
        )
    assert detail["min_reduced"] == 5.0
    assert detail["minutes_forecast"]["blowout_reduction"] == 5.0


def test_minutes_forecast_gate_reads_the_decision_field_and_fails_closed(tmp_path):
    """v10 fix: MINUTES_FORECAST_ENABLED had no equivalent to _artifact_for_season's
    automatic season-cutoff gate -- a bare hardcoded bool that anyone could flip
    without re-checking reports/phase3_minutes_gate.json. _minutes_forecast_gate_passed()
    closes that gap; a missing/corrupt/undecided file fails closed like every other
    gated artifact in this module."""
    import playerlinepredictor as predictor

    # The real, committed gate's own decision is "off" right now (it failed a
    # playoff validation check) -- the default path must reflect that.
    assert predictor._minutes_forecast_gate_passed() is False

    on = tmp_path / "on.json"
    on.write_text(json.dumps({"decision": "on"}))
    assert predictor._minutes_forecast_gate_passed(str(on)) is True

    off = tmp_path / "off.json"
    off.write_text(json.dumps({"decision": "off"}))
    assert predictor._minutes_forecast_gate_passed(str(off)) is False

    undecided = tmp_path / "undecided.json"
    undecided.write_text(json.dumps({"decision": "inconclusive"}))
    assert predictor._minutes_forecast_gate_passed(str(undecided)) is False

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("not json")
    assert predictor._minutes_forecast_gate_passed(str(corrupt)) is False

    assert predictor._minutes_forecast_gate_passed(str(tmp_path / "missing.json")) is False


def test_minutes_forecast_flag_is_forced_off_if_the_replay_gate_has_not_passed():
    """The import-time wiring is `_resolve_minutes_forecast_flag(MINUTES_FORECAST_ENABLED,
    _minutes_forecast_gate_passed())` -- a pure decision, tested directly here rather than
    by reimporting the module with a different hardcoded source default."""
    import playerlinepredictor as predictor

    assert predictor._resolve_minutes_forecast_flag(True, gate_passed=False) is False
    assert predictor._resolve_minutes_forecast_flag(True, gate_passed=True) is True
    assert predictor._resolve_minutes_forecast_flag(False, gate_passed=False) is False
    assert predictor._resolve_minutes_forecast_flag(False, gate_passed=True) is False
