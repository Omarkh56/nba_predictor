"""Position fallback must be supported and reach the real prop SD call site."""

import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import requests

import playerlinepredictor as predictor
from tests.test_train_model import _tm as trainer


def supported_variance():
    support = {"train_n": 1000, "train_players": 20, "val_n": 200,
               "val_players": 10, "val_nll_gain": 0.1, "enabled": True}
    return {"league_average": {"PTS": 6},
            "by_position": {pg: {"PTS": sd} for pg, sd in (("G", 7), ("F", 5), ("C", 4))},
            "position_support": {pg: {"PTS": support.copy()} for pg in ("G", "F", "C")}}


@pytest.fixture(autouse=True)
def clean_variance(monkeypatch):
    monkeypatch.setattr(predictor, "_LEARNED_VARIANCE", supported_variance())
    monkeypatch.setattr(predictor, "PLAYER_STD_CACHE", {})
    monkeypatch.setattr(predictor, "PROP_STD_DEV_PO", {"player_points": 6, "combo": 9})
    monkeypatch.setattr(predictor, "PROP_STD_DEV_RS", {"player_points": 5.7, "combo": 8})


@pytest.fixture
def projection_setup(monkeypatch):
    from tests.test_player_regression import neutral_projection
    return neutral_projection.__wrapped__(monkeypatch)


@pytest.mark.parametrize("pos,expected", [("PG", 7), ("SG", 7), ("SF", 5), ("PF", 5),
                                          ("G", 7), ("F", 5), ("C", 4)])
def test_supported_position_split_is_used_for_sparse_logs(pos, expected):
    empty = pd.DataFrame()
    assert predictor.get_player_std(1, empty, empty, "player_points", ["PTS"], 0,
                                    pos_group=pos) == expected


def test_empirical_sd_and_playoff_blend_remain_primary():
    rs = pd.DataFrame({"PTS": [15, 20, 23, 27, 34]})
    po = pd.DataFrame({"PTS": [12, 19, 21, 33, 40]})
    weight = predictor._adaptive_po_weight(4)
    expected = weight * po.PTS.std() + (1 - weight) * rs.PTS.std()
    assert predictor.get_player_std(1, po, rs, "player_points", ["PTS"], 4,
                                    pos_group="C") == pytest.approx(expected)


def test_empirical_one_source_can_blend_with_position_fallback():
    po = pd.DataFrame({"PTS": [12, 19, 21, 33, 40]})
    weight = predictor._adaptive_po_weight(4)
    actual = predictor.get_player_std(1, po, pd.DataFrame(), "player_points", ["PTS"], 4,
                                      pos_group="C")
    assert actual == pytest.approx(weight * po.PTS.std() + (1 - weight) * 4)


@pytest.mark.parametrize("change", [
    {"enabled": False}, {"train_n": 20}, {"train_players": 2}, {"val_n": 10},
    {"val_players": 2}, {"val_nll_gain": -0.01}, {"val_nll_gain": float("nan")},
])
def test_thin_or_unhelpful_splits_use_league_fallback(change):
    predictor._LEARNED_VARIANCE["position_support"]["G"]["PTS"].update(change)
    assert predictor.get_player_std(1, pd.DataFrame(), pd.DataFrame(), "player_points", ["PTS"], 0,
                                    pos_group="PG") == 5.7


@pytest.mark.parametrize("pos", [None, "unknown"])
def test_unknown_position_does_not_become_guard(pos):
    assert predictor.get_learned_pos_std(pos, "PTS", 5.7) == 5.7


@pytest.mark.parametrize("sd", [0, -1, float("nan"), float("inf"), "invalid", 6.01])
def test_invalid_or_immaterial_estimates_use_fallback(sd):
    predictor._LEARNED_VARIANCE["by_position"]["G"]["PTS"] = sd
    assert predictor.get_learned_pos_std("G", "PTS", 5.7) == 5.7


def test_legacy_unverified_artifact_does_not_activate_even_if_different():
    predictor._LEARNED_VARIANCE.pop("position_support")
    assert predictor.get_learned_pos_std("G", "PTS", 5.7) == 5.7


def test_sd_cache_cannot_reuse_unknown_position_for_known_position():
    empty = pd.DataFrame()
    args = (1, empty, empty, "player_points", ["PTS"], 0)
    assert predictor.get_player_std(*args) == 5.7
    assert predictor.get_player_std(*args, pos_group="PG") == 7
    assert predictor.get_player_std(*args, pos_group="C") == 4


def test_combo_keeps_empirical_or_joint_league_variance():
    logs = pd.DataFrame({"PTS": [10, 15, 20, 25, 30], "REB": [9, 8, 7, 6, 5]})
    with mock.patch.object(predictor, "get_learned_pos_std", side_effect=AssertionError("No joint variance")):
        assert predictor.get_player_std(1, pd.DataFrame(), logs, "combo", ["PTS", "REB"], 0,
                                        pos_group="C") == pytest.approx((logs.PTS + logs.REB).std())
        assert predictor.get_player_std(2, pd.DataFrame(), pd.DataFrame(), "combo", ["PTS", "REB"], 0,
                                        pos_group="C") == 8


def test_real_projection_threads_resolved_position_into_actual_sd(projection_setup, monkeypatch):
    projection_setup({})  # Set projection factors to neutral; real project_stat still runs.
    monkeypatch.setattr(predictor, "PLAYER_POS_CACHE", {"Probe": "PG"})
    monkeypatch.setattr(predictor, "get_player_id", lambda _: 1)
    monkeypatch.setattr(predictor, "get_player_logs_blended", lambda _: (
        pd.DataFrame(), pd.DataFrame({"PTS": [20], "MIN": [20]}), False, 0, 0))
    monkeypatch.setattr(predictor, "_compute_prop_range", lambda *_: (None, None))
    monkeypatch.setattr(predictor, "_apply_combo_and_usg", lambda total, *_: (total, 1, 1))
    monkeypatch.setattr(predictor, "_build_prop_metadata", lambda *_: {})
    monkeypatch.setattr(predictor, "CALIB", {})
    with mock.patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")):
        projection, details, sd, _ = predictor.project_prop(
            "Probe", "player_points", "BOS", 2, [], None, True, "NYK")
    assert projection == 20
    assert details["PTS"]["pos_group"] == "PG"
    assert sd == 7
    evaluated = predictor.evaluate_prop(projection, 21.5, "player_points", sd=sd)
    assert evaluated == predictor.evaluate_prop(20, 21.5, "player_points", sd=7)
    assert evaluated != predictor.evaluate_prop(20, 21.5, "player_points", sd=5.7)


@pytest.fixture
def position_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(trainer, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(trainer, "_sleep", lambda: None)
    monkeypatch.setattr(trainer.requests, "exceptions", requests.exceptions, raising=False)
    from nba_api.stats.endpoints import leaguedashplayerstats
    fetch = mock.Mock()
    monkeypatch.setattr(leaguedashplayerstats, "LeagueDashPlayerStats", fetch)
    return fetch


def test_position_fetch_uses_explicit_filters_without_position_column(position_fetch, tmp_path):
    groups = {"G": list(range(15)), "F": list(range(20, 35)), "C": list(range(40, 55))}
    groups["G"].append(99)
    groups["F"].append(99)  # Ambiguous hybrids are left unknown.
    position_fetch.side_effect = lambda **kw: mock.Mock(get_data_frames=lambda: [
        pd.DataFrame({"PLAYER_ID": groups[kw["player_position_abbreviation_nullable"]]})])
    positions = trainer.fetch_player_positions(["2024-25"])
    assert positions[0] == "G" and positions[20] == "F" and positions[40] == "C"
    assert 99 not in positions
    assert position_fetch.call_count == 3
    assert all(call.kwargs["per_mode_detailed"] == "PerGame" for call in position_fetch.call_args_list)
    cached = json.loads((tmp_path / "player_positions.json").read_text())
    assert cached["source"] == "nba-position-filters-v1"
    position_fetch.reset_mock()
    assert trainer.fetch_player_positions(["2024-25"], allow_fetch=False) == positions
    position_fetch.assert_not_called()


def test_no_fetch_rejects_old_all_guard_cache_and_does_not_network(position_fetch, tmp_path):
    (tmp_path / "player_positions.json").write_text(json.dumps({"1": "G", "2": "G"}))
    assert trainer.fetch_player_positions(["2024-25"], allow_fetch=False) == {}
    position_fetch.assert_not_called()


def test_ignored_position_filters_are_rejected(position_fetch, tmp_path):
    position_fetch.return_value.get_data_frames.return_value = [pd.DataFrame({"PLAYER_ID": range(30)})]
    assert trainer.fetch_player_positions(["2024-25"]) == {}
    assert not (tmp_path / "player_positions.json").exists()


def test_unknown_player_position_remains_unknown(monkeypatch):
    monkeypatch.setattr(trainer, "PLAYER_STATS", ["PTS"])
    logs = pd.DataFrame({"PLAYER_ID": [1] * 6, "PLAYER_NAME": ["Probe"] * 6,
                         "GAME_ID": range(6), "GAME_DATE": pd.date_range("2024-01-01", periods=6),
                         "MATCHUP": ["BOS vs. LAL"] * 6, "MIN": [30] * 6, "PTS": [20] * 6,
                         "season": ["2023-24"] * 6})
    frame = trainer.build_player_dataset(logs, {})
    assert not frame.empty
    assert frame.pos_group.isna().all()


@pytest.mark.parametrize("thin", [False, True])
def test_fitted_position_variance_support_is_saved_and_honored(thin, monkeypatch):
    rng = np.random.default_rng(13)
    rows = []
    for season, n in (("2022-23", 700), ("2023-24", 400)):
        for pg, sd in (("G", 1), ("F", 3), ("C", 7)):
            for i in range(n):
                ewma = rng.uniform(30, 50)
                home, rest = rng.integers(0, 2), rng.integers(1, 7)
                rows.append({"season": season, "pos_group": pg, "player_id": i % (2 if thin else 20),
                             "ewma_PTS": ewma, "actual_PTS": 2 + 0.9 * ewma + home + rng.normal(0, sd),
                             "is_home": home, "rest_days": rest})
    monkeypatch.setattr(trainer, "PLAYER_STATS", ["PTS"])
    result = trainer.fit_player_models(pd.DataFrame(rows))
    variance = json.loads(json.dumps(result["variance"]))
    monkeypatch.setattr(predictor, "_LEARNED_VARIANCE", variance)
    for pg in ("G", "F", "C"):
        support = variance["position_support"][pg]["PTS"]
        assert support["train_n"] == 700
        assert support["val_n"] == 400
        assert support["enabled"] is not thin
        fallback = variance["league_average"]["PTS"]
        sd = predictor.get_learned_pos_std(pg, "PTS", fallback)
        assert (sd == fallback) is thin
        if not thin:
            assert abs(sd / fallback - 1) >= 0.05
