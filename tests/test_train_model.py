"""
tests/test_train_model.py — Unit tests for the actual learning/fitting
functions in train_model.py, using small synthetic datasets (no network,
no data_cache/ dependency).

Covers:
  _build_three_way_split()  — train/val/test partitioning has no leakage
  fit_game_models()         — logistic/ridge fit actually beats a trivial
                               always-0.5 baseline on held-out data
  ablation_study()          — a feature that's pure signal gets accepted,
                               a feature that's pure noise gets rejected
"""

import importlib.util
import sys
import types
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

_proj = Path(__file__).parent.parent
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))


def _import_train_model():
    """Load train_model.py with NBA-API/requests stubbed out (no network at import)."""
    nba_stubs = {}
    for name in [
        "nba_api", "nba_api.stats", "nba_api.stats.library",
        "nba_api.stats.library.http", "nba_api.stats.endpoints",
        "nba_api.stats.endpoints.leaguegamefinder",
        "nba_api.stats.endpoints.leaguegamelog",
        "nba_api.stats.endpoints.leaguedashplayerstats",
    ]:
        nba_stubs[name] = types.ModuleType(name)
    nba_stubs["nba_api.stats.library.http"].STATS_HEADERS = {}
    nba_stubs["nba_api.stats.library.http"].STATS_TIMEOUT = 30

    req_stub = types.ModuleType("requests")
    req_stub.get = mock.Mock()
    req_stub.RequestException = Exception
    req_stub.Session = mock.MagicMock

    with mock.patch.dict("sys.modules", {**nba_stubs, "requests": req_stub}):
        spec = importlib.util.spec_from_file_location(
            "train_model_stub", str(_proj / "train_model.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


_tm = _import_train_model()

_build_three_way_split = _tm._build_three_way_split
fit_game_models        = _tm.fit_game_models
ablation_study          = _tm.ablation_study
hand_tuned_predict      = _tm.hand_tuned_predict
GAME_FEATURES           = _tm.GAME_FEATURES
VAL_SEASON              = _tm.VAL_SEASON
TEST_SEASON             = _tm.TEST_SEASON


# ── Synthetic dataset builder ─────────────────────────────────────────────────

def _make_game_df(n_per_season: int, seasons: list, signal_feature: str,
                   signal_weight: float = 3.0, seed: int = 0) -> pd.DataFrame:
    """
    Build a synthetic df_game: `signal_feature` truly drives home_win (via a
    logistic relationship); every other GAME_FEATURES column plus
    altitude_penalty/venue_residual/star_form_diff is pure independent noise.
    """
    rng = np.random.default_rng(seed)
    all_feats = list(GAME_FEATURES) + ["altitude_penalty", "venue_residual", "star_form_diff"]
    rows = []
    for season in seasons:
        for _ in range(n_per_season):
            feats = {f: float(rng.normal(0, 1)) for f in all_feats}
            # home_b2b / away_b2b are boolean-ish in the real pipeline
            feats["home_b2b"] = bool(rng.random() < 0.15)
            feats["away_b2b"] = bool(rng.random() < 0.15)
            logit = signal_weight * feats[signal_feature]
            p = 1.0 / (1.0 + np.exp(-logit))
            home_win = int(rng.random() < p)
            rows.append({
                **feats,
                "season": season,
                "home_win": home_win,
                "actual_margin": float(rng.normal(logit * 3.0, 5.0)),
            })
    return pd.DataFrame(rows)


# ── _build_three_way_split ────────────────────────────────────────────────────

class TestThreeWaySplit:
    def test_no_leakage_across_splits(self):
        seasons = ["2020-21", "2021-22", VAL_SEASON, TEST_SEASON]
        df = _make_game_df(40, seasons, signal_feature="net_rtg_diff")

        df_train, df_val, df_test = _build_three_way_split(df)

        assert set(df_val["season"].unique())  == {VAL_SEASON}
        assert set(df_test["season"].unique()) == {TEST_SEASON}
        assert VAL_SEASON  not in df_train["season"].unique()
        assert TEST_SEASON not in df_train["season"].unique()

        # every row accounted for exactly once
        assert len(df_train) + len(df_val) + len(df_test) == len(df)

    def test_exits_when_val_season_missing(self):
        df = _make_game_df(20, ["2020-21", "2021-22"], signal_feature="net_rtg_diff")
        with pytest.raises(SystemExit):
            _build_three_way_split(df)


# ── fit_game_models ────────────────────────────────────────────────────────────

class TestFitGameModels:
    def test_learned_model_beats_trivial_baseline(self):
        seasons = ["2020-21", "2021-22", "2022-23", VAL_SEASON]
        df = _make_game_df(150, seasons, signal_feature="net_rtg_diff", signal_weight=2.5)
        df_train = df[df["season"] != VAL_SEASON]
        df_val   = df[df["season"] == VAL_SEASON]

        result = fit_game_models(df_train, df_val)

        trivial_ll = log_loss(df_val["home_win"].values, np.full(len(df_val), 0.5))
        assert result["learned_log_loss"] < trivial_ll, (
            "logistic regression should beat a constant p=0.5 baseline when one "
            "feature is a genuine (if noisy) predictor of the outcome"
        )
        assert result["recommendation"] in {"swap", "monitor", "keep_hand_tuned"}
        assert result["feature_names"] == GAME_FEATURES
        assert len(result["scaler_mean"]) == len(GAME_FEATURES)

    def test_test_set_scored_independently_of_val(self):
        seasons = ["2020-21", "2021-22", "2022-23", VAL_SEASON, TEST_SEASON]
        df = _make_game_df(120, seasons, signal_feature="net_rtg_diff", signal_weight=2.5)
        df_train = df[~df["season"].isin({VAL_SEASON, TEST_SEASON})]
        df_val   = df[df["season"] == VAL_SEASON]
        df_test  = df[df["season"] == TEST_SEASON]

        result = fit_game_models(df_train, df_val, df_test=df_test)

        assert result["test_log_loss"] is not None
        assert result["test_games"] == len(df_test)
        # test metrics must not simply mirror val metrics (independent split)
        assert result["test_log_loss"] != result["learned_log_loss"]


# ── hand_tuned_predict ─────────────────────────────────────────────────────────

class TestHandTunedPredict:
    def test_output_is_valid_probability(self):
        df = _make_game_df(30, ["2020-21"], signal_feature="net_rtg_diff")
        probs = hand_tuned_predict(df)
        assert len(probs) == len(df)
        assert np.all((probs > 0) & (probs < 1))


# ── ablation_study ─────────────────────────────────────────────────────────────

class TestAblationStudy:
    def test_signal_feature_passes_noise_feature_fails(self):
        # star_form_diff genuinely drives the outcome; the 10 base features,
        # altitude_penalty and venue_residual are all independent noise.
        seasons = ["2020-21", "2021-22", "2022-23", VAL_SEASON]
        df = _make_game_df(300, seasons, signal_feature="star_form_diff", signal_weight=3.0)
        df_train = df[df["season"] != VAL_SEASON]
        df_val   = df[df["season"] == VAL_SEASON]

        result = ablation_study(df_train, df_val)

        assert "star_form_diff" in result["winning_features"], (
            "a feature that truly drives the outcome should beat the noise "
            "baseline on both log loss and Brier"
        )
        assert result["features"]["star_form_diff"]["passed"] is True
        assert result["new_signal_rec"] == "use_extended"

        for noise_feat in ("altitude_penalty", "venue_residual"):
            assert result["features"][noise_feat]["passed"] is False, (
                f"{noise_feat} is pure noise here and should not pass ablation"
            )
