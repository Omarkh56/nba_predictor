"""Training artifact handoff, approval decisions, and real game-model inference."""

import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def predictor(monkeypatch, tmp_path):
    with mock.patch(
        "requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")
    ):
        import newnbapredictor

        # Redirect only this module's artifact lookup; never modify the user's model.
        monkeypatch.setattr(newnbapredictor, "__file__", str(tmp_path / "newnbapredictor.py"))
        monkeypatch.setattr(newnbapredictor, "_LEARNED_GAME_PARAMS", {})
        monkeypatch.setattr(newnbapredictor, "USE_LEARNED_GAME_MODEL", False)
        monkeypatch.setattr(newnbapredictor, "_VENUE_RESIDUALS", {})
        yield newnbapredictor


def _save_model(tmp_path, model, **sections):
    path = tmp_path / "learned_params.json"
    path.write_text(json.dumps({"game_model": model, **sections}))
    return path


@pytest.fixture
def model():
    return {
        "recommendation": "swap",
        "feature_version": "season-history-v1",
        "feature_names": ["net_rtg_diff", "home_b2b"],
        "scaler_mean": [2.0, 0.2],
        "scaler_scale": [4.0, 0.4],
        "log_coefficients": {"net_rtg_diff": 1.5, "home_b2b": -0.3},
        "log_intercept": 0.1,
    }


@pytest.mark.parametrize("recommendation,active", [
    ("swap", True), ("use_learned", True), ("monitor", False),
    ("keep_hand_tuned", False), ("unknown", False), (None, False),
])
def test_loader_applies_the_saved_recommendation(predictor, tmp_path, model,
                                               recommendation, active):
    model["recommendation"] = recommendation
    _save_model(tmp_path, model)
    predictor._load_learned_params()
    assert predictor.USE_LEARNED_GAME_MODEL is active
    probability = predictor._sigmoid_learned({"net_rtg_diff": 6.0, "home_b2b": 0})
    assert (probability is not None) is active


def test_missing_recommendation_keeps_hand_tuned(predictor, tmp_path, model):
    del model["recommendation"]
    _save_model(tmp_path, model)
    predictor._load_learned_params()
    assert predictor.USE_LEARNED_GAME_MODEL is False


@pytest.mark.parametrize("version", [None, "old-inputs"])
def test_old_input_definition_requires_retraining(predictor, tmp_path, model, version, caplog):
    model["feature_version"] = version
    _save_model(tmp_path, model)
    predictor._load_learned_params()
    assert predictor.USE_LEARNED_GAME_MODEL is False
    assert "rerun train_model.py" in caplog.text


@pytest.mark.parametrize("replacement", [None, "{broken json", "[]", '{"game_model": []}'])
def test_reload_clears_stale_approval_and_parameters(predictor, tmp_path, model, replacement):
    path = _save_model(tmp_path, model, venue_residuals={"1": 0.08})
    predictor._load_learned_params()
    assert predictor.USE_LEARNED_GAME_MODEL is True
    if replacement is None:
        path.unlink()
    else:
        path.write_text(replacement)
    predictor._load_learned_params()
    assert predictor.USE_LEARNED_GAME_MODEL is False
    assert predictor._LEARNED_GAME_PARAMS == {}
    assert predictor._VENUE_RESIDUALS == {}


def test_missing_live_feature_falls_back_instead_of_using_zero(
    predictor, tmp_path, model, caplog
):
    _save_model(tmp_path, model)
    predictor._load_learned_params()
    assert predictor._sigmoid_learned({"net_rtg_diff": 6.0}) is None
    assert "missing live inputs: home_b2b" in caplog.text
    assert predictor._sigmoid_learned({"net_rtg_diff": 6.0, "home_b2b": 0.0}) is not None


@pytest.mark.parametrize("change", [
    {"scaler_mean": [2.0]}, {"scaler_scale": [4.0]},
    {"scaler_scale": []},
    {"scaler_scale": [4.0, 0.0]}, {"scaler_scale": [4.0, float("nan")]},
    {"log_coefficients": {"net_rtg_diff": 1.5}},
    {"log_intercept": float("inf")}, {"log_intercept": None},
    {"feature_names": ["net_rtg_diff", "net_rtg_diff"]},
])
def test_invalid_saved_parameters_select_the_fallback(
    predictor, tmp_path, model, change, caplog
):
    model.update(change)
    _save_model(tmp_path, model)
    predictor._load_learned_params()
    assert predictor._sigmoid_learned({"net_rtg_diff": 6.0, "home_b2b": 0.0}) is None
    assert "Using hand-tuned game probability" in caplog.text


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None])
def test_invalid_live_values_select_the_fallback(predictor, tmp_path, model, value):
    _save_model(tmp_path, model)
    predictor._load_learned_params()
    assert predictor._sigmoid_learned({"net_rtg_diff": value, "home_b2b": 0.0}) is None


@pytest.mark.parametrize("intercept,expected", [(-1000.0, 0.0), (1000.0, 1.0)])
def test_extreme_logits_do_not_overflow(predictor, tmp_path, model, intercept, expected):
    model["log_intercept"] = intercept
    _save_model(tmp_path, model)
    predictor._load_learned_params()
    assert predictor._sigmoid_learned({"net_rtg_diff": 2.0, "home_b2b": 0.2}) == expected


def test_real_training_artifact_activates_and_matches_the_fitted_estimator(
    predictor, tmp_path, monkeypatch
):
    # Training currently patches requests.get at import, so restore it after the
    # import. Fit/save below exercise the actual trainer without making requests.
    with mock.patch("requests.get", side_effect=AssertionError("Network forbidden")):
        import train_model as trainer

    rng = np.random.default_rng(42)

    def frame(count):
        data = pd.DataFrame({feature: np.zeros(count) for feature in trainer.GAME_FEATURES})
        signal = rng.normal(0.0, 3.0, count)
        data["net_rtg_diff"] = signal
        data["home_win"] = (signal > 0).astype(int)
        data["actual_margin"] = 3.0 * signal
        return data

    train, validation = frame(200), frame(100)
    fitted = trainer.fit_game_models(train, validation)
    assert fitted["recommendation"] == "swap"
    monkeypatch.setattr(trainer, "PARAMS_FILE", tmp_path / "learned_params.json")
    trainer.save_params(fitted, {"stat_models": {}, "variance": {}}, {},
                        ["2022-23", trainer.VAL_SEASON], no_bootstrap=True)

    predictor._load_learned_params()
    assert predictor.USE_LEARNED_GAME_MODEL is True
    for _, row in validation.iloc[:10].iterrows():
        features = row[trainer.GAME_FEATURES].to_dict()
        raw = np.array([[features[feature] for feature in fitted["feature_names"]]])
        expected = fitted["log_model"].predict_proba(fitted["scaler"].transform(raw))[0, 1]
        assert predictor._sigmoid_learned(features) == pytest.approx(expected)


@pytest.fixture
def predict_game(predictor, monkeypatch):
    """Use the real prediction engine with its external data sources held constant."""
    form = {"l10_net_rtg": 0.0, "days_rest": 3, "b2b": False, "consec_road": 0, "w_pct": 0.5}
    monkeypatch.setattr(predictor, "_fetch_last_n", mock.Mock(return_value=form))
    monkeypatch.setattr(predictor, "_fetch_h2h", mock.Mock(return_value={
        "avg_margin": 0.0, "shrinkage": 0.0, "total": 0, "team1_wins": 0,
    }))
    monkeypatch.setattr(predictor.time, "sleep", mock.Mock())
    for function in ("compute_injury_adj", "compute_four_factors_edge", "compute_matchup_edge",
                     "compute_clutch_adj", "compute_altitude_adj", "compute_star_form_adj"):
        monkeypatch.setattr(predictor, function, mock.Mock(return_value=(0.0, {})))
    monkeypatch.setattr(predictor, "_compute_hca", mock.Mock(return_value=2.8))
    monkeypatch.setattr(predictor, "USE_KALMAN_GAME_MODEL", False)
    monkeypatch.setattr(predictor, "_fetch_learned_features", mock.Mock(return_value={
        "net_rtg_diff": 6.0, "home_b2b": 0.0,
    }))
    monkeypatch.setattr(predictor, "_KALMAN_EKF", None)
    ratings = pd.DataFrame([
        {"TEAM_NAME": "Boston Celtics", "NET_RATING": 0.0},
        {"TEAM_NAME": "New York Knicks", "NET_RATING": 6.0},
    ], index=[2, 1])
    matchup = {
        "home_name": "New York Knicks", "away_name": "Boston Celtics",
        "home_abbr": "NYK", "away_abbr": "BOS", "status": "scheduled",
        "game_date": "2026-09-18",
    }

    def run():
        return predictor._compute_prediction(matchup, ratings, "2026-27", {}, {})

    return run


@pytest.mark.parametrize("recommendation", ["swap", "monitor"])
def test_prediction_engine_uses_the_approved_model_or_hand_tuned_fallback(
    predictor, tmp_path, model, predict_game, recommendation
):
    model["recommendation"] = recommendation
    _save_model(tmp_path, model)
    predictor._load_learned_params()
    result = predict_game()
    if recommendation == "swap":
        # logit = 0.1 + 1.5*((6-2)/4) - 0.3*((0-0.2)/0.4) = 1.75.
        expected = 1.0 / (1.0 + np.exp(-1.75))
    else:
        expected = predictor._sigmoid(result["expected_margin"])
    assert result["home_prob"] == pytest.approx(expected)


def test_prediction_engine_falls_back_for_extended_model_missing_live_inputs(
    predictor, tmp_path, model, predict_game, caplog
):
    model["feature_names"].append("star_form_diff")
    model["scaler_mean"].append(0.0)
    model["scaler_scale"].append(1.0)
    model["log_coefficients"]["star_form_diff"] = 2.0
    _save_model(tmp_path, model)
    predictor._load_learned_params()
    result = predict_game()
    assert result["home_prob"] == predictor._sigmoid(result["expected_margin"])
    assert "missing live inputs: star_form_diff" in caplog.text


def test_prediction_engine_preserves_kalman_priority(
    predictor, tmp_path, model, predict_game, monkeypatch
):
    _save_model(tmp_path, model)
    predictor._load_learned_params()
    monkeypatch.setattr(predictor, "USE_KALMAN_GAME_MODEL", True)
    ekf = mock.Mock(feature_names=["net_rtg_diff"])
    ekf.predict_proba.return_value = 0.73
    monkeypatch.setattr(predictor, "_KALMAN_EKF", ekf)
    with mock.patch.object(predictor, "_sigmoid_learned", side_effect=AssertionError("Static path used")):
        result = predict_game()
    assert result["home_prob"] == 0.73
