"""Saved player-model inference and its real projection/blending call site."""

import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from player_regression import predict_player_stat


@pytest.fixture
def fitted_model():
    """A real fitted estimator, serialized using the training artifact schema."""
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        "ewma_PTS": rng.uniform(5, 30, 200),
        "is_home": rng.integers(0, 2, 200),
        "rest_days": rng.integers(1, 8, 200),
    })
    target = 4 + 0.8 * frame.ewma_PTS + 1.5 * frame.is_home - 0.6 * frame.rest_days
    scaler = StandardScaler().fit(frame.values)
    model = Ridge(alpha=5.0).fit(scaler.transform(frame.values), target)
    params = {
        "feature_names": list(frame.columns),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "intercept": float(model.intercept_),
        "ewma_weight": float(model.coef_[0]),
        "home_boost": float(model.coef_[1]),
        "rest_coef": float(model.coef_[2]),
    }
    return json.loads(json.dumps(params)), scaler, model


@pytest.mark.parametrize("raw", [[20, 1, 1], [8, 0, 7], [30, 1, 4]])
def test_saved_prediction_matches_fitted_sklearn_estimator(fitted_model, raw):
    params, scaler, model = fitted_model
    expected = model.predict(scaler.transform([raw]))[0]
    actual = predict_player_stat(params, "PTS", raw[0], bool(raw[1]), raw[2])
    assert actual == pytest.approx(expected)


def test_absent_rest_is_neutral_in_standardized_space(fitted_model):
    params, scaler, model = fitted_model
    expected = model.predict(scaler.transform([[20, 1, scaler.mean_[2]]]))[0]
    assert predict_player_stat(params, "PTS", 20, True) == pytest.approx(expected)


def test_artifact_feature_order_is_respected(fitted_model):
    params, scaler, model = fitted_model
    permutation = [2, 0, 1]
    for key in ("feature_names", "scaler_mean", "scaler_scale"):
        params[key] = [params[key][i] for i in permutation]
    expected = model.predict(scaler.transform([[20, 0, 2]]))[0]
    assert predict_player_stat(params, "PTS", 20, False, 2) == pytest.approx(expected)


@pytest.mark.parametrize("change", [
    {"scaler_mean": [0, 0]},
    {"scaler_scale": [1, 0, 1]},
    {"scaler_scale": [1, float("nan"), 1]},
    {"feature_names": ["ewma_PTS", "is_home", "unknown_feature"]},
    {"ewma_weight": float("inf")},
])
def test_invalid_artifacts_are_rejected(fitted_model, change):
    params, _, _ = fitted_model
    params.update(change)
    with pytest.raises(ValueError):
        predict_player_stat(params, "PTS", 20, True)


@pytest.fixture
def neutral_projection(monkeypatch):
    """Exercise real project_stat with a 20-point base and external factors neutral."""
    # Import the real module without replacing its functions with copied test code.
    with mock.patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")):
        import playerlinepredictor as predictor

    returns = {
        "_per_min_rate": (1.0, 20.0),
        "_blend_rate": 1.0,
        "_blend_minutes": 20.0,
        "detect_injury_return": 100,
        "injury_rust_discount": 0.0,
        "get_series_h2h": 1.0,
        "get_series_def_factor": 1.0,
        "_ha_factor": 1.0,
        "_player_season_avg": 20.0,
        "get_team_stat_avg": 100.0,
        "compute_usage_boost": 1.0,
    }
    for name, value in returns.items():
        monkeypatch.setattr(predictor, name, mock.Mock(return_value=value))
    monkeypatch.setattr(predictor.dvp_engine, "get_dvp_factor", mock.Mock(return_value=1.0))
    for name in ("MARKET_PROJ_ADJ", "MANUAL_PLAYER_PENALTIES", "PLAYER_PROJ_ADJ"):
        monkeypatch.setattr(predictor, name, {})
    monkeypatch.setattr(predictor, "USE_KALMAN_PLAYER_MODEL", False)

    def run(params):
        monkeypatch.setattr(predictor, "_LEARNED_PLAYER_PARAMS", {"PTS": params})
        with mock.patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")):
            return predictor.project_stat(
                "Probe", "PTS", "BOS", 2, [], "PG", pd.DataFrame(),
                pd.DataFrame({"PTS": [20], "MIN": [20]}), True, "NYK", False, 0,
            )

    return run


def test_project_stat_blends_the_fitted_prediction(neutral_projection, fitted_model):
    params, scaler, model = fitted_model
    expected_ridge = model.predict(scaler.transform([[20, 1, scaler.mean_[2]]]))[0]
    expected = 0.85 * 20 + 0.15 * expected_ridge
    projection, details = neutral_projection(params)
    assert projection == pytest.approx(expected)
    assert details["learned_adj"] == pytest.approx(round(expected - 20, 2))


def test_reviewed_20_point_projection_does_not_inflate(neutral_projection):
    # PTS coefficients from the artifact inspected during the review; no local
    # learned_params.json dependency. Old raw-input arithmetic yielded 36.445.
    params = {
        "feature_names": ["ewma_PTS", "is_home", "rest_days"],
        "scaler_mean": [11.611214355431411, 0.5005596992941898, 2.425430379299374],
        "scaler_scale": [6.743661021001347, 0.499999686736602, 1.4146085917963227],
        "intercept": 12.321302744294012,
        "ewma_weight": 5.8625807203876,
        "home_boost": 0.06122605576363455,
        "rest_coef": -0.21412808741744604,
    }
    projection, _ = neutral_projection(params)
    assert projection == pytest.approx(19.95, abs=0.005)


def test_bad_artifact_preserves_base_projection(neutral_projection, fitted_model, caplog):
    params, _, _ = fitted_model
    del params["scaler_mean"]
    projection, details = neutral_projection(params)
    assert projection == 20.0
    assert details["learned_adj"] == 0.0
    assert "Skipping learned player projection" in caplog.text


def test_absent_model_preserves_base_projection(neutral_projection):
    projection, details = neutral_projection({})
    assert projection == 20.0
    assert details["learned_adj"] == 0.0
