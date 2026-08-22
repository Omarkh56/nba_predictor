"""
tests/test_kalman.py — Unit tests for kalman_filter.py.

Tests cover:
  - EKF filter math (correct direction, P properties, Kalman gain)
  - Standard KF player model (linear case, convergence)
  - State serialisation round-trip (save → load → identical prediction)
  - Named-constant values are within expected ranges
  - Q magnitude controls adaptation speed
"""

import importlib.util
import json
import math
import types
from pathlib import Path

import numpy as np
import pytest

# ── Load kalman_filter without network deps ─────────────────────────────────────
# kalman_filter.py imports nothing from nba_api, so a direct import is safe.

import sys
_proj = Path(__file__).parent.parent
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

from kalman_filter import (
    GameEKF,
    PlayerKF,
    P_GAME_INIT_DIAG,
    Q_GAME_SCALAR,
    P_PLAYER_INIT_DIAG,
    Q_PLAYER_SCALAR,
    _V_MIN,
    save_tvp_state,
    load_game_ekf,
    load_player_kf,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────────

N_FEATURES = 10
FEATURE_NAMES = [
    "net_rtg_diff", "efg_diff", "tov_diff", "orb_diff", "ftr_diff",
    "form_diff", "home_b2b", "away_b2b", "rest_diff", "h2h_margin",
]


def _make_ekf(q: float = Q_GAME_SCALAR, intercept: float = 0.0) -> GameEKF:
    """Return a GameEKF with zero coefficients and identity-P prior."""
    beta           = np.zeros(N_FEATURES + 1)
    beta[0]        = intercept
    P              = np.eye(N_FEATURES + 1) * P_GAME_INIT_DIAG
    scaler_mean    = np.zeros(N_FEATURES)
    scaler_scale   = np.ones(N_FEATURES)
    return GameEKF(
        beta          = beta,
        P             = P,
        feature_names = FEATURE_NAMES,
        scaler_mean   = scaler_mean,
        scaler_scale  = scaler_scale,
        n_updates     = 0,
        q_scalar      = q,
    )


def _make_player_kf(obs_var: float = 25.0) -> PlayerKF:
    """Return a PlayerKF with a single 'PTS' stat, zero weights, identity P."""
    stat_states = {
        "PTS": {
            "beta":         np.array([5.0, 1.0, 0.5, 0.2]),   # intercept + 3 features
            "P":            np.eye(4) * P_PLAYER_INIT_DIAG,
            "obs_var":      obs_var,
            "feat_names":   ["ewma_PTS", "is_home", "rest_days"],
            "scaler_mean":  np.array([15.0, 0.5, 3.0]),
            "scaler_scale": np.array([5.0,  0.5, 1.5]),
            "n_updates":    0,
        }
    }
    return PlayerKF(stat_states=stat_states, q_scalar=Q_PLAYER_SCALAR)


# ── Named constants ───────────────────────────────────────────────────────────────

def test_p_game_init_range():
    """P_GAME_INIT_DIAG should be strictly positive and <= 1.0."""
    assert 0 < P_GAME_INIT_DIAG <= 1.0


def test_q_game_scalar_range():
    """Q_GAME_SCALAR should be small but not negligibly so."""
    assert 1e-6 < Q_GAME_SCALAR < 1e-1


def test_p_player_init_range():
    assert 0 < P_PLAYER_INIT_DIAG <= 5.0


def test_q_player_scalar_range():
    assert 1e-7 < Q_PLAYER_SCALAR < 1e-2


def test_v_min_is_positive():
    assert _V_MIN > 0


# ── GameEKF: predict_step ─────────────────────────────────────────────────────────

def test_predict_step_increases_P_trace():
    """predict_step should strictly increase trace(P) by q_scalar * (d+1)."""
    ekf    = _make_ekf()
    tr_pre = np.trace(ekf.P)
    ekf.predict_step()
    tr_post = np.trace(ekf.P)
    assert pytest.approx(tr_post - tr_pre) == Q_GAME_SCALAR * (N_FEATURES + 1)


def test_predict_step_preserves_symmetry():
    ekf = _make_ekf()
    ekf.predict_step()
    assert np.allclose(ekf.P, ekf.P.T)


# ── GameEKF: update ───────────────────────────────────────────────────────────────

def test_update_beta_shifts_toward_observation_home_win():
    """If home wins (y=1) and p < 0.5, intercept should increase after update."""
    ekf = _make_ekf(intercept=-0.5)   # p < 0.5
    ekf.predict_step()
    x   = np.zeros(N_FEATURES)
    p0  = ekf.predict_proba(x)
    assert p0 < 0.5, "Precondition: model should favour away win"
    ekf.update(x, y=1)
    p1  = ekf.predict_proba(x)
    assert p1 > p0, "Win observation should push p upward"


def test_update_beta_shifts_toward_observation_away_win():
    """If away wins (y=0) and p > 0.5, intercept should decrease after update."""
    ekf = _make_ekf(intercept=0.5)
    ekf.predict_step()
    x  = np.zeros(N_FEATURES)
    p0 = ekf.predict_proba(x)
    ekf.update(x, y=0)
    p1 = ekf.predict_proba(x)
    # One KF step reduces p; it may not cross 0.5 in a single update (small gain
    # with P_GAME_INIT_DIAG=0.25) but must move in the right direction.
    assert p1 < p0, "Loss observation should push p downward"


def test_update_returns_pre_update_probability():
    """update() should return p BEFORE the state is changed."""
    ekf = _make_ekf(intercept=1.0)
    ekf.predict_step()
    x       = np.zeros(N_FEATURES)
    p_pre   = ekf.predict_proba(x)
    p_ret   = ekf.update(x, y=1)
    assert pytest.approx(p_ret, abs=1e-8) == p_pre


def test_update_increments_n_updates():
    ekf = _make_ekf()
    ekf.predict_step()
    ekf.update(np.zeros(N_FEATURES), y=1)
    assert ekf.n_updates == 1


def test_update_P_is_symmetric():
    ekf = _make_ekf()
    ekf.predict_step()
    ekf.update(np.random.default_rng(0).standard_normal(N_FEATURES), y=1)
    assert np.allclose(ekf.P, ekf.P.T, atol=1e-10)


def test_update_P_diagonal_non_negative():
    """Diagonal of P should stay non-negative after update."""
    ekf = _make_ekf()
    for _ in range(20):
        ekf.predict_step()
        x = np.random.default_rng(42).standard_normal(N_FEATURES)
        ekf.update(x, y=int(np.random.default_rng(42).integers(0, 2)))
    assert np.all(np.diag(ekf.P) >= -1e-12)


def test_correct_prediction_small_beta_shift():
    """
    When the prediction is already exactly right (y matches round(p)),
    K*(y-p) should be much smaller than when the prediction is wrong.
    """
    ekf_right = _make_ekf(intercept=2.0)   # strong home-win prediction
    ekf_wrong = _make_ekf(intercept=-2.0)  # wrong (predicts away)
    x = np.zeros(N_FEATURES)

    ekf_right.predict_step()
    beta_before_r = ekf_right.beta.copy()
    ekf_right.update(x, y=1)   # correct
    shift_right = np.linalg.norm(ekf_right.beta - beta_before_r)

    ekf_wrong.predict_step()
    beta_before_w = ekf_wrong.beta.copy()
    ekf_wrong.update(x, y=1)   # wrong
    shift_wrong = np.linalg.norm(ekf_wrong.beta - beta_before_w)

    assert shift_right < shift_wrong, (
        "Correct prediction should produce smaller β shift than wrong prediction"
    )


def test_larger_q_means_larger_gain():
    """With a larger Q, the Kalman gain K should be larger (filter reacts faster)."""
    q_small = 1e-6
    q_large = 1e-2
    ekf_s   = _make_ekf(q=q_small)
    ekf_l   = _make_ekf(q=q_large)
    x       = np.zeros(N_FEATURES)

    ekf_s.predict_step()
    b_s_before = ekf_s.beta.copy()
    ekf_s.update(x, y=1)
    shift_s = np.linalg.norm(ekf_s.beta - b_s_before)

    ekf_l.predict_step()
    b_l_before = ekf_l.beta.copy()
    ekf_l.update(x, y=1)
    shift_l = np.linalg.norm(ekf_l.beta - b_l_before)

    assert shift_l > shift_s, "Larger Q → larger gain → bigger β shift"


# ── GameEKF: from_dict / to_dict round-trip ───────────────────────────────────────

def test_ekf_to_from_dict_round_trip():
    ekf = _make_ekf(intercept=0.7)
    ekf.predict_step()
    ekf.update(np.ones(N_FEATURES) * 0.3, y=1)
    d      = ekf.to_dict()
    ekf2   = GameEKF.from_dict(d)
    x      = np.random.default_rng(7).standard_normal(N_FEATURES)
    assert pytest.approx(ekf.predict_proba(x), abs=1e-10) == ekf2.predict_proba(x)


def test_ekf_to_from_dict_preserves_n_updates():
    ekf = _make_ekf()
    for _ in range(5):
        ekf.predict_step()
        ekf.update(np.zeros(N_FEATURES), y=0)
    ekf2 = GameEKF.from_dict(ekf.to_dict())
    assert ekf2.n_updates == 5


# ── GameEKF: save / load via tvp_state.json ──────────────────────────────────────

def test_save_load_tvp_state(tmp_path):
    ekf = _make_ekf(intercept=0.4)
    ekf.predict_step()
    ekf.update(np.zeros(N_FEATURES), y=1)
    pkf = _make_player_kf()

    path = tmp_path / "tvp_state.json"
    save_tvp_state(ekf, pkf, last_updated="2026-08-22", path=path)

    ekf2 = load_game_ekf(path)
    assert ekf2 is not None
    x    = np.random.default_rng(3).standard_normal(N_FEATURES)
    assert pytest.approx(ekf.predict_proba(x), abs=1e-10) == ekf2.predict_proba(x)


def test_load_game_ekf_returns_none_if_missing(tmp_path):
    assert load_game_ekf(tmp_path / "nonexistent.json") is None


def test_load_player_kf_returns_none_if_missing(tmp_path):
    assert load_player_kf(tmp_path / "nonexistent.json") is None


# ── PlayerKF ─────────────────────────────────────────────────────────────────────

def test_player_kf_predict_step_increases_P():
    pkf    = _make_player_kf()
    tr_pre = np.trace(pkf._states["PTS"]["P"])
    pkf.predict_step("PTS")
    tr_post = np.trace(pkf._states["PTS"]["P"])
    assert tr_post > tr_pre


def test_player_kf_update_shifts_beta_toward_observation():
    """If actual > predicted, intercept should increase after update."""
    pkf    = _make_player_kf()
    x_raw  = np.array([15.0, 1.0, 3.0])          # ewma_PTS, is_home, rest_days
    y_pred = pkf.predict("PTS", x_raw)
    y_high = y_pred + 10.0                         # much higher than predicted
    pkf.predict_step("PTS")
    pkf.update("PTS", x_raw, y_high)
    y_pred2 = pkf.predict("PTS", x_raw)
    assert y_pred2 > y_pred


def test_player_kf_returns_pre_update_pred():
    pkf    = _make_player_kf()
    x_raw  = np.array([15.0, 1.0, 3.0])
    y_pre  = pkf.predict("PTS", x_raw)
    pkf.predict_step("PTS")
    ret    = pkf.update("PTS", x_raw, y=y_pre + 5.0)
    assert pytest.approx(ret, abs=1e-8) == y_pre


def test_player_kf_linear_convergence():
    """
    With a linear model and known true β, KF should converge toward it.
    Use zero observation noise to make convergence obvious.
    """
    true_beta   = np.array([10.0, 2.0, 1.5, 0.5])   # intercept + 3 features
    pkf         = _make_player_kf(obs_var=1e-6)       # near-zero noise
    # Manually override beta to be far from truth
    pkf._states["PTS"]["beta"] = np.zeros(4)

    rng     = np.random.default_rng(99)
    scaler_mean  = np.array([15.0, 0.5, 3.0])
    scaler_scale = np.array([5.0,  0.5, 1.5])

    for _ in range(200):
        x_raw   = rng.standard_normal(3)
        x_scaled = (x_raw - scaler_mean) / scaler_scale
        x_aug   = np.concatenate([[1.0], x_scaled])
        y       = float(true_beta @ x_aug)             # noiseless observation
        pkf.predict_step("PTS")
        pkf.update("PTS", x_raw, y)

    err = np.linalg.norm(pkf._states["PTS"]["beta"] - true_beta)
    assert err < 0.5, f"KF should converge close to true β; error = {err:.3f}"


def test_player_kf_P_symmetric():
    pkf = _make_player_kf()
    for i in range(10):
        x = np.array([15.0 + i, float(i % 2), 3.0])
        pkf.predict_step("PTS")
        pkf.update("PTS", x, y=float(15 + i))
    assert np.allclose(pkf._states["PTS"]["P"], pkf._states["PTS"]["P"].T, atol=1e-10)


def test_player_kf_to_dict_round_trip():
    pkf  = _make_player_kf()
    pkf.predict_step("PTS")
    pkf.update("PTS", np.array([20.0, 1.0, 2.0]), y=22.0)
    d    = pkf.to_dict()
    pkf2 = PlayerKF(stat_states=d, q_scalar=pkf.q_scalar)
    x    = np.array([18.0, 0.0, 4.0])
    assert pytest.approx(pkf.predict("PTS", x), abs=1e-8) == pkf2.predict("PTS", x)
