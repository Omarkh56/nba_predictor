"""
kalman_filter.py — Adaptive time-varying parameter models for NBA predictions.

Game model:  Extended Kalman Filter (EKF) on a logistic regression state β.
Player model: Standard Kalman Filter on linear Ridge regression states.

Both models are initialized from learned_params.json (the batch-fit from
train_model.py), so the prior is informed rather than zero.

State persistence: tvp_state.json  (β, P, metadata per model).

Named constants — rationale documented in-line:
  P_GAME_INIT_DIAG   Initial P diagonal: ≈ 95% CI width of LR coefficients
                     on ~1 000 training games (standardized-feature space).
  Q_GAME_SCALAR      Per-game random-walk drift set by kalman_validate.py
                     grid search; default is conservative 1e-4.
  P_PLAYER_INIT_DIAG Wider than game model — player stats are more volatile.
  Q_PLAYER_SCALAR    Slow drift; player tendencies change over weeks, not days.

EKF derivation (game model):
  State:       β ∈ ℝ^{d+1}  (intercept at β[0], standardized-feature weights β[1:])
  Process:     β_t = β_{t-1} + w_t,    w_t ~ N(0, Q)
  Observation: y_t ~ Bernoulli(σ(x̃_t · β_t)),   x̃ = [1, x_scaled]
  Linearise:   H_t = p_t(1-p_t) x̃_t  (Jacobian of σ(·) at current estimate)
               R_t = p_t(1-p_t)       (Bernoulli variance — observation noise)
  Kalman gain: K_t = P_t|t-1 x̃_t / (v_t · x̃_t^T P_t|t-1 x̃_t + 1)
               where v_t = p_t(1-p_t)
  Updates:     β_t = β_{t-1} + K_t (y_t - p_t)
               P_t = P_{t-1} - v_t · outer(K_t, P_{t-1} x̃_t)
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from game_features import GAME_FEATURE_VERSION

_HERE          = Path(__file__).parent
PARAMS_FILE    = _HERE / "learned_params.json"
TVP_STATE_FILE = _HERE / "tvp_state.json"

# ── Named constants ─────────────────────────────────────────────────────────────

# Initial state uncertainty (diagonal of P₀) in standardized-feature space.
# Logistic regression on ~1 000 games gives typical SEs of 0.05–0.15 per
# standardized coefficient; 0.25 is intentionally wider to allow early
# adaptation without anchoring too hard to the batch estimate.
P_GAME_INIT_DIAG: float = 0.25

# Per-game process noise (Q diagonal scalar).
# Tuned on the 2024-25 validation season by kalman_validate.py; the default
# is a conservative starting point meaning each coefficient drifts by at most
# ~0.01 per game on average (sqrt(Q) ≈ 0.01).
Q_GAME_SCALAR: float = 1e-4

# Player model: wider initial uncertainty (player stats are more variable than
# binary win/loss outcomes; smaller signal-to-noise ratio).
P_PLAYER_INIT_DIAG: float = 1.0

# Player process noise: slower drift than the game model because individual
# player tendencies evolve over weeks (injury recovery, role changes), not day-
# to-day.
Q_PLAYER_SCALAR: float = 5e-5

# Minimum local Bernoulli variance to prevent near-zero denominators when
# predictions approach 0 or 1.
_V_MIN: float = 1e-3


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))


# ── Game model: Extended Kalman Filter ───────────────────────────────────────────

class GameEKF:
    """EKF for a time-varying logistic-regression game model."""

    def __init__(
        self,
        beta: np.ndarray,
        P: np.ndarray,
        feature_names: list,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        n_updates: int = 0,
        q_scalar: float = Q_GAME_SCALAR,
        feature_version: Optional[str] = None,
    ) -> None:
        self.beta          = beta.copy()
        self.P             = P.copy()
        self.feature_names = feature_names
        self.scaler_mean   = scaler_mean
        self.scaler_scale  = scaler_scale
        self.n_updates     = n_updates
        self.q_scalar      = q_scalar
        self.feature_version = feature_version
        self._d            = len(feature_names)

    # ── Internal ────────────────────────────────────────────────────────────────

    def _augment(self, x_raw: np.ndarray) -> np.ndarray:
        """Standardize raw features and prepend the intercept column (1.0)."""
        x_scaled = (x_raw - self.scaler_mean) / self.scaler_scale
        return np.concatenate([[1.0], x_scaled])

    # ── Core EKF steps ─────────────────────────────────────────────────────────

    def predict_step(self) -> None:
        """Time update: add Q to P (models random-walk drift of β between games)."""
        self.P = self.P + np.eye(self._d + 1) * self.q_scalar

    def update(self, x_raw: np.ndarray, y: int) -> float:
        """
        Measurement update.
        x_raw: raw feature vector (same order as GAME_FEATURES).
        y:     observed outcome — 1 = home win, 0 = away win.
        Returns the pre-update win probability (use this for scoring, not the
        post-update value, to avoid peeking at the outcome).
        """
        x_aug = self._augment(x_raw)
        logit = float(x_aug @ self.beta)
        p     = _sigmoid(logit)
        v     = max(p * (1.0 - p), _V_MIN)

        Px    = self.P @ x_aug
        denom = v * float(x_aug @ Px) + 1.0
        K     = Px / denom

        self.beta      = self.beta + K * (y - p)
        self.P         = self.P - v * np.outer(K, Px)
        self.n_updates += 1
        return p

    def predict_proba(self, x_raw: np.ndarray) -> float:
        """Win probability without updating state."""
        x_aug = self._augment(x_raw)
        return _sigmoid(float(x_aug @ self.beta))

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "beta":          self.beta.tolist(),
            "P":             self.P.tolist(),
            "feature_names": self.feature_names,
            "scaler_mean":   self.scaler_mean.tolist(),
            "scaler_scale":  self.scaler_scale.tolist(),
            "n_updates":     self.n_updates,
            "q_scalar":      self.q_scalar,
            "feature_version": self.feature_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GameEKF":
        return cls(
            beta          = np.array(d["beta"]),
            P             = np.array(d["P"]),
            feature_names = d["feature_names"],
            scaler_mean   = np.array(d["scaler_mean"]),
            scaler_scale  = np.array(d["scaler_scale"]),
            n_updates     = d.get("n_updates", 0),
            q_scalar      = d.get("q_scalar", Q_GAME_SCALAR),
            feature_version = d.get("feature_version"),
        )

    @classmethod
    def from_learned_params(
        cls,
        params_path: Path = PARAMS_FILE,
        q_scalar: float = Q_GAME_SCALAR,
    ) -> "GameEKF":
        """Initialise EKF from the batch LogReg fit in learned_params.json."""
        with open(params_path) as fh:
            p = json.load(fh)
        gm = p["game_model"]
        if gm.get("feature_version") != GAME_FEATURE_VERSION:
            raise ValueError("Game input definitions changed; rerun train_model.py first")

        feat_names   = gm["feature_names"]
        d            = len(feat_names)
        intercept    = gm["log_intercept"]
        coef_dict    = gm["log_coefficients"]
        scaler_mean  = np.array(gm["scaler_mean"])
        scaler_scale = np.array(gm["scaler_scale"])

        beta    = np.zeros(d + 1)
        beta[0] = intercept
        for i, f in enumerate(feat_names):
            beta[i + 1] = coef_dict[f]

        P = np.eye(d + 1) * P_GAME_INIT_DIAG

        return cls(
            beta          = beta,
            P             = P,
            feature_names = feat_names,
            scaler_mean   = scaler_mean,
            scaler_scale  = scaler_scale,
            n_updates     = 0,
            q_scalar      = q_scalar,
            feature_version = gm.get("feature_version"),
        )


# ── Player model: Standard Kalman Filter ─────────────────────────────────────────

class PlayerKF:
    """
    Linear Kalman Filter for time-varying Ridge regression player-stat models.

    State per stat: β ∈ ℝ^{k+1} (intercept + standardized-feature weights).
    Observation noise R is the empirical residual variance from the batch fit.
    """

    def __init__(
        self,
        stat_states: dict,
        q_scalar: float = Q_PLAYER_SCALAR,
    ) -> None:
        self.q_scalar = q_scalar
        self._states: dict = {}
        for stat, s in stat_states.items():
            self._states[stat] = {
                "beta":         np.array(s["beta"]),
                "P":            np.array(s["P"]),
                "obs_var":      float(s["obs_var"]),
                "feat_names":   s["feat_names"],
                "scaler_mean":  np.array(s["scaler_mean"]),
                "scaler_scale": np.array(s["scaler_scale"]),
                "n_updates":    int(s.get("n_updates", 0)),
            }

    def stats(self) -> list:
        return list(self._states.keys())

    def _augment(self, stat: str, x_raw: np.ndarray) -> np.ndarray:
        s = self._states[stat]
        x_scaled = (x_raw - s["scaler_mean"]) / s["scaler_scale"]
        return np.concatenate([[1.0], x_scaled])

    def predict_step(self, stat: str) -> None:
        """Time update: add Q to P for the given stat."""
        s = self._states[stat]
        k = len(s["beta"])
        s["P"] = s["P"] + np.eye(k) * self.q_scalar

    def update(self, stat: str, x_raw: np.ndarray, y: float) -> float:
        """
        Standard KF measurement update for a player stat observation.
        x_raw: raw feature vector (e.g. [ewma_stat, is_home, rest_days]).
        y:     actual stat value observed.
        Returns pre-update prediction (for scoring).
        """
        s      = self._states[stat]
        x_aug  = self._augment(stat, x_raw)
        y_pred = float(x_aug @ s["beta"])
        R      = s["obs_var"]

        Px    = s["P"] @ x_aug
        denom = float(x_aug @ Px) + R
        K     = Px / denom

        s["beta"]      = s["beta"] + K * (y - y_pred)
        s["P"]         = s["P"] - np.outer(K, Px)
        s["n_updates"] += 1
        return y_pred

    def predict(self, stat: str, x_raw: np.ndarray) -> float:
        """Predict a stat without updating state."""
        s     = self._states[stat]
        x_aug = self._augment(stat, x_raw)
        return float(x_aug @ s["beta"])

    def to_dict(self) -> dict:
        out: dict = {}
        for stat, s in self._states.items():
            out[stat] = {
                "beta":         s["beta"].tolist(),
                "P":            s["P"].tolist(),
                "obs_var":      s["obs_var"],
                "feat_names":   s["feat_names"],
                "scaler_mean":  s["scaler_mean"].tolist(),
                "scaler_scale": s["scaler_scale"].tolist(),
                "n_updates":    s["n_updates"],
            }
        return out

    @classmethod
    def from_learned_params(
        cls,
        params_path: Path = PARAMS_FILE,
        q_scalar: float = Q_PLAYER_SCALAR,
    ) -> "PlayerKF":
        """Initialise KF from batch Ridge fits in learned_params.json."""
        with open(params_path) as fh:
            p = json.load(fh)
        pm = p.get("player_models", {})
        pv = p.get("player_variance", {}).get("league_average", {})

        stat_states: dict = {}
        for stat, m in pm.items():
            feat_names   = m["feature_names"]
            k            = len(feat_names)
            scaler_mean  = np.array(m["scaler_mean"])
            scaler_scale = np.array(m["scaler_scale"])

            # β = [intercept, ewma_weight, home_boost, rest_coef, ...]
            beta       = np.zeros(k + 1)
            beta[0]    = m["intercept"]
            beta[1]    = m["ewma_weight"]
            if k > 1:
                beta[2] = m.get("home_boost", 0.0)
            if k > 2:
                beta[3] = m.get("rest_coef", 0.0)

            league_std = pv.get(stat, 5.0)
            obs_var    = league_std ** 2

            stat_states[stat] = {
                "beta":         beta,
                "P":            np.eye(k + 1) * P_PLAYER_INIT_DIAG,
                "obs_var":      obs_var,
                "feat_names":   feat_names,
                "scaler_mean":  scaler_mean,
                "scaler_scale": scaler_scale,
                "n_updates":    0,
            }

        return cls(stat_states=stat_states, q_scalar=q_scalar)


# ── Top-level persistence ─────────────────────────────────────────────────────────

def save_tvp_state(
    game_ekf: GameEKF,
    player_kf: PlayerKF,
    last_updated: str,
    recommendation: str = "use_kalman",
    path: Path = TVP_STATE_FILE,
) -> None:
    """Write full Kalman filter state to tvp_state.json."""
    state = {
        "version":        last_updated,
        "last_updated":   last_updated,
        "recommendation": recommendation,
        "game_model":     game_ekf.to_dict(),
        "player_models":  player_kf.to_dict(),
    }
    with open(path, "w") as fh:
        json.dump(state, fh, indent=2)


def load_tvp_state(path: Path = TVP_STATE_FILE) -> Optional[dict]:
    """Return raw dict from tvp_state.json, or None if it doesn't exist."""
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def load_game_ekf(path: Path = TVP_STATE_FILE) -> Optional[GameEKF]:
    """Load the game EKF from tvp_state.json, or None."""
    state = load_tvp_state(path)
    if state is None or "game_model" not in state:
        return None
    return GameEKF.from_dict(state["game_model"])


def load_player_kf(path: Path = TVP_STATE_FILE) -> Optional[PlayerKF]:
    """Load the player KF from tvp_state.json, or None."""
    state = load_tvp_state(path)
    if state is None or "player_models" not in state:
        return None
    return PlayerKF(stat_states=state["player_models"])
