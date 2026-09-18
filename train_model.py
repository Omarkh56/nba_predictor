"""
train_model.py — NBA Prediction Model Training Pipeline
=========================================================
Fetches 3-5 seasons of historical data, engineers walk-forward features,
fits game + player regression models, evaluates against the hand-tuned
baseline, bootstraps calibration.json, and saves learned_params.json.

Re-run at any point in the season — cached CSVs avoid re-fetching.

Usage:
    python3 train_model.py                  # full run
    python3 train_model.py --no-fetch       # reuse cached CSVs, refit only
    python3 train_model.py --eval-only      # load existing params, evaluate only
    python3 train_model.py --seasons 2022-23 2023-24 2024-25
    python3 train_model.py --no-bootstrap   # skip calibration bootstrap

Outputs:
    learned_params.json  — game + player model coefficients
    calibration.json     — bootstrapped prior for calibrate.py (if not present)
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from config import training_seasons
from game_features import (
    _ALTITUDE_HOME_TIDS as _ALTITUDE_HOME_TIDS,
)
from game_features import (
    _STAR_FORM_SHRINK_K as _STAR_FORM_SHRINK_K,
)
from game_features import (
    GAME_FEATURE_VERSION,
    GAME_FEATURES,
    build_game_dataset,
    venue_residual_from_record,
)
from game_features import (
    GAME_FEATURES_EXTENDED as GAME_FEATURES_EXTENDED,
)
from game_features import (
    GAME_FEATURES_NEW as GAME_FEATURES_NEW,
)
from game_features import (
    altitude_feature as altitude_feature,
)
from game_features import (
    compute_star_form_index as compute_star_form_index,
)

warnings.filterwarnings("ignore")

# ── NBA-API patch (same as newnbapredictor.py) ────────────────────────────────
import nba_api.stats.library.http as _nba_http
_nba_http.STATS_HEADERS.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
})
_nba_http.STATS_TIMEOUT = 90

import requests as _req
import requests          # needed for requests.exceptions in except clauses
_orig_get = _req.get
def _patched_get(url, **kwargs):
    kwargs.setdefault("verify", False)
    return _orig_get(url, **kwargs)
_req.get = _patched_get
# ─────────────────────────────────────────────────────────────────────────────

from nba_api.stats.endpoints import leaguegamefinder, leaguegamelog

_HERE       = Path(__file__).parent
CACHE_DIR   = _HERE / "data_cache"
PARAMS_FILE    = _HERE / "learned_params.json"
CALIB_FILE     = _HERE / "calibration.json"
ABLATION_FILE  = _HERE / "ablation_results.json"
CACHE_DIR.mkdir(exist_ok=True)

# v9: were hardcoded ["2020-21", ..., "2024-25"] / "2023-24" / "2024-25" — see
# config.py. training_seasons(5) returns the last 5 COMPLETED seasons,
# oldest first; the two most recent of those are held out as val/test so the
# remaining 3 are pure training data (unchanged from the original split).
DEFAULT_SEASONS = training_seasons(5)
VAL_SEASON  = DEFAULT_SEASONS[-2]   # drives swap/monitor/keep recommendation
TEST_SEASON = DEFAULT_SEASONS[-1]   # touched exactly once, after model is already selected

# Current hand-tuned constants (mirrored from newnbapredictor.py)
RTG_SCALE            = 9.5
BASE_HCA             = 2.8
FOUR_FACTORS_WEIGHT  = 0.35
FORM_WEIGHT          = 0.20
H2H_WEIGHT           = 0.08
BACK_TO_BACK_PENALTY = 2.2
REST_DAY_ADVANTAGE   = 0.8

# Game inputs have one implementation shared with live predictions.
PLAYER_STATS  = ["PTS", "REB", "AST", "FG3M", "STL", "BLK"]
POS_GROUPS    = ["G", "F", "C"]


# =============================================================================
# SECTION 1 — DATA FETCHING (with disk cache)
# =============================================================================

def _sleep():
    time.sleep(1.5)


def fetch_team_game_logs(seasons: list, force: bool = False) -> pd.DataFrame:
    """Pull one row per team per game for each season. Cached to CSV."""
    frames = []
    for season in seasons:
        cache_f = CACHE_DIR / f"team_games_{season.replace('-','')}.csv"
        if cache_f.exists() and not force:
            print(f"  [cache] team games {season}")
            frames.append(pd.read_csv(cache_f))
            continue

        print(f"  [fetch] team games {season} ...", end=" ", flush=True)
        try:
            df = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable="00",
                season_type_nullable="Regular Season",
                player_or_team_abbreviation="T",
            ).get_data_frames()[0]
            df["season"] = season
            df.to_csv(cache_f, index=False)
            frames.append(df)
            print(f"  {len(df)} rows")
        except (requests.exceptions.RequestException, json.JSONDecodeError, OSError, IndexError) as e:
            print(f"  FAILED: {e}")
        _sleep()

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_player_game_logs(seasons: list, force: bool = False) -> pd.DataFrame:
    """Pull one row per player per game for each season. Cached to CSV."""
    frames = []
    for season in seasons:
        cache_f = CACHE_DIR / f"player_games_{season.replace('-','')}.csv"
        if cache_f.exists() and not force:
            print(f"  [cache] player games {season}")
            frames.append(pd.read_csv(cache_f))
            continue

        print(f"  [fetch] player games {season} ...", end=" ", flush=True)
        try:
            df = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation="P",
            ).get_data_frames()[0]
            df["season"] = season
            df.to_csv(cache_f, index=False)
            frames.append(df)
            print(f"  {len(df)} rows")
        except (requests.exceptions.RequestException, json.JSONDecodeError, OSError, IndexError) as e:
            print(f"  FAILED: {e}")
        _sleep()

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_player_positions(seasons: list, force: bool = False, allow_fetch: bool = True) -> dict:
    """Map IDs using explicit NBA position filters, never a missing column.

    Ambiguous/unmapped players retain unknown positions. Old all-guard caches
    are rejected because LeagueDashPlayerStats does not return PLAYER_POSITION.
    """
    from nba_api.stats.endpoints import leaguedashplayerstats

    cache_f = CACHE_DIR / "player_positions.json"
    if cache_f.exists() and not force:
        try:
            with open(cache_f) as fh:
                cached = json.load(fh)
            if (isinstance(cached, dict) and cached.get("source") == "nba-position-filters-v1"
                    and cached.get("season") == seasons[-1]):
                mapping = {int(k): v for k, v in cached["positions"].items() if v in POS_GROUPS}
                if set(mapping.values()) == set(POS_GROUPS):
                    return mapping
        except (AttributeError, OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass
    if not allow_fetch:
        print("  No verified position cache; position-specific SDs will remain inactive.")
        return {}

    pos_map = {}
    season = seasons[-1]
    print(f"  [fetch] player positions ({season}) ...", end=" ", flush=True)
    try:
        memberships = {}
        groups = []
        for pg in POS_GROUPS:
            df = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season, season_type_all_star="Regular Season",
                per_mode_detailed="PerGame", player_position_abbreviation_nullable=pg,
            ).get_data_frames()[0]
            ids = set(df["PLAYER_ID"].astype(int))
            groups.append(ids)
            for pid in ids:
                memberships.setdefault(pid, set()).add(pg)
            _sleep()
        if any(len(ids) < 10 for ids in groups) or any(a == b for i, a in enumerate(groups) for b in groups[i + 1:]):
            raise ValueError("NBA position filters returned insufficient/identical groups")
        pos_map = {pid: next(iter(pgs)) for pid, pgs in memberships.items() if len(pgs) == 1}
        if any(sum(value == pg for value in pos_map.values()) < 10 for pg in POS_GROUPS):
            raise ValueError("too few unambiguous players per position")
        print(f"  {len(pos_map)} players")
        with open(cache_f, "w") as fh:
            json.dump({"source": "nba-position-filters-v1", "season": season,
                       "positions": {str(k): v for k, v in pos_map.items()}}, fh)
    except (requests.exceptions.RequestException, json.JSONDecodeError,
            OSError, IndexError, KeyError, ValueError, TypeError) as e:
        pos_map = {}
        print(f"  FAILED: {type(e).__name__}")
    _sleep()
    return pos_map


# =============================================================================
# SECTION 2 — GAME FEATURE ENGINEERING (walk-forward, no lookahead)
# =============================================================================

# ── New signal helpers ─────────────────────────────────────────────────────────

# =============================================================================
# SECTION 3 — HAND-TUNED MODEL PREDICTION (for baseline comparison)
# =============================================================================

def _sigmoid(x, scale=RTG_SCALE):
    return 1.0 / (1.0 + math.exp(-x / scale))


def hand_tuned_predict(df: pd.DataFrame) -> np.ndarray:
    """Replicate newnbapredictor.py logic on historical features."""
    probs = []
    for _, r in df.iterrows():
        base    = r["net_rtg_diff"]
        hca     = BASE_HCA + (r.get("home_wpct", 0.55) - 0.57) * 2.0
        hca     = max(1.5, min(5.0, hca))

        ff_adj  = (
            r["efg_diff"]  * 120.0   # eFG% pts equivalent
          + r["tov_diff"]  *  80.0
          + r["orb_diff"]  *  50.0
          + r["ftr_diff"]  *  30.0
        ) * FOUR_FACTORS_WEIGHT

        form_adj = r["form_diff"] * FORM_WEIGHT

        rest_adj = 0.0
        if r["home_b2b"] and not r["away_b2b"]:
            rest_adj = -BACK_TO_BACK_PENALTY
        elif r["away_b2b"] and not r["home_b2b"]:
            rest_adj = +BACK_TO_BACK_PENALTY
        else:
            rest_adj = r["rest_diff"] * REST_DAY_ADVANTAGE

        h2h_adj  = r["h2h_margin"] * H2H_WEIGHT
        margin   = base + hca + ff_adj + form_adj + rest_adj + h2h_adj
        probs.append(_sigmoid(margin))

    return np.array(probs)


# =============================================================================
# SECTION 4 — GAME MODEL FITTING
# =============================================================================

def fit_game_models(df_train: pd.DataFrame, df_val: pd.DataFrame,
                    df_test: pd.DataFrame = None,
                    features: list = None) -> dict:
    """
    Fit logistic regression (win prob) + ridge regression (margin).
    features  — the exact list of column names to use; defaults to GAME_FEATURES.
    df_test   — independent holdout (TEST_SEASON); scored AFTER recommendation is
                made so it never influences model selection.
    Returns dict with model objects, scalers, and evaluation results.
    """
    if features is None:
        features = GAME_FEATURES

    X_train  = df_train[features].values
    y_win_tr = df_train["home_win"].values
    y_mgn_tr = df_train["actual_margin"].values

    X_val   = df_val[features].values
    y_win_v = df_val["home_win"].values
    y_mgn_v = df_val["actual_margin"].values

    scaler = StandardScaler().fit(X_train)
    Xt = scaler.transform(X_train)
    Xv = scaler.transform(X_val)

    # ── Logistic regression (win probability) ────────────────────────────────
    log_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    log_model.fit(Xt, y_win_tr)

    learned_probs_val  = log_model.predict_proba(Xv)[:, 1]
    handtuned_probs_val = hand_tuned_predict(df_val)

    ll_learned  = log_loss(y_win_v, learned_probs_val)
    ll_hand     = log_loss(y_win_v, handtuned_probs_val)
    bs_learned  = brier_score_loss(y_win_v, learned_probs_val)
    bs_hand     = brier_score_loss(y_win_v, handtuned_probs_val)

    n_new = len(features) - len(GAME_FEATURES)
    feat_label = (f"{len(features)} features ({len(GAME_FEATURES)} base + {n_new} new)"
                  if n_new > 0 else f"{len(features)} features")
    print(f"\n  === GAME MODEL EVALUATION ({feat_label}) — val set: {len(df_val)} games ===")
    print(f"  {'Model':<20} {'Log Loss':>10} {'Brier':>10}")
    print(f"  {'-'*40}")
    print(f"  {'Hand-tuned':<20} {ll_hand:>10.4f} {bs_hand:>10.4f}")
    print(f"  {'Learned (LogReg)':<20} {ll_learned:>10.4f} {bs_learned:>10.4f}")
    delta_ll = ll_learned - ll_hand
    delta_bs = bs_learned - bs_hand
    print(f"  {'Delta':<20} {delta_ll:>+10.4f} {delta_bs:>+10.4f}")

    if ll_learned < ll_hand and bs_learned < bs_hand:
        recommendation = "swap"
        print("  ✓ Learned model wins on both metrics → recommend swap")
    elif ll_learned < ll_hand or bs_learned < bs_hand:
        recommendation = "monitor"
        print("  ~ Learned model wins on one metric → monitor, don't swap yet")
    else:
        recommendation = "keep_hand_tuned"
        print("  ✗ Hand-tuned baseline wins → keeping hand-tuned")

    # ── Ridge regression (margin) ─────────────────────────────────────────────
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(Xt, y_mgn_tr)
    margin_preds = ridge_model.predict(Xv)
    rmse = float(np.sqrt(np.mean((margin_preds - y_mgn_v) ** 2)))
    mae  = float(np.mean(np.abs(margin_preds - y_mgn_v)))
    print(f"\n  Margin model RMSE={rmse:.2f} MAE={mae:.2f}")

    # Feature importance (standardized coefficients)
    coef_dict = dict(zip(features, log_model.coef_[0].tolist()))
    print("\n  Learned logistic regression coefficients (standardized):")
    for feat, coef in sorted(coef_dict.items(), key=lambda x: -abs(x[1])):
        bar = "+" * int(abs(coef) * 5) if coef > 0 else "-" * int(abs(coef) * 5)
        print(f"    {feat:<22} {coef:>+7.4f}  {bar}")

    # ── Test-set evaluation (scored AFTER recommendation — never influences it) ─
    test_log_loss = None
    test_brier    = None
    test_games    = None
    ht_test_ll    = None
    ht_test_bs    = None
    if df_test is not None and not df_test.empty:
        missing = [f for f in features if f not in df_test.columns]
        if missing:
            print(f"\n  [test] skipped — missing columns: {missing}")
        else:
            Xte = scaler.transform(df_test[features].values)
            yte = df_test["home_win"].values
            probs_te   = log_model.predict_proba(Xte)[:, 1]
            ht_probs_te = hand_tuned_predict(df_test)
            test_log_loss = round(float(log_loss(yte, probs_te)), 6)
            test_brier    = round(float(brier_score_loss(yte, probs_te)), 6)
            ht_test_ll    = round(float(log_loss(yte, ht_probs_te)), 6)
            ht_test_bs    = round(float(brier_score_loss(yte, ht_probs_te)), 6)
            test_games    = len(df_test)
            print(f"\n  === TEST SET ({TEST_SEASON}) — {test_games} games ===")
            print(f"  {'Model':<20} {'Log Loss':>10} {'Brier':>10}")
            print(f"  {'-'*40}")
            print(f"  {'Hand-tuned':<20} {ht_test_ll:>10.4f} {ht_test_bs:>10.4f}")
            print(f"  {'Learned (LogReg)':<20} {test_log_loss:>10.4f} {test_brier:>10.4f}")

    return {
        "log_model":              log_model,
        "ridge_model":            ridge_model,
        "scaler":                 scaler,
        "feature_names":          features,
        "feature_version":        GAME_FEATURE_VERSION,
        "scaler_mean":            scaler.mean_.tolist(),
        "scaler_scale":           scaler.scale_.tolist(),
        "log_intercept":          float(log_model.intercept_[0]),
        "log_coefficients":       {f: float(c) for f, c in coef_dict.items()},
        "ridge_intercept":        float(ridge_model.intercept_),
        "ridge_coefficients":     {f: float(c) for f, c
                                    in zip(features, ridge_model.coef_.tolist())},
        "hand_tuned_log_loss":    ll_hand,
        "learned_log_loss":       ll_learned,
        "hand_tuned_brier":       bs_hand,
        "learned_brier":          bs_learned,
        "ht_test_log_loss":       ht_test_ll,
        "ht_test_brier":          ht_test_bs,
        "test_log_loss":          test_log_loss,
        "test_brier":             test_brier,
        "test_games":             test_games,
        "margin_rmse":            rmse,
        "margin_mae":             mae,
        "val_games":              len(df_val),
        "train_games":            len(df_train),
        "recommendation":         recommendation,
    }


# =============================================================================
# SECTION 4b — NEW-SIGNAL ABLATION STUDY
# =============================================================================

def ablation_study(df_train: pd.DataFrame, df_val: pd.DataFrame) -> dict:
    """
    Compare baseline (GAME_FEATURES) against each new signal individually.
    Pass criterion: beats baseline on BOTH log loss AND Brier (stricter than
    one-metric wins — the same bar used for the overall swap/keep decision).

    Returns:
      baseline         — {log_loss, brier}
      features         — per-feature {log_loss, brier, delta_ll, delta_brier,
                          passed, reason}  ← inspectable in learned_params.json
      winning_features — list of new feature names that passed (may be empty)
      new_signal_rec   — "use_extended" | "keep_base"
    """
    y_val = df_val["home_win"].values

    def _fit_eval(feats: list) -> dict:
        available = [f for f in feats if f in df_train.columns and f in df_val.columns]
        if len(available) != len(feats):
            return {}
        X_tr = df_train[available].values
        X_vl = df_val[available].values
        sc   = StandardScaler().fit(X_tr)
        clf  = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(sc.transform(X_tr), df_train["home_win"].values)
        probs = clf.predict_proba(sc.transform(X_vl))[:, 1]
        return {
            "log_loss": float(log_loss(y_val, probs)),
            "brier":    float(brier_score_loss(y_val, probs)),
        }

    base = _fit_eval(GAME_FEATURES)
    if not base:
        return {"winning_features": [], "baseline": {}, "features": {}, "new_signal_rec": "keep_base"}

    base_ll = base["log_loss"]
    base_bs = base["brier"]

    feature_results = {}
    winning_features = []

    for feat_name in ["altitude_penalty", "venue_residual", "star_form_diff"]:
        r = _fit_eval(GAME_FEATURES + [feat_name])
        if not r:
            feature_results[feat_name] = {
                "log_loss": None, "brier": None,
                "delta_ll": None, "delta_brier": None,
                "passed": False,
                "reason": "feature column missing from dataset",
            }
            continue

        d_ll  = r["log_loss"] - base_ll
        d_bs  = r["brier"]    - base_bs
        beats_ll = r["log_loss"] < base_ll
        beats_bs = r["brier"]    < base_bs
        passed   = beats_ll and beats_bs

        if passed:
            reason = "beats baseline on both metrics"
            winning_features.append(feat_name)
        elif not beats_ll and not beats_bs:
            reason = f"worse log loss ({d_ll:+.4f}) and worse Brier ({d_bs:+.4f})"
        elif not beats_ll:
            reason = f"worse log loss ({d_ll:+.4f}); Brier improves but not enough alone"
        else:
            reason = f"worse Brier ({d_bs:+.4f}); log loss improves but not enough alone"

        feature_results[feat_name] = {
            "log_loss":    round(r["log_loss"], 6),
            "brier":       round(r["brier"], 6),
            "delta_ll":    round(d_ll, 6),
            "delta_brier": round(d_bs, 6),
            "passed":      passed,
            "reason":      reason,
        }

    # ── Print table ──────────────────────────────────────────────────────────
    W = 72
    print(f"\n  {'─'*W}")
    print(f"  ABLATION STUDY — new signals vs baseline (val set: {len(df_val)} games)")
    print(f"  {'─'*W}")
    print(f"  {'Feature':<22} {'LL':>8} {'ΔLL':>8} {'Brier':>8} {'ΔBrier':>8} {'':>5}")
    print(f"  {'─'*W}")
    print(f"  {'Baseline (10 feats)':<22} {base_ll:>8.4f} {'—':>8} {base_bs:>8.4f} {'—':>8} {'★':>5}")
    for feat_name, r in feature_results.items():
        if r["log_loss"] is None:
            print(f"  {feat_name:<22}  MISSING (column not in dataset)")
            continue
        mark = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"  {feat_name:<22} {r['log_loss']:>8.4f} {r['delta_ll']:>+8.4f} "
              f"{r['brier']:>8.4f} {r['delta_brier']:>+8.4f} {mark:>5}")
        if not r["passed"]:
            print(f"  {'':>22}  → excluded: {r['reason']}")
    print(f"  {'─'*W}")

    if winning_features:
        print(f"  Winning features (added to production model): {winning_features}")
    else:
        print("  No new features beat baseline on both metrics → production stays at 10 features")

    return {
        "baseline":          {"log_loss": round(base_ll, 6), "brier": round(base_bs, 6)},
        "features":          feature_results,
        "winning_features":  winning_features,
        "new_signal_rec":    "use_extended" if winning_features else "keep_base",
    }


# =============================================================================
# SECTION 4c — GAME FEATURE ABLATION (leave-one-out, --ablation flag only)
# =============================================================================

def run_ablation(df_train: pd.DataFrame, df_val: pd.DataFrame,
                 feature_names: list) -> list:
    """
    Leave-one-out ablation over feature_names.

    Baseline = logistic regression trained on all feature_names.
    For each feature: refit without it, compare val log_loss and Brier vs baseline.
    Positive delta = removing the feature made the model WORSE → feature was helping.
    Negative delta = removing improved the model → feature was hurting.

    Sorted by |delta log_loss|, largest impact first.
    Writes ablation_results.json and returns a list of result dicts.
    """
    y_val = df_val["home_win"].values

    def _fit_eval(feats: list):
        avail = [f for f in feats if f in df_train.columns and f in df_val.columns]
        if len(avail) != len(feats):
            return None
        sc  = StandardScaler().fit(df_train[avail].values)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(sc.transform(df_train[avail].values), df_train["home_win"].values)
        probs = clf.predict_proba(sc.transform(df_val[avail].values))[:, 1]
        return {
            "log_loss": float(log_loss(y_val, probs)),
            "brier":    float(brier_score_loss(y_val, probs)),
        }

    baseline = _fit_eval(feature_names)
    if baseline is None:
        print("  [ablation] one or more feature columns missing — cannot run")
        return []

    base_ll = baseline["log_loss"]
    base_bs = baseline["brier"]

    rows = []
    for feat in feature_names:
        reduced = [f for f in feature_names if f != feat]
        result  = _fit_eval(reduced)
        if result is None:
            continue
        delta_ll = result["log_loss"] - base_ll
        delta_bs = result["brier"]    - base_bs
        rows.append({
            "feature":           feat,
            "full_val_log_loss": round(base_ll, 6),
            "drop_val_log_loss": round(result["log_loss"], 6),
            "delta_log_loss":    round(delta_ll, 6),
            "full_val_brier":    round(base_bs, 6),
            "drop_val_brier":    round(result["brier"], 6),
            "delta_brier":       round(delta_bs, 6),
            "was_helping":       delta_ll > 0 or delta_bs > 0,
        })

    rows.sort(key=lambda r: -abs(r["delta_log_loss"]))

    W = 82
    print(f"\n  {'─'*W}")
    print(f"  GAME FEATURES ABLATION (leave-one-out) — val set: {len(df_val)} games")
    print(f"  Baseline ({len(feature_names)} features):  "
          f"log_loss={base_ll:.4f}  brier={base_bs:.4f}")
    print(f"  ΔLogLoss > 0  → removing hurt  (feature was contributing)")
    print(f"  ΔLogLoss < 0  → removing helped (feature was noise/harmful)")
    print(f"  {'─'*W}")
    print(f"  {'Feature':<22} {'ΔLogLoss':>10} {'ΔBrier':>8}  Assessment")
    print(f"  {'─'*W}")
    for r in rows:
        note = "was helping" if r["was_helping"] else "was noise/harmful"
        sign = "↑" if r["was_helping"] else "↓"
        print(f"  {r['feature']:<22} {r['delta_log_loss']:>+10.4f} "
              f"{r['delta_brier']:>+8.4f}  {sign} {note}")
    print(f"  {'─'*W}")
    print(f"  (No features were dropped — review ablation_results.json before "
          f"removing anything from GAME_FEATURES)")

    payload = {
        "val_season":    VAL_SEASON,
        "feature_names": feature_names,
        "baseline":      {"log_loss": round(base_ll, 6), "brier": round(base_bs, 6)},
        "features":      rows,
    }
    with open(ABLATION_FILE, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  ✓ Saved {ABLATION_FILE.name}")

    return rows


# =============================================================================
# SECTION 5 — PLAYER FEATURE ENGINEERING (walk-forward)
# =============================================================================

def _ewm_series(series: pd.Series, half_life: float = 5.0) -> pd.Series:
    """Exponentially weighted mean, shifted so current game is excluded."""
    return series.shift(1).ewm(halflife=half_life, min_periods=3).mean()


def build_player_dataset(player_logs: pd.DataFrame,
                         pos_map: dict) -> pd.DataFrame:
    """
    Build per-player per-game feature rows.
    Features are pre-game EWMA of rate and minutes; target is actual stat.
    """
    if player_logs.empty:
        return pd.DataFrame()

    pl = player_logs.copy()
    pl["GAME_DATE"] = pd.to_datetime(pl["GAME_DATE"])
    pl["is_home"]   = pl["MATCHUP"].str.contains(r"vs\.", na=False)
    pl["MIN"]       = pd.to_numeric(pl["MIN"], errors="coerce").fillna(0)

    # Guarantee numeric stat columns
    for col in PLAYER_STATS + ["MIN"]:
        if col in pl.columns:
            pl[col] = pd.to_numeric(pl[col], errors="coerce").fillna(0)

    records = []
    for pid, grp in pl.groupby("PLAYER_ID"):
        g = grp.sort_values("GAME_DATE").reset_index(drop=True)
        pos_group = pos_map.get(int(pid))

        g["prev_date"] = g["GAME_DATE"].shift(1)
        g["rest_days"] = ((g["GAME_DATE"] - g["prev_date"]).dt.days
                          .fillna(3).clip(upper=7).astype(float))
        g["ewma_min"] = _ewm_series(g["MIN"])

        for stat in PLAYER_STATS:
            if stat not in g.columns:
                continue
            g[f"rate_{stat}"] = g[stat] / g["MIN"].clip(lower=1)
            g[f"ewma_rate_{stat}"] = _ewm_series(g[f"rate_{stat}"])
            g[f"ewma_proj_{stat}"] = g[f"ewma_rate_{stat}"] * g["ewma_min"]

        for idx, row in g.iterrows():
            if row["ewma_min"] < 8 or pd.isna(row["ewma_min"]):
                continue
            rec = {
                "player_id":  int(pid),
                "player_name": str(row.get("PLAYER_NAME", "")),
                "pos_group":   pos_group,
                "game_id":     row["GAME_ID"],
                "game_date":   row["GAME_DATE"],
                "season":      row.get("season", ""),
                "is_home":     int(row["is_home"]),
                "rest_days":   float(row["rest_days"]),
                "actual_min":  float(row["MIN"]),
            }
            for stat in PLAYER_STATS:
                if f"ewma_proj_{stat}" in g.columns:
                    rec[f"ewma_{stat}"]   = float(row.get(f"ewma_proj_{stat}", np.nan) or np.nan)
                    rec[f"actual_{stat}"] = float(row.get(stat, np.nan) or np.nan)
            records.append(rec)

    return pd.DataFrame(records)


# =============================================================================
# SECTION 6 — PLAYER MODEL FITTING (Ridge + partial pooling for variance)
# =============================================================================

def fit_player_models(df_players: pd.DataFrame) -> dict:
    """
    For each stat, fit a Ridge regression predicting actual_stat from the
    EWMA projection + home/rest adjustments. Then estimate residual variance
    with partial pooling across position groups (shrinkage toward group mean).
    """
    player_results = {}
    variance_results = {"by_position": {pg: {} for pg in POS_GROUPS},
                        "position_support": {pg: {} for pg in POS_GROUPS},
                        "league_average": {},
                        "shrinkage_n": 15}

    print("\n  === PLAYER MODEL RESULTS ===")

    for stat in PLAYER_STATS:
        ewma_col   = f"ewma_{stat}"
        actual_col = f"actual_{stat}"

        cols_needed = [ewma_col, actual_col, "is_home", "rest_days"]
        available   = [c for c in cols_needed if c in df_players.columns]
        if ewma_col not in available or actual_col not in available:
            continue

        sub = df_players[available + ["pos_group", "player_id"]].copy()
        sub = sub.dropna(subset=[ewma_col, actual_col])
        sub = sub[sub[actual_col] >= 0]
        if len(sub) < 200:
            continue

        # Simple train/val split — use most recent season as val
        seasons_sorted = sorted(df_players["season"].unique())
        val_season     = seasons_sorted[-1]
        tr = sub[df_players.loc[sub.index, "season"] != val_season]
        va = sub[df_players.loc[sub.index, "season"] == val_season]

        X_cols = [ewma_col] + [c for c in ["is_home", "rest_days"] if c in available]
        Xtr = tr[X_cols].values
        ytr = tr[actual_col].values
        Xva = va[X_cols].values
        yva = va[actual_col].values

        sc  = StandardScaler().fit(Xtr)
        model = Ridge(alpha=5.0)
        model.fit(sc.transform(Xtr), ytr)

        preds_val = model.predict(sc.transform(Xva))
        rmse = float(np.sqrt(np.mean((preds_val - yva) ** 2)))
        mae  = float(np.mean(np.abs(preds_val - yva)))

        # Residual-based variance per position group
        preds_tr  = model.predict(sc.transform(Xtr))
        resid_tr  = ytr - preds_tr
        resid_va = yva - preds_val
        league_std = float(np.std(resid_tr))

        pos_stds = {}
        for pg in POS_GROUPS:
            mask = tr["pos_group"] == pg
            val_mask = va["pos_group"] == pg
            pg_resid, val_resid = resid_tr[mask.values], resid_va[val_mask.values]
            n, vn = len(pg_resid), len(val_resid)
            players = int(tr.loc[mask, "player_id"].nunique())
            val_players = int(va.loc[val_mask, "player_id"].nunique())
            # Pool thin estimates toward the league residual variance.
            weight = n / (n + variance_results["shrinkage_n"])
            variance = float(np.var(pg_resid)) if n > 1 else league_std ** 2
            pos_stds[pg] = math.sqrt(weight * variance + (1 - weight) * league_std ** 2)
            gain = 0.0
            if vn and pos_stds[pg] > 0 and league_std > 0:
                mse = float(np.mean(val_resid ** 2))
                gain = (math.log(league_std) + mse / (2 * league_std ** 2)
                        - math.log(pos_stds[pg]) - mse / (2 * pos_stds[pg] ** 2))
            variance_results["position_support"][pg][stat] = {
                "train_n": n, "train_players": players, "val_n": vn,
                "val_players": val_players, "val_nll_gain": gain,
                "enabled": bool(n >= 100 and players >= 10 and vn >= 50 and val_players >= 5
                                and league_std > 0 and abs(pos_stds[pg] / league_std - 1) >= 0.05
                                and math.isfinite(gain) and gain > 0),
            }
            variance_results["by_position"][pg][stat] = round(pos_stds[pg], 3)

        variance_results["league_average"][stat] = round(league_std, 3)

        player_results[stat] = {
            "intercept":       float(model.intercept_),
            "ewma_weight":     float(model.coef_[0]),
            "home_boost":      float(model.coef_[1]) if len(X_cols) > 1 else 0.0,
            "rest_coef":       float(model.coef_[2]) if len(X_cols) > 2 else 0.0,
            "scaler_mean":     sc.mean_.tolist(),
            "scaler_scale":    sc.scale_.tolist(),
            "feature_names":   X_cols,
            "val_rmse":        round(rmse, 3),
            "val_mae":         round(mae, 3),
            "val_n":           len(va),
            "train_n":         len(tr),
        }

        naive_rmse = float(np.sqrt(np.mean((Xva[:, 0] - yva) ** 2)))  # EWMA alone
        print(f"  {stat:<5} RMSE={rmse:.3f}  MAE={mae:.3f}  "
              f"(vs EWMA-only RMSE={naive_rmse:.3f})  n_val={len(va)}")

    return {"stat_models": player_results, "variance": variance_results}


# =============================================================================
# SECTION 7 — CALIBRATION BOOTSTRAP
# =============================================================================

def bootstrap_calibration(df_game: pd.DataFrame,
                           df_players: pd.DataFrame,
                           player_result: dict) -> dict:
    """
    Build a calibration.json-compatible starting state from historical data.
    This seeds the file before live bets so calibrate.py has a valid prior.

    Computes:
      - overall: historical win rate of home-team favorites using hand-tuned model
      - market_direction: hit rates split by stat / over vs under based on EWMA
      - player_market: per-player per-stat mean projection bias
    """
    print("\n  === CALIBRATION BOOTSTRAP ===")
    calib: dict = {"overall": {}, "market_direction": {}, "player_market": {}}

    # ── Overall (game model) ─────────────────────────────────────────────────
    if not df_game.empty:
        ht_probs = hand_tuned_predict(df_game)
        fav_mask = ht_probs >= 0.5
        fav_hit  = df_game["home_win"].values[fav_mask]
        if len(fav_hit):
            calib["overall"] = {
                "bets": len(fav_hit),
                "hits": int(fav_hit.sum()),
                "rate": round(float(fav_hit.mean()), 4),
            }
            print(f"  Game model favorite hit rate: {fav_hit.mean():.3f} "
                  f"({fav_hit.sum()}/{len(fav_hit)})")

    # ── Player market direction (per stat) ───────────────────────────────────
    for stat in PLAYER_STATS:
        ewma_col   = f"ewma_{stat}"
        actual_col = f"actual_{stat}"
        if ewma_col not in df_players.columns or actual_col not in df_players.columns:
            continue

        sub = df_players[[ewma_col, actual_col]].dropna()
        if sub.empty:
            continue

        # "Line" proxy = EWMA projection itself
        # OVER pick = when projection > rolling median (i.e. model bullish)
        over_mask  = sub[ewma_col] > sub[ewma_col].expanding().median().shift(1).fillna(sub[ewma_col].median())
        under_mask = ~over_mask

        for pick_label, mask in [("OVER", over_mask), ("UNDER", under_mask)]:
            picks  = sub[mask]
            if picks.empty:
                continue
            if pick_label == "OVER":
                hits = (picks[actual_col] > picks[ewma_col]).sum()
            else:
                hits = (picks[actual_col] < picks[ewma_col]).sum()
            key = f"{stat}_{pick_label}"
            calib["market_direction"][key] = {
                "hits": int(hits),
                "bets": len(picks),
                "rate": round(hits / len(picks), 4),
            }

    # ── Player-market bias (additive residual per player) ────────────────────
    for stat in PLAYER_STATS:
        ewma_col   = f"ewma_{stat}"
        actual_col = f"actual_{stat}"
        if ewma_col not in df_players.columns or actual_col not in df_players.columns:
            continue

        sub = df_players[["player_name", ewma_col, actual_col]].dropna()
        sub["residual"] = sub[actual_col] - sub[ewma_col]

        for pname, grp in sub.groupby("player_name"):
            n    = len(grp)
            if n < 10:
                continue
            mean = float(grp["residual"].mean())
            std  = float(grp["residual"].std())
            shrink = n / (n + 15)
            nudge  = mean * shrink
            if abs(nudge) > 0.2:   # only store non-trivial biases
                key = f"{pname}|{stat}"
                calib["player_market"][key] = {
                    "n":     n,
                    "mean":  round(mean, 4),
                    "std":   round(std, 4),
                    "nudge": round(nudge, 4),
                }

    print(f"  player_market entries: {len(calib['player_market'])}")
    return calib


# =============================================================================
# SECTION 8 — SAVE RESULTS
# =============================================================================

def save_params(game_result: dict, player_result: dict,
                calib: dict, seasons: list, no_bootstrap: bool,
                venue_residuals: dict = None,
                ablation_result: dict = None) -> None:
    gm_section = {
        k: v for k, v in game_result.items()
        if k not in ("log_model", "ridge_model", "scaler")
    }

    params = {
        "version":      datetime.now().strftime("%Y-%m-%d"),
        "trained_on":   [s for s in seasons if s not in {VAL_SEASON, TEST_SEASON}],
        "validated_on": VAL_SEASON,
        "tested_on":    TEST_SEASON,
        "game_model":   gm_section,
        "player_models":   player_result["stat_models"],
        "player_variance": player_result["variance"],
    }
    if venue_residuals:
        params["venue_residuals"] = venue_residuals
    if ablation_result:
        base      = ablation_result.get("baseline", {})
        feat_dict = ablation_result.get("features", {})
        # Stored as a list so each entry is self-contained and inspectable.
        ablation_list = [
            {
                "feature":                   feat_name,
                "baseline_val_log_loss":     base.get("log_loss"),
                "with_feature_val_log_loss": r.get("log_loss"),
                "baseline_val_brier":        base.get("brier"),
                "with_feature_val_brier":    r.get("brier"),
                "passed":                    r.get("passed", False),
                "reason":                    r.get("reason", ""),
            }
            for feat_name, r in feat_dict.items()
        ]
        params["game_model"]["ablation"]      = ablation_list
        params["game_model"]["new_signal_rec"] = ablation_result.get("new_signal_rec", "keep_base")
        params["game_model"]["winning_features"] = ablation_result.get("winning_features", [])

    with open(PARAMS_FILE, "w") as fh:
        json.dump(params, fh, indent=2)
    print(f"\n  ✓ Saved learned_params.json  ({PARAMS_FILE})")
    print(f"    recommendation: {game_result['recommendation']}")

    if no_bootstrap:
        return

    if CALIB_FILE.exists():
        print(f"  calibration.json already exists — not overwriting "
              f"(delete it to re-bootstrap from history)")
        return

    with open(CALIB_FILE, "w") as fh:
        json.dump(calib, fh, indent=2)
    print(f"  ✓ Bootstrapped calibration.json  ({CALIB_FILE})")


# =============================================================================
# SECTION 9 — MAIN
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="NBA model training pipeline")
    p.add_argument("--no-fetch",     action="store_true",
                   help="Use cached CSVs only, do not call NBA API")
    p.add_argument("--force-fetch",  action="store_true",
                   help="Force re-fetch even if cache exists")
    p.add_argument("--eval-only",    action="store_true",
                   help="Load existing learned_params.json and just print metrics")
    p.add_argument("--no-bootstrap", action="store_true",
                   help="Skip calibration.json bootstrap")
    p.add_argument("--ablation",     action="store_true",
                   help="Run leave-one-out ablation over GAME_FEATURES and write "
                        "ablation_results.json (adds ~10 extra model fits)")
    p.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS,
                   help="Seasons to use (e.g. 2021-22 2022-23 2023-24 2024-25)")
    p.add_argument("--model-name",   default=None,
                   help="Short identifier for this run in eval_log.jsonl "
                        "(e.g. 'logreg_v2').  Defaults to logreg_YYYYMMDD.")
    p.add_argument("--no-log",       action="store_true",
                   help="Skip appending this run to eval_log.jsonl")
    return p.parse_args()


def _run_eval_only():
    if not PARAMS_FILE.exists():
        print("learned_params.json not found — run without --eval-only first.")
        sys.exit(1)
    with open(PARAMS_FILE) as fh:
        params = json.load(fh)
    gm = params.get("game_model", {})
    print(f"\nlearned_params.json ({params.get('version', 'unknown')})")
    print(f"  trained on:  {params.get('trained_on', [])}")
    print(f"  val season:  {params.get('validated_on', '?')}  "
          f"({gm.get('val_games', '?')} games)")
    print(f"  test season: {params.get('tested_on', '?')}  "
          f"({gm.get('test_games', '?')} games)")
    print(f"  game model:  {gm.get('recommendation', '?')}")
    print(f"    val   log_loss  hand={gm.get('hand_tuned_log_loss','?'):.4f}  "
          f"learned={gm.get('learned_log_loss','?'):.4f}")
    print(f"    val   brier     hand={gm.get('hand_tuned_brier','?'):.4f}  "
          f"learned={gm.get('learned_brier','?'):.4f}")
    if gm.get("test_log_loss") is not None:
        print(f"    test  log_loss  hand={gm.get('ht_test_log_loss','?'):.4f}  "
              f"learned={gm.get('test_log_loss','?'):.4f}")
        print(f"    test  brier     hand={gm.get('ht_test_brier','?'):.4f}  "
              f"learned={gm.get('test_brier','?'):.4f}")


def _fetch_pipeline_data(seasons, no_fetch, force_fetch):
    if not no_fetch:
        print("\n[1/5] Fetching team game logs ...")
        team_logs = fetch_team_game_logs(seasons, force=force_fetch)
        print("\n[2/5] Fetching player game logs ...")
        player_logs = fetch_player_game_logs(seasons, force=force_fetch)
        print("\n[3/5] Fetching player positions ...")
        pos_map = fetch_player_positions(seasons, force=force_fetch)
    else:
        print("\n[1-3/5] Loading from cache ...")
        team_logs   = fetch_team_game_logs(seasons, force=False)
        player_logs = fetch_player_game_logs(seasons, force=False)
        pos_map     = fetch_player_positions(seasons, force=False, allow_fetch=False)
    return team_logs, player_logs, pos_map


def _build_three_way_split(df_game):
    """Three-way walk-forward split:
    df_train — everything excluding both holdout seasons
    df_val   — VAL_SEASON (drives recommendation / ablation)
    df_test  — TEST_SEASON (scored exactly once, after recommendation is made)
    """
    train_mask = ~df_game["season"].isin({VAL_SEASON, TEST_SEASON})
    df_train = df_game[train_mask].copy()
    df_val   = df_game[df_game["season"] == VAL_SEASON].copy()
    df_test  = df_game[df_game["season"] == TEST_SEASON].copy()

    if df_val.empty or df_train.empty:
        # Guard: refuse to silently degrade to single-split if seasons are missing.
        # The user said they'd rather add a season to DEFAULT_SEASONS.
        missing = []
        if df_val.empty:
            missing.append(f"VAL_SEASON={VAL_SEASON}")
        if df_train.empty:
            missing.append("training data")
        print(f"\n  ERROR: Three-way split impossible — {', '.join(missing)} not in "
              f"fetched seasons.\n"
              f"  Fetched seasons: {sorted(df_game['season'].unique().tolist())}\n"
              f"  Add the missing season to DEFAULT_SEASONS or --seasons and re-run.\n"
              f"  (Refusing to silently fall back to single-split.)")
        sys.exit(1)

    n_train_seasons = df_train["season"].nunique()
    if n_train_seasons < 2:
        print(f"\n  WARNING: Only {n_train_seasons} training season(s) available after "
              f"removing VAL_SEASON={VAL_SEASON} and TEST_SEASON={TEST_SEASON}.\n"
              f"  Consider adding an older season to DEFAULT_SEASONS for a more robust fit.")

    print(f"  Train: {len(df_train)} games ({n_train_seasons} seasons) | "
          f"Val ({VAL_SEASON}): {len(df_val)} games | "
          f"Test ({TEST_SEASON}): {len(df_test)} games")

    return df_train, df_val, df_test


def _compute_venue_residuals(team_logs):
    """Season-end venue residuals per team, for live-predictor loading."""
    venue_residuals: dict = {}
    for tid, grp in team_logs.groupby("TEAM_ID"):
        sorted_g = grp.sort_values("GAME_DATE")
        is_win  = (sorted_g["WL"] == "W").astype(float)
        is_home = sorted_g["MATCHUP"].str.contains(r"vs\.", na=False).astype(float)
        total_wins  = int(is_win.sum())
        total_games = len(sorted_g)
        home_wins   = int((is_win * is_home).sum())
        home_games  = int(is_home.sum())
        venue_residuals[str(int(tid))] = round(
            venue_residual_from_record(home_wins, home_games, total_wins, total_games),
            4,
        )
    return venue_residuals


def _fit_player_stage(player_logs, pos_map):
    print(f"\n[5/5] Engineering player features ...")
    df_players = build_player_dataset(player_logs, pos_map)
    print(f"  Player dataset: {len(df_players)} player-game rows")

    if len(df_players) >= 1000:
        player_result = fit_player_models(df_players)
    else:
        print("  ⚠ Too few player rows — skipping player model fit")
        player_result = {"stat_models": {}, "variance": {}}
    return df_players, player_result


def _print_next_steps(rec):
    print("\n  Done. Next steps:")
    if rec == "swap":
        print("  → Learned model outperforms baseline on both metrics.")
        print("    newnbapredictor.py selects it automatically when all required inputs are available.")
    elif rec == "monitor":
        print("  → Mixed results. Re-run after more games accumulate.")
        print("    newnbapredictor.py keeps the hand-tuned model for this recommendation.")
    else:
        print("  → Hand-tuned baseline still wins. No swap needed yet.")
        print("    Re-run at midseason or after significant lineup changes.")


def main():
    args = _parse_args()

    if args.eval_only:
        _run_eval_only()
        return

    seasons     = args.seasons
    force_fetch = args.force_fetch and not args.no_fetch

    print("=" * 60)
    print("NBA Model Training Pipeline")
    print(f"Seasons: {seasons}")
    print("=" * 60)

    team_logs, player_logs, pos_map = _fetch_pipeline_data(
        seasons, args.no_fetch, force_fetch)

    if team_logs.empty:
        print("No team log data — aborting.")
        sys.exit(1)

    print(f"\n[4/5] Engineering game features ...")
    df_game = build_game_dataset(team_logs, player_logs=player_logs)
    print(f"  Game dataset: {len(df_game)} games "
          f"({df_game['home_win'].sum()} home wins = "
          f"{df_game['home_win'].mean():.1%} HWP)")

    if len(df_game) < 500:
        print("  ⚠ Very few games — check fetch or cache.")
        sys.exit(1)

    df_train, df_val, df_test = _build_three_way_split(df_game)

    # ── Ablation study — determines which new features (if any) pass ─────
    ablation_res = ablation_study(df_train, df_val)
    winning_new   = ablation_res.get("winning_features", [])
    final_features = GAME_FEATURES + winning_new
    if winning_new:
        print(f"\n  Production feature set: {len(final_features)} features "
              f"({len(GAME_FEATURES)} base + {len(winning_new)} new: {winning_new})")
    else:
        print(f"\n  Production feature set: {len(final_features)} features (base only)")

    # Fit production model; df_test is scored inside but never used for model
    # selection (recommendation is already printed before df_test numbers
    # appear — see fit_game_models for the ordering guarantee).
    game_result = fit_game_models(df_train, df_val, df_test=df_test,
                                  features=final_features)

    if args.ablation:
        print("\n  Running game-feature ablation study ...")
        run_ablation(df_train, df_val, final_features)

    venue_residuals = _compute_venue_residuals(team_logs)

    df_players, player_result = _fit_player_stage(player_logs, pos_map)

    calib = {}
    if not args.no_bootstrap:
        calib = bootstrap_calibration(df_game, df_players, player_result)

    save_params(game_result, player_result, calib, seasons, args.no_bootstrap,
                venue_residuals=venue_residuals, ablation_result=ablation_res)

    # ── Evaluation framework — score val + test, log to eval_log.jsonl ───
    try:
        from evaluate import evaluate, print_report
        from datetime import datetime as _dt

        model_tag = args.model_name or f"logreg_{_dt.now().strftime('%Y%m%d')}"

        # Reconstruct model probabilities for val and test sets
        from sklearn.preprocessing import StandardScaler as _SS
        from sklearn.linear_model import LogisticRegression as _LR

        # Refit scaler + model on train only (same as fit_game_models did)
        _avail = [f for f in final_features
                  if f in df_train.columns and f in df_val.columns]
        _sc  = _SS().fit(df_train[_avail].values)
        _clf = _LR(C=1.0, max_iter=1000, random_state=42)
        _clf.fit(_sc.transform(df_train[_avail].values),
                 df_train["home_win"].values)

        # Val set evaluation
        if not df_val.empty and "game_date" in df_val.columns:
            _Xv   = _sc.transform(df_val[_avail].values)
            _pv   = _clf.predict_proba(_Xv)[:, 1]
            _bv   = hand_tuned_predict(df_val)
            _yv   = df_val["home_win"].values
            _dv   = df_val["game_date"].values
            print(f"\n  Evaluating {model_tag} on val ({VAL_SEASON}) ...")
            eval_val = evaluate(
                y_true         = _yv,
                probs          = _pv,
                dates          = _dv,
                model_name     = model_tag,
                season         = VAL_SEASON,
                split          = "val",
                baseline_probs = _bv,
                n_boot         = 2_000,
                save           = not args.no_log,
            )
            print_report(eval_val)

        # Test set evaluation — only if test data is available
        if df_test is not None and not df_test.empty and "game_date" in df_test.columns:
            _avail_te = [f for f in final_features if f in df_test.columns]
            if len(_avail_te) == len(final_features):
                _Xte  = _sc.transform(df_test[_avail_te].values)
                _pte  = _clf.predict_proba(_Xte)[:, 1]
                _bte  = hand_tuned_predict(df_test)
                _yte  = df_test["home_win"].values
                _dte  = df_test["game_date"].values
                print(f"\n  Evaluating {model_tag} on test ({TEST_SEASON}) ...")
                eval_test = evaluate(
                    y_true         = _yte,
                    probs          = _pte,
                    dates          = _dte,
                    model_name     = model_tag,
                    season         = TEST_SEASON,
                    split          = "test",
                    baseline_probs = _bte,
                    n_boot         = 2_000,
                    save           = not args.no_log,
                )
                print_report(eval_test)
    except ImportError:
        print("\n  [evaluate] evaluate.py not found — skipping framework report.")
    except Exception:
        import logging as _lg
        _lg.exception("evaluate() failed — skipping")

    _print_next_steps(game_result["recommendation"])


if __name__ == "__main__":
    main()
