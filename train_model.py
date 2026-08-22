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
_orig_get = _req.get
def _patched_get(url, **kwargs):
    kwargs.setdefault("verify", False)
    return _orig_get(url, **kwargs)
_req.get = _patched_get
# ─────────────────────────────────────────────────────────────────────────────

from nba_api.stats.endpoints import leaguegamefinder, leaguegamelog

_HERE       = Path(__file__).parent
CACHE_DIR   = _HERE / "data_cache"
PARAMS_FILE = _HERE / "learned_params.json"
CALIB_FILE  = _HERE / "calibration.json"
CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
TRAIN_CUTOFF    = "2024-25"   # this season is validation/test only

# Current hand-tuned constants (mirrored from newnbapredictor.py)
RTG_SCALE            = 9.5
BASE_HCA             = 2.8
FOUR_FACTORS_WEIGHT  = 0.35
FORM_WEIGHT          = 0.20
H2H_WEIGHT           = 0.08
BACK_TO_BACK_PENALTY = 2.2
REST_DAY_ADVANTAGE   = 0.8

GAME_FEATURES = [
    "net_rtg_diff",   # season-to-date rolling net rating gap (home - away)
    "efg_diff",       # eFG% gap (home off - away off adjusted for each D)
    "tov_diff",       # TOV% edge (positive = home forces more / commits fewer)
    "orb_diff",       # OREB% gap
    "ftr_diff",       # FTR gap
    "form_diff",      # L10 rolling margin gap (home - away)
    "home_b2b",       # 1 = home on back-to-back
    "away_b2b",       # 1 = away on back-to-back
    "rest_diff",      # home rest days - away rest days (capped ±5)
    "h2h_margin",     # historical H2H avg margin (shrinkage-smoothed)
]

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
        except Exception as e:
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
        except Exception as e:
            print(f"  FAILED: {e}")
        _sleep()

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_player_positions(seasons: list, force: bool = False) -> dict:
    """Return {player_id: pos_group} mapping. pos_group in {G, F, C}."""
    from nba_api.stats.endpoints import leaguedashplayerstats

    cache_f = CACHE_DIR / "player_positions.json"
    if cache_f.exists() and not force:
        with open(cache_f) as fh:
            return {int(k): v for k, v in json.load(fh).items()}

    pos_map = {}
    season = seasons[-1]
    print(f"  [fetch] player positions ({season}) ...", end=" ", flush=True)
    try:
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_simple="PerGame",
        ).get_data_frames()[0]
        for _, row in df.iterrows():
            pos_raw = str(row.get("PLAYER_POSITION", "") or "").upper()
            if "C" in pos_raw:
                pg = "C"
            elif "F" in pos_raw:
                pg = "F"
            else:
                pg = "G"
            pos_map[int(row["PLAYER_ID"])] = pg
        print(f"  {len(pos_map)} players")
        with open(cache_f, "w") as fh:
            json.dump({str(k): v for k, v in pos_map.items()}, fh)
    except Exception as e:
        print(f"  FAILED: {e}")
    _sleep()
    return pos_map


# =============================================================================
# SECTION 2 — GAME FEATURE ENGINEERING (walk-forward, no lookahead)
# =============================================================================

def _four_factors_from_row(row: pd.Series) -> dict:
    fga  = max(float(row.get("FGA", 1)), 1)
    fta  = float(row.get("FTA", 0))
    fgm  = float(row.get("FGM", 0))
    fg3m = float(row.get("FG3M", 0))
    tov  = float(row.get("TOV", 0))
    oreb = float(row.get("OREB", 0))
    reb  = max(float(row.get("REB", 1)), 1)
    return {
        "efg":  (fgm + 0.5 * fg3m) / fga,
        "tov":  tov / max(fga + 0.44 * fta + tov, 1),
        "orb":  oreb / reb,
        "ftr":  fta / fga,
        "pm":   float(row.get("PLUS_MINUS", 0)),
        "poss": fga + 0.44 * fta + tov - oreb,   # possessions estimate
    }


def build_game_dataset(team_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward game dataset. For each game G on date D, features are
    computed exclusively from games before D (no lookahead).

    Returns one row per game (home team perspective).
    """
    if team_logs.empty:
        return pd.DataFrame()

    team_logs = team_logs.copy()
    team_logs["GAME_DATE"] = pd.to_datetime(team_logs["GAME_DATE"])
    team_logs["is_home"]   = team_logs["MATCHUP"].str.contains(r"vs\.", na=False)

    # Compute per-game four-factor stats
    for col, fn in [
        ("efg",  lambda r: (float(r["FGM"]) + 0.5*float(r["FG3M"])) / max(float(r["FGA"]),1)),
        ("tov_r",lambda r: float(r["TOV"]) / max(float(r["FGA"]) + 0.44*float(r["FTA"]) + float(r["TOV"]),1)),
        ("orb_r",lambda r: float(r["OREB"]) / max(float(r["REB"]),1)),
        ("ftr",  lambda r: float(r["FTA"]) / max(float(r["FGA"]),1)),
    ]:
        team_logs[col] = team_logs.apply(fn, axis=1)

    # Build rolling pre-game stats per team
    roll_stat_cols = ["PLUS_MINUS", "efg", "tov_r", "orb_r", "ftr"]
    team_stats = {}
    for tid, grp in team_logs.groupby("TEAM_ID"):
        g = grp.sort_values("GAME_DATE").reset_index(drop=True)
        g["prev_date"] = g["GAME_DATE"].shift(1)
        g["rest_days"] = ((g["GAME_DATE"] - g["prev_date"]).dt.days
                          .clip(upper=7).fillna(3).astype(int))
        g["b2b"] = (g["rest_days"] <= 1).astype(int)

        for col in roll_stat_cols:
            g[f"roll_{col}"] = g[col].shift(1).expanding(min_periods=3).mean()

        g["l10_pm"] = g["PLUS_MINUS"].shift(1).rolling(10, min_periods=4).mean()
        g["wpct"]   = (g["WL"] == "W").shift(1).expanding(min_periods=3).mean()

        team_stats[int(tid)] = g.set_index("GAME_ID")

    # Match home and away sides of each game
    records = []
    for game_id, pair in team_logs.groupby("GAME_ID"):
        home = pair[pair["is_home"]]
        away = pair[~pair["is_home"]]
        if home.empty or away.empty:
            continue
        hr = home.iloc[0]
        ar = away.iloc[0]
        htid = int(hr["TEAM_ID"])
        atid = int(ar["TEAM_ID"])

        if htid not in team_stats or atid not in team_stats:
            continue
        if game_id not in team_stats[htid].index or game_id not in team_stats[atid].index:
            continue

        hs = team_stats[htid].loc[game_id]
        as_ = team_stats[atid].loc[game_id]

        # Skip games with insufficient history (first few games of season)
        if any(pd.isna(hs[f"roll_{c}"]) for c in roll_stat_cols):
            continue
        if any(pd.isna(as_[f"roll_{c}"]) for c in roll_stat_cols):
            continue

        h_l10 = hs.get("l10_pm", 0) if not pd.isna(hs.get("l10_pm", np.nan)) else hs["roll_PLUS_MINUS"]
        a_l10 = as_.get("l10_pm", 0) if not pd.isna(as_.get("l10_pm", np.nan)) else as_["roll_PLUS_MINUS"]

        rec = {
            # Features
            "net_rtg_diff": hs["roll_PLUS_MINUS"] - as_["roll_PLUS_MINUS"],
            "efg_diff":     hs["roll_efg"]        - as_["roll_efg"],
            "tov_diff":     as_["roll_tov_r"]     - hs["roll_tov_r"],   # positive = home advantage
            "orb_diff":     hs["roll_orb_r"]      - as_["roll_orb_r"],
            "ftr_diff":     hs["roll_ftr"]         - as_["roll_ftr"],
            "form_diff":    float(h_l10)           - float(a_l10),
            "home_b2b":     int(hs["b2b"]),
            "away_b2b":     int(as_["b2b"]),
            "rest_diff":    float(np.clip(hs["rest_days"] - as_["rest_days"], -5, 5)),
            "h2h_margin":   0.0,   # filled below for within-season H2H
            "home_wpct":    float(hs["wpct"]) if not pd.isna(hs["wpct"]) else 0.5,
            # Targets
            "actual_margin": float(hr["PLUS_MINUS"]),   # home_pts - away_pts
            "home_win":      int(hr["WL"] == "W"),
            # Metadata
            "game_id":   game_id,
            "game_date": hr["GAME_DATE"],
            "season":    hr.get("season", ""),
            "home_tid":  htid,
            "away_tid":  atid,
        }
        records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Fill in within-season H2H margin (shrinkage toward 0, only prior games)
    df = df.sort_values("game_date").reset_index(drop=True)
    h2h_seen: dict = {}
    h2h_margins = []
    for _, row in df.iterrows():
        key = (min(row["home_tid"], row["away_tid"]),
               max(row["home_tid"], row["away_tid"]))
        prior = h2h_seen.get(key, [])
        if prior:
            raw   = np.mean(prior)
            shrink = len(prior) / (len(prior) + 5)
            h2h_val = raw * shrink
        else:
            h2h_val = 0.0
        h2h_margins.append(h2h_val)
        # Update with current game margin (from home_tid's perspective)
        margin_for_key = (row["actual_margin"]
                          if row["home_tid"] == key[1] else -row["actual_margin"])
        h2h_seen.setdefault(key, []).append(margin_for_key)

    df["h2h_margin"] = h2h_margins
    return df


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

def fit_game_models(df_train: pd.DataFrame, df_val: pd.DataFrame) -> dict:
    """
    Fit logistic regression (win prob) + ridge regression (margin).
    Returns dict with model objects, scalers, and evaluation results.
    """
    X_train = df_train[GAME_FEATURES].values
    y_win_tr = df_train["home_win"].values
    y_mgn_tr = df_train["actual_margin"].values

    X_val   = df_val[GAME_FEATURES].values
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

    print(f"\n  === GAME MODEL EVALUATION (val set: {len(df_val)} games) ===")
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
    coef_dict = dict(zip(GAME_FEATURES, log_model.coef_[0].tolist()))
    print("\n  Learned logistic regression coefficients (standardized):")
    for feat, coef in sorted(coef_dict.items(), key=lambda x: -abs(x[1])):
        bar = "+" * int(abs(coef) * 5) if coef > 0 else "-" * int(abs(coef) * 5)
        print(f"    {feat:<18} {coef:>+7.4f}  {bar}")

    return {
        "log_model":           log_model,
        "ridge_model":         ridge_model,
        "scaler":              scaler,
        "feature_names":       GAME_FEATURES,
        "scaler_mean":         scaler.mean_.tolist(),
        "scaler_scale":        scaler.scale_.tolist(),
        "log_intercept":       float(log_model.intercept_[0]),
        "log_coefficients":    {f: float(c) for f, c in coef_dict.items()},
        "ridge_intercept":     float(ridge_model.intercept_),
        "ridge_coefficients":  {f: float(c) for f, c
                                 in zip(GAME_FEATURES, ridge_model.coef_.tolist())},
        "hand_tuned_log_loss": ll_hand,
        "learned_log_loss":    ll_learned,
        "hand_tuned_brier":    bs_hand,
        "learned_brier":       bs_learned,
        "margin_rmse":         rmse,
        "margin_mae":          mae,
        "val_games":           len(df_val),
        "train_games":         len(df_train),
        "recommendation":      recommendation,
    }


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
        pos_group = pos_map.get(int(pid), "G")

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
        tr_pos    = df_players.loc[tr.index, "pos_group"] if hasattr(tr.index, "__iter__") else tr["pos_group"]

        pos_stds = {}
        for pg in POS_GROUPS:
            mask = tr.get("pos_group", pd.Series(dtype=str)) == pg if isinstance(tr, pd.DataFrame) else (tr_pos == pg)
            pg_resid = resid_tr[mask.values if hasattr(mask, "values") else mask]
            pos_stds[pg] = float(np.std(pg_resid)) if len(pg_resid) > 10 else float(np.std(resid_tr))
            variance_results["by_position"][pg][stat] = round(pos_stds[pg], 3)

        league_std = float(np.std(resid_tr))
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
                calib: dict, seasons: list, no_bootstrap: bool) -> None:
    params = {
        "version":    datetime.now().strftime("%Y-%m-%d"),
        "trained_on": [s for s in seasons if s != TRAIN_CUTOFF],
        "validated_on": TRAIN_CUTOFF,
        "game_model": {
            k: v for k, v in game_result.items()
            if k not in ("log_model", "ridge_model", "scaler")
        },
        "player_models": player_result["stat_models"],
        "player_variance": player_result["variance"],
    }

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

def main():
    p = argparse.ArgumentParser(description="NBA model training pipeline")
    p.add_argument("--no-fetch",     action="store_true",
                   help="Use cached CSVs only, do not call NBA API")
    p.add_argument("--force-fetch",  action="store_true",
                   help="Force re-fetch even if cache exists")
    p.add_argument("--eval-only",    action="store_true",
                   help="Load existing learned_params.json and just print metrics")
    p.add_argument("--no-bootstrap", action="store_true",
                   help="Skip calibration.json bootstrap")
    p.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS,
                   help="Seasons to use (e.g. 2022-23 2023-24 2024-25)")
    args = p.parse_args()

    if args.eval_only:
        if not PARAMS_FILE.exists():
            print("learned_params.json not found — run without --eval-only first.")
            sys.exit(1)
        with open(PARAMS_FILE) as fh:
            params = json.load(fh)
        gm = params.get("game_model", {})
        print(f"\nlearned_params.json ({params.get('version', 'unknown')})")
        print(f"  trained on:  {params.get('trained_on', [])}")
        print(f"  val season:  {params.get('validated_on', '?')}")
        print(f"  game model:  {gm.get('recommendation', '?')}")
        print(f"    log_loss   hand={gm.get('hand_tuned_log_loss','?'):.4f} "
              f"learned={gm.get('learned_log_loss','?'):.4f}")
        print(f"    brier      hand={gm.get('hand_tuned_brier','?'):.4f} "
              f"learned={gm.get('learned_brier','?'):.4f}")
        return

    seasons     = args.seasons
    force_fetch = args.force_fetch and not args.no_fetch

    print("=" * 60)
    print("NBA Model Training Pipeline")
    print(f"Seasons: {seasons}")
    print("=" * 60)

    # ── 1. Fetch data ──────────────────────────────────────────────────────
    if not args.no_fetch:
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
        pos_map     = fetch_player_positions(seasons, force=False)

    if team_logs.empty:
        print("No team log data — aborting.")
        sys.exit(1)

    # ── 2. Build game dataset ─────────────────────────────────────────────
    print(f"\n[4/5] Engineering game features ...")
    df_game = build_game_dataset(team_logs)
    print(f"  Game dataset: {len(df_game)} games "
          f"({df_game['home_win'].sum()} home wins = "
          f"{df_game['home_win'].mean():.1%} HWP)")

    if len(df_game) < 500:
        print("  ⚠ Very few games — check fetch or cache.")
        sys.exit(1)

    # Walk-forward split: all seasons except last = train; last = val
    df_train = df_game[df_game["season"] != TRAIN_CUTOFF].copy()
    df_val   = df_game[df_game["season"] == TRAIN_CUTOFF].copy()
    if df_val.empty:
        # If the cutoff season wasn't fetched, use last 20% as validation
        cutoff_idx = int(len(df_game) * 0.80)
        df_train = df_game.iloc[:cutoff_idx].copy()
        df_val   = df_game.iloc[cutoff_idx:].copy()
    print(f"  Train: {len(df_train)} games | Val: {len(df_val)} games")

    # ── 3. Fit game model ─────────────────────────────────────────────────
    game_result = fit_game_models(df_train, df_val)

    # ── 4. Build player dataset & fit player models ───────────────────────
    print(f"\n[5/5] Engineering player features ...")
    df_players = build_player_dataset(player_logs, pos_map)
    print(f"  Player dataset: {len(df_players)} player-game rows")

    if len(df_players) >= 1000:
        player_result = fit_player_models(df_players)
    else:
        print("  ⚠ Too few player rows — skipping player model fit")
        player_result = {"stat_models": {}, "variance": {}}

    # ── 5. Bootstrap calibration ──────────────────────────────────────────
    calib = {}
    if not args.no_bootstrap:
        calib = bootstrap_calibration(df_game, df_players, player_result)

    # ── 6. Save ────────────────────────────────────────────────────────────
    save_params(game_result, player_result, calib, seasons, args.no_bootstrap)

    print("\n  Done. Next steps:")
    rec = game_result["recommendation"]
    if rec == "swap":
        print("  → Learned model outperforms baseline on both metrics.")
        print("    Set USE_LEARNED_GAME_MODEL = True in newnbapredictor.py")
    elif rec == "monitor":
        print("  → Mixed results. Re-run after more games accumulate.")
        print("    Keep USE_LEARNED_GAME_MODEL = False for now.")
    else:
        print("  → Hand-tuned baseline still wins. No swap needed yet.")
        print("    Re-run at midseason or after significant lineup changes.")


if __name__ == "__main__":
    main()
