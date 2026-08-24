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

DEFAULT_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
VAL_SEASON  = "2023-24"   # drives swap/monitor/keep recommendation
TEST_SEASON = "2024-25"   # touched exactly once, after model is already selected

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

# New candidate features — included when --new-signals flag is passed.
# Validated against GAME_FEATURES baseline before wiring in as defaults.
GAME_FEATURES_NEW = [
    "altitude_penalty",   # away team acclimatization deficit at DEN/UTA (0=neutral)
    "venue_residual",     # home team's home-W% minus overall-W% (isolated building effect)
    "star_form_diff",     # PPG-weighted, shrunk star-player EWMA deviation (home − away)
]
GAME_FEATURES_EXTENDED = GAME_FEATURES + GAME_FEATURES_NEW

# Nuggets (1610612743) and Jazz (1610612762) — the only NBA venues above 4 000 ft.
_ALTITUDE_HOME_TIDS: frozenset = frozenset({1610612743, 1610612762})
# Star-form shrinkage constant: games needed before trusting a streak (same n/(n+k) used
# in calibrate.py's player-projection bias).
_STAR_FORM_SHRINK_K: int = 10

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
    except (requests.exceptions.RequestException, json.JSONDecodeError,
            OSError, KeyError, ValueError, TypeError) as e:
        print(f"  FAILED: {e}")
    _sleep()
    return pos_map


# =============================================================================
# SECTION 2 — GAME FEATURE ENGINEERING (walk-forward, no lookahead)
# =============================================================================

# ── New signal helpers ─────────────────────────────────────────────────────────

def altitude_feature(home_tid: int, away_b2b: int, away_rest_days: float) -> float:
    """
    Away-team acclimatization deficit at elevation venues (DEN, UTA only).
    Returns 0.0 for all other home venues.
    Range [0, 1]: 1 = maximally unacclimatized (B2B into altitude); 0 = well-rested.

    Acclimatization is approximated by rest-days only; a proper model would
    also track the previous city's elevation, but that data isn't in the NBA
    game log.  Conservative: teams that arrive 3+ days early are treated as
    fully acclimatized (factor → 0).
    """
    if int(home_tid) not in _ALTITUDE_HOME_TIDS:
        return 0.0
    if away_b2b:
        return 1.0
    # Linear interpolation: 0 rest_days → 1.0, 3+ rest_days → 0.0
    return float(max(0.0, (3.0 - float(away_rest_days)) / 3.0))


def venue_residual_from_record(home_win_count: int, home_game_count: int,
                                total_win_count: int, total_game_count: int) -> float:
    """
    Isolated venue/crowd effect: home win% minus overall win%.
    A positive value means the team wins MORE at home than their overall quality
    predicts; a negative value means they under-perform at home.
    Returns 0.0 when fewer than 3 home or 3 total games have been played.
    """
    if home_game_count < 3 or total_game_count < 3:
        return 0.0
    home_wp  = home_win_count  / home_game_count
    total_wp = total_win_count / total_game_count
    return float(home_wp - total_wp)


def compute_star_form_index(
    player_logs: pd.DataFrame,
    ewma_halflife: float = 5.0,
    top_k: int = 2,
    shrink_k: int = _STAR_FORM_SHRINK_K,
) -> pd.DataFrame:
    """
    Walk-forward star-player form index per (team_id, game_id).

    For each game G, computes: PPG-weighted mean of the top-k players' EWMA
    deviations from their own season average, shrunk by n/(n+k) where n is
    the number of games played before G.

    Returns DataFrame with columns [team_id, game_id, star_form].
    The value is in points-above-baseline units; positive = hot streak.
    All features use only data from games BEFORE game G (no lookahead).
    """
    if player_logs.empty:
        return pd.DataFrame(columns=["team_id", "game_id", "star_form"])

    pl = player_logs.copy()
    pl["GAME_DATE"] = pd.to_datetime(pl["GAME_DATE"])
    pl["PTS"] = pd.to_numeric(pl["PTS"], errors="coerce").fillna(0.0)
    pl["MIN"] = pd.to_numeric(pl["MIN"], errors="coerce").fillna(0.0)

    records = []
    for (team_id, player_id), grp in pl.groupby(["TEAM_ID", "PLAYER_ID"]):
        g = grp.sort_values("GAME_DATE").reset_index(drop=True)
        g = g[g["MIN"] >= 8].reset_index(drop=True)   # skip DNPs
        if len(g) < 5:
            continue

        g["season_avg"]  = g["PTS"].shift(1).expanding(min_periods=3).mean()
        g["ewma_pts"]    = g["PTS"].shift(1).ewm(halflife=ewma_halflife, min_periods=3).mean()
        g["deviation"]   = g["ewma_pts"] - g["season_avg"]
        g["games_before"] = np.arange(len(g))
        g["shrink"]      = g["games_before"] / (g["games_before"] + shrink_k)
        g["shrunk_dev"]  = g["deviation"] * g["shrink"]
        # Player value proxy: expanding PPG average before this game
        g["ppg_value"]   = g["PTS"].shift(1).expanding(min_periods=1).mean().fillna(0)

        for _, row in g.iterrows():
            if pd.isna(row["shrunk_dev"]) or row["ppg_value"] <= 0:
                continue
            records.append({
                "team_id":    int(team_id),
                "player_id":  int(player_id),
                "game_id":    str(row["GAME_ID"]),
                "shrunk_dev": float(row["shrunk_dev"]),
                "ppg_value":  float(row["ppg_value"]),
            })

    if not records:
        return pd.DataFrame(columns=["team_id", "game_id", "star_form"])

    df = pd.DataFrame(records)

    # For each (team, game): weight deviations by PPG value, take top-k contributors
    out_rows = []
    for (team_id, game_id), grp in df.groupby(["team_id", "game_id"]):
        top = grp.nlargest(top_k, "ppg_value")
        total_val = top["ppg_value"].sum()
        if total_val <= 0:
            continue
        weighted_dev = (top["shrunk_dev"] * top["ppg_value"]).sum() / total_val
        out_rows.append({"team_id": team_id, "game_id": game_id, "star_form": weighted_dev})

    return pd.DataFrame(out_rows) if out_rows else pd.DataFrame(
        columns=["team_id", "game_id", "star_form"]
    )


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


def build_game_dataset(team_logs: pd.DataFrame,
                       player_logs: pd.DataFrame = None) -> pd.DataFrame:
    """
    Walk-forward game dataset. For each game G on date D, features are
    computed exclusively from games before D (no lookahead).

    Returns one row per game (home team perspective).
    Optional player_logs enables the star_form_diff feature.
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

        # Walk-forward venue residual: home_win% - overall_win% before this game.
        # Positive = team wins MORE at home than their overall quality predicts.
        g["is_win"]      = (g["WL"] == "W").astype(float)
        g["cum_wins"]    = g["is_win"].shift(1).expanding().sum().fillna(0)
        g["cum_games"]   = pd.Series(np.arange(len(g)), index=g.index).values  # 0,1,2,...
        g["cum_hm_wins"] = (g["is_win"] * g["is_home"].astype(float)).shift(1).expanding().sum().fillna(0)
        g["cum_hm_games"]= g["is_home"].astype(float).shift(1).expanding().sum().fillna(0)
        g["venue_res"]   = g.apply(
            lambda r: venue_residual_from_record(
                int(r["cum_hm_wins"]), int(r["cum_hm_games"]),
                int(r["cum_wins"]),    int(r["cum_games"]),
            ),
            axis=1,
        )

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

        away_rest  = float(as_["rest_days"]) if not pd.isna(as_["rest_days"]) else 3.0
        away_b2b_v = int(as_["b2b"])
        h_venue_res = float(hs.get("venue_res", 0.0)) if not pd.isna(hs.get("venue_res", np.nan)) else 0.0
        a_venue_res = float(as_.get("venue_res", 0.0)) if not pd.isna(as_.get("venue_res", np.nan)) else 0.0

        rec = {
            # Features — original 10
            "net_rtg_diff":    hs["roll_PLUS_MINUS"] - as_["roll_PLUS_MINUS"],
            "efg_diff":        hs["roll_efg"]        - as_["roll_efg"],
            "tov_diff":        as_["roll_tov_r"]     - hs["roll_tov_r"],
            "orb_diff":        hs["roll_orb_r"]      - as_["roll_orb_r"],
            "ftr_diff":        hs["roll_ftr"]         - as_["roll_ftr"],
            "form_diff":       float(h_l10)           - float(a_l10),
            "home_b2b":        int(hs["b2b"]),
            "away_b2b":        int(as_["b2b"]),
            "rest_diff":       float(np.clip(hs["rest_days"] - as_["rest_days"], -5, 5)),
            "h2h_margin":      0.0,   # filled below for within-season H2H
            "home_wpct":       float(hs["wpct"]) if not pd.isna(hs["wpct"]) else 0.5,
            # New signal 1: altitude
            "altitude_penalty": altitude_feature(htid, away_b2b_v, away_rest),
            # New signal 2: residualized venue effect
            "venue_residual":  h_venue_res - a_venue_res,
            # New signal 3: star form (filled after player_logs merge)
            "star_form_diff":  0.0,
            # Targets
            "actual_margin":   float(hr["PLUS_MINUS"]),
            "home_win":        int(hr["WL"] == "W"),
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

    # Signal 3: merge star form from player logs (walk-forward, pre-computed)
    if player_logs is not None and not player_logs.empty:
        sf = compute_star_form_index(player_logs)
        if not sf.empty:
            sf = sf.rename(columns={"star_form": "_h_star"})
            sf["game_id"] = sf["game_id"].astype(str)
            df["game_id"] = df["game_id"].astype(str)

            # Home team star form
            df = df.merge(
                sf[["team_id", "game_id", "_h_star"]],
                left_on=["home_tid", "game_id"],
                right_on=["team_id", "game_id"],
                how="left",
            ).drop(columns="team_id", errors="ignore")

            # Away team star form
            sf2 = sf.rename(columns={"_h_star": "_a_star"})
            df = df.merge(
                sf2[["team_id", "game_id", "_a_star"]],
                left_on=["away_tid", "game_id"],
                right_on=["team_id", "game_id"],
                how="left",
            ).drop(columns="team_id", errors="ignore")

            df["star_form_diff"] = (
                df["_h_star"].fillna(0.0) - df["_a_star"].fillna(0.0)
            )
            df = df.drop(columns=["_h_star", "_a_star"], errors="ignore")

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
    p.add_argument("--ablation",     action="store_true",
                   help="Run leave-one-out ablation over GAME_FEATURES and write "
                        "ablation_results.json (adds ~10 extra model fits)")
    p.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS,
                   help="Seasons to use (e.g. 2021-22 2022-23 2023-24 2024-25)")
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
    df_game = build_game_dataset(team_logs, player_logs=player_logs)
    print(f"  Game dataset: {len(df_game)} games "
          f"({df_game['home_win'].sum()} home wins = "
          f"{df_game['home_win'].mean():.1%} HWP)")

    if len(df_game) < 500:
        print("  ⚠ Very few games — check fetch or cache.")
        sys.exit(1)

    # ── Three-way walk-forward split ──────────────────────────────────────────
    # df_train   — everything excluding both holdout seasons
    # df_val     — VAL_SEASON (drives recommendation / ablation)
    # df_test    — TEST_SEASON (scored exactly once, after recommendation is made)
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

    # ── 3a. Ablation study — determines which new features (if any) pass ─────
    ablation_res = ablation_study(df_train, df_val)
    winning_new   = ablation_res.get("winning_features", [])
    final_features = GAME_FEATURES + winning_new
    if winning_new:
        print(f"\n  Production feature set: {len(final_features)} features "
              f"({len(GAME_FEATURES)} base + {len(winning_new)} new: {winning_new})")
    else:
        print(f"\n  Production feature set: {len(final_features)} features (base only)")

    # ── 3b. Fit production model; df_test is scored inside but never used for
    #        model selection (recommendation is already printed before df_test
    #        numbers appear — see fit_game_models for the ordering guarantee).
    game_result = fit_game_models(df_train, df_val, df_test=df_test,
                                  features=final_features)

    # ── 3c. Optional leave-one-out ablation over GAME_FEATURES ───────────────
    if args.ablation:
        print("\n  Running game-feature ablation study ...")
        run_ablation(df_train, df_val, final_features)

    # Compute season-end venue residuals per team for live-predictor loading
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
    save_params(game_result, player_result, calib, seasons, args.no_bootstrap,
                venue_residuals=venue_residuals, ablation_result=ablation_res)

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
