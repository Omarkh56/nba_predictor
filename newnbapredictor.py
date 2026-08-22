"""
NBA Betting Intel v4 — Complete Multi-Factor Prediction Engine
================================================================
10 factors, weighted by predictive power:

  1. Net Rating (base)           — Foundation of the spread
  2. Injuries & Lineup           — Who's actually playing
  3. Four Factors                — eFG%, TOV%, OREB%, FT rate
  4. Matchup-Specific            — Stylistic mismatches (3PT D vs 3PT O, size, pace)
  5. Rest & Schedule             — B2B, road trip length, days off
  6. Recent Form (L10)           — Rolling net rating captures current team state
  7. Home Court Advantage        — ~3 pts, venue-adjusted
  8. Off/Def Ratings             — The "how" behind net rating, informs matchup analysis
  9. Pace                        — Shapes variance; high pace favors talent
  10. Clutch Performance         — Close-game execution for projected tight matchups

Install:  pip install nba_api pandas requests urllib3
Run:      python3 bettingnba_v4.py
"""

import logging
import math
import time
import requests
import warnings
import urllib3
from datetime import date, timedelta, datetime
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)

import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashteamstats,
    leaguedashteamclutch,
    leaguegamefinder,
    leaguedashplayerstats,
)

warnings.filterwarnings("ignore")
urllib3.disable_warnings()

# ── Patch nba_api ─────────────────────────────────────────────────────────────
import nba_api.stats.library.http as _nba_http
_nba_http.STATS_HEADERS.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
    "DNT": "1",
})
_nba_http.STATS_TIMEOUT = 90

# ─────────────────────────────────────────────────────────────────────────────
# TLS NOTE: verify=False was previously set globally here as a workaround for
# macOS LibreSSL 2.8.3 / Python 3.9 cert issues.  The correct fix is to ensure
# the 'certifi' package is installed and up-to-date; requests uses it
# automatically.  If you see SSLError on ESPN calls, run:
#   pip install --upgrade certifi
# ─────────────────────────────────────────────────────────────────────────────

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_INJURIES_URL   = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
ESPN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
}

ESPN_ABBR_TO_NAME = {
    "ATL": "Atlanta Hawks",           "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",           "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",           "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",        "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",         "GSW": "Golden State Warriors",
    "GS":  "Golden State Warriors",   "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",          "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",      "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",              "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",  "NOP": "New Orleans Pelicans",
    "NO":  "New Orleans Pelicans",    "NYK": "New York Knicks",
    "NY":  "New York Knicks",         "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",           "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",            "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",        "SAS": "San Antonio Spurs",
    "SA":  "San Antonio Spurs",       "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",               "UTAH": "Utah Jazz",
    "WAS": "Washington Wizards",      "WSH": "Washington Wizards",
}

NAME_TO_ESPN_ABBR = {}
for _a, _n in ESPN_ABBR_TO_NAME.items():
    if _n not in NAME_TO_ESPN_ABBR:
        NAME_TO_ESPN_ABBR[_n] = _a


# =============================================================================
# MODEL CONSTANTS
# =============================================================================

# Sigmoid scale — calibrated to NBA historical ATS data
# 3-pt spread ≈ 58%, 7-pt ≈ 70%, 10-pt ≈ 76%, 15-pt ≈ 85%
RTG_SCALE = 9.5

# ─── Factor weights (sum to 1.0 conceptually, but applied as pts) ───────────
# These control how much each factor can shift the margin from the base.

# Factor 1: NET RATING — not "weighted" separately, it IS the base margin.
# Everything else adjusts relative to it.

# Factor 2: INJURIESºº
INJURY_REPLACEMENT_FACTOR = 0.55   # how much of a player's value is truly lost
INJURY_MISS_PROB = {
    "Out":          1.00,
    "Doubtful":     0.85,
    "Questionable": 0.50,
    "Day-To-Day":   0.40,
    "Probable":     0.15,
}
DEFAULT_PLAYER_VALUE = 1.5

# Factor 3: FOUR FACTORS matchup edge
# When one team dominates 3+ of the four factors vs their opponent,
# it's a strong signal. We measure the gap in each factor and convert
# to a margin adjustment.
FOUR_FACTORS_WEIGHT = 0.35  # max ~3-4 pts swing for extreme mismatches

# Factor 4: MATCHUP-SPECIFIC (3PT offense vs 3PT defense, pace mismatch)
MATCHUP_WEIGHT = 0.25       # max ~2-3 pts swing

# Factor 5: REST & SCHEDULE
BACK_TO_BACK_PENALTY = 2.2  # pts
REST_DAY_ADVANTAGE   = 0.8  # per day (capped ±2)
ROAD_TRIP_PEN_PER_GAME = 0.15  # per consecutive road game beyond 3

# Factor 6: RECENT FORM (L10 rolling net rating)
FORM_GAMES  = 10
FORM_WEIGHT = 0.20  # how much L10 form shifts margin vs season baseline

# Factor 7: HOME COURT ADVANTAGE
# Base HCA + venue intensity factor (some arenas are louder/harder)
BASE_HCA = 2.8
# Teams with extreme home W% (>65%) get a small boost; <50% get a reduction
HCA_VENUE_SCALE = 2.0  # pts range for venue adjustment

# Factor 8: OFF/DEF RATINGS — used within matchup analysis, not standalone

# Factor 9: PACE — variance adjustment
# High-pace games favor the better team (more possessions = less variance)
# Low-pace games create more upsets
PACE_VARIANCE_SCALE = 0.03  # pts per pace unit above/below league avg

# Factor 10: CLUTCH
CLUTCH_WEIGHT = 0.12  # only applied when projected margin is close (< 5 pts)

# H2H (small but real)
H2H_WEIGHT      = 0.08
H2H_SHRINKAGE_K = 5

# =============================================================================
# LEARNED MODEL LOADING (from train_model.py output)
# =============================================================================
# Set True only after reviewing backtest results from train_model.py.
# The flag is intentionally conservative — keep False until you've confirmed
# the learned model beats the baseline on both log loss AND Brier score.
USE_LEARNED_GAME_MODEL = False

_LEARNED_GAME_PARAMS: dict = {}


def _load_learned_params() -> None:
    """Load learned_params.json once at startup if it exists."""
    global _LEARNED_GAME_PARAMS, RTG_SCALE
    params_file = Path(__file__).parent / "learned_params.json"
    if not params_file.exists():
        return
    try:
        with open(params_file) as fh:
            data = json.load(fh)
        gm = data.get("game_model", {})
        if gm:
            _LEARNED_GAME_PARAMS = gm
            rec = gm.get("recommendation", "keep_hand_tuned")
            ll_h = gm.get("hand_tuned_log_loss", "?")
            ll_l = gm.get("learned_log_loss", "?")
            print(f"  [train_model] game params loaded  "
                  f"(rec={rec}  ll_hand={ll_h:.4f}  ll_learned={ll_l:.4f})")
    except Exception as e:
        print(f"  [train_model] failed to load learned_params.json: {e}")


def _sigmoid_learned(features: dict) -> float:
    """
    Compute win probability using the logistic regression from train_model.py.
    features: dict keyed by GAME_FEATURES names.
    Falls back to hand-tuned sigmoid if params are missing.
    """
    gm = _LEARNED_GAME_PARAMS
    if not gm or not USE_LEARNED_GAME_MODEL:
        return None  # signal: use hand-tuned path

    coefs  = gm.get("log_coefficients", {})
    means  = gm.get("scaler_mean", [])
    scales = gm.get("scaler_scale", [])
    feats  = gm.get("feature_names", [])
    intercept = gm.get("log_intercept", 0.0)

    if not (coefs and means and scales and feats):
        return None

    logit = intercept
    for i, feat in enumerate(feats):
        raw = features.get(feat, 0.0)
        if i < len(means) and i < len(scales) and scales[i] > 0:
            scaled = (raw - means[i]) / scales[i]
        else:
            scaled = raw
        logit += coefs.get(feat, 0.0) * scaled

    return 1.0 / (1.0 + math.exp(-logit))


# Run once at import time
import json
from pathlib import Path
_load_learned_params()


# =============================================================================
# API HELPERS
# =============================================================================
def _api_call(fn, retries=3, base_delay=5.0):
    for attempt in range(retries):
        try:
            result = fn()
            time.sleep(2.0)
            return result
        except Exception as e:
            wait = base_delay * (2 ** attempt)
            print(f"    Retry {attempt+1}/{retries} in {wait:.0f}s… ({type(e).__name__})")
            time.sleep(wait)
    raise RuntimeError("NBA API call failed after retries.")

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x / RTG_SCALE))


# =============================================================================
# ESPN — GAMES & INJURIES
# =============================================================================
def get_todays_games() -> List[Dict]:
    for days_ahead in range(-1, 8):
        target = date.today() + timedelta(days=days_ahead)
        date_str = target.strftime("%Y%m%d")
        try:
            resp = requests.get(f"{ESPN_SCOREBOARD_URL}?dates={date_str}",
                                headers=ESPN_HEADERS, timeout=12)
            resp.raise_for_status()
            events = resp.json().get("events", [])
        except requests.RequestException as e:
            print(f"  ESPN error: {e}")
            return []
        if not events:
            continue
        matchups = []
        for event in events:
            try:
                comp = event["competitions"][0]
                status_type = event["status"]["type"]
                status_txt = status_type.get("shortDetail", "")
                state = status_type.get("state", "pre")
                if state == "post":
                    continue
                competitors = comp["competitors"]
                home = next(c for c in competitors if c["homeAway"] == "home")
                away = next(c for c in competitors if c["homeAway"] == "away")
                ha = home["team"]["abbreviation"]
                aa = away["team"]["abbreviation"]
                matchups.append({
                    "away_abbr": aa,
                    "away_name": ESPN_ABBR_TO_NAME.get(aa, away["team"]["displayName"]),
                    "home_abbr": ha,
                    "home_name": ESPN_ABBR_TO_NAME.get(ha, home["team"]["displayName"]),
                    "status": status_txt,
                    "game_date": target,
                })
            except (KeyError, StopIteration):
                continue
        if matchups:
            return matchups
    return []


def get_espn_injuries() -> Dict[str, List[Dict]]:
    try:
        r = requests.get(ESPN_INJURIES_URL, headers=ESPN_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ESPN injuries error: {e}")
        return {}

    # Build a rough name→PPG lookup from a known-good source so injury
    # weighting works even though ESPN's injury endpoint has no stats.
    # We use position-tier defaults: stars (~25+ PPG), starters (~15 PPG),
    # rotation (~8 PPG).  Actual PPG is fetched via nba_api later if available.
    injuries_by_team = {}
    for team_entry in data.get("injuries", []):
        # Bug fix: ESPN top-level key is "displayName", not nested under "team"
        team_name = team_entry.get("displayName", "") or \
                    team_entry.get("team", {}).get("displayName", "")
        if not team_name:
            continue
        team_injuries = []
        for inj in team_entry.get("injuries", []):
            athlete = inj.get("athlete", {})
            # ESPN injury endpoint does NOT include statistics —
            # set ppg=0 and let _player_margin_value use DEFAULT_PLAYER_VALUE.
            ppg = 0.0
            team_injuries.append({
                "player": athlete.get("displayName", ""),
                "status": inj.get("status", ""),
                "position": athlete.get("position", {}).get("abbreviation", ""),
                "ppg": ppg,
            })
        if team_injuries:
            injuries_by_team[team_name] = team_injuries
    return injuries_by_team


def _enrich_injuries_with_ppg(injuries: dict, season: str) -> dict:
    """
    ESPN's injury endpoint has no player stats.
    Fetch per-game PPG from nba_api for every team that has injuries today,
    then stamp each injury entry with the player's real PPG.
    One API call per injured team (already filtered to only today's games).
    """
    enriched = {}
    for team_name, inj_list in injuries.items():
        enriched[team_name] = list(inj_list)   # copy

    # Build player_name → PPG lookup with a single league-wide call
    try:
        df = _api_call(lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
        ).get_data_frames()[0])
        ppg_lookup = {row["PLAYER_NAME"]: float(row["PTS"]) for _, row in df.iterrows()}
    except Exception as exc:
        logger.warning("PPG enrichment fetch failed, using defaults: %s", exc)
        return enriched

    for team_name, inj_list in enriched.items():
        for entry in inj_list:
            name = entry.get("player", "")
            if name in ppg_lookup:
                entry["ppg"] = ppg_lookup[name]

    return enriched


# =============================================================================
# NBA API — FULL DATA PULL
# =============================================================================
def _fetch_all_team_data(season: str) -> pd.DataFrame:
    """
    Fetches all data needed in as few API calls as possible:
      1. Advanced stats: NET_RATING, OFF_RATING, DEF_RATING, PACE
      2. Base stats: PTS, W, L, W_PCT
      3. Four Factors: EFG_PCT, TOV_PCT, OREB_PCT, FTA_RATE (off & def)
      4. Clutch stats: NET_RATING in clutch situations
      5. Opponent stats: OPP_FG3_PCT, etc. for matchup analysis
    """
    # ─── Call 1: Advanced (overall) ──────────────────────────────────────
    print("  [1/6] Advanced ratings…")
    adv = _api_call(lambda: leaguedashteamstats.LeagueDashTeamStats(
        season=season, measure_type_detailed_defense="Advanced",
    ).get_data_frames()[0])
    # FTA_RATE is NOT in Advanced — it lives in Four Factors.
    # We compute it from base stats (FTA / FGA) which is its exact definition.
    adv = adv[["TEAM_ID", "TEAM_NAME", "GP", "W", "L", "W_PCT",
               "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE",
               "EFG_PCT", "TM_TOV_PCT", "OREB_PCT"]].copy()

    # ─── Call 2: Base stats (PTS for PPG) ────────────────────────────────
    print("  [2/6] Base stats…")
    base = _api_call(lambda: leaguedashteamstats.LeagueDashTeamStats(
        season=season, measure_type_detailed_defense="Base",
    ).get_data_frames()[0])
    base = base[["TEAM_ID", "PTS", "FG3M", "FG3A", "FG3_PCT", "FTA", "FGA"]].copy()
    base = base.rename(columns={"PTS": "TOTAL_PTS", "FG3M": "FG3M_TOTAL",
                                 "FG3A": "FG3A_TOTAL", "FG3_PCT": "FG3_PCT_OFF"})
    # FTA_RATE = free throw attempts per field goal attempt (Four Factors definition)
    base["FTA_RATE"] = (base["FTA"] / base["FGA"].clip(lower=1)).round(4)

    # ─── Call 3a: Four Factors (OPP_EFG_PCT / TOV_PCT / OREB_PCT / FTA_RATE) ─
    # These metrics ONLY exist in the "Four Factors" measure type, not "Opponent".
    print("  [3/6] Opponent stats (Four Factors + raw)…")
    try:
        opp_ff = _api_call(lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed_defense="Four Factors",
        ).get_data_frames()[0])
        opp_ff = opp_ff[["TEAM_ID", "OPP_EFG_PCT", "OPP_TOV_PCT",
                          "OPP_OREB_PCT", "OPP_FTA_RATE"]].copy()
    except Exception as e:
        print(f"    ⚠ Four Factors unavailable ({e}), using defaults")
        opp_ff = pd.DataFrame({"TEAM_ID": adv["TEAM_ID"],
                                "OPP_EFG_PCT": 0.50, "OPP_TOV_PCT": 0.14,
                                "OPP_OREB_PCT": 0.27, "OPP_FTA_RATE": 0.24})

    # ─── Call 3b: Opponent raw (OPP_FG3_PCT / OPP_PTS) ──────────────────
    try:
        opp_raw = _api_call(lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed_defense="Opponent",
        ).get_data_frames()[0])
        opp_raw = opp_raw[["TEAM_ID", "OPP_FG3_PCT", "OPP_PTS"]].copy()
    except Exception as e:
        print(f"    ⚠ Opponent raw stats unavailable ({e}), using defaults")
        opp_raw = pd.DataFrame({"TEAM_ID": adv["TEAM_ID"],
                                 "OPP_FG3_PCT": 0.36, "OPP_PTS": 0.0})

    opp = opp_ff.merge(opp_raw, on="TEAM_ID", how="left")

    # ─── Call 4: Home W% ────────────────────────────────────────────────
    print("  [4/6] Home win %…")
    home_df = _api_call(lambda: leaguedashteamstats.LeagueDashTeamStats(
        season=season, location_nullable="Home",
    ).get_data_frames()[0])[["TEAM_ID", "W_PCT", "GP"]].copy()
    home_df = home_df.rename(columns={"W_PCT": "HOME_W_PCT", "GP": "HOME_GP"})

    # ─── Call 5: Clutch stats (last 5 min, ≤5 pt game) ──────────────────
    # Uses LeagueDashTeamClutch — the correct endpoint for this filter.
    # LeagueDashTeamStats does NOT have clutch_time/ahead_behind params.
    print("  [5/6] Clutch stats…")
    try:
        clutch = _api_call(lambda: leaguedashteamclutch.LeagueDashTeamClutch(
            season=season,
            measure_type_detailed_defense="Advanced",
            clutch_time="Last 5 Minutes",
            ahead_behind="Ahead or Behind",
            point_diff="5",
        ).get_data_frames()[0])
        clutch = clutch[["TEAM_ID", "NET_RATING", "W_PCT", "GP"]].copy()
        clutch = clutch.rename(columns={"NET_RATING": "CLUTCH_NET_RTG",
                                         "W_PCT": "CLUTCH_W_PCT",
                                         "GP": "CLUTCH_GP"})
    except Exception:
        print("    ⚠ Clutch stats unavailable, using defaults")
        clutch = pd.DataFrame({"TEAM_ID": adv["TEAM_ID"],
                                "CLUTCH_NET_RTG": 0.0,
                                "CLUTCH_W_PCT": 0.5,
                                "CLUTCH_GP": 0})

    # ─── Call 6: Road stats (for road trip estimation) ───────────────────
    print("  [6/6] Road win %…")
    road_df = _api_call(lambda: leaguedashteamstats.LeagueDashTeamStats(
        season=season, location_nullable="Road",
    ).get_data_frames()[0])[["TEAM_ID", "W_PCT"]].copy()
    road_df = road_df.rename(columns={"W_PCT": "ROAD_W_PCT"})

    # ─── Merge everything ────────────────────────────────────────────────
    df = (adv.merge(base, on="TEAM_ID", how="left")
             .merge(opp, on="TEAM_ID", how="left")
             .merge(home_df, on="TEAM_ID", how="left")
             .merge(clutch, on="TEAM_ID", how="left")
             .merge(road_df, on="TEAM_ID", how="left"))

    df["PPG"] = df["TOTAL_PTS"] / df["GP"].clip(lower=1)
    df["FG3A_PG"] = df["FG3A_TOTAL"] / df["GP"].clip(lower=1)
    df["FG3M_PG"] = df["FG3M_TOTAL"] / df["GP"].clip(lower=1)

    # Fill NaNs
    df["HOME_W_PCT"]    = df["HOME_W_PCT"].fillna(df["W_PCT"])
    df["ROAD_W_PCT"]    = df["ROAD_W_PCT"].fillna(df["W_PCT"])
    df["CLUTCH_NET_RTG"] = df["CLUTCH_NET_RTG"].fillna(0.0)
    df["CLUTCH_W_PCT"]  = df["CLUTCH_W_PCT"].fillna(0.5)
    df["CLUTCH_GP"]     = df["CLUTCH_GP"].fillna(0)

    return df.set_index("TEAM_ID")


def _fetch_last_n(team_id: int, season: str, all_ratings: pd.DataFrame,
                  n: int = FORM_GAMES) -> dict:
    """L10 SOS-adjusted form + rest + road trip detection."""
    df = _api_call(lambda: leaguegamefinder.LeagueGameFinder(
        team_id_nullable=team_id, season_nullable=season,
        player_or_team_abbreviation="T",
    ).get_data_frames()[0]).head(max(n, 15))  # grab 15 to detect road trips

    if df.empty:
        return {"w_pct": 0.5, "adj_margin": 0.0, "l10_net_rtg": 0.0,
                "days_rest": 3, "b2b": False, "consec_road": 0}

    league_avg_net = all_ratings["NET_RATING"].mean()
    adj_margins = []

    for _, row in df.head(n).iterrows():
        raw = float(row["PLUS_MINUS"])
        opp_abbr = row["MATCHUP"].split()[-1]
        opp_net = league_avg_net
        for _, r in all_ratings.iterrows():
            if opp_abbr.lower() in r["TEAM_NAME"].lower():
                opp_net = r["NET_RATING"]
                break
        sos_adj = (opp_net - league_avg_net) / 4.0
        adj_margins.append(raw + sos_adj)

    wins = (df.head(n)["WL"] == "W").sum()
    last_date = datetime.strptime(df.iloc[0]["GAME_DATE"], "%Y-%m-%d")
    days_rest = (datetime.today() - last_date).days

    # Detect consecutive road games (for road trip fatigue)
    consec_road = 0
    for _, row in df.iterrows():
        if "@" in str(row["MATCHUP"]):
            consec_road += 1
        else:
            break

    # L10 rolling net rating (simple: avg margin / avg possessions proxy)
    # We use adj_margin as a proxy for rolling net rating
    l10_net_rtg = float(pd.Series(adj_margins).mean())

    return {
        "w_pct":       wins / len(df.head(n)),
        "adj_margin":  l10_net_rtg,
        "l10_net_rtg": l10_net_rtg,
        "days_rest":   days_rest,
        "b2b":         days_rest <= 1,
        "consec_road": consec_road,
    }


_H2H_EMPTY = {"team1_wins": 0, "total": 0, "avg_margin": 0.0, "shrinkage": 0.0}

def _fetch_h2h(t1_id: int, t2_id: int, season: str) -> dict:
    try:
        df = _api_call(lambda: leaguegamefinder.LeagueGameFinder(
            team_id_nullable=t1_id, vs_team_id_nullable=t2_id,
            season_nullable=season, player_or_team_abbreviation="T",
        ).get_data_frames()[0])
    except Exception as exc:
        logger.debug("H2H fetch failed for %d vs %d: %s", t1_id, t2_id, exc)
        return _H2H_EMPTY
    if df.empty:
        return _H2H_EMPTY
    n = len(df)
    return {
        "team1_wins": int((df["WL"] == "W").sum()),
        "total": n,
        "avg_margin": float(df["PLUS_MINUS"].mean()),
        "shrinkage": n / (n + H2H_SHRINKAGE_K),
    }


# =============================================================================
# FACTOR 2: INJURY ADJUSTMENT
# =============================================================================
def _player_margin_value(inj: dict, team_ppg: float) -> float:
    ppg = inj.get("ppg", 0)
    if ppg <= 0:
        return DEFAULT_PLAYER_VALUE
    if team_ppg <= 0:
        team_ppg = 112.0
    share = ppg / team_ppg
    return min(share * 20.0, 7.0)


def compute_injury_adj(team_name: str, injuries: dict, team_ppg: float) -> Tuple[float, list]:
    team_inj = injuries.get(team_name, [])
    if not team_inj:
        return 0.0, []
    total = 0.0
    details = []
    for inj in team_inj:
        miss_p = INJURY_MISS_PROB.get(inj.get("status", ""), 0.3)
        if miss_p <= 0:
            continue
        val = _player_margin_value(inj, team_ppg)
        loss = val * miss_p * INJURY_REPLACEMENT_FACTOR
        total -= loss
        details.append({
            "player": inj["player"], "status": inj["status"],
            "miss_prob": miss_p, "value": round(val, 2),
            "adj": round(-loss, 2), "ppg": inj.get("ppg", 0),
        })
    details.sort(key=lambda d: d["adj"])
    return round(total, 2), details


# =============================================================================
# FACTOR 3: FOUR FACTORS MATCHUP
# =============================================================================
def compute_four_factors_edge(home_row, away_row, league_avgs: dict) -> Tuple[float, dict]:
    """
    Compare each team's four factors vs the opponent's corresponding defense.
    Returns a margin adjustment and breakdown.

    The four factors (Dean Oliver):
      1. eFG% — shooting efficiency
      2. TOV% — turnover rate (lower is better for offense)
      3. OREB% — offensive rebounding rate
      4. FTA rate — free throw attempt rate

    We compare: team's offensive factor vs opponent's defensive allowance.
    A team with 55% eFG facing a defense that allows 48% eFG has a +7% edge.
    """
    factors = {}
    home_edge_count = 0

    # eFG%: home offense vs away defense, and vice versa
    h_efg_off = home_row.get("EFG_PCT", 0.50)
    a_efg_def = away_row.get("OPP_EFG_PCT", 0.50)
    a_efg_off = away_row.get("EFG_PCT", 0.50)
    h_efg_def = home_row.get("OPP_EFG_PCT", 0.50)

    efg_home_edge = (h_efg_off - a_efg_def) - (a_efg_off - h_efg_def)
    factors["eFG%"] = {"home_off": h_efg_off, "away_def": a_efg_def,
                       "away_off": a_efg_off, "home_def": h_efg_def,
                       "edge": round(efg_home_edge, 4)}
    if efg_home_edge > 0.005:
        home_edge_count += 1

    # TOV%: lower is better for offense, higher opponent TOV% is better for defense
    h_tov_off = home_row.get("TM_TOV_PCT", 0.14)
    a_tov_def = away_row.get("OPP_TOV_PCT", 0.14)
    a_tov_off = away_row.get("TM_TOV_PCT", 0.14)
    h_tov_def = home_row.get("OPP_TOV_PCT", 0.14)

    # Home advantage = home forces more TOVs than away forces, AND home commits fewer
    tov_home_edge = (h_tov_def - a_tov_def) + (a_tov_off - h_tov_off)
    factors["TOV%"] = {"home_off": h_tov_off, "home_forces": h_tov_def,
                       "away_off": a_tov_off, "away_forces": a_tov_def,
                       "edge": round(tov_home_edge, 4)}
    if tov_home_edge > 0.005:
        home_edge_count += 1

    # OREB%
    h_oreb = home_row.get("OREB_PCT", 0.27)
    a_oreb_def = away_row.get("OPP_OREB_PCT", 0.27)
    a_oreb = away_row.get("OREB_PCT", 0.27)
    h_oreb_def = home_row.get("OPP_OREB_PCT", 0.27)

    oreb_home_edge = (h_oreb - a_oreb_def) - (a_oreb - h_oreb_def)
    factors["OREB%"] = {"home_off": h_oreb, "away_off": a_oreb,
                        "edge": round(oreb_home_edge, 4)}
    if oreb_home_edge > 0.005:
        home_edge_count += 1

    # FTA Rate
    h_fta = home_row.get("FTA_RATE", 0.24)
    a_fta_def = away_row.get("OPP_FTA_RATE", 0.24)
    a_fta = away_row.get("FTA_RATE", 0.24)
    h_fta_def = home_row.get("OPP_FTA_RATE", 0.24)

    fta_home_edge = (h_fta - a_fta_def) - (a_fta - h_fta_def)
    factors["FTA_RATE"] = {"home_off": h_fta, "away_off": a_fta,
                           "edge": round(fta_home_edge, 4)}
    if fta_home_edge > 0.005:
        home_edge_count += 1

    # Convert edges to margin points
    # eFG% is the most impactful (1% eFG ≈ 1.2 pts per game)
    # TOV% is second (1% TOV ≈ 0.8 pts)
    # OREB% and FTA rate are smaller
    margin_adj = (
        efg_home_edge * 120.0    # 1% = 1.2 pts
        + tov_home_edge * 80.0   # 1% = 0.8 pts
        + oreb_home_edge * 50.0  # 1% = 0.5 pts
        + fta_home_edge * 30.0   # 1% = 0.3 pts
    ) * FOUR_FACTORS_WEIGHT

    factors["home_factors_won"] = home_edge_count
    factors["margin_adj"] = round(margin_adj, 2)

    return margin_adj, factors


# =============================================================================
# FACTOR 4: MATCHUP-SPECIFIC (3PT, PACE, SIZE)
# =============================================================================
def compute_matchup_edge(home_row, away_row, league_avgs: dict) -> Tuple[float, dict]:
    """
    Stylistic matchup analysis:
    - 3PT offense vs opponent's 3PT defense
    - Pace mismatch (fast team vs slow team — who benefits?)
    """
    details = {}
    total_adj = 0.0

    # ─── 3PT matchup ────────────────────────────────────────────────────
    # A team that shoots a lot of 3s vs a team that defends the 3 poorly
    h_fg3a_pg = home_row.get("FG3A_PG", 35)
    a_opp_fg3_pct = away_row.get("OPP_FG3_PCT", 0.36)
    a_fg3a_pg = away_row.get("FG3A_PG", 35)
    h_opp_fg3_pct = home_row.get("OPP_FG3_PCT", 0.36)

    league_fg3_pct = league_avgs.get("FG3_PCT", 0.36)
    league_fg3a = league_avgs.get("FG3A_PG", 35)

    # Home 3PT edge: how many extra 3PT pts home gets vs away's D
    h_3pt_edge_pct = a_opp_fg3_pct - league_fg3_pct  # positive = away D allows more 3s
    a_3pt_edge_pct = h_opp_fg3_pct - league_fg3_pct

    # Scale by attempt volume (a team shooting 40 3PA benefits more from bad 3PT D)
    h_3pt_boost = h_3pt_edge_pct * h_fg3a_pg * 3.0  # 3 pts per made 3
    a_3pt_boost = a_3pt_edge_pct * a_fg3a_pg * 3.0

    three_pt_adj = (h_3pt_boost - a_3pt_boost) * MATCHUP_WEIGHT
    details["3PT_matchup"] = round(three_pt_adj, 2)
    total_adj += three_pt_adj

    # ─── Pace mismatch ──────────────────────────────────────────────────
    # When two teams with very different paces meet, the game pace
    # tends toward the average. The team playing out of their comfort
    # zone is at a disadvantage.
    h_pace = home_row.get("PACE", 100)
    a_pace = away_row.get("PACE", 100)
    league_pace = league_avgs.get("PACE", 100)
    expected_pace = (h_pace + a_pace) / 2.0
    pace_diff = abs(h_pace - a_pace)

    # The team forced to play at an unusual pace is disadvantaged
    # Home teams control pace more (home court), so away team deviates more
    h_pace_dev = abs(expected_pace - h_pace) * 0.4  # home controls more
    a_pace_dev = abs(expected_pace - a_pace) * 0.6  # away forced to adjust

    pace_adj = (a_pace_dev - h_pace_dev) * PACE_VARIANCE_SCALE
    details["pace_mismatch"] = round(pace_adj, 2)
    total_adj += pace_adj

    details["total"] = round(total_adj, 2)
    return total_adj, details


# =============================================================================
# FACTOR 9: PACE-TALENT INTERACTION
# =============================================================================
def compute_pace_variance_adj(home_row, away_row, base_margin: float,
                               league_pace: float) -> float:
    """
    High-pace games favor the better team (more possessions = more signal).
    Low-pace games create more variance = more upsets.

    If the better team is playing in a high-pace game, their edge is amplified.
    If it's a grind-it-out game, the edge is compressed.
    """
    expected_pace = (home_row.get("PACE", 100) + away_row.get("PACE", 100)) / 2.0
    pace_deviation = expected_pace - league_pace  # positive = faster than average

    # Amplify or compress the base margin
    # +5 pace → 5 * 0.03 = 0.15 → multiply margin by 1.15 (15% more of edge)
    # -5 pace → -0.15 → multiply margin by 0.85 (15% less)
    multiplier = 1.0 + pace_deviation * PACE_VARIANCE_SCALE
    multiplier = max(0.80, min(1.20, multiplier))  # cap at ±20%

    return base_margin * (multiplier - 1.0)


# =============================================================================
# FACTOR 10: CLUTCH ADJUSTMENT
# =============================================================================
def compute_clutch_adj(home_row, away_row, pre_clutch_margin: float) -> Tuple[float, dict]:
    """
    Only applies when the game is projected to be close (margin < 5 pts).
    Teams with strong clutch performance get a small boost in close games.
    """
    if abs(pre_clutch_margin) > 5.0:
        return 0.0, {"applied": False, "reason": "margin > 5 pts"}

    h_clutch = home_row.get("CLUTCH_NET_RTG", 0)
    a_clutch = away_row.get("CLUTCH_NET_RTG", 0)
    h_clutch_gp = home_row.get("CLUTCH_GP", 0)
    a_clutch_gp = away_row.get("CLUTCH_GP", 0)

    # Shrink clutch data — small samples are noisy
    h_shrink = h_clutch_gp / (h_clutch_gp + 15) if h_clutch_gp > 0 else 0
    a_shrink = a_clutch_gp / (a_clutch_gp + 15) if a_clutch_gp > 0 else 0

    clutch_edge = (h_clutch * h_shrink - a_clutch * a_shrink)
    adj = clutch_edge * CLUTCH_WEIGHT

    # Cap clutch adjustment at ±1.5 pts
    adj = max(-1.5, min(1.5, adj))

    return adj, {
        "applied": True,
        "home_clutch_rtg": round(h_clutch, 1),
        "away_clutch_rtg": round(a_clutch, 1),
        "adj": round(adj, 2),
    }


# =============================================================================
# PREDICTION ENGINE — PUTTING IT ALL TOGETHER
# =============================================================================
def _estimate_total(home_row, away_row, league_avg_off):
    ep = (home_row.get("PACE", 100) + away_row.get("PACE", 100)) / 2.0
    league_def = league_avg_off
    h100 = league_avg_off + (home_row.get("OFF_RATING",112) - league_avg_off) + \
           (away_row.get("DEF_RATING",112) - league_def)
    a100 = league_avg_off + (away_row.get("OFF_RATING",112) - league_avg_off) + \
           (home_row.get("DEF_RATING",112) - league_def)
    return round(h100 * ep/100 + a100 * ep/100, 1)


def _compute_prediction(m: dict, ratings: pd.DataFrame, season: str,
                         injuries: dict, league_avgs: dict) -> Optional[Dict]:
    def resolve(name):
        for tid, row in ratings.iterrows():
            if row["TEAM_NAME"].lower() in name.lower() or name.lower() in row["TEAM_NAME"].lower():
                return int(tid), row
        return None, None

    away_id, away_row = resolve(m["away_name"])
    home_id, home_row = resolve(m["home_name"])
    if away_id is None or home_id is None:
        print(f"  Could not resolve: {m['away_name']} / {m['home_name']}")
        return None

    print(f"    {m['away_abbr']} @ {m['home_abbr']} — computing…")
    _form_default = {"w_pct":0.5, "adj_margin":0, "l10_net_rtg":0, "days_rest":3, "b2b":False, "consec_road":0}
    try:
        away_form = _fetch_last_n(away_id, season, ratings)
    except Exception as exc:
        logger.warning("Away form fetch failed for %d: %s", away_id, exc)
        away_form = _form_default
    try:
        home_form = _fetch_last_n(home_id, season, ratings)
    except Exception as exc:
        logger.warning("Home form fetch failed for %d: %s", home_id, exc)
        home_form = _form_default
    h2h = _fetch_h2h(home_id, away_id, season)
    time.sleep(1.0)

    breakdown = {}

    # ═══════════════════════════════════════════════════════════════════════
    # FACTOR 1: NET RATING (base)
    # ═══════════════════════════════════════════════════════════════════════
    base_margin = float(home_row["NET_RATING"]) - float(away_row["NET_RATING"])
    breakdown["net_rtg"] = round(base_margin, 2)

    # ═══════════════════════════════════════════════════════════════════════
    # FACTOR 7: HOME COURT ADVANTAGE (applied early since it's part of base)
    # Venue-adjusted: teams with exceptional home records get a small boost
    # ═══════════════════════════════════════════════════════════════════════
    home_wpct = float(home_row.get("HOME_W_PCT", 0.55))
    # Scale HCA: league avg home W% is ~0.57. Deviation from that adjusts HCA.
    venue_adj = (home_wpct - 0.57) * HCA_VENUE_SCALE
    hca = BASE_HCA + venue_adj
    hca = max(1.5, min(5.0, hca))  # cap between 1.5 and 5.0
    breakdown["hca"] = round(hca, 2)

    margin = base_margin + hca

    # ═══════════════════════════════════════════════════════════════════════
    # FACTOR 2: INJURIES
    # ═══════════════════════════════════════════════════════════════════════
    h_ppg = float(home_row.get("PPG", 112))
    a_ppg = float(away_row.get("PPG", 112))
    h_inj_adj, h_inj_det = compute_injury_adj(m["home_name"], injuries, h_ppg)
    a_inj_adj, a_inj_det = compute_injury_adj(m["away_name"], injuries, a_ppg)
    injury_margin = h_inj_adj - a_inj_adj
    margin += injury_margin
    breakdown["injuries"] = round(injury_margin, 2)

    # ═══════════════════════════════════════════════════════════════════════
    # FACTOR 3: FOUR FACTORS
    # ═══════════════════════════════════════════════════════════════════════
    ff_adj, ff_details = compute_four_factors_edge(home_row, away_row, league_avgs)
    margin += ff_adj
    breakdown["four_factors"] = round(ff_adj, 2)

    # ═══════════════════════════════════════════════════════════════════════
    # FACTOR 4: MATCHUP-SPECIFIC
    # ═══════════════════════════════════════════════════════════════════════
    mu_adj, mu_details = compute_matchup_edge(home_row, away_row, league_avgs)
    margin += mu_adj
    breakdown["matchup"] = round(mu_adj, 2)

    # ═══════════════════════════════════════════════════════════════════════
    # FACTOR 5: REST & SCHEDULE
    # ═══════════════════════════════════════════════════════════════════════
    rest_adj = 0.0
    if home_form["b2b"] and not away_form["b2b"]:
        rest_adj = -BACK_TO_BACK_PENALTY
    elif away_form["b2b"] and not home_form["b2b"]:
        rest_adj = +BACK_TO_BACK_PENALTY
    elif not home_form["b2b"] and not away_form["b2b"]:
        dd = min(max(home_form["days_rest"] - away_form["days_rest"], -2), 2)
        rest_adj = dd * REST_DAY_ADVANTAGE

    # Road trip fatigue (away team on extended road trip)
    if away_form["consec_road"] > 3:
        rest_adj += (away_form["consec_road"] - 3) * ROAD_TRIP_PEN_PER_GAME
    if home_form["consec_road"] > 3:
        rest_adj -= (home_form["consec_road"] - 3) * ROAD_TRIP_PEN_PER_GAME

    margin += rest_adj
    breakdown["rest"] = round(rest_adj, 2)

    # ═══════════════════════════════════════════════════════════════════════
    # FACTOR 6: RECENT FORM (L10 rolling net rating)
    # ═══════════════════════════════════════════════════════════════════════
    form_adj = (home_form["l10_net_rtg"] - away_form["l10_net_rtg"]) * FORM_WEIGHT
    margin += form_adj
    breakdown["form"] = round(form_adj, 2)

    # ═══════════════════════════════════════════════════════════════════════
    # FACTOR 8: OFF/DEF RATINGS — already incorporated in factors 3 & 4
    # (They inform the four factors and matchup analysis rather than
    #  being a separate additive term, avoiding double-counting with net rtg)
    # ═══════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════
    # FACTOR 9: PACE-TALENT INTERACTION
    # ═══════════════════════════════════════════════════════════════════════
    pace_adj = compute_pace_variance_adj(
        home_row, away_row, margin, league_avgs.get("PACE", 100))
    margin += pace_adj
    breakdown["pace"] = round(pace_adj, 2)

    # H2H
    h2h_adj = h2h["avg_margin"] * H2H_WEIGHT * h2h["shrinkage"] if h2h["total"] > 0 else 0
    margin += h2h_adj
    breakdown["h2h"] = round(h2h_adj, 2)

    # ═══════════════════════════════════════════════════════════════════════
    # FACTOR 10: CLUTCH (only for close games)
    # ═══════════════════════════════════════════════════════════════════════
    clutch_adj, clutch_det = compute_clutch_adj(home_row, away_row, margin)
    margin += clutch_adj
    breakdown["clutch"] = round(clutch_adj, 2)

    # ─── Final calculations ──────────────────────────────────────────────
    expected_margin = round(margin, 2)

    # Use learned logistic model if enabled and loaded
    _learned_features = {
        "net_rtg_diff": float(home_row["NET_RATING"]) - float(away_row["NET_RATING"]),
        "efg_diff":  home_row.get("EFG_PCT", 0.50) - away_row.get("EFG_PCT", 0.50),
        "tov_diff":  away_row.get("TM_TOV_PCT", 0.14) - home_row.get("TM_TOV_PCT", 0.14),
        "orb_diff":  home_row.get("OREB_PCT", 0.27) - away_row.get("OREB_PCT", 0.27),
        "ftr_diff":  home_row.get("FTA_RATE", 0.24) - away_row.get("FTA_RATE", 0.24),
        "form_diff": home_form["l10_net_rtg"] - away_form["l10_net_rtg"],
        "home_b2b":  int(home_form["b2b"]),
        "away_b2b":  int(away_form["b2b"]),
        "rest_diff": float(home_form["days_rest"] - away_form["days_rest"]),
        "h2h_margin": h2h["avg_margin"] * h2h.get("shrinkage", 0.0),
    }
    _lp = _sigmoid_learned(_learned_features)
    home_prob = _lp if _lp is not None else _sigmoid(expected_margin)
    away_prob = 1.0 - home_prob

    league_avg_off = league_avgs.get("OFF_RATING", 112)
    predicted_total = _estimate_total(home_row, away_row, league_avg_off)
    home_score = round((predicted_total + expected_margin) / 2.0, 1)
    away_score = round((predicted_total - expected_margin) / 2.0, 1)

    return {
        "matchup": f"{m['away_abbr']} @ {m['home_abbr']}",
        "away_abbr": m["away_abbr"], "home_abbr": m["home_abbr"],
        "away_name": m["away_name"], "home_name": m["home_name"],
        "status": m["status"], "game_date": m["game_date"],
        "away_prob": away_prob, "home_prob": home_prob,
        "favorite": m["home_abbr"] if home_prob >= 0.5 else m["away_abbr"],
        "fav_prob": max(home_prob, away_prob),
        "home_spread": round(-expected_margin, 1),
        "away_spread": round(expected_margin, 1),
        "expected_margin": expected_margin,
        "predicted_total": predicted_total,
        "home_score": home_score, "away_score": away_score,
        # Team data
        "away_net_rtg": float(away_row["NET_RATING"]),
        "home_net_rtg": float(home_row["NET_RATING"]),
        "away_off_rtg": float(away_row.get("OFF_RATING", 0)),
        "home_off_rtg": float(home_row.get("OFF_RATING", 0)),
        "away_def_rtg": float(away_row.get("DEF_RATING", 0)),
        "home_def_rtg": float(home_row.get("DEF_RATING", 0)),
        "away_pace": float(away_row.get("PACE", 100)),
        "home_pace": float(home_row.get("PACE", 100)),
        "away_l10": away_form["w_pct"], "home_l10": home_form["w_pct"],
        "away_rest": "B2B" if away_form["b2b"] else f"{away_form['days_rest']}d",
        "home_rest": "B2B" if home_form["b2b"] else f"{home_form['days_rest']}d",
        "away_road_trip": away_form["consec_road"],
        "home_road_trip": home_form["consec_road"],
        # Breakdown
        "breakdown": breakdown,
        "ff_details": ff_details,
        "mu_details": mu_details,
        "clutch_details": clutch_det,
        "h2h_wins": h2h["team1_wins"],
        "h2h_losses": h2h["total"] - h2h["team1_wins"],
        "h2h_shrinkage": round(h2h["shrinkage"], 3),
        "h_inj_adj": h_inj_adj, "a_inj_adj": a_inj_adj,
        "h_inj_det": h_inj_det, "a_inj_det": a_inj_det,
    }


def predict_games_data(season="2025-26", matchups=None, injuries=None,
                        progress_cb=None) -> List[Dict]:
    if matchups is None:
        matchups = get_todays_games()
    if not matchups:
        return []
    if injuries is None:
        injuries = {}

    msg = progress_cb or (lambda s: print(f"\n{s}"))
    msg("Fetching team data (6 API calls)…")
    ratings = _fetch_all_team_data(season)

    # Enrich injury data with real PPG from nba_api so injury penalties
    # correctly reflect star vs. bench player value (ESPN has no stats)
    if injuries:
        msg("Enriching injuries with player PPG…")
        injuries = _enrich_injuries_with_ppg(injuries, season)

    # Compute league averages for reference
    league_avgs = {
        "OFF_RATING": float(ratings["OFF_RATING"].mean()),
        "DEF_RATING": float(ratings["DEF_RATING"].mean()),
        "PACE": float(ratings["PACE"].mean()),
        "FG3_PCT": float(ratings["FG3_PCT_OFF"].mean()) if "FG3_PCT_OFF" in ratings.columns else 0.36,
        "FG3A_PG": float(ratings["FG3A_PG"].mean()) if "FG3A_PG" in ratings.columns else 35,
        "EFG_PCT": float(ratings["EFG_PCT"].mean()),
    }

    results = []
    for i, m in enumerate(matchups):
        if progress_cb:
            progress_cb(f"Analyzing {m['away_abbr']} @ {m['home_abbr']} ({i+1}/{len(matchups)})…")
        r = _compute_prediction(m, ratings, season, injuries, league_avgs)
        if r:
            results.append(r)
    return results


# =============================================================================
# CLI OUTPUT
# =============================================================================
def _print_results(results: List[Dict], season: str):
    w = 74
    print(f"\n{'═'*w}")
    print(f"  NBA MULTI-FACTOR PREDICTION ENGINE  ({season})")
    print(f"  Factors: NetRtg · Injuries · FourFactors · Matchups · Rest")
    print(f"           Form · HCA · Pace · Clutch · H2H")
    print(f"{'═'*w}")

    for r in results:
        b = r["breakdown"]
        margin = r["expected_margin"]
        spread_str = (f"{r['home_abbr']} {r['home_spread']}" if margin > 0
                      else f"{r['away_abbr']} {r['away_spread']}")

        print(f"\n  {'─'*70}")
        print(f"  {r['away_abbr']:<4}  @  {r['home_abbr']:<4}  │  {r['status']}")
        print(f"  {'─'*70}")

        # Team comparison
        print(f"  {'':4}  {'WIN%':>7}  {'NetRtg':>7}  {'OffRtg':>7}  {'DefRtg':>7}"
              f"  {'Pace':>5}  {'L10':>4}  {'Rest':>4}")
        print(f"  {r['away_abbr']:<4}  {r['away_prob']:>6.1%}  "
              f"{r['away_net_rtg']:>+7.1f}  {r['away_off_rtg']:>7.1f}  "
              f"{r['away_def_rtg']:>7.1f}  {r['away_pace']:>5.1f}  "
              f"{r['away_l10']:>4.0%}  {r['away_rest']:>4}")
        print(f"  {r['home_abbr']:<4}  {r['home_prob']:>6.1%}  "
              f"{r['home_net_rtg']:>+7.1f}  {r['home_off_rtg']:>7.1f}  "
              f"{r['home_def_rtg']:>7.1f}  {r['home_pace']:>5.1f}  "
              f"{r['home_l10']:>4.0%}  {r['home_rest']:>4}")

        # Prediction box
        print(f"\n  ┌───────────────────────────────────────────────┐")
        print(f"  │  Spread:    {spread_str:<14}                  │")
        print(f"  │  Total:     O/U {r['predicted_total']:<8}                    │")
        print(f"  │  Score:     {r['away_abbr']} {r['away_score']:.0f}  -  "
              f"{r['home_abbr']} {r['home_score']:.0f}              │")
        print(f"  │  Favorite:  {r['favorite']} ({r['fav_prob']:.1%})"
              f"{'':20}│")
        print(f"  └───────────────────────────────────────────────┘")

        # Factor breakdown (ordered by weight/importance)
        print(f"\n  Factor breakdown (positive = home favored):")
        print(f"    ① Net Rating diff:       {b['net_rtg']:>+6.2f} pts")
        print(f"    ② Injuries:              {b['injuries']:>+6.2f} pts")
        print(f"    ③ Four Factors:          {b['four_factors']:>+6.2f} pts  "
              f"(home wins {r['ff_details'].get('home_factors_won',0)}/4)")
        print(f"    ④ Matchup (3PT/pace):    {b['matchup']:>+6.2f} pts")
        print(f"    ⑤ Rest/schedule:         {b['rest']:>+6.2f} pts")
        print(f"    ⑥ L{FORM_GAMES} form:             {b['form']:>+6.2f} pts")
        print(f"    ⑦ Home court:            {b['hca']:>+6.2f} pts")
        print(f"    ⑧ Pace-talent interact:  {b['pace']:>+6.2f} pts")
        print(f"    ⑨ H2H {r['h2h_wins']}-{r['h2h_losses']} "
              f"(shrk={r['h2h_shrinkage']:.2f}): {b['h2h']:>+6.2f} pts")
        cd = r["clutch_details"]
        if cd.get("applied"):
            print(f"    ⑩ Clutch:                {b['clutch']:>+6.2f} pts  "
                  f"(H:{cd['home_clutch_rtg']:+.1f} A:{cd['away_clutch_rtg']:+.1f})")
        else:
            print(f"    ⑩ Clutch:                  n/a  (margin > 5 pts)")
        print(f"    {'─'*40}")
        print(f"    Total margin:            {r['expected_margin']:>+6.2f} pts")

        # Injury details
        if r["h_inj_det"] or r["a_inj_det"]:
            print(f"\n  Injury impact:")
            if r["h_inj_det"]:
                print(f"    {r['home_abbr']} ({r['h_inj_adj']:+.1f} pts):")
                for d in r["h_inj_det"][:5]:
                    ppg_s = f" ({d['ppg']:.1f} PPG)" if d['ppg'] > 0 else ""
                    print(f"      {d['player']:<22} {d['status']:<13} "
                          f"{d['adj']:>+5.1f} pts{ppg_s}")
            if r["a_inj_det"]:
                print(f"    {r['away_abbr']} ({r['a_inj_adj']:+.1f} pts):")
                for d in r["a_inj_det"][:5]:
                    ppg_s = f" ({d['ppg']:.1f} PPG)" if d['ppg'] > 0 else ""
                    print(f"      {d['player']:<22} {d['status']:<13} "
                          f"{d['adj']:>+5.1f} pts{ppg_s}")

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'═'*w}")
    print(f"  PREDICTED WINNERS — FULL SLATE SUMMARY")
    print(f"  {'─'*70}")
    print(f"  {'#':<3}  {'MATCHUP':<14}  {'PICK':<5}  {'WIN%':>6}  {'SPREAD':>8}  {'O/U':>6}  CONFIDENCE")
    print(f"  {'─'*70}")
    sorted_results = sorted(results, key=lambda x: x["fav_prob"], reverse=True)
    for i, r in enumerate(sorted_results, 1):
        fav   = r["favorite"]
        prob  = r["fav_prob"]
        spread_val = abs(r["home_spread"] if r["expected_margin"] > 0 else r["away_spread"])
        spread_str = f"{fav} -{spread_val}"
        if prob >= 0.68:
            conf = "Strong"
        elif prob >= 0.57:
            conf = "Lean"
        else:
            conf = "Toss-up"
        dep = ""
        if r["h_inj_adj"] <= -5:
            dep = f"  ⚠ {r['home_abbr']} inj"
        elif r["a_inj_adj"] <= -5:
            dep = f"  ⚠ {r['away_abbr']} inj"
        b2b = ""
        if r["away_rest"] == "B2B":
            b2b = f"  ⚠ {r['away_abbr']} B2B"
        elif r["home_rest"] == "B2B":
            b2b = f"  ⚠ {r['home_abbr']} B2B"
        print(
            f"  {i:<3}  {r['away_abbr']} @ {r['home_abbr']:<6}  "
            f"{fav:<5}  {prob:>5.1%}  {spread_str:>8}  "
            f"{r['predicted_total']:>6.1f}  {conf}{dep}{b2b}"
        )
    print(f"  {'─'*70}")
    print(f"  Strong ≥68%  |  Lean 57–68%  |  Toss-up <57%")
    print(f"{'═'*w}")
    print("  Not financial advice.")
    print(f"{'═'*w}\n")


# =============================================================================
# MAIN
# =============================================================================
def main(season="2025-26"):
    print("Fetching today's games from ESPN…")
    matchups = get_todays_games()
    if not matchups:
        print("No games found.")
        return

    gd = matchups[0]["game_date"]
    tag = "TODAY" if gd == date.today() else gd.strftime("%A %B %d")
    print(f"\n{'='*62}")
    print(f"  NBA — {tag}  ({len(matchups)} games)")
    print(f"{'='*62}")
    for m in matchups:
        print(f"  {m['away_abbr']:<5} @  {m['home_abbr']:<5}  │  {m['status']}")

    print("\nFetching injury report…")
    injuries = get_espn_injuries()
    inj_count = sum(len(v) for v in injuries.values())
    print(f"  {inj_count} player(s) across {len(injuries)} team(s)")
    for tn, injs in injuries.items():
        sig = [i for i in injs if i["status"] in ("Out", "Doubtful")]
        if sig:
            ab = NAME_TO_ESPN_ABBR.get(tn, "???")
            names = ", ".join(f"{i['player']} ({i['status']})" for i in sig[:5])
            print(f"    {ab}: {names}")

    results = predict_games_data(season, matchups, injuries)
    if results:
        _print_results(results, season)


if __name__ == "__main__":
    main()