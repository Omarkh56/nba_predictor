"""
NBA Player Props Predictor — Playoff Edition v6
================================================
v2 upgrades (adaptive blend, series DEF, pace deflator, bench flags, etc.)
v3 fixes:
  1.  Series DEF uses actual GAME_IDs from player H2H logs, not recent opp games
  2.  _blend_rate fallback: if RS missing but PO data exists below threshold, use PO
  3.  Pace deflator applied to RS minutes too (consistent with rate deflation)
  4.  Road B2B only triggers when both games are vs the same opponent
  5.  Players on own injury report are skipped (Out/Doubtful) or penalised (GTD/PROB)
  6.  Book count tracked; thin-market lines penalised (1-BOOK -4pp, 2-BOOK -2pp)
  7.  H2H clamp is dynamic: ±6% per game, max ±25% (prevents 1-game 25% swings)
  8.  Combo markets: AST reduced 3% when usage boost >1.05 (PTS/AST negative correlation)
  9.  GAME_SPREADS auto-populated from Odds API spreads market
  10. --verbose CLI flag prints per-factor breakdown for top-5-by-edge bets
v4 fixes (from Apr 15 2026 backtesting — OVERs 9-4/69% vs UNDERs 8-11/42%):
  11. Context-aware pace deflation: PLAYOFF_PACE_FACTOR only applied when po_count>0;
      pre-playoff RS context skips the 2.5% deflation that was over-biasing toward UNDER
  12. UNDER confidence penalty: -5pp applied to all UNDER picks to correct for
      systematic UNDER underperformance observed in backtesting
  13. Combo market UNDER SD inflation: ×1.20 SD for PTS+AST/PRA/PTS+REB UNDER picks;
      multi-stat composite UNDERs have higher star blowup risk — wider uncertainty band
  14. Star blowup buffer: extra -5pp when proj ≥ 28 + UNDER (Curry/Maxey-type blowups)
  15. UNDER edge% floor: UNDER picks need edge% ≥ 20% to appear in top-10 display;
      low-edge UNDERs (Garland 17.9%, Curry 17.7%, Kawhi 10.1%) all missed Apr 15
v5 fixes:
  1.  Substring matching bugs: _parse_matchup_teams() for determine_team (exact split
      on "vs."/"@"), word-boundary regex in get_series_h2h, exact name match in
      _auto_fetch_spread — prevents "IND" hitting "INDIANA", "LA" hitting "LAL"/"LAC"
  2.  Dict mutation: details["AST"] copied with {**d} before reassignment so mutations
      in one market don't corrupt the same dict object used by other markets
  3.  Matchup-specific pace factor: nba_combined.py computes (home_pace+away_pace)/2/99.5
      and threads it through process_game → project_prop → project_stat → _blend_rate/
      _blend_minutes, replacing the flat PLAYOFF_PACE_FACTOR global constant
  4.  Negative Binomial probability for single-stat markets (PTS/REB/AST/3PM): discrete
      count distribution more accurate than Normal CDF; combo markets unchanged
  5.  SD inflation scaling: sd *= sqrt(total/raw_total) so wider distributions are
      correctly reflected when defensive/usage factors push the projection up/down
  6.  Unified defensive signal: def_unified = 0.20×H2H + 0.40×team_DEF + 0.40×DvP
      replaces multiplicative chain; avoids compounding independent defensive signals
v6 fixes (Apr 20 2026 — 85% OVER bias, impossible 50-145% edges, po_count=1 EWM):
  1.  po_count=1 EWM explosion: _adaptive_po_weight now requires po_count≥2 before
      blending (single game = noise); _blend_minutes ignores playoff minutes until
      po_count≥2; LOW-PO flag replaces PO-MIN+ for po_count==1 players
  2.  DvP clamp tightened [0.78,1.28]→[0.85,1.18]; dvp.py MIN_TOTAL_MINUTES 80→240
      (~10 games) so early-playoffs noise no longer pins every pick to the ceiling
  3.  Suspicious edge% filter: edge_pct>40% docks confidence −20pp and adds
      ⚠SUSP flag — real markets don't misprice by >40%
  4.  Hard thin-market filter: picks with book_count<3 excluded from all top-N
      displays (previously only got a confidence penalty, still appeared)
  5.  USG_PCT fuzzy lookup: _lookup_usg() strips diacritics (Jokić→Jokic) and
      name suffixes (Jr./III) before matching PLAYER_USG_CACHE
  6.  Line anchor: proj = 0.88×model + 0.12×line applied before evaluate_prop;
      dampens runaway projections driven by thin or stale historical data

Install:  pip install nba_api pandas numpy requests urllib3
Run:      python3 newplayerbets
          python3 newplayerbets --verbose
"""

import logging
import sys
import json
import math
import os
import time
import unicodedata
import requests
import warnings
import pandas as pd
import numpy as np
import urllib3
from datetime import date, datetime

logger = logging.getLogger(__name__)

from nba_api.stats.endpoints import (
    playergamelog,
    leaguegamefinder,
    leaguedashteamstats,
    leaguedashplayerbiostats,  # USG_PCT per player
    commonteamroster,          # roster positions for DvP
)
from nba_api.stats.static import players, teams

import dvp as dvp_engine   # Defense vs Position module

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests as _req
_original_get = _req.get
def _patched_get(url, **kwargs):
    kwargs.setdefault("verify", False)
    kwargs.setdefault("timeout", 30)
    return _original_get(url, **kwargs)
_req.get = _patched_get

# =============================================================================
# CONFIG
# =============================================================================
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "4e7c237908d67ae79a6e3a50352e2a18")  # v7 Fix 4a
SPORT        = "basketball_nba"
BASE_URL     = "https://api.the-odds-api.com/v4"

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

ESPN_TO_NBA_ABBR = {
    "GS": "GSW", "NO": "NOP", "NY": "NYK",
    "SA": "SAS", "WSH": "WAS", "UTH": "UTA",
}

TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL",         "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",         "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",         "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",      "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",       "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",       "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",  "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",     "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",       "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",  "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR","Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",     "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",             "Washington Wizards": "WAS",
}

MARKETS = [
    "player_points", "player_rebounds", "player_assists", "player_threes",
    "player_points_rebounds", "player_points_assists",
    "player_rebounds_assists", "player_points_rebounds_assists",
]

MARKET_LABELS = {
    "player_points": "PTS", "player_rebounds": "REB",
    "player_assists": "AST", "player_threes": "3PM",
    "player_points_rebounds": "PTS+REB", "player_points_assists": "PTS+AST",
    "player_rebounds_assists": "REB+AST", "player_points_rebounds_assists": "PRA",
}

MARKET_TO_STATS = {
    "player_points":                  ["PTS"],
    "player_rebounds":                ["REB"],
    "player_assists":                 ["AST"],
    "player_threes":                  ["FG3M"],
    "player_points_rebounds":         ["PTS", "REB"],
    "player_points_assists":          ["PTS", "AST"],
    "player_rebounds_assists":        ["REB", "AST"],
    "player_points_rebounds_assists": ["PTS", "REB", "AST"],
}

# Fix 4: single-stat markets use Negative Binomial for discrete count stats;
# combo markets (PTS+REB, PRA, etc.) stay on Normal CDF.
SINGLE_STAT_MARKETS = {
    "player_points", "player_rebounds", "player_assists", "player_threes",
}

SEASON         = "2025-26"
SEASON_TYPE_RS = "Regular Season"
SEASON_TYPE_PO = "Playoffs"

PLAYOFF_GAMES_THRESHOLD = 3
PLAYOFF_PACE_FACTOR     = 0.975   # RS pace ~100 poss → playoff ~97.5

# Playoff rotation minute adjustments — applied when playoffs have started but
# a player has < PLAYOFF_GAMES_THRESHOLD games (model falls back to RS minutes).
# Playoff rotations shrink from ~10 men to ~7–8: stars play more, bench less.
PLAYOFF_MIN_STARTER_MULT  = 1.06   # ≥ STARTER_MIN_THRESHOLD RS avg → +6%
PLAYOFF_MIN_ROTATION_MULT = 0.93   # 22–28 min RS avg → −7%  (rotation squeeze)
PLAYOFF_MIN_BENCH_MULT    = 0.72   # < 22 min RS avg → −28% (many go DNP/minimal)
PLAYOFF_ROTATION_RISK_MIN = 22.0   # RS avg below this → ROT-RISK flag

MIN_MINUTES_RS        = 15
MIN_MINUTES_PO        = 10   # lowered from 18 — rotation players with 10-17 min PO appearances were being excluded
BENCH_MIN_THRESHOLD   = 20
STARTER_MIN_THRESHOLD = 28

# Fix 9: auto-populated from Odds API before processing games; can also be filled manually.
# Key: "{AWAY}@{HOME}", value: float spread (positive = home favored).
GAME_SPREADS = {}
BLOWOUT_SPREAD_THRESHOLD  = 8.0
BLOWOUT_MAX_MIN_REDUCTION = 5.0

TOP_BETS_PER_GAME = 10

# v4 Fix 11-15: direction-aware confidence tuning
# Apr 19 postmortem: OVER bias was 87.5% — blanket UNDER penalty too aggressive.
# Calibration handles market-specific direction now. Reduce flat penalty.
UNDER_CONF_PENALTY   = 2.0   # reduced from 5.0; calibration handles market-specific direction
COMBO_UNDER_SD_MULT  = 1.20  # inflate SD for combo markets + UNDER (blowup risk on multi-stat)
STAR_PROJ_THRESHOLD  = 28.0  # proj at/above this triggers extra UNDER blowup buffer
STAR_UNDER_PENALTY   = 5.0   # extra pp off UNDER when proj >= STAR_PROJ_THRESHOLD
UNDER_EDGE_PCT_FLOOR = 15.0  # reduced from 20.0 — was suppressing valid UNDER picks
OVER_EDGE_PCT_FLOOR  = 10.0  # OVER picks minimum edge% (less restrictive)
COMBO_UNDER_MARKETS  = {     # markets where combo UNDER SD inflation applies
    "player_points_assists", "player_points_rebounds_assists", "player_points_rebounds",
}

# v5 postmortem fixes: playoff role-uncertainty and low-line/market penalties
PLAYOFF_NO_PO_PENALTY     = 8.0   # extra penalty when IS_PLAYOFF and po_count==0 (RS rates ≠ playoff role)
PTS_REB_OVER_PENALTY      = 5.0   # PTS+REB OVER hits only 25%; add extra confidence penalty
LOW_LINE_THRESHOLD        = 1.5   # lines at/below this get a confidence penalty for ranking purposes
LOW_LINE_CONF_PENALTY     = 5.0   # penalty applied to very short lines

# v6 fixes: sanity filters and projection anchoring
SUSPICIOUS_EDGE_PCT   = 40.0   # v6 Fix 3: edge% above this docks confidence −20pp (model error likely)
MARKET_ANCHOR_WEIGHT  = 0.12   # v6 Fix 6: 12% pull toward market line dampens runaway projections

# -----------------------------------------------------------------------------
# Defense vs Position (DvP) — computed from league logs + team rosters
# -----------------------------------------------------------------------------
# Weight when blending position-specific DvP with team-level def_f.
# 0.40 = 40% DvP, 60% existing team-level factor.
DVP_BLEND_WEIGHT = 0.40

# Minimum number of games a position must have been tracked against a team
# before we trust the DvP factor (fallback 1.0 when below this).
DVP_MIN_GAMES = 10   # v6 Fix 2: raised from 6; matches dvp.py MIN_TOTAL_MINUTES=240

# -----------------------------------------------------------------------------
# Usage Rate (USG_PCT) — from leaguedashplayerbiostats
# -----------------------------------------------------------------------------
# League-average usage across all players (~2025-26 season baseline).
LEAGUE_AVG_USG = 0.215

# How much projection weight we give to the USG deviation.
# 0.10 = a player 10% above avg USG gets +1.0% projection on scoring markets.
USG_PROJ_WEIGHT = 0.10

# PTS-related markets where USG_PCT adjustment applies (others unaffected).
USG_MARKETS = {
    "player_points",
    "player_points_rebounds",
    "player_points_assists",
    "player_points_rebounds_assists",
}

INJURY_MISS_PROB = {
    "Out":          1.00,
    "Doubtful":     0.85,
    "Questionable": 0.50,
    "Day-To-Day":   0.40,
    "Probable":     0.15,
}

USAGE_ABSORPTION_RS = 0.70
USAGE_ABSORPTION_PO = 0.82
SAME_POS_SHARE = 0.45
DIFF_POS_SHARE = 0.55

PROP_STD_DEV_PO = {
    "player_points": 7.0, "player_rebounds": 3.2, "player_assists": 2.6,
    "player_threes": 1.6,
    "player_points_rebounds": 7.8, "player_points_assists": 7.6,
    "player_rebounds_assists": 4.3, "player_points_rebounds_assists": 8.2,
}
PROP_STD_DEV_RS = {
    "player_points": 6.5, "player_rebounds": 3.0, "player_assists": 2.8,
    "player_threes": 1.5,
    "player_points_rebounds": 7.2, "player_points_assists": 7.1,
    "player_rebounds_assists": 4.1, "player_points_rebounds_assists": 7.8,
}

# 5-position mapping (matches hashtagbasketball.com DvP groupings: PG/SG/SF/PF/C).
# NBA API rosters often return generic "G"/"F" or combo "G-F"/"F-C" strings;
# we map these to the closest single-position equivalent.
POSITION_MAP = {
    "PG":  "PG",  "SG":  "SG",  "SF":  "SF",  "PF":  "PF",  "C": "C",
    "G":   "SG",  # generic guard → SG (PG is usually labeled more specifically)
    "F":   "SF",  # generic forward → SF
    "G-F": "SF",  # guard-forward hybrid → SF
    "F-G": "SF",
    "F-C": "PF",  # forward-center → PF
    "C-F": "PF",
}
# Broad groupings still used by compute_usage_boost position-matching
GUARD_POSITIONS   = {"PG", "SG", "G"}
FORWARD_POSITIONS = {"SF", "PF", "F", "G-F", "F-G"}
CENTER_POSITIONS  = {"C", "F-C", "C-F"}

STAT_TO_INJ_FIELD = {"PTS": "ppg", "REB": "rpg", "AST": "apg", "FG3M": "ppg"}
FG3M_PPG_RATIO = 0.12

# Calibration data loaded from calibration.json (generated by calibrate.py).
# Empty dict = no calibration applied (safe default when file is absent).
CALIB      = {}
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")

# =============================================================================
# LEARNED MODEL PARAMS (from train_model.py → learned_params.json)
# =============================================================================
_LEARNED_PLAYER_PARAMS: dict = {}   # stat → {"ewma_weight", "home_boost", "rest_coef", ...}
_LEARNED_VARIANCE:      dict = {}   # {"by_position": {G/F/C: {PTS/REB/...: std}}, ...}

_LEARNED_PARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned_params.json")

def _load_learned_player_params() -> None:
    """Load player model params and variance estimates from learned_params.json."""
    global _LEARNED_PLAYER_PARAMS, _LEARNED_VARIANCE, PROP_STD_DEV_PO, PROP_STD_DEV_RS
    if not os.path.exists(_LEARNED_PARAMS_FILE):
        return
    try:
        with open(_LEARNED_PARAMS_FILE) as fh:
            data = json.load(fh)
        _LEARNED_PLAYER_PARAMS = data.get("player_models", {})
        var_block = data.get("player_variance", {})
        _LEARNED_VARIANCE = var_block

        # Update fallback std-dev tables from learned position-averaged variances
        league = var_block.get("league_average", {})
        _STAT_TO_MARKET = {
            "PTS": ("player_points", "player_points"),
            "REB": ("player_rebounds", "player_rebounds"),
            "AST": ("player_assists", "player_assists"),
            "FG3M": ("player_threes", "player_threes"),
        }
        for stat, (mk_po, mk_rs) in _STAT_TO_MARKET.items():
            league_std = league.get(stat)
            if league_std and league_std > 0:
                PROP_STD_DEV_PO[mk_po] = round(league_std, 3)
                PROP_STD_DEV_RS[mk_rs] = round(league_std * 0.95, 3)

        print(f"  [train_model] player params loaded  "
              f"(stats={list(_LEARNED_PLAYER_PARAMS.keys())}  "
              f"version={data.get('version','?')})")
    except Exception as e:
        print(f"  [train_model] failed to load player params: {e}")


def get_learned_pos_std(pos_group: str, stat: str, fallback: float) -> float:
    """Return learned position-group std dev for a stat, or fallback if not available."""
    broad_pg = {"G": "G", "PG": "G", "SG": "G", "SF": "F", "PF": "F", "F": "F", "C": "C"}.get(pos_group, "G")
    by_pos = _LEARNED_VARIANCE.get("by_position", {})
    pg_stds = by_pos.get(broad_pg, {})
    return pg_stds.get(stat, fallback)


_load_learned_player_params()   # run once at import time

MARKET_PROJ_ADJ  = {}   # v7 Fix 1a: per-stat projection multiplier derived from calibration hit rates
PLAYER_PROJ_ADJ  = {}   # v8 Fix 3: per-player per-stat additive residual nudges from player_market calib
NULL_POS_PLAYERS = set() # v8 Fix 6: players whose position lookup failed (DvP=1.0 for these)

# v8 Fix 5: manual position overrides for players missing from the auto-detect roster fetch.
# Values must match the 5-group DvP system (PG/SG/SF/PF/C) used by _pos_group().
MANUAL_POSITIONS = {
    "Devin Booker":              "SG",
    "Anthony Black":             "SG",
    "Nickeil Alexander-Walker":  "SG",
    "Quentin Grimes":            "SG",
    "Isaiah Hartenstein":        "C",
    "Keldon Johnson":            "SF",
    "Isaiah Joe":                "SG",
    "Goga Bitadze":              "C",
    "Jonathan Kuminga":          "SF",
    "Jaxson Hayes":              "C",
}

# v8 Fix 2: per-player projection multipliers for players with confirmed PTS projection
# inflation (0W on 3+ PTS OVER bets in bet_log.csv). Suppresses these players at the
# player level without touching the broader PTS market for other players.
MANUAL_PLAYER_PENALTIES = {
    ("Devin Booker",             "PTS"): 0.85,
    ("Anthony Black",            "PTS"): 0.85,
    ("Nickeil Alexander-Walker", "PTS"): 0.85,
    ("Quentin Grimes",           "PTS"): 0.85,
    ("Isaiah Hartenstein",       "PTS"): 0.85,
    ("Keldon Johnson",           "PTS"): 0.85,
    ("Isaiah Joe",               "PTS"): 0.85,
    ("Goga Bitadze",             "PTS"): 0.85,
    ("Jonathan Kuminga",         "PTS"): 0.85,
    ("Jaxson Hayes",             "PTS"): 0.85,
}

# Set to True by get_league_logs() when playoff game logs are available.
# Used by _blend_minutes() to apply rotation adjustments for early playoff games.
IS_PLAYOFF_SEASON = False

# =============================================================================
# CACHES
# =============================================================================
PLAYER_LOGS_CACHE     = {}
LEAGUE_GAMELOGS_CACHE = {}
TEAM_DEF_CACHE        = {}
TEAM_STATS_CACHE      = {}
PLAYER_STD_CACHE      = {}
PLAYER_USG_CACHE      = {}   # {player_name: usg_pct float}  — leaguedashplayerbiostats
PLAYER_POS_CACHE      = {}   # {player_name: "G"/"F"/"C"}    — built from team rosters
TEAM_ROSTER_CACHE     = {}   # {team_id: {player_name: pos_group}} — commonteamroster
TEAM_DVP_CACHE        = {}   # {(team_id, pos_group, stat_col): factor} — computed


# =============================================================================
# CALIBRATION
# =============================================================================
def _load_calibration():
    """Load calibration.json into the CALIB global (no-op if file is absent).

    calibration.json is produced by calibrate.py after graded bets accumulate
    in bet_log.csv.  The file is optional — if absent, all adjustments are 0.
    """
    global CALIB, MARKET_PROJ_ADJ, PLAYER_PROJ_ADJ
    if not os.path.isfile(CALIB_FILE):
        return
    try:
        with open(CALIB_FILE) as f:
            CALIB = json.load(f)
        total = CALIB.get("total_bets", 0)
        md    = len(CALIB.get("market_direction", {}))
        pm    = len(CALIB.get("player_market", {}))
        print(f"  Calibration loaded ({total} bets, "
              f"{md} market-direction, {pm} player-market adjustments)")

        # v7 Fix 1b: build projection multipliers from OVER hit rates.
        # v8 Fix 1: PTS is excluded — flat market-wide punishment doesn't fix the
        # root cause (a specific set of bad-projection players). Player-level
        # penalties in MANUAL_PLAYER_PENALTIES handle those cases more precisely.
        _STAT_TO_MKT   = {"PTS": "PTS", "REB": "REB", "AST": "AST", "FG3M": "3PM"}
        SKIP_MARKET_ADJ = {"PTS"}   # v8 Fix 1: skip PTS market-level multiplier
        MARKET_PROJ_ADJ.clear()
        for key, info in CALIB.get("market_direction", {}).items():
            if "_OVER" not in key or info.get("n", 0) < 20:
                continue
            mkt_label = key.replace("_OVER", "")
            stat_key  = next((s for s, m in _STAT_TO_MKT.items() if m == mkt_label), None)
            if stat_key is None or stat_key in SKIP_MARKET_ADJ:  # v8 Fix 1
                continue
            over_rate  = info["raw_rate"]
            bias       = (0.5 - over_rate) * 0.6
            multiplier = max(0.92, min(1.08, 1.0 - bias))   # v8 Fix 1: tighter clamp ±8%
            MARKET_PROJ_ADJ[stat_key] = round(multiplier, 3)
        if MARKET_PROJ_ADJ:
            print(f"  Projection multipliers (excl. PTS): {MARKET_PROJ_ADJ}")

        # v8 Fix 3: load per-player per-stat residual adjustments from player_market.
        # Keys: (player_name, stat_col); values: shrinkage-weighted mean residual
        # (actual − projection) in stat units. Applied as additive nudge in project_stat.
        # Only applied when n >= 3 graded bets exist for that player-market pair.
        _MARKET_TO_STAT = {"PTS": "PTS", "REB": "REB", "AST": "AST", "3PM": "FG3M"}
        PLAYER_PROJ_ADJ.clear()
        for key, info in CALIB.get("player_market", {}).items():
            if "|" not in key or info.get("n", 0) < 3:
                continue
            player, market_label = key.split("|", 1)
            stat_col = _MARKET_TO_STAT.get(market_label)
            if stat_col is None:
                continue
            adj = info.get("adj", 0.0)
            if adj != 0.0:
                PLAYER_PROJ_ADJ[(player, stat_col)] = round(adj, 2)
        if PLAYER_PROJ_ADJ:
            print(f"  Player-stat nudges loaded: {len(PLAYER_PROJ_ADJ)} adjustments")
    except Exception as e:
        print(f"  Warning: could not load calibration.json — {e}")
        CALIB = {}


# =============================================================================
# UTILS
# =============================================================================
def api_sleep():
    time.sleep(1.2)

def _parse_minutes(val):
    try:
        if isinstance(val, str) and ":" in val:
            p = val.split(":")
            return int(p[0]) + int(p[1]) / 60
        return float(val)
    except (ValueError, TypeError, IndexError):
        return 0.0

def _local_date(ct):
    try:
        utc_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        return utc_dt.astimezone().date()
    except (ValueError, AttributeError, OSError) as exc:
        logger.debug("_local_date parse failed for %r: %s", ct, exc)
        return None

def _pos_group(pos):
    """Map NBA API position string to one of PG/SG/SF/PF/C (5-group DvP system).
    Falls back to broad G/F/C inference for any unrecognised strings."""
    if not pos: return None
    p = pos.upper().strip()
    if p in POSITION_MAP:
        return POSITION_MAP[p]
    # Fallback: infer from substrings (handles "PG-SG", "SG-SF", etc.)
    if "PG" in p: return "PG"
    if "SG" in p: return "SG"
    if "SF" in p: return "SF"
    if "PF" in p: return "PF"
    if "C"  in p: return "C"
    if "G"  in p: return "SG"
    if "F"  in p: return "SF"
    return None

def _adaptive_po_weight(po_count):
    """v6 Fix 1a: require 2+ playoff games before blending — a single playoff
    game is unreliable noise that inflates projections through the EWM.
    Smooth ramp: 2 games = 15%, 4 = 45%, 8 = 85%, capped at 85%."""
    if po_count < 2:
        return 0.0
    return min(0.85, (po_count - 1) * 0.15)


# =============================================================================
# ESPN INJURIES
# =============================================================================
def get_espn_injuries():
    try:
        r = requests.get(ESPN_INJURIES_URL, timeout=15, verify=False)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ESPN injuries error: {e}")
        return {}

    out = {}
    for te in data.get("injuries", []):
        name = te.get("team", {}).get("displayName", "")
        injs = []
        for inj in te.get("injuries", []):
            a = inj.get("athlete", {})
            ppg = rpg = apg = 0.0
            for block in a.get("statistics", []):
                for split in block.get("splits", []):
                    for cat in split.get("categories", []):
                        for s in cat.get("stats", []):
                            nm = s.get("name", "").lower()
                            v  = float(s.get("value", 0))
                            if nm in ("avgpoints",   "ppg"): ppg = v
                            elif nm in ("avgrebounds","rpg"): rpg = v
                            elif nm in ("avgassists", "apg"): apg = v
            injs.append({
                "player":   a.get("displayName", ""),
                "status":   inj.get("status", ""),
                "position": a.get("position", {}).get("abbreviation", ""),
                "ppg": ppg, "rpg": rpg, "apg": apg,
            })
        if injs:
            out[name] = injs
    return out


# =============================================================================
# USAGE RATE  (leaguedashplayerbiostats → USG_PCT)
# =============================================================================
def get_all_player_usage():
    """Fetch season USG_PCT for every player in one API call. Results cached.

    USG_PCT (usage rate) measures the fraction of team possessions a player
    uses while on the floor.  League average ≈ 21.5%.  High-usage players
    (≥28%) have more self-determined scoring and tend to project more reliably
    on PTS-heavy markets; low-usage players (<13%) are more scheme-dependent.
    """
    if PLAYER_USG_CACHE:          # already fetched this run
        return PLAYER_USG_CACHE
    print("  Loading player usage rates (USG_PCT)...")
    try:
        df = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
            season=SEASON, timeout=30,
        ).get_data_frames()[0]
        api_sleep()
        for _, row in df.iterrows():
            name = row.get("PLAYER_NAME", "")
            usg  = row.get("USG_PCT", 0)
            if name:
                PLAYER_USG_CACHE[name] = float(usg or 0)
        print(f"  Loaded USG_PCT for {len(PLAYER_USG_CACHE)} player(s).")
    except Exception as e:
        print(f"  USG_PCT fetch error: {e}")
    return PLAYER_USG_CACHE


# =============================================================================
# DEFENSE vs POSITION (DvP)  — team rosters + league logs
# =============================================================================
def _fetch_team_roster(team_id):
    """Fetch current roster for one team; populate PLAYER_POS_CACHE.

    commonteamroster returns {PLAYER: name, POSITION: 'G'/'F'/'C'/'G-F'/...}.
    Results stored in TEAM_ROSTER_CACHE and merged into PLAYER_POS_CACHE.
    """
    if team_id in TEAM_ROSTER_CACHE:
        return TEAM_ROSTER_CACHE[team_id]
    try:
        df = commonteamroster.CommonTeamRoster(
            team_id=team_id, season=SEASON, timeout=30,
        ).get_data_frames()[0]
        api_sleep()
        result = {}
        for _, row in df.iterrows():
            name = row.get("PLAYER", "")
            pos  = row.get("POSITION", "")
            if name:
                pg = _pos_group(pos)
                if pg:
                    result[name] = pg
                    PLAYER_POS_CACHE[name] = pg
        TEAM_ROSTER_CACHE[team_id] = result
        return result
    except Exception as e:
        print(f"  Roster fetch error (team {team_id}): {e}")
        TEAM_ROSTER_CACHE[team_id] = {}
        return {}


def prefetch_rosters_for_games(events):
    """Pre-load rosters for every team playing today so DvP is computable.

    Called once in main() before the game loop.  Populates PLAYER_POS_CACHE
    for all players on today's teams so get_dvp_factor() has coverage.
    """
    seen = set()
    for ev in events:
        for tname in [ev.get("home_team", ""), ev.get("away_team", "")]:
            abbr = TEAM_NAME_TO_ABBR.get(tname, tname[:3].upper())
            tid  = get_team_id(abbr)
            if tid and tid not in seen:
                seen.add(tid)
                _fetch_team_roster(tid)
    if seen:
        # v8 Fix 5: apply manual position overrides before injecting into DvP engine.
        # Auto-detect missed these players; overrides ensure they get real DvP factors.
        _manual_added = {p for p in MANUAL_POSITIONS if p not in PLAYER_POS_CACHE}
        PLAYER_POS_CACHE.update(MANUAL_POSITIONS)
        if _manual_added:
            print(f"  Manual positions applied for {len(_manual_added)} player(s): "
                  f"{', '.join(sorted(_manual_added))}")
        print(f"  DvP roster positions loaded for {len(seen)} team(s) "
              f"({len(PLAYER_POS_CACHE)} players mapped).")
        # Pass positions to DvP engine and pre-build the table
        dvp_engine.inject_positions(PLAYER_POS_CACHE)
        _dvp_window = "playoffs" if IS_PLAYOFF_SEASON else "season"
        dvp_engine.build_dvp_table(_dvp_window)
        print(f"  DvP table built (window={_dvp_window}).")


def get_dvp_factor(opp_team_id, pos_group, stat_col):
    """Position-specific defensive factor for the opponent team.

    Answers: "Relative to league average, how many {stat_col} does {opp_team}
    allow per game to {pos_group} players?"

    Methodology
    -----------
    1. Pull all games the opponent team was involved in (from league logs).
    2. Filter to opposing-team rows (the players who FACED the defense).
    3. Keep only players whose position maps to pos_group.
    4. Average the stat per game → compare to league-wide per-game avg for
       the same position group.
    5. Clamp factor to [0.78, 1.28] and require DVP_MIN_GAMES data points.

    Returns 1.0 (neutral) if position unknown or insufficient data.
    """
    if not pos_group:
        return 1.0

    ck = (opp_team_id, pos_group, stat_col)
    if ck in TEAM_DVP_CACHE:
        return TEAM_DVP_CACHE[ck]

    logs = get_league_logs()
    if logs.empty or stat_col not in logs.columns or "PLAYER_NAME" not in logs.columns:
        TEAM_DVP_CACHE[ck] = 1.0
        return 1.0

    # Players with known positions for this group
    pos_players = {n for n, g in PLAYER_POS_CACHE.items() if g == pos_group}
    if not pos_players:
        TEAM_DVP_CACHE[ck] = 1.0
        return 1.0

    # Games involving the opponent team
    opp_game_ids = set(logs[logs["TEAM_ID"] == opp_team_id]["GAME_ID"].tolist())
    if not opp_game_ids:
        TEAM_DVP_CACHE[ck] = 1.0
        return 1.0

    # Opponent-team rows = players who faced this defense
    facing = logs[
        (logs["GAME_ID"].isin(opp_game_ids)) &
        (logs["TEAM_ID"] != opp_team_id) &
        (logs["PLAYER_NAME"].isin(pos_players))
    ]

    if len(facing) < DVP_MIN_GAMES:
        TEAM_DVP_CACHE[ck] = 1.0
        return 1.0

    # Per-game total for this position vs this defense
    opp_pg_avg = facing.groupby("GAME_ID")[stat_col].sum().mean()

    # League-wide per-game total for the same position (all teams)
    all_pos_rows = logs[logs["PLAYER_NAME"].isin(pos_players)]
    if all_pos_rows.empty:
        TEAM_DVP_CACHE[ck] = 1.0
        return 1.0

    league_pg_avg = all_pos_rows.groupby("GAME_ID")[stat_col].sum().mean()

    if not league_pg_avg or league_pg_avg <= 0:
        TEAM_DVP_CACHE[ck] = 1.0
        return 1.0

    raw_factor = opp_pg_avg / league_pg_avg
    factor = max(0.78, min(raw_factor, 1.28))
    TEAM_DVP_CACHE[ck] = factor
    return factor


# =============================================================================
# ODDS API
# =============================================================================
def get_all_odds_events():
    from datetime import datetime, timezone
    r = requests.get(f"{BASE_URL}/sports/{SPORT}/events",
                     params={"apiKey": ODDS_API_KEY}, timeout=20, verify=False)
    r.raise_for_status()
    events = r.json()
    now = datetime.now(timezone.utc).isoformat()
    upcoming = [e for e in events if e.get("commence_time", "") > now]
    in_play  = len(events) - len(upcoming)
    if in_play:
        print(f"  Skipping {in_play} in-play game(s) — pre-game prop markets are closed.")
    return upcoming

def get_event_player_props(event_id):
    for regions in ["us", "us,us2,uk,eu,au"]:
        r = requests.get(
            f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds",
            params={"apiKey": ODDS_API_KEY, "regions": regions,
                    "markets": ",".join(MARKETS), "oddsFormat": "american",
                    "dateFormat": "iso"},
            timeout=20, verify=False)
        r.raise_for_status()
        data = r.json()
        if data.get("bookmakers"):
            return data
    return data

def _auto_fetch_spread(event_id, home_name):
    """Fix 9: fetch spreads market from Odds API to auto-populate GAME_SPREADS.
    Returns float: positive = home favored, 0.0 if unavailable.
    Convention: Odds API 'point' for home team = negative when home is favorite;
    we negate it so GAME_SPREADS positive = home favored."""
    try:
        r = requests.get(
            f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds",
            params={"apiKey": ODDS_API_KEY, "regions": "us",
                    "markets": "spreads", "oddsFormat": "american"},
            timeout=15, verify=False)
        r.raise_for_status()
        data = r.json()
        for bk in data.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt.get("key") != "spreads":
                    continue
                for oc in mkt.get("outcomes", []):
                    # Fix 1c: exact name match to avoid "LA" hitting both "LAL"/"LAC"
                    if oc.get("name", "").strip().lower() == home_name.strip().lower():
                        point = oc.get("point")
                        if point is not None:
                            # Odds API: home team point=-5.5 means home favored by 5.5
                            # Our convention: positive = home favored → negate the point
                            return float(-point)
    except (KeyError, TypeError, ValueError, requests.RequestException) as exc:
        logger.error("_auto_fetch_spread failed: %s", exc)
    return 0.0

def flatten_props(props_json):
    rows = []
    for bk in props_json.get("bookmakers", []):
        book = bk.get("title", "Unknown")
        for mkt in bk.get("markets", []):
            mk = mkt.get("key")
            for oc in mkt.get("outcomes", []):
                rows.append({"bookmaker": book, "market": mk,
                             "player": oc.get("description", ""),
                             "side": oc.get("name", ""),
                             "line": oc.get("point"), "odds": oc.get("price")})
    df = pd.DataFrame(rows)
    return df.dropna(subset=["player","market","line"]) if not df.empty else df

def consensus_lines(props_df):
    """Fix 6: also compute book_count (distinct bookmakers per player/market)
    so thin-market lines can be penalised downstream."""
    over_df = props_df[props_df["side"] == "Over"].copy()
    if over_df.empty:
        over_df = props_df.drop_duplicates(subset=["player","market","bookmaker"]).copy()

    lines_df = (over_df.groupby(["player","market"])["line"]
                .median().reset_index()
                .rename(columns={"line": "consensus_line"}))

    # Count distinct bookmakers offering each line — thin markets are less reliable
    book_counts = (over_df.groupby(["player","market"])["bookmaker"]
                   .nunique().reset_index()
                   .rename(columns={"bookmaker": "book_count"}))

    return lines_df.merge(book_counts, on=["player","market"])


# =============================================================================
# NBA API — PLAYER LOGS
# =============================================================================
def get_player_id(name):
    m = players.find_players_by_full_name(name)
    if not m: return None
    a = [p for p in m if p.get("is_active")]
    return a[0]["id"] if a else m[0]["id"]

def get_team_id(abbr):
    for t in teams.get_teams():
        if t["abbreviation"] == abbr: return t["id"]
    return None

def _fetch_logs(pid, season, season_type, min_minutes, retries=3):
    # v8 Fix Bug 1+2: min_minutes included in cache key so po_logs (min=10) and
    # po_raw (min=0) don't collide; bare except replaced with Exception capture.
    ck = (pid, season, season_type, min_minutes)
    if ck in PLAYER_LOGS_CACHE:
        return PLAYER_LOGS_CACHE[ck]

    last_error = None
    for att in range(retries):
        try:
            df = playergamelog.PlayerGameLog(
                player_id=pid, season=season,
                season_type_all_star=season_type, timeout=30,
            ).get_data_frames()[0]
            api_sleep()
            if not df.empty:
                df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="mixed")
                df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
                if "MIN" in df.columns:
                    df["_MIN_F"] = df["MIN"].apply(_parse_minutes)
                    df = df[df["_MIN_F"] >= min_minutes].reset_index(drop=True)
                    df = df.drop(columns=["_MIN_F"])
            PLAYER_LOGS_CACHE[ck] = df
            return df
        except Exception as e:
            last_error = e
            if att == 0:
                print(f"  [_fetch_logs] pid={pid} {season_type}: "
                      f"{type(e).__name__}: {e}  (retrying...)")
            time.sleep(2 * (att + 1))

    print(f"  ⚠ _fetch_logs FAILED after {retries} attempts: "
          f"pid={pid} season={season} type={season_type} "
          f"→ {type(last_error).__name__}: {last_error}")
    empty = pd.DataFrame()
    PLAYER_LOGS_CACHE[ck] = empty
    return empty

def get_player_logs_blended(pid):
    po_logs     = _fetch_logs(pid, SEASON, SEASON_TYPE_PO, MIN_MINUTES_PO)
    # Raw count (no minute filter) — used only for NO-PO flag so it fires only
    # when the player has truly had zero playoff appearances, not just short ones.
    po_raw      = _fetch_logs(pid, SEASON, SEASON_TYPE_PO, 0)
    rs_logs     = _fetch_logs(pid, SEASON, SEASON_TYPE_RS, MIN_MINUTES_RS)
    if not rs_logs.empty:
        rs_logs = rs_logs.head(20)
    using_playoffs = len(po_logs) >= PLAYOFF_GAMES_THRESHOLD
    # Return raw_po_count as 5th element for flag logic
    return po_logs, rs_logs, using_playoffs, len(po_logs), len(po_raw)

def _avg_minutes(logs, half_life=4):
    """EWM average minutes — recent games weighted more (half-life = 4 games).
    Helps reflect late-season rotation changes heading into the playoffs."""
    if logs.empty or "MIN" not in logs.columns: return 0.0
    mins = logs["MIN"].apply(_parse_minutes).values
    if len(mins) == 0: return 0.0
    w = np.array([0.5 ** (i / half_life) for i in range(len(mins))])
    return float(np.average(mins, weights=w))


def _playoff_rotation_adj(rs_min: float) -> float:
    """Role-based minute adjustment for early-playoff context.
    Starters play more; rotation/bench players play significantly less."""
    if rs_min >= STARTER_MIN_THRESHOLD:
        return rs_min * PLAYOFF_MIN_STARTER_MULT
    if rs_min >= PLAYOFF_ROTATION_RISK_MIN:
        return rs_min * PLAYOFF_MIN_ROTATION_MULT
    return rs_min * PLAYOFF_MIN_BENCH_MULT


def _std_minutes(logs):
    if logs.empty or "MIN" not in logs.columns: return 0.0
    mins = logs["MIN"].apply(_parse_minutes)
    return float(mins.std()) if len(mins) > 1 else 0.0


# =============================================================================
# LEAGUE LOGS
# =============================================================================
def safe_lgf(retries=3, **kw):
    for att in range(retries):
        try:
            df = leaguegamefinder.LeagueGameFinder(timeout=30, **kw).get_data_frames()[0]
            api_sleep()
            return df
        except Exception as exc:
            logger.warning("safe_lgf attempt %d/%d failed: %s", att + 1, retries, exc)
            time.sleep(2 * (att + 1))
    return pd.DataFrame()

def get_league_logs():
    global IS_PLAYOFF_SEASON
    ck = (SEASON, SEASON_TYPE_PO)
    if ck in LEAGUE_GAMELOGS_CACHE:
        return LEAGUE_GAMELOGS_CACHE[ck]
    print("  Loading league game logs (Playoffs)...")
    df = safe_lgf(season_nullable=SEASON, season_type_nullable=SEASON_TYPE_PO,
                  player_or_team_abbreviation="P")
    if df.empty:
        print("  Playoff logs empty — falling back to regular season.")
        df = safe_lgf(season_nullable=SEASON, season_type_nullable=SEASON_TYPE_RS,
                      player_or_team_abbreviation="P")
    else:
        IS_PLAYOFF_SEASON = True
        print("  Playoffs detected — rotation minute adjustments active.")
    if not df.empty and "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    print(f"  {len(df):,} rows loaded.")
    LEAGUE_GAMELOGS_CACHE[ck] = df
    # Inject into DvP engine so it doesn't need its own fetch
    dvp_engine.inject_logs(df, "playoffs" if IS_PLAYOFF_SEASON else "season")
    return df


# =============================================================================
# TEAM STATS
# =============================================================================
def get_team_season_stats():
    if "stats" in TEAM_STATS_CACHE:
        return TEAM_STATS_CACHE["stats"]
    try:
        df = leaguedashteamstats.LeagueDashTeamStats(
            season=SEASON, measure_type_detailed_defense="Base", timeout=30,
        ).get_data_frames()[0]
        api_sleep()
        df["PPG"]     = df["PTS"]  / df["GP"].clip(lower=1)
        df["RPG"]     = df["REB"]  / df["GP"].clip(lower=1)
        df["APG"]     = df["AST"]  / df["GP"].clip(lower=1)
        df["FG3M_PG"] = df["FG3M"] / df["GP"].clip(lower=1)
        result = df.set_index("TEAM_ID")
        TEAM_STATS_CACHE["stats"] = result
        return result
    except Exception as e:
        print(f"  Team stats error: {e}")
        TEAM_STATS_CACHE["stats"] = pd.DataFrame()
        return pd.DataFrame()

def get_team_stat_avg(team_abbr, stat_col):
    ts = get_team_season_stats()
    if ts.empty: return 0.0
    tid = get_team_id(team_abbr)
    if tid is None or tid not in ts.index: return 0.0
    col_map = {"PTS": "PPG", "REB": "RPG", "AST": "APG", "FG3M": "FG3M_PG"}
    col = col_map.get(stat_col)
    if col and col in ts.columns:
        return float(ts.loc[tid, col])
    return 0.0


# =============================================================================
# DEFENSIVE FACTOR
# =============================================================================
def get_def_factor(opp_team_id, stat_col):
    ck = (opp_team_id, stat_col)
    if ck in TEAM_DEF_CACHE: return TEAM_DEF_CACHE[ck]
    logs = get_league_logs()
    if logs.empty or stat_col not in logs.columns:
        TEAM_DEF_CACHE[ck] = 1.0; return 1.0
    opp = logs[logs["TEAM_ID"] == opp_team_id]
    if opp.empty:
        TEAM_DEF_CACHE[ck] = 1.0; return 1.0
    gids   = set(opp["GAME_ID"].unique())
    facing = logs[(logs["GAME_ID"].isin(gids)) & (logs["TEAM_ID"] != opp_team_id)]
    if facing.empty:
        TEAM_DEF_CACHE[ck] = 1.0; return 1.0
    allowed = facing.groupby("GAME_ID")[stat_col].sum().mean()
    league  = logs.groupby("GAME_ID")[stat_col].sum().mean() / 2.0
    f = allowed / league if league and league > 0 else 1.0
    f = max(0.70, min(f, 1.30))
    TEAM_DEF_CACHE[ck] = f
    return f

def get_series_def_factor(opp_team_id, stat_col, h2h_logs):
    """Fix 1: Use actual GAME_IDs from the player's H2H-filtered playoff logs
    (games where MATCHUP contains the opponent abbreviation) rather than
    approximating series games as the opponent's N most recent games against anyone."""
    season_factor = get_def_factor(opp_team_id, stat_col)

    if h2h_logs.empty or stat_col not in h2h_logs.columns:
        return season_factor

    # Fix 1: extract real series GAME_IDs directly from the player's H2H logs
    if "GAME_ID" not in h2h_logs.columns:
        return season_factor  # can't proceed without GAME_ID; fall back to season factor

    series_game_ids = set(h2h_logs["GAME_ID"].tolist())
    series_count    = len(series_game_ids)
    if series_count == 0:
        return season_factor

    logs = get_league_logs()
    if logs.empty or stat_col not in logs.columns:
        return season_factor

    # Pull opposing-player rows from exactly those series game IDs
    series_facing = logs[
        (logs["GAME_ID"].isin(series_game_ids)) & (logs["TEAM_ID"] != opp_team_id)
    ]
    if series_facing.empty:
        return season_factor

    series_allowed = series_facing.groupby("GAME_ID")[stat_col].sum().mean()
    league_avg     = logs.groupby("GAME_ID")[stat_col].sum().mean() / 2.0
    if league_avg <= 0:
        return season_factor

    raw_series_factor = max(0.70, min(series_allowed / league_avg, 1.30))

    # Shrinkage: more series games → trust series factor more
    series_weight = series_count / (series_count + 5)
    return series_weight * raw_series_factor + (1 - series_weight) * season_factor


# =============================================================================
# USG LOOKUP
# =============================================================================
def _nfkd(s):
    """Strip diacritics: 'Nikola Jokić' → 'Nikola Jokic'."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

def _lookup_usg(player_name):
    """v6 Fix 5: fuzzy USG lookup with diacritic normalization and suffix stripping.
    Falls back to 0.0 if no match found (treated as 'no adjustment' by caller)."""
    if player_name in PLAYER_USG_CACHE:
        return PLAYER_USG_CACHE[player_name]
    # Pass 1: diacritic normalization — catches Jokić/Jokic, Schröder/Schroder, etc.
    pn = _nfkd(player_name)
    for cached_name, usg in PLAYER_USG_CACHE.items():
        if _nfkd(cached_name) == pn:
            return usg
    # Pass 2: first two name tokens only — strips "Jr.", "III", etc.
    parts = player_name.split()
    if len(parts) >= 2:
        base = _nfkd(" ".join(parts[:2]))
        for cached_name, usg in PLAYER_USG_CACHE.items():
            if _nfkd(" ".join(cached_name.split()[:2])) == base:
                return usg
    return 0.0


# =============================================================================
# USAGE REDISTRIBUTION
# =============================================================================
def _get_injured_player_recent_stat(injured_name: str, inj_field: str,
                                    n_games: int = 10):
    """v9 Fix 5: look up injured player's recent average from our own cached logs.
    Returns None when logs are unavailable; caller falls back to ESPN season avg."""
    pid = get_player_id(injured_name)
    if not pid:
        return None
    try:
        po_logs, rs_logs, _, _, _ = get_player_logs_blended(pid)
    except Exception:
        return None
    logs = po_logs if not po_logs.empty else rs_logs
    if logs.empty:
        return None
    stat_col_map = {"ppg": "PTS", "rpg": "REB", "apg": "AST"}
    sc = stat_col_map.get(inj_field)
    if not sc or sc not in logs.columns:
        return None
    recent = logs.head(n_games)[sc].astype(float)
    return float(recent.mean()) if not recent.empty else None


def compute_usage_boost(player_name, player_pos, stat_col, player_avg,
                        team_avg, injuries_for_team, using_playoffs=False):
    """Absorption rate 0.82 in playoffs (tighter rotations) vs 0.70 RS."""
    if not injuries_for_team or team_avg <= 0 or player_avg <= 0:
        return 1.0

    absorption_rate = USAGE_ABSORPTION_PO if using_playoffs else USAGE_ABSORPTION_RS
    my_group  = _pos_group(player_pos)
    inj_field = STAT_TO_INJ_FIELD.get(stat_col)
    if not inj_field: return 1.0

    total_absorbed = 0.0
    for inj in injuries_for_team:
        if inj.get("player") == player_name: continue
        miss_prob = INJURY_MISS_PROB.get(inj.get("status", ""), 0.0)
        if miss_prob <= 0: continue

        if stat_col == "FG3M" and inj_field == "ppg":
            # FG3M has no direct ESPN field; derive from PPG using ratio.
            # v9 Fix 5: prefer recent PPG over ESPN season average for the base.
            base_ppg = inj.get("ppg", 0)
            recent_ppg = _get_injured_player_recent_stat(inj.get("player", ""), "ppg")
            if recent_ppg is not None and recent_ppg > 0:
                base_ppg = recent_ppg
            inj_stat = base_ppg * FG3M_PPG_RATIO
        else:
            # v9 Fix 5: prefer recent game-log average over ESPN season average.
            # ESPN's season stats don't reflect role changes or injury returns.
            recent_stat = _get_injured_player_recent_stat(inj.get("player", ""), inj_field)
            if recent_stat is not None and recent_stat > 0:
                inj_stat = recent_stat
            else:
                inj_stat = inj.get(inj_field, 0)

        if inj_stat <= 0:
            inj_stat = team_avg * 0.10

        redistributed = inj_stat * absorption_rate * miss_prob
        inj_group = _pos_group(inj.get("position", ""))
        same_pos  = my_group and inj_group and my_group == inj_group
        player_share_of_team = player_avg / team_avg

        player_claim = player_share_of_team * (SAME_POS_SHARE / 0.20 if same_pos else DIFF_POS_SHARE / 0.80)
        player_claim = min(player_claim, 0.40)
        total_absorbed += redistributed * player_claim

    boost = min(total_absorbed / player_avg if player_avg > 0 else 0, 0.25)
    return 1.0 + boost


# =============================================================================
# PLAYER STD DEV
# =============================================================================
def get_player_std(pid, po_logs, rs_logs, market_key, stat_cols, po_count, min_games=5):
    """Blend po_sd and rs_sd using the same adaptive weight as rates."""
    ck = (pid, market_key)
    if ck in PLAYER_STD_CACHE: return PLAYER_STD_CACHE[ck]

    po_fallback = PROP_STD_DEV_PO.get(market_key, 6.0)
    rs_fallback = PROP_STD_DEV_RS.get(market_key, 5.5)

    def _calc_sd(logs, fallback):
        if logs.empty or len(logs) < min_games: return fallback
        avail = [sc for sc in stat_cols if sc in logs.columns]
        if len(avail) != len(stat_cols): return fallback
        combined = logs[stat_cols].sum(axis=1) if len(stat_cols) > 1 else logs[stat_cols[0]]
        sd = float(combined.std())
        return sd if sd and not math.isnan(sd) and sd > 0 else fallback

    po_sd     = _calc_sd(po_logs, po_fallback)
    rs_sd     = _calc_sd(rs_logs, rs_fallback)
    po_weight = _adaptive_po_weight(po_count)
    blended   = po_weight * po_sd + (1 - po_weight) * rs_sd
    PLAYER_STD_CACHE[ck] = blended
    return blended


# =============================================================================
# PROJECTION ENGINE
# =============================================================================
def _ewm(values, half_life=5):
    n = len(values)
    if n == 0: return None
    w = np.array([0.5 ** (i / half_life) for i in range(n)])
    return float(np.average(values, weights=w))

def _per_min_rate(logs, stat_col, max_g=20):
    sub = logs.head(max_g).copy()
    if sub.empty or stat_col not in sub.columns: return None, None
    if "MIN" not in sub.columns: return None, None
    sub["_m"] = sub["MIN"].apply(_parse_minutes)
    sub = sub[sub["_m"] > 0].copy()
    if sub.empty: return None, None
    sub["_r"] = sub[stat_col].astype(float) / sub["_m"]
    return _ewm(sub["_r"].values), _ewm(sub["_m"].values)

def _trend_factor(logs, stat_col):
    """Compare last-5 average vs last-20 average for a stat to detect form trends.

    Returns (ratio, label) where:
      ratio > 1.10 → "improving"   (last-5 is 10%+ above last-20)
      ratio < 0.88 → "declining"   (last-5 is 12%+ below last-20)
      otherwise    → "neutral"

    Multiplier applied to projection: clamp [0.94, 1.06].
    Uses raw stat totals (not per-minute) since the question is "is this player
    producing more of this stat recently", independent of minute fluctuations.
    """
    if stat_col not in logs.columns or len(logs) < 5:
        return 1.0, "neutral"
    vals = logs[stat_col].astype(float).values
    vals = vals[~np.isnan(vals)]
    if len(vals) < 5:
        return 1.0, "neutral"
    last5_avg  = float(np.mean(vals[:5]))
    last20_avg = float(np.mean(vals[:20])) if len(vals) >= 10 else last5_avg
    if last20_avg <= 0:
        return 1.0, "neutral"
    ratio = last5_avg / last20_avg
    if ratio > 1.10:
        label = "improving"
    elif ratio < 0.88:
        label = "declining"
    else:
        label = "neutral"
    # Clamp multiplier: 0.94 at ratio=0.80, 1.0 at neutral, 1.06 at ratio=1.20
    multiplier = max(0.94, min(1.06, 1.0 + (ratio - 1.0) * 0.5))
    return round(multiplier, 3), label


def _projection_range(logs, stat_col, n=15):
    """Return (P25, P75) of recent raw stat totals as a projected range.
    Returns (None, None) if not enough data."""
    if stat_col not in logs.columns or len(logs) < 5:
        return None, None
    vals = logs.head(n)[stat_col].astype(float).values
    vals = vals[~np.isnan(vals)]
    if len(vals) < 5:
        return None, None
    return round(float(np.percentile(vals, 25)), 1), round(float(np.percentile(vals, 75)), 1)


def _blend_rate(po_rate, rs_rate, po_count, pace_factor=None):
    """Fix 2 + v4 Fix 11: pace deflation only when we have real playoff context.
    At po_count=0 (NO-PO / pre-playoffs), RS rates are used without deflation —
    the 2.5% haircut was over-biasing projections downward and inflating UNDER picks.
    Fix 3: pace_factor=None falls back to PLAYOFF_PACE_FACTOR; pass matchup-specific
    factor from nba_combined.py to avoid using a flat global constant."""
    po_weight = _adaptive_po_weight(po_count)
    # Fix 3: use matchup-specific pace_factor when provided, else fallback constant
    pf = pace_factor if pace_factor is not None else PLAYOFF_PACE_FACTOR
    if po_count > 0 and rs_rate is not None:
        rs_adjusted = rs_rate * pf
    else:
        rs_adjusted = rs_rate

    if po_weight == 0 or po_rate is None:
        # Fix 2: if RS is also None but PO data exists (below threshold), use it
        return rs_adjusted if rs_adjusted is not None else po_rate
    if rs_adjusted is None:
        return po_rate
    return po_weight * po_rate + (1 - po_weight) * rs_adjusted

def _blend_minutes(po_min, rs_min, po_count, pace_factor=None):
    """v6 Fix 1b: ignore playoff minutes until po_count >= 2 — a single playoff
    game's EWM-weighted minutes are round integers that override the RS baseline.
    Playoff rotation adjustment still applies to RS minutes when IS_PLAYOFF_SEASON.
    Fix 3: pace_factor=None falls back to PLAYOFF_PACE_FACTOR."""
    pf = pace_factor if pace_factor is not None else PLAYOFF_PACE_FACTOR

    # Not enough playoff games — use RS minutes entirely
    if po_count < 2:
        if rs_min and rs_min > 0:
            factor = pf if po_count > 0 else 1.0
            base   = rs_min * factor
            if IS_PLAYOFF_SEASON and po_count < PLAYOFF_GAMES_THRESHOLD:
                base = _playoff_rotation_adj(base)
            return base
        factor = pf if po_count > 0 else 1.0
        return 28.0 * factor

    # Enough playoff data to fully trust it
    if po_count >= PLAYOFF_GAMES_THRESHOLD and po_min and po_min > 0:
        return po_min  # actual playoff minutes — already reflect rotation reality

    # Partial trust: blend with adaptive weight
    po_weight = _adaptive_po_weight(po_count)
    if po_min and po_min > 0 and rs_min and rs_min > 0:
        return po_weight * po_min + (1.0 - po_weight) * rs_min * pf

    # Fallback to RS with rotation adjustment
    if rs_min and rs_min > 0:
        base = rs_min * pf
        if IS_PLAYOFF_SEASON:
            base = _playoff_rotation_adj(base)
        return base
    return 28.0 * pf

def get_series_h2h(logs, opp_abbr, stat_col, base_rate, max_games=7):
    """Fix 7: dynamic clamp range = min(0.25, n_games * 0.06) instead of fixed ±0.25.
    1 game = ±6%, 4 games = ±24%, 5+ games = ±25% — prevents single-game 25% swings."""
    if logs.empty or stat_col not in logs.columns or not base_rate or base_rate <= 0:
        return 1.0
    # Fix 1b: word boundary prevents "IND" matching "INDIANA", "LA" matching "LAL"/"LAC"
    h = logs[logs["MATCHUP"].str.contains(rf"\b{opp_abbr}\b", na=False, regex=True)].head(max_games).copy()
    if len(h) < 1: return 1.0
    if "MIN" not in h.columns: return 1.0
    h["_m"] = h["MIN"].apply(_parse_minutes)
    h = h[h["_m"] > 0]
    if h.empty: return 1.0

    total_w = 0.0
    weighted_rate = 0.0
    for i, (_, row) in enumerate(h.iterrows()):
        w = 0.5 ** (i / 1.5)  # exponential decay
        r = float(row[stat_col]) / float(row["_m"])
        weighted_rate += r * w
        total_w       += w

    if total_w == 0: return 1.0
    h2h_rate = weighted_rate / total_w
    ratio    = h2h_rate / base_rate
    reg      = max(0.2, 1.0 - len(h) * 0.15)
    factor   = 1.0 + (ratio - 1.0) * (1.0 - reg)

    # Fix 7: dynamic clamp — scales with number of H2H games available
    clamp_range = min(0.25, len(h) * 0.06)
    return max(1.0 - clamp_range, min(factor, 1.0 + clamp_range))

def _ha_factor(logs, is_home, stat_col):
    if logs.empty or "MATCHUP" not in logs.columns or stat_col not in logs.columns:
        return 1.0
    mask = (logs["MATCHUP"].str.contains(r"\bvs\.", na=False)
            if is_home else logs["MATCHUP"].str.contains(r"@", na=False))
    vg = logs[mask]
    aa = float(logs[stat_col].mean()) if not logs.empty else 0
    if vg.empty or aa <= 0: return 1.0
    va  = float(vg[stat_col].mean())
    n   = len(vg)
    reg = max(0.3, 1.0 - n * 0.03)
    return max(0.85, min(1.0 + (va / aa - 1.0) * (1.0 - reg), 1.15))

def _player_season_avg(logs, stat_col, n=20):
    if logs.empty or stat_col not in logs.columns: return 0.0
    return float(logs.head(n)[stat_col].mean())

def _is_road_b2b(po_logs, is_home):
    """Fix 4: also verify both road games are vs the same opponent abbreviation
    (i.e., they belong to the same playoff series, not two different rounds)."""
    if is_home: return False
    if po_logs.empty or "MATCHUP" not in po_logs.columns or "GAME_DATE" not in po_logs.columns:
        return False
    if len(po_logs) < 2: return False

    g0 = po_logs.iloc[0]
    g1 = po_logs.iloc[1]

    is_away_0 = "@" in str(g0["MATCHUP"]) and "vs." not in str(g0["MATCHUP"])
    is_away_1 = "@" in str(g1["MATCHUP"]) and "vs." not in str(g1["MATCHUP"])
    if not (is_away_0 and is_away_1): return False

    # Fix 4: extract opponent from each MATCHUP and confirm they match
    def _opp_from_matchup(m):
        s = str(m)
        if "@" in s:
            parts = s.split("@")
            return parts[-1].strip() if len(parts) > 1 else ""
        return ""

    opp0 = _opp_from_matchup(g0["MATCHUP"])
    opp1 = _opp_from_matchup(g1["MATCHUP"])
    if opp0 != opp1 or not opp0:
        return False  # different opponents = different rounds, not a true B2B

    try:
        d0 = pd.Timestamp(g0["GAME_DATE"]).date()
        d1 = pd.Timestamp(g1["GAME_DATE"]).date()
        return 0 < (d0 - d1).days <= 3
    except (KeyError, ValueError, TypeError) as exc:
        logger.debug("_is_road_b2b date parse failed: %s", exc)
        return False


def project_stat(player_name, stat_col, opp_abbr, opp_team_id,
                 injuries, pos, po_logs, rs_logs, is_home, team_abbr,
                 using_playoffs, po_count, spread=0.0, pace_factor=None):
    po_rate, po_min = _per_min_rate(po_logs, stat_col) if not po_logs.empty else (None, None)
    rs_rate, rs_min = _per_min_rate(rs_logs, stat_col) if not rs_logs.empty else (None, None)

    # Fix 3: thread matchup-specific pace_factor through rate and minute blending
    rate    = _blend_rate(po_rate, rs_rate, po_count, pace_factor=pace_factor)
    avg_min = _blend_minutes(po_min, rs_min, po_count, pace_factor=pace_factor)

    if rate is None or avg_min is None:
        return None, {}

    mins_reduced = 0.0
    if abs(spread) > BLOWOUT_SPREAD_THRESHOLD and avg_min >= STARTER_MIN_THRESHOLD:
        favored_is_home   = spread > 0
        player_is_favored = (is_home == favored_is_home)
        if player_is_favored:
            mins_reduced = min(BLOWOUT_MAX_MIN_REDUCTION,
                               (abs(spread) - BLOWOUT_SPREAD_THRESHOLD) * 0.5)
            avg_min = max(0.0, avg_min - mins_reduced)

    h2h_logs   = po_logs if (not po_logs.empty and using_playoffs) else rs_logs
    series_h2h = get_series_h2h(h2h_logs, opp_abbr, stat_col, rate)
    def_f      = get_series_def_factor(opp_team_id, stat_col, h2h_logs)
    ha_f       = _ha_factor(h2h_logs, is_home, stat_col)

    # Defense vs Position: blend team-level def_f with per-48 position-specific DvP.
    # dvp_engine uses the same window as IS_PLAYOFF_SEASON (playoffs or season).
    pos_group    = _pos_group(pos) or PLAYER_POS_CACHE.get(player_name)
    # v8 Fix 6: track players with no position data so they can be added to MANUAL_POSITIONS
    if pos_group is None:
        NULL_POS_PLAYERS.add(player_name)
    _dvp_window  = "playoffs" if IS_PLAYOFF_SEASON else "season"
    dvp_f        = dvp_engine.get_dvp_factor(opp_abbr, pos_group, stat_col, _dvp_window)
    # Fix 6: unified additive blend — H2H 20%, team DEF 40%, DvP 40%.
    # Avoids compounding errors from multiplying three independent defensive signals.
    def_unified = 0.20 * series_h2h + 0.40 * def_f + 0.40 * dvp_f

    player_avg  = _player_season_avg(rs_logs, stat_col)
    team_avg    = get_team_stat_avg(team_abbr, stat_col)
    usage_boost = compute_usage_boost(player_name, pos, stat_col,
                                      player_avg, team_avg, injuries,
                                      using_playoffs=using_playoffs)

    # v9 Fix 1: cap the combined multiplier at ±20% to prevent compounding errors.
    # Individual factors are each bounded, but they correlate — a player with an
    # injured teammate (usage↑), a hot streak (trend↑), and a favourable matchup
    # (def/DvP↑) can see all three fire at once, producing runaway projections.
    # v9 Fix 2: trend_factor removed from the pipeline — EWM in _per_min_rate
    # already captures recency; applying trend_mult on top triple-counts the same
    # 3-game hot streak. Keep the function for reference but don't call it here.
    combined_mult          = def_unified * ha_f * usage_boost
    combined_mult_uncapped = combined_mult
    combined_mult          = max(0.80, min(1.20, combined_mult))

    base_proj  = rate * avg_min * combined_mult
    final_proj = base_proj

    # v7 Fix 1c: apply calibration-based market projection correction
    proj_mult  = MARKET_PROJ_ADJ.get(stat_col, 1.0)
    final_proj = final_proj * proj_mult

    # v8 Fix 2: apply player-specific projection penalty for confirmed bad-projection
    # players (overrides if more restrictive than market-level). Suppresses at the
    # player level so the rest of the market flows through normally.
    player_penalty = MANUAL_PLAYER_PENALTIES.get((player_name, stat_col), 1.0)
    final_proj = final_proj * player_penalty

    # v8 Fix 3: apply data-driven per-player per-stat residual nudge from bet_log.
    # Additive in stat units (actual − projection mean residual, shrinkage-weighted).
    # Positive → model historically under-projects → nudge up; negative → nudge down.
    data_adj = PLAYER_PROJ_ADJ.get((player_name, stat_col), 0.0)
    if data_adj != 0.0:
        final_proj = max(0.0, final_proj + data_adj)

    # Blend with learned regression model if train_model.py has been run and
    # learned_params.json exists. The learned model uses EWMA projection directly
    # as its primary feature; we blend (80% current / 20% learned) so the
    # existing feature engineering is preserved while benefiting from fitted weights.
    lp = _LEARNED_PLAYER_PARAMS.get(stat_col)
    learned_proj_adj = 0.0
    if lp:
        try:
            ewma_proj_for_stat = rate * avg_min  # base EWMA projection (pre-multiplier)
            _lp_intercept   = lp.get("intercept", 0.0)
            _lp_ewma_weight = lp.get("ewma_weight", 1.0)
            _lp_home_boost  = lp.get("home_boost", 0.0)
            _lp_is_home     = 1.0 if is_home else 0.0
            learned_pred = (_lp_intercept
                            + _lp_ewma_weight * ewma_proj_for_stat
                            + _lp_home_boost  * _lp_is_home)
            if learned_pred > 0:
                blend = 0.15   # 15% learned, 85% existing pipeline
                blended = (1 - blend) * final_proj + blend * learned_pred * combined_mult
                learned_proj_adj = blended - final_proj
                final_proj = max(0.0, blended)
        except Exception:
            pass

    detail = {
        "rate":                    round(rate, 4),
        "min":                     round(avg_min, 1),
        "min_reduced":             round(mins_reduced, 1),
        "series_h2h":              round(series_h2h, 3),
        "def":                     round(def_unified, 3),
        "def_team":                round(def_f, 3),
        "dvp":                     round(dvp_f, 3),
        "ha":                      round(ha_f, 3),
        "usage":                   round(usage_boost, 3),
        "combined_mult":           round(combined_mult, 3),           # v9 Fix 1
        "combined_mult_uncapped":  round(combined_mult_uncapped, 3),  # v9 Fix 1
        "trend_mult":              1.0,                               # v9 Fix 2: removed
        "trend":                   "n/a (EWM)",                       # v9 Fix 2: removed
        "proj_mult":               round(proj_mult, 3),
        "player_penalty":          round(player_penalty, 3),          # v8 Fix 2
        "data_adj":                round(data_adj, 2),                # v8 Fix 3
        "learned_adj":             round(learned_proj_adj, 2),        # from train_model.py
        "base":                    round(base_proj, 1),
        "final":                   round(final_proj, 1),
        "po_games":                len(po_logs),
    }
    return final_proj, detail


def project_prop(player_name, market_key, opp_abbr, opp_team_id,
                 injuries, pos, is_home, team_abbr, spread=0.0, pace_factor=None):
    """Returns (projection, details, sd, metadata)."""
    stat_cols = MARKET_TO_STATS.get(market_key)
    if not stat_cols: return None, None, None, None

    # Fix 5: skip or penalise players who are themselves on the injury report
    self_status = next((inj["status"] for inj in injuries
                        if inj.get("player") == player_name), None)
    self_inj_penalty = 0.0
    self_inj_flag    = ""
    if self_status in ("Out", "Doubtful"):
        return None, None, None, None  # no props should exist, but guard anyway
    elif self_status in ("Questionable", "Day-To-Day"):
        self_inj_penalty = 5.0
        self_inj_flag    = "⚠GTD"
    elif self_status == "Probable":
        self_inj_penalty = 1.0
        self_inj_flag    = "PROB"

    pid = get_player_id(player_name)
    if not pid: return None, None, None, None

    po_logs, rs_logs, using_playoffs, po_count, po_raw_count = get_player_logs_blended(pid)
    if po_logs.empty and rs_logs.empty: return None, None, None, None

    # Projected range (P25–P75) using best available logs for the primary stat
    primary_stat = MARKET_TO_STATS.get(market_key, [None])[0]
    range_logs   = po_logs if (not po_logs.empty and po_count >= 3) else rs_logs
    proj_low, proj_high = (
        _projection_range(range_logs, primary_stat)
        if primary_stat else (None, None)
    )
    # For combo markets (PTS+REB etc.), scale range proportionally by #stats
    if proj_low is not None and len(MARKET_TO_STATS.get(market_key, [])) > 1:
        n_stats = len(MARKET_TO_STATS[market_key])
        proj_low  = round(proj_low  * n_stats * 0.85, 1)  # slight discount for combos
        proj_high = round(proj_high * n_stats * 1.00, 1)

    total   = 0.0
    details = {}
    for sc in stat_cols:
        p, d = project_stat(player_name, sc, opp_abbr, opp_team_id,
                            injuries, pos, po_logs, rs_logs,
                            is_home, team_abbr, using_playoffs, po_count,
                            spread=spread, pace_factor=pace_factor)
        if p is None: return None, None, None, None
        total   += p
        details[sc] = d

    # Compute max usage across all stat components
    max_usage = max((d.get("usage", 1.0) for d in details.values()), default=1.0)

    # Fix 8: negative correlation correction for combo markets containing PTS + AST.
    # When a player absorbs extra usage/scoring load, playmaking opportunities drop —
    # iso/post usage means the ball ends up in their hands more, not distributed.
    if "PTS" in stat_cols and "AST" in stat_cols and max_usage > 1.05:
        ast_raw = details.get("AST", {}).get("final", 0.0)
        ast_correction = ast_raw * 0.03  # reduce AST component by 3%
        total -= ast_correction
        # Fix 2: shallow copy prevents mutation of shared dict across markets
        details["AST"] = {**details["AST"], "final": round(details["AST"]["final"] - ast_correction, 1)}

    # Usage Rate adjustment: high-usage players generate more of their own PTS;
    # low-usage players are more scheme-dependent and typically over-project.
    # Applied only to markets containing PTS (USG measures scoring possession use).
    usg_factor = 1.0
    if market_key in USG_MARKETS:
        usg = _lookup_usg(player_name)  # v6 Fix 5: fuzzy lookup handles diacritics/suffixes
        if usg > 0:
            # Deviation from league average, scaled by USG_PROJ_WEIGHT.
            # e.g. Giannis 35% USG → (0.35-0.215)/0.215 * 0.10 = +0.063 → ×1.063
            # e.g. bench G  12% USG → (0.12-0.215)/0.215 * 0.10 = -0.044 → ×0.956
            raw_adj = (usg - LEAGUE_AVG_USG) / LEAGUE_AVG_USG * USG_PROJ_WEIGHT
            usg_factor = max(0.93, min(1.0 + raw_adj, 1.07))  # clamp ±7%
            total *= usg_factor

    sd = get_player_std(pid, po_logs, rs_logs, market_key, stat_cols, po_count)

    # Fix 5: scale SD proportionally when defensive/usage factors inflate the projection.
    # raw_total = sum(rate × min) — the projection without any multiplier adjustments.
    # If total >> raw_total (big inflation), the true distribution is wider; scale SD up.
    raw_total = sum(d.get("rate", 0.0) * d.get("min", 0.0) for d in details.values())
    if raw_total > 0 and total > 0 and sd and sd > 0:
        inflation_ratio = total / raw_total
        sd = sd * (inflation_ratio ** 0.5)

    # Calibration: apply per-(player, market) projection bias learned from bet_log.csv.
    # adj is the shrinkage-weighted mean residual (actual − projection).
    # Positive adj → model historically under-projects this player → nudge up.
    calib_label    = MARKET_LABELS.get(market_key, market_key)
    calib_pm_key   = f"{player_name}|{calib_label}"
    calib_proj_adj = CALIB.get("player_market", {}).get(calib_pm_key, {}).get("adj", 0.0)
    if calib_proj_adj != 0.0:
        total += calib_proj_adj

    po_min  = _avg_minutes(po_logs)
    rs_min  = _avg_minutes(rs_logs)
    avg_min = po_min if po_min > 0 else rs_min

    # Playoff rotation multiplier — computed for flagging purposes.
    # Reflects whether this player's minutes were adjusted up/down for early playoffs.
    po_min_adj_mult = 1.0
    if IS_PLAYOFF_SEASON and po_count < PLAYOFF_GAMES_THRESHOLD and rs_min > 0:
        if rs_min >= STARTER_MIN_THRESHOLD:
            po_min_adj_mult = PLAYOFF_MIN_STARTER_MULT
        elif rs_min >= PLAYOFF_ROTATION_RISK_MIN:
            po_min_adj_mult = PLAYOFF_MIN_ROTATION_MULT
        else:
            po_min_adj_mult = PLAYOFF_MIN_BENCH_MULT

    # Proportional bench penalty
    bench_penalty = max(0.0, (BENCH_MIN_THRESHOLD - avg_min) * 1.5) if avg_min > 0 else 0.0

    # Minute-volatility penalty
    mins_vol_penalty = 0.0
    mins_vol_flag    = False
    if not po_logs.empty and len(po_logs) >= 3:
        cv = _std_minutes(po_logs) / po_min if po_min > 0 else 0.0
        if cv > 0.25:
            mins_vol_penalty = min(8.0, (cv - 0.20) * 25)
            mins_vol_flag    = True

    # Road B2B penalty
    road_b2b         = _is_road_b2b(po_logs, is_home)
    road_b2b_penalty = 3.0 if road_b2b else 0.0

    total_conf_penalty = bench_penalty + mins_vol_penalty + road_b2b_penalty + self_inj_penalty

    metadata = {
        "avg_min":          avg_min,
        "conf_penalty":     total_conf_penalty,
        "bench_penalty":    bench_penalty,
        "mins_vol_penalty": mins_vol_penalty,
        "road_b2b_penalty": road_b2b_penalty,
        "self_inj_penalty": self_inj_penalty,
        "is_bench":         avg_min > 0 and avg_min < BENCH_MIN_THRESHOLD,
        "mins_vol_flag":    mins_vol_flag,
        "road_b2b":         road_b2b,
        "self_inj_flag":    self_inj_flag,
        "po_count":         po_count,
        "po_raw_count":     po_raw_count,   # unfiltered — for NO-PO flag only
        "using_playoffs":   using_playoffs,
        "max_usage":        max_usage,
        "usg_pct":          _lookup_usg(player_name),              # v6 Fix 5
        "usg_factor":       round(usg_factor, 3),
        "po_min_adj_mult":  round(po_min_adj_mult, 3),
        "proj_low":         proj_low,
        "proj_high":        proj_high,
    }

    return round(total, 1), details, sd, metadata


# =============================================================================
# PROBABILITY
# =============================================================================
def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _nb_over_prob(proj, sd, line):
    """Fix 4: Negative Binomial P(X > line) for discrete counting stats.
    Parameterised by mean=proj and variance=sd^2.  Pure Python (no scipy).
    Falls back to Normal CDF when variance <= mean (Poisson-like regime)."""
    if proj <= 0 or sd <= 0:
        return 0.5
    var = sd * sd
    if var <= proj:
        # variance ≤ mean → use Normal CDF as fallback
        z = (proj - line) / sd
        return normal_cdf(z)
    # NB parameters: r = mean^2 / (var - mean),  p = mean / var
    r = proj * proj / (var - proj)
    p = proj / var          # probability of success per trial
    q = 1.0 - p
    if r <= 0 or p <= 0 or p >= 1:
        z = (proj - line) / sd
        return normal_cdf(z)

    # P(X <= floor(line)) via NB CDF using log-gamma recursion
    k_max = max(0, int(math.floor(line)))
    log_q = math.log(q)
    log_p = math.log(p)
    # log P(X=0) = r * log(p)
    log_pmf = r * log_p
    cdf = math.exp(log_pmf)
    for k in range(1, k_max + 1):
        log_pmf += math.log(r + k - 1) - math.log(k) + log_q
        cdf += math.exp(log_pmf)
        if cdf >= 1.0:
            return 0.0   # essentially zero chance of going over
    return max(0.0, min(1.0, 1.0 - cdf))

def evaluate_prop(line, proj, market_key, sd=None, conf_penalty=0.0):
    if sd is None or sd <= 0:
        sd = PROP_STD_DEV_PO.get(market_key, 5.0)
    edge     = proj - line
    edge_pct = abs(edge) / line * 100 if line and line > 0 else 0

    # Fix 4: use Negative Binomial for single-stat discrete count markets
    if market_key in SINGLE_STAT_MARKETS:
        over_p = _nb_over_prob(proj, sd, line)
    else:
        z      = edge / sd
        over_p = normal_cdf(z)
    under_p  = 1.0 - over_p
    conf     = max(50.0, max(over_p, under_p) * 100 - conf_penalty)

    return {
        "edge":       round(edge, 1),
        "edge_pct":   round(edge_pct, 1),
        "over_prob":  round(over_p * 100, 1),
        "under_prob": round(under_p * 100, 1),
        "confidence": round(conf, 1),
        "pick":       "OVER" if edge > 0 else "UNDER",
    }


# =============================================================================
# TEAM ASSIGNMENT
# =============================================================================
def _parse_matchup_teams(matchup_str):
    """Fix 1a: parse 'TEAM vs. OPP' or 'TEAM @ OPP' into exact abbreviations.
    Returns (player_team_abbr, opp_abbr) or (None, None) if unparseable."""
    for sep in (" vs. ", " @ "):
        if sep in matchup_str:
            parts = matchup_str.split(sep, 1)
            return parts[0].strip(), parts[1].strip()
    return None, None

def determine_team(po_logs, rs_logs, home_abbr, away_abbr):
    """Count how often the player's own team (t — always the first MATCHUP token)
    matches home_abbr vs away_abbr.  Never use the opponent token (o): in the
    playoffs every recent game is vs the SAME opponent, so counting o creates a
    tie on every row and makes the function fall through to wrong data.
    Normalizes ESPN abbreviations (SA→SAS, GS→GSW, etc.) before comparing against
    the NBA API MATCHUP field which always uses canonical 3-letter codes."""
    h = ESPN_TO_NBA_ABBR.get(home_abbr, home_abbr)
    a = ESPN_TO_NBA_ABBR.get(away_abbr, away_abbr)
    for logs in [po_logs, rs_logs]:
        if logs.empty or "MATCHUP" not in logs.columns: continue
        recent = logs.head(5)
        hc = 0; ac = 0
        for _, r in recent.iterrows():
            t, _ = _parse_matchup_teams(str(r["MATCHUP"]))
            if t is None: continue
            if t == h:   hc += 1
            elif t == a: ac += 1
        if hc > ac: return {"team_abbr": home_abbr, "opp_abbr": away_abbr, "is_home": True}
        elif ac > hc: return {"team_abbr": away_abbr, "opp_abbr": home_abbr, "is_home": False}
    return None

def get_player_position_from_injuries(player_name, injuries_by_team):
    for _, injs in injuries_by_team.items():
        for inj in injs:
            if inj["player"] == player_name:
                return inj.get("position", "")
    return ""


# =============================================================================
# GAME PROCESSOR
# =============================================================================
def process_game(home_name, away_name, home_abbr, away_abbr,
                 odds_event, injuries_by_team, spread=0.0, pace_factor=None):
    home_tid = get_team_id(home_abbr)
    away_tid = get_team_id(away_abbr)
    try:
        pj = get_event_player_props(odds_event["id"])
    except Exception as e:
        print(f"  Error fetching props: {e}"); return []

    bookmakers = pj.get("bookmakers", [])
    if not bookmakers:
        commence = odds_event.get("commence_time", "")
        time_hint = f" (tip-off: {commence[:16].replace('T',' ')} UTC)" if commence else ""
        print(f"  No bookmakers returned props — lines not posted yet{time_hint}.")
        print(f"  Props typically go up 2–4 hours before tip-off. Try again later.")
        return []

    n_markets = sum(len(b.get("markets",[])) for b in bookmakers)
    print(f"  {len(bookmakers)} book(s), {n_markets} market(s)")

    pdf = flatten_props(pj)
    if pdf.empty:
        print("  Props empty after parsing."); return []
    cdf = consensus_lines(pdf)  # Fix 6: now returns book_count column too
    if cdf.empty:
        print("  No consensus lines."); return []

    print(f"  {len(cdf)} consensus lines, {cdf['player'].nunique()} players")

    ctx = {}
    for pl in cdf["player"].unique():
        pid = get_player_id(pl)
        if not pid: continue
        po_logs, rs_logs, _, _, _ = get_player_logs_blended(pid)
        ti = determine_team(po_logs, rs_logs, home_abbr, away_abbr)
        if ti is None: continue
        ih      = ti["is_home"]
        team_nm = home_name if ih else away_name
        pos     = get_player_position_from_injuries(pl, injuries_by_team)
        ctx[pl] = {
            "team_abbr": ti["team_abbr"],
            "opp_abbr":  ti["opp_abbr"],
            "opp_tid":   away_tid if ih else home_tid,
            "injuries":  injuries_by_team.get(team_nm, []),
            "is_home":   ih,
            "pos":       pos,
        }

    results = []
    for _, row in cdf.iterrows():
        pl, mkt, line = row["player"], row["market"], row["consensus_line"]
        label      = MARKET_LABELS.get(mkt, mkt)
        book_count = int(row.get("book_count", 0))  # Fix 6
        if pd.isna(line) or pl not in ctx: continue
        c = ctx[pl]

        proj, details, sd, meta = project_prop(
            pl, mkt, c["opp_abbr"], c["opp_tid"],
            c["injuries"], c["pos"], c["is_home"], c["team_abbr"],
            spread=spread, pace_factor=pace_factor)
        if proj is None: continue

        # v6 Fix 6: soft anchor toward the market line — dampens runaway projections.
        # Real markets don't misprice by 50%+; if the model disagrees that much,
        # it's usually the model that's wrong. 12% pull toward line reduces
        # catastrophic overfitting to historical data quirks (tiny samples, stale H2H).
        proj = (1.0 - MARKET_ANCHOR_WEIGHT) * proj + MARKET_ANCHOR_WEIGHT * line

        # Fix 6: thin-market confidence penalty based on book count
        if book_count == 1:
            book_penalty = 4.0
            book_flag    = "1-BOOK"
        elif book_count == 2:
            book_penalty = 2.0
            book_flag    = "2-BOOK"
        else:
            book_penalty = 0.0
            book_flag    = ""

        # v4 Fixes 12-14: direction-aware confidence adjustments
        pick_dir          = "UNDER" if proj < line else "OVER"
        direction_penalty = 0.0
        adj_sd            = sd

        if pick_dir == "UNDER":
            # Fix 12: flat UNDER penalty (reduced from 5pp — calibration handles market-specific)
            direction_penalty += UNDER_CONF_PENALTY
            # Fix 13: combo market SD inflation — widen uncertainty for multi-stat UNDERs
            if mkt in COMBO_UNDER_MARKETS and sd:
                adj_sd = sd * COMBO_UNDER_SD_MULT
            # Fix 14: star blowup buffer — high-proj players can easily eclipse UNDER lines
            if proj >= STAR_PROJ_THRESHOLD:
                direction_penalty += STAR_UNDER_PENALTY

        # Postmortem fix: PTS+REB OVER hits only 25% — add structural penalty
        if mkt == "player_points_rebounds" and pick_dir == "OVER":
            direction_penalty += PTS_REB_OVER_PENALTY

        # Postmortem fix: NO-PO during playoffs = RS rates don't reflect playoff role.
        # Coaches tighten rotations, scoring drops, bench players get DNP'd.
        po_count_local     = meta["po_count"]
        po_raw_count_local = meta.get("po_raw_count", po_count_local)
        if IS_PLAYOFF_SEASON and po_raw_count_local == 0:
            direction_penalty += PLAYOFF_NO_PO_PENALTY

        # Low-line penalty: tiny lines (≤ 1.5) generate inflated edge% that dominate rankings.
        if line <= LOW_LINE_THRESHOLD:
            direction_penalty += LOW_LINE_CONF_PENALTY

        # Calibration: market-direction hit-rate adjustment from bet_log history.
        # conf_adj > 0 means this (market, direction) hits above baseline → ease penalty.
        # conf_adj < 0 means it hits below baseline → tighten penalty.
        calib_md_key = f"{label}_{pick_dir}"
        calib_adj    = CALIB.get("market_direction", {}).get(calib_md_key, {}).get("conf_adj", 0.0)
        direction_penalty -= calib_adj   # positive adj reduces penalty; negative adds to it

        total_penalty = meta["conf_penalty"] + book_penalty + direction_penalty

        ev = evaluate_prop(line, proj, mkt, sd=adj_sd, conf_penalty=total_penalty)

        # v6 Fix 3: suspicious edge% filter — real markets don't misprice by >40%.
        # High edge% signals model error (bad data, tiny sample) not alpha.
        # Dock confidence heavily so these fall out of top-N rankings.
        if ev["edge_pct"] > SUSPICIOUS_EDGE_PCT:
            ev["confidence"] = max(50.0, ev["confidence"] - 20.0)
            ev["_suspicious"] = True
        else:
            ev["_suspicious"] = False

        po_count = meta["po_count"]
        min_red  = max((d.get("min_reduced", 0.0) for d in details.values()), default=0.0)

        flags = []
        if ev.get("_suspicious"):  flags.append(f"⚠SUSP-{ev['edge_pct']:.0f}%")   # v6 Fix 3
        if meta["is_bench"]:       flags.append("⚠BENCH")
        if po_raw_count_local == 0:    flags.append("NO-PO")       # truly no playoff appearances
        elif po_count_local == 1:      flags.append("LOW-PO")     # 1 qualifying game — EWM unreliable
        if IS_PLAYOFF_SEASON and po_raw_count_local == 0: flags.append("⚠PO-ROLE")
        if line <= LOW_LINE_THRESHOLD: flags.append("LOW-LINE")
        if meta["mins_vol_flag"]:  flags.append("⚠MINS-VOL")
        if meta["road_b2b"]:       flags.append("ROAD-B2B")
        if meta["self_inj_flag"]:  flags.append(meta["self_inj_flag"])  # Fix 5
        if book_flag:              flags.append(book_flag)              # Fix 6
        if meta["max_usage"] > 1.05: flags.append(f"USG+{round((meta['max_usage']-1)*100):.0f}%")
        # DvP and USG_PCT flags — 'details' is the stat breakdown dict in scope here
        dvp_vals = [d.get("dvp", 1.0) for d in details.values() if isinstance(d, dict)]
        if dvp_vals:
            avg_dvp = sum(dvp_vals) / len(dvp_vals)
            if avg_dvp >= 1.12:   flags.append(f"DvP+{avg_dvp:.2f}")  # soft matchup
            elif avg_dvp <= 0.88: flags.append(f"DvP-{avg_dvp:.2f}") # tough matchup
        usg_f = meta.get("usg_factor", 1.0)
        if usg_f >= 1.04:   flags.append("HI-USG")   # high-usage projection boost
        elif usg_f <= 0.96: flags.append("LO-USG")   # low-usage projection penalty
        if min_red > 0:            flags.append(f"BLW-{min_red:.0f}m")
        # v4 Fix 14: flag star blowup buffer
        if pick_dir == "UNDER" and proj >= STAR_PROJ_THRESHOLD:
            flags.append("★BLW-BUF")
        # Playoff rotation minute adjustment — ROT-RISK only; PO-MIN+ removed (v6 Fix 1c)
        po_adj = meta.get("po_min_adj_mult", 1.0)
        if po_adj <= PLAYOFF_MIN_BENCH_MULT + 0.05:   flags.append("ROT-RISK")   # heavy bench cut

        # Trend flags — derived from per-stat trend labels in details dict
        trend_labels = [d.get("trend", "neutral") for d in details.values() if isinstance(d, dict)]
        dominant_trend = (
            "declining"  if trend_labels.count("declining")  > len(trend_labels) // 2 else
            "improving"  if trend_labels.count("improving")  > len(trend_labels) // 2 else
            "neutral"
        )
        if dominant_trend == "declining":
            flags.append("TREND↓")
            if pick_dir == "OVER":
                ev["confidence"] = max(50.0, ev["confidence"] - 3.0)
        elif dominant_trend == "improving":
            flags.append("TREND↑")

        # STRONG flag: P25 (OVER) or P75 (UNDER) clears the line — even pessimistic range beats it
        proj_low  = meta.get("proj_low")
        proj_high = meta.get("proj_high")
        is_strong = (
            (pick_dir == "OVER"  and proj_low  is not None and proj_low  > line) or
            (pick_dir == "UNDER" and proj_high is not None and proj_high < line)
        )
        if is_strong:
            flags.append("★STRONG")

        # Confidence label (Low / Med / High) based on final confidence
        final_conf = ev["confidence"]
        conf_label = "High" if final_conf >= 66 else ("Med" if final_conf >= 56 else "Low")

        results.append({
            "Player":           pl,
            "Team":             c["team_abbr"],
            "Market":           label,
            "Line":             line,
            "Projection":       proj,
            "Edge":             ev["edge"],
            "Edge%":            ev["edge_pct"],
            "Confidence":       ev["confidence"],
            "Conf_Label":       conf_label,
            "Pick":             ev["pick"],
            "PO_Games":         po_count_local,
            "Is_Bench":         meta["is_bench"],
            "Flags":            " ".join(flags),
            "Proj_Low":         proj_low,
            "Proj_High":        proj_high,
            "details":          details,
            "meta":             meta,
            "book_count":       book_count,
            "direction_penalty": direction_penalty,  # v4: for verbose output
        })

    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Fix 10: --verbose flag prints per-factor breakdown for top-5-by-edge bets
    verbose = "--verbose" in sys.argv

    # Load calibration data produced by calibrate.py (no-op if file absent)
    _load_calibration()

    print("\n" + "=" * 68)
    print("  NBA PLAYER PROPS — PLAYOFF EDITION v4")
    print("  Adaptive blend · series DEF · UNDER penalty · blowup buffer · edge floor")
    if verbose:
        print("  [VERBOSE MODE ON]")
    print("=" * 68)

    print("\n  Fetching events...")
    try:
        all_events = get_all_odds_events()
    except Exception as e:
        print(f"  Error: {e}"); return
    if not all_events:
        print("  No upcoming NBA events."); return

    dates = []
    for ev in all_events:
        ct = ev.get("commence_time", "")
        if ct:
            d = _local_date(ct)
            if d: dates.append(d)
    if not dates:
        print("  No dates found."); return

    soonest = min(dates)
    gday = [ev for ev in all_events if _local_date(ev.get("commence_time","")) == soonest]
    tag  = "Today" if soonest == date.today() else soonest.strftime("%A, %B %d")
    print(f"  {len(gday)} game(s) on {tag}\n")

    # Fix 9: auto-populate GAME_SPREADS from Odds API spreads market
    print("  Fetching spreads...")
    for ev in gday:
        hn       = ev.get("home_team", "")
        an       = ev.get("away_team", "")
        ha       = TEAM_NAME_TO_ABBR.get(hn, hn[:3].upper())
        aa       = TEAM_NAME_TO_ABBR.get(an, an[:3].upper())
        game_key = f"{aa}@{ha}"
        if game_key not in GAME_SPREADS:
            # Only fetch if not already manually set — auto-fetch from API
            sprd = _auto_fetch_spread(ev["id"], hn)
            GAME_SPREADS[game_key] = sprd
            if sprd != 0.0:
                print(f"    {game_key}: spread {sprd:+.1f} (home favored)" if sprd > 0
                      else f"    {game_key}: spread {sprd:+.1f} (away favored)")

    print("  Fetching injuries...")
    injuries = get_espn_injuries()
    inj_count = sum(len(v) for v in injuries.values())
    print(f"  {inj_count} player(s) on injury report")
    for tn, injs in injuries.items():
        sig = [i for i in injs if i["status"] in ("Out", "Doubtful")]
        if sig:
            abbr  = TEAM_NAME_TO_ABBR.get(tn, "???")
            names = ", ".join(f"{i['player']} ({i['status']})" for i in sig[:4])
            print(f"    {abbr}: {names}")

    print("\n  Loading team stats and supplementary data...")
    get_team_season_stats()
    get_league_logs()
    get_all_player_usage()          # USG_PCT for all players (one API call, cached)
    prefetch_rosters_for_games(gday) # team roster positions for DvP (2 calls/team)

    for ev in gday:
        hn = ev.get("home_team", "")
        an = ev.get("away_team", "")
        ha = TEAM_NAME_TO_ABBR.get(hn, hn[:3].upper())
        aa = TEAM_NAME_TO_ABBR.get(an, an[:3].upper())

        game_key = f"{aa}@{ha}"
        spread   = GAME_SPREADS.get(game_key, 0.0)

        ct = ev.get("commence_time", "")
        tip_str = ""
        if ct:
            try:
                tip_utc   = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                tip_local = tip_utc.astimezone()
                tip_str   = f"  │  {tip_local.strftime('%I:%M %p')}"
            except (ValueError, OSError) as exc:
                logger.debug("tip-time format failed: %s", exc)

        spread_str = f"  │  Sprd {spread:+.1f}" if spread != 0.0 else ""
        print(f"\n{'━'*68}")
        print(f"  {aa} @ {ha}{tip_str}{spread_str}")
        print(f"{'━'*68}")

        results = process_game(hn, an, ha, aa, ev, injuries, spread=spread)
        if not results:
            print("  No props available."); continue

        df = pd.DataFrame(results)

        # v4 Fix 15: apply per-direction edge% floors before building top-10 lists.
        # UNDERs must clear UNDER_EDGE_PCT_FLOOR (20%) — low-edge UNDERs were 0-3
        # in Apr 15 backtesting (Garland 17.9%, Curry 17.7%, Kawhi 10.1% all missed).
        def _edge_floored(df_in):
            if df_in.empty:
                return df_in
            keep = df_in.apply(
                lambda r: r["Edge%"] >= UNDER_EDGE_PCT_FLOOR if r["Pick"] == "UNDER"
                          else r["Edge%"] >= OVER_EDGE_PCT_FLOOR,
                axis=1,
            )
            return df_in[keep].reset_index(drop=True)

        df_top = _edge_floored(df)

        top_edge = (df_top.sort_values("Edge%", ascending=False)
                          .drop_duplicates(subset=["Player"], keep="first")
                          .head(TOP_BETS_PER_GAME)
                          .reset_index(drop=True))

        top_conf = (df_top.sort_values("Confidence", ascending=False)
                          .drop_duplicates(subset=["Player"], keep="first")
                          .head(TOP_BETS_PER_GAME)
                          .reset_index(drop=True))

        def _print_top(title, top_df, show_verbose=False, verbose_limit=5):
            if top_df.empty: return
            print(f"\n  {title}")
            print(f"  {'Player':<22} {'Mkt':<9} {'Line':>5} {'Proj':>5} "
                  f"{'Edge':>5} {'E%':>5} {'Conf':>5}  Pick    Flags")
            print(f"  {'─'*22} {'─'*9} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5}  {'─'*6}  {'─'*16}")
            for rank, (_, r) in enumerate(top_df.iterrows()):
                print(f"  {r['Player']:<22} {r['Market']:<9} {r['Line']:>5.1f} "
                      f"{r['Projection']:>5.1f} {r['Edge']:>+5.1f} "
                      f"{r['Edge%']:>4.1f}% {r['Confidence']:>4.1f}%  "
                      f"{r['Pick']:<6}  {r['Flags']}")

                # Fix 10: verbose breakdown for top-5-by-edge bets only
                if show_verbose and rank < verbose_limit:
                    dets = r.get("details", {})
                    meta = r.get("meta", {})
                    for sc, d in dets.items():
                        line_parts = (
                            f"    └─ {sc}: rate={d.get('rate',0):.4f}/min"
                            f" × {d.get('min',0):.1f}min"
                            f" × H2H={d.get('series_h2h',1):.3f}"
                            f" × DEF={d.get('def_team',d.get('def',1)):.3f}"
                            f" × DvP={d.get('dvp',1):.3f}"
                            f" × HA={d.get('ha',1):.3f}"
                            f" × USG={d.get('usage',1):.3f}"
                            f" → {d.get('final',0):.1f}"
                        )
                        if d.get("min_reduced", 0) > 0:
                            line_parts += f"  [blowout -{d['min_reduced']:.1f}min]"
                        print(line_parts)
                    # Confidence penalties breakdown
                    pen_parts = []
                    if meta.get("bench_penalty",   0) > 0: pen_parts.append(f"bench={meta['bench_penalty']:.1f}")
                    if meta.get("mins_vol_penalty", 0) > 0: pen_parts.append(f"vol={meta['mins_vol_penalty']:.1f}")
                    if meta.get("road_b2b_penalty", 0) > 0: pen_parts.append(f"b2b={meta['road_b2b_penalty']:.1f}")
                    if meta.get("self_inj_penalty", 0) > 0: pen_parts.append(f"inj={meta['self_inj_penalty']:.1f}")
                    bk = r.get("book_count", 0)
                    book_pen = 4.0 if bk == 1 else (2.0 if bk == 2 else 0.0)
                    if book_pen > 0: pen_parts.append(f"book={book_pen:.1f}")
                    dp = r.get("direction_penalty", 0.0)
                    if dp > 0: pen_parts.append(f"under_dir={dp:.1f}")  # v4
                    if pen_parts:
                        print(f"    └─ penalties: {', '.join(pen_parts)}")

        _print_top(f"TOP {TOP_BETS_PER_GAME} BY EDGE%  (biggest mispricing)",
                   top_edge, show_verbose=verbose, verbose_limit=5)
        _print_top(f"TOP {TOP_BETS_PER_GAME} BY CONFIDENCE  (most likely to hit)",
                   top_conf, show_verbose=False)

        no_po  = df[df["PO_Games"] == 0]
        bench  = df[df["Is_Bench"] == True]
        b2b    = df[df["Flags"].str.contains("ROAD-B2B", na=False)]
        vol    = df[df["Flags"].str.contains("MINS-VOL", na=False)]
        thin   = df[df["Flags"].str.contains("BOOK", na=False)]
        gtd    = df[df["Flags"].str.contains("GTD|PROB", na=False)]
        if not no_po.empty:
            print(f"\n  ⚠  {no_po['Player'].nunique()} player(s) with no playoff data — RS only")
        if not bench.empty:
            print(f"  ⚠  {bench['Player'].nunique()} bench player(s) — sliding confidence penalty")
        if not vol.empty:
            print(f"  ⚠  {vol['Player'].nunique()} player(s) with high minute volatility (⚠MINS-VOL)")
        if not b2b.empty:
            print(f"  ⚠  {b2b['Player'].nunique()} player(s) on road back-to-back (ROAD-B2B, -3pp)")
        if not thin.empty:
            print(f"  ⚠  {thin['Player'].nunique()} player(s) with thin market coverage (1/2-BOOK)")
        if not gtd.empty:
            print(f"  ⚠  {gtd['Player'].nunique()} player(s) on own injury report (⚠GTD/PROB)")

        print(f"\n  {len(df)} prop(s) analyzed")

    print(f"\n{'='*68}")
    print("  Not financial advice.")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
