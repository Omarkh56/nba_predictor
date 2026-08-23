"""
dvp.py — Defense vs Position (DvP) Engine
==========================================
Builds per-48 defensive tables for all (team × position) pairs, matching the
methodology of hashtagbasketball.com's DvP tables.

Methodology
-----------
For every player game log:
  • The player's team is the OFFENSE
  • The opponent extracted from MATCHUP is the DEFENSE
  • The player's position (PG/SG/SF/PF/C) comes from PLAYER_POS_CACHE
  → Stat accumulates to: defense_team × player_position

Aggregation per (defense_team, position) bucket:
  • Counting stats (PTS, REB, AST, STL, BLK, TO, 3PM) → per-48 minutes
  • Percentages (FG%, FT%) → aggregate numerator / aggregate denominator

DvP factor (used by projection model)
  factor = team_per48 / league_avg_per48_for_same_position
  > 1.0 → soft defense (allows more than avg) → boost projection
  < 1.0 → tough defense (allows less than avg) → reduce projection
  Clamped to [0.78, 1.28].

Time windows supported: "season", "playoffs", "last_7", "last_14", "last_30"

Integration with playerlinepredictor.py
-----------------------------------------
  import dvp
  dvp.inject_logs(df, "playoffs")        # pass already-fetched logs
  dvp.inject_positions(PLAYER_POS_CACHE) # pass roster position map
  dvp.build_dvp_table("playoffs")        # pre-build (optional — lazy otherwise)
  factor = dvp.get_dvp_factor("BOS", "PG", "PTS", "playoffs")

Standalone use
--------------
  python3 dvp.py                  # print PTS DvP table (playoffs)
  python3 dvp.py --stat REB --window last_14 --pos SF
"""

import json
import logging
import re
import sys
import time
import warnings
import numpy as np
import pandas as pd
import requests.exceptions
from datetime import date, timedelta

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

try:
    from nba_api.stats.endpoints import leaguegamefinder
    _NBA_API_AVAILABLE = True
except ImportError:
    _NBA_API_AVAILABLE = False

# =============================================================================
# CONSTANTS
# =============================================================================
SEASON         = "2025-26"
SEASON_TYPE_RS = "Regular Season"
SEASON_TYPE_PO = "Playoffs"

POSITIONS = ["PG", "SG", "SF", "PF", "C"]

# Maps NBA API / ESPN position strings to one of the 5 canonical positions.
POSITION_MAP = {
    "PG": "PG",  "SG": "SG",  "SF": "SF",  "PF": "PF",  "C": "C",
    "G":  "SG",  # generic guard → SG
    "F":  "SF",  # generic forward → SF
    "G-F": "SF", "F-G": "SF",
    "F-C": "PF", "C-F": "PF",
}

# stat_col (playerlinepredictor convention) → DvP column name
STAT_TO_DVP_COL = {
    "PTS":  "PTS/48",
    "REB":  "REB/48",
    "AST":  "AST/48",
    "FG3M": "3PM/48",
    "STL":  "STL/48",
    "BLK":  "BLK/48",
    "TOV":  "TO/48",
}

# ESPN short abbreviations → standard NBA API abbreviations
_ESPN_NORM = {
    "NY": "NYK", "GS": "GSW", "NO": "NOP",
    "SA": "SAS", "WSH": "WAS", "UTH": "UTA",
    "UTAH": "UTA",
}

MIN_MINUTES_THRESHOLD = 10   # rows below this are noise
MIN_TOTAL_MINUTES     = 240  # v6 Fix 2: ~10 games × 24 avg min; was 80 (too noisy)
DVP_CLAMP_LOW         = 0.85   # v6 Fix 2: tighter clamp (was 0.78)
DVP_CLAMP_HIGH        = 1.18   # v6 Fix 2: tighter clamp (was 1.28)

# =============================================================================
# MODULE-LEVEL STATE
# =============================================================================
_INJECTED_LOGS  = {}    # {window_label: DataFrame}  — from playerlinepredictor
_PLAYER_POS     = {}    # {player_name: pos_group}   — from PLAYER_POS_CACHE
_DVP_TABLE      = {}    # {window_label: DataFrame}  — computed results
_DVP_FACTOR     = {}    # {(team, pos, stat, window): float}


# =============================================================================
# INJECT (called by playerlinepredictor at startup)
# =============================================================================
def inject_logs(df: pd.DataFrame, window_label: str):
    """Receive already-fetched game logs. window_label: 'playoffs' or 'season'."""
    if df is not None and not df.empty:
        _INJECTED_LOGS[window_label] = df.copy()
        # Invalidate dependent caches
        _DVP_TABLE.pop(window_label, None)
        for k in [k for k in _DVP_FACTOR if k[3] == window_label]:
            del _DVP_FACTOR[k]


def inject_positions(pos_dict: dict):
    """Receive player→position mapping from PLAYER_POS_CACHE."""
    _PLAYER_POS.update(pos_dict)
    # Invalidate all computed tables since positions changed
    _DVP_TABLE.clear()
    _DVP_FACTOR.clear()


def clear_cache():
    _DVP_TABLE.clear()
    _DVP_FACTOR.clear()


# =============================================================================
# HELPERS
# =============================================================================
def _parse_min(m) -> float:
    if not m or pd.isna(m): return 0.0
    try:
        s = str(m)
        if ":" in s:
            p = s.split(":")
            return float(p[0]) + float(p[1]) / 60
        return float(s)
    except ValueError:
        return 0.0


def _extract_opponent(matchup: str, team_abbr: str):
    """Extract opponent abbreviation from 'BOS vs. MIA' or 'BOS @ MIA'."""
    parts = re.split(r'\s+(?:vs\.?|@)\s+', str(matchup).strip())
    if len(parts) == 2:
        a, b = parts[0].strip(), parts[1].strip()
        return b if a == team_abbr else a
    return None


def _normalize_abbr(abbr: str) -> str:
    if not abbr: return abbr
    return _ESPN_NORM.get(abbr.upper(), abbr.upper())


def _pos_group(pos: str):
    if not pos: return None
    p = str(pos).upper().strip()
    if p in POSITION_MAP: return POSITION_MAP[p]
    if "PG" in p: return "PG"
    if "SG" in p: return "SG"
    if "SF" in p: return "SF"
    if "PF" in p: return "PF"
    if "C"  in p: return "C"
    if "G"  in p: return "SG"
    if "F"  in p: return "SF"
    return None


# =============================================================================
# LOG FETCHING (used when no injected logs available)
# =============================================================================
def _fetch_own_logs(season_type: str) -> pd.DataFrame:
    if not _NBA_API_AVAILABLE:
        return pd.DataFrame()
    try:
        df = leaguegamefinder.LeagueGameFinder(
            season_nullable=SEASON,
            season_type_nullable=season_type,
            player_or_team_abbreviation="P",
            timeout=30,
        ).get_data_frames()[0]
        time.sleep(1.0)
        return df
    except (requests.exceptions.RequestException, json.JSONDecodeError, IndexError) as e:
        print(f"  [dvp] log fetch error: {e}")
        return pd.DataFrame()


def _get_logs(window: str) -> pd.DataFrame:
    """Return game logs for the given window, using injected logs when available."""
    today = date.today()

    # ── Time-windowed slices always come from the full season logs ────────────
    if window in ("last_7", "last_14", "last_30"):
        days = {"last_7": 7, "last_14": 14, "last_30": 30}[window]
        cutoff = pd.Timestamp(today - timedelta(days=days))
        base = (_INJECTED_LOGS.get("season") or
                _INJECTED_LOGS.get("playoffs") or
                _fetch_own_logs(SEASON_TYPE_RS))
        if base.empty: return pd.DataFrame()
        if "GAME_DATE" in base.columns:
            base = base.copy()
            base["GAME_DATE"] = pd.to_datetime(base["GAME_DATE"])
            return base[base["GAME_DATE"] >= cutoff].reset_index(drop=True)
        return base

    if window == "playoffs":
        if "playoffs" in _INJECTED_LOGS:
            return _INJECTED_LOGS["playoffs"]
        return _fetch_own_logs(SEASON_TYPE_PO)

    # default: "season"
    if "season" in _INJECTED_LOGS:
        return _INJECTED_LOGS["season"]
    return _fetch_own_logs(SEASON_TYPE_RS)


# =============================================================================
# CORE COMPUTATION
# =============================================================================
def build_dvp_table(window: str = "playoffs") -> pd.DataFrame:
    """Compute per-48 DvP table for all (team × position) pairs.

    Returns DataFrame with columns:
        team, position, n_rows, total_min,
        PTS/48, REB/48, AST/48, STL/48, BLK/48, TO/48, 3PM/48, FG%, FT%
        plus <stat>_rank columns (1 = best defense = fewest allowed)
    """
    if window in _DVP_TABLE:
        return _DVP_TABLE[window]

    raw = _get_logs(window)
    if raw is None or raw.empty:
        _DVP_TABLE[window] = pd.DataFrame()
        return pd.DataFrame()

    df = raw.copy()

    # ── Map positions ─────────────────────────────────────────────────────────
    # v7 Fix 6: warn loudly when position data was never injected
    if not _PLAYER_POS:
        print(f"  ⚠ DvP build_dvp_table({window!r}): _PLAYER_POS is empty — "
              f"call inject_player_positions() before build_dvp_table(). "
              f"Falling back to PLAYER_POSITION column if available; "
              f"otherwise DvP factors will be 1.0 (neutral).")

    if _PLAYER_POS:
        df["_pos"] = df["PLAYER_NAME"].map(_PLAYER_POS)
    else:
        df["_pos"] = None

    # Fill gaps from PLAYER_POSITION column if present (some log endpoints have it)
    if "_pos" not in df.columns or df["_pos"].isna().all():
        if "PLAYER_POSITION" in df.columns:
            df["_pos"] = df["PLAYER_POSITION"].apply(_pos_group)
        else:
            _DVP_TABLE[window] = pd.DataFrame()
            return pd.DataFrame()

    df = df[df["_pos"].notna()].copy()
    if df.empty:
        _DVP_TABLE[window] = pd.DataFrame()
        return pd.DataFrame()

    # ── Parse minutes ─────────────────────────────────────────────────────────
    df["_min"] = df["MIN"].apply(_parse_min) if "MIN" in df.columns else 0.0
    df = df[df["_min"] >= MIN_MINUTES_THRESHOLD].copy()

    # ── Extract opponent (= defensive team) ───────────────────────────────────
    df["_opp"] = df.apply(
        lambda r: _extract_opponent(r["MATCHUP"], r["TEAM_ABBREVIATION"]), axis=1
    )
    df["_opp"] = df["_opp"].apply(lambda x: _normalize_abbr(x) if x else None)
    df = df[df["_opp"].notna()].copy()

    # ── Ensure numeric columns exist ──────────────────────────────────────────
    needed = {
        "PTS": 0, "REB": 0, "AST": 0, "STL": 0, "BLK": 0,
        "TOV": 0, "FG3M": 0, "FGM": 0, "FGA": 0, "FTM": 0, "FTA": 0,
    }
    for col, default in needed.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # ── Aggregate by (defensive_team, position) ───────────────────────────────
    rows = []
    for (opp, pos), grp in df.groupby(["_opp", "_pos"]):
        total_min = float(grp["_min"].sum())
        if total_min < MIN_TOTAL_MINUTES:
            continue

        r = {
            "team":      opp,
            "position":  pos,
            "n_rows":    len(grp),
            "total_min": round(total_min, 1),
        }

        # Per-48 counting stats
        for api_col, display in [("PTS","PTS"), ("REB","REB"), ("AST","AST"),
                                  ("STL","STL"), ("BLK","BLK"), ("TOV","TO"),
                                  ("FG3M","3PM")]:
            r[f"{display}/48"] = round((float(grp[api_col].sum()) / total_min) * 48, 2)

        # Aggregate percentage stats
        fga = float(grp["FGA"].sum())
        fta = float(grp["FTA"].sum())
        r["FG%"] = round(float(grp["FGM"].sum()) / fga * 100, 1) if fga > 0 else 0.0
        r["FT%"] = round(float(grp["FTM"].sum()) / fta * 100, 1) if fta > 0 else 0.0

        rows.append(r)

    if not rows:
        _DVP_TABLE[window] = pd.DataFrame()
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # ── Rank within each position (1 = fewest allowed = best defense) ─────────
    rank_cols = ["PTS/48", "REB/48", "AST/48", "STL/48", "BLK/48",
                 "TO/48", "3PM/48", "FG%", "FT%"]
    for col in rank_cols:
        if col not in result.columns:
            continue
        rank_key = col.replace("/48","").replace("%","pct") + "_rank"
        for pos in POSITIONS:
            mask = result["position"] == pos
            if mask.sum() < 2:
                continue
            result.loc[mask, rank_key] = (
                result.loc[mask, col]
                .rank(method="min", ascending=True)  # lower allowed = rank 1
                .astype(int)
            )

    _DVP_TABLE[window] = result
    return result


# =============================================================================
# FACTOR API (used by playerlinepredictor.project_stat)
# =============================================================================
def get_dvp_factor(opp_team_abbr: str, pos_group: str,
                   stat: str, window: str = "playoffs") -> float:
    """Return DvP multiplier for the projection model.

    > 1.0 = team allows more than league avg for this position → boost projection
    < 1.0 = team allows less than league avg → reduce projection
    Returns 1.0 (neutral) if insufficient data.
    """
    if not opp_team_abbr or not pos_group or not stat:
        return 1.0

    opp = _normalize_abbr(opp_team_abbr)
    ck  = (opp, pos_group, stat, window)
    if ck in _DVP_FACTOR:
        return _DVP_FACTOR[ck]

    col = STAT_TO_DVP_COL.get(stat)
    if not col:
        _DVP_FACTOR[ck] = 1.0
        return 1.0

    tbl = build_dvp_table(window)
    if tbl.empty or col not in tbl.columns:
        _DVP_FACTOR[ck] = 1.0
        return 1.0

    pos_rows = tbl[tbl["position"] == pos_group]
    if pos_rows.empty:
        _DVP_FACTOR[ck] = 1.0
        return 1.0

    team_row = pos_rows[pos_rows["team"] == opp]
    if team_row.empty:
        _DVP_FACTOR[ck] = 1.0
        return 1.0

    team_val    = float(team_row[col].iloc[0])
    league_avg  = float(pos_rows[col].mean())

    if league_avg <= 0:
        _DVP_FACTOR[ck] = 1.0
        return 1.0

    raw_factor = max(DVP_CLAMP_LOW, min(team_val / league_avg, DVP_CLAMP_HIGH))

    # v9 Fix 3: shrink raw factor toward 1.0 based on sample size.
    # n_rows is player-game appearances vs this team at this position — a reliable
    # proxy for games played (more rows = more data = more trust).
    # Formula n/(n+5): 10 rows → 67% trust, 20 rows → 80%, 50+ rows → 91%.
    # Prevents noisy 8-game playoff samples from swinging DvP to the ±18% clamp.
    n_rows           = float(team_row["n_rows"].iloc[0]) if "n_rows" in team_row.columns else 0.0
    shrinkage_weight = n_rows / (n_rows + 5.0) if n_rows > 0 else 0.0
    factor           = shrinkage_weight * raw_factor + (1.0 - shrinkage_weight) * 1.0

    _DVP_FACTOR[ck] = factor
    return factor


# =============================================================================
# DISPLAY
# =============================================================================
def print_dvp_table(stat: str = "PTS", window: str = "playoffs",
                    pos: str = None, top_n: int = 30):
    """Print DvP table sorted worst→best defense (most favourable for props first).

    stat: PTS | REB | AST | 3PM | STL | BLK | TO | FG% | FT%
    pos:  PG | SG | SF | PF | C  (None = all positions, pivot layout)
    """
    col_map = {
        "PTS": "PTS/48", "REB": "REB/48", "AST": "AST/48",
        "3PM": "3PM/48", "STL": "STL/48", "BLK": "BLK/48",
        "TO":  "TO/48",  "FG%": "FG%",    "FT%": "FT%",
    }
    col = col_map.get(stat.upper(), stat)
    tbl = build_dvp_table(window)

    if tbl.empty:
        print(f"  [dvp] No data for window='{window}'")
        return

    if col not in tbl.columns:
        print(f"  [dvp] Column '{col}' not in table. "
              f"Available: {[c for c in tbl.columns if '/' in c or '%' in c]}")
        return

    rank_col = col.replace("/48","").replace("%","pct") + "_rank"

    if pos:
        # Single-position view: sorted by allowed stat descending (worst defense first)
        sub = tbl[tbl["position"] == pos.upper()][["team", col, rank_col]].copy()
        sub = sub.sort_values(col, ascending=False).head(top_n)
        print(f"\n  DvP — {col} allowed to {pos.upper()} ({window})"
              f"  [1 = best defense]")
        print(f"  {'#':<4} {'Team':<6} {col:>8}  {'Rank':>4}")
        print(f"  {'─'*4} {'─'*6} {'─'*8}  {'─'*4}")
        for i, (_, row) in enumerate(sub.iterrows(), 1):
            rk = int(row.get(rank_col, 0)) if rank_col in sub.columns else ""
            print(f"  {i:<4} {row['team']:<6} {row[col]:>8.2f}  {rk:>4}")
    else:
        # Pivot: rows=teams, columns=positions
        pivot = tbl.pivot_table(index="team", columns="position",
                                values=col, aggfunc="first")
        pos_cols = [p for p in POSITIONS if p in pivot.columns]
        pivot["_avg"] = pivot[pos_cols].mean(axis=1)
        pivot = pivot.sort_values("_avg", ascending=False).drop(columns="_avg").head(top_n)

        print(f"\n  DvP — {col} allowed per position ({window})"
              f"  [sorted: worst defense first]")
        header = f"  {'Team':<6}" + "".join(f" {p:>7}" for p in pos_cols)
        print(header)
        print("  " + "─" * (6 + 8 * len(pos_cols)))
        for team, row in pivot.iterrows():
            vals = "".join(
                f" {row[p]:>7.2f}" if p in row.index and pd.notna(row[p]) else f" {'—':>7}"
                for p in pos_cols
            )
            print(f"  {team:<6}{vals}")

    print()


def print_dvp_matchup(opp_team: str, window: str = "playoffs"):
    """Print all stats for one defensive team — useful for pre-game analysis."""
    tbl = build_dvp_table(window)
    if tbl.empty:
        return
    opp = _normalize_abbr(opp_team)
    sub = tbl[tbl["team"] == opp].sort_values("position")
    if sub.empty:
        print(f"  [dvp] No data for team '{opp}'")
        return

    stat_cols = ["PTS/48", "REB/48", "AST/48", "3PM/48", "FG%", "FT%",
                 "STL/48", "BLK/48", "TO/48"]
    stat_cols = [c for c in stat_cols if c in sub.columns]

    print(f"\n  DvP — {opp} defense by position ({window})")
    hdr = f"  {'Pos':<5}" + "".join(f" {c:>8}" for c in stat_cols)
    print(hdr)
    print("  " + "─" * (5 + 9 * len(stat_cols)))
    for _, row in sub.iterrows():
        vals = "".join(f" {row[c]:>8.1f}" if pd.notna(row.get(c)) else f" {'—':>8}"
                       for c in stat_cols)
        print(f"  {row['position']:<5}{vals}")
    print()


# =============================================================================
# STANDALONE CLI
# =============================================================================
if __name__ == "__main__":
    import argparse
    from nba_api.stats.endpoints import leaguegamefinder  # noqa

    ap = argparse.ArgumentParser(description="DvP table viewer")
    ap.add_argument("--stat",   default="PTS",      help="Stat: PTS REB AST 3PM FG% FT% STL BLK TO")
    ap.add_argument("--window", default="playoffs",  help="Window: season playoffs last_7 last_14 last_30")
    ap.add_argument("--pos",    default=None,        help="Position filter: PG SG SF PF C")
    ap.add_argument("--team",   default=None,        help="Show one team's full DvP breakdown")
    ap.add_argument("--top",    default=30, type=int,help="Rows to show")
    args = ap.parse_args()

    print(f"\nBuilding DvP table (window={args.window})…")
    tbl = build_dvp_table(args.window)
    if tbl.empty:
        print("No data — position cache is empty when running standalone.")
        print("Run via nba_combined.py or playerlinepredictor.py to use injected positions.")
        sys.exit(0)

    if args.team:
        print_dvp_matchup(args.team, window=args.window)
    else:
        print_dvp_table(stat=args.stat, window=args.window,
                        pos=args.pos, top_n=args.top)
