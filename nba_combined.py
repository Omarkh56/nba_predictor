"""
NBA Full Analysis — Team Model + Player Props (Unified Engine)
==============================================================
Single script that replaces running playerlinepredictor.py and
newnbapredictor.py separately.  Every feature from both is active.

  newnbapredictor.py  — 10-factor team model
    Win probability · spread · O/U · injury impact · form · clutch · pace

  playerlinepredictor.py  — Player props engine (v4+)a
    Projections vs market lines · DvP · USG_PCT · calibration
    UNDER penalty · combo SD inflation · star blowup buffer · edge floor

  Fixes vs old nba_combined.py
    [FIX-1] Calibration now loaded — _load_calibration() called at startup
    [FIX-2] Props injuries fetched separately (full ppg/rpg/apg fields)
    [FIX-3] USG_PCT + DvP roster prefetch called before game loop
    [FIX-4] predict_games_data wrapped in try-except (no crash cascade)
    [FIX-5] All 10 team factors shown (clutch + matchup were silently dropped)
    [FIX-6] Pace context drives cross-reference narrative
    [FIX-7] Picks exported to predictions_YYYY-MM-DD.json after run
    [FIX-8] Breakeven threshold (53%) flagged in confidence column
    [FIX-9] Verbose shows DvP + USG factors per stat component

Run:
    python3 nba_combined.py
    python3 nba_combined.py --verbose
"""

import json
import logging
import os
import sys
from datetime import date

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import requests.exceptions

import newnbapredictor as team_model
import oddstracker
import playerlinepredictor as props_model
import predictions_db as pred_db
from roi_analysis import american_to_decimal

logger = logging.getLogger(__name__)

SEASON = "2026-27"
VERBOSE = "--verbose" in sys.argv
WIDTH = 80
_HERE = os.path.dirname(os.path.abspath(__file__))

# Minimum confidence to display with a star in the best-bets section.
# 53.0% = breakeven at standard -110 juice.
BREAKEVEN_CONF = 53.0

# Market ranking weights — applied only to the sort score, not displayed values.
# 3PM lines are tiny (0.5 / 1.5) so even a small absolute edge produces huge
# Edge%, drowning out PTS/REB/AST picks.  Weight < 1.0 de-prioritises a market.
MARKET_RANK_WEIGHTS = {
    "3PM": 0.40,  # heavily de-prioritised
}
# Hard cap: max this many picks from a single market per ranked table.
MARKET_CAP = {
    "3PM": 2,
}

# Flag added to a row's Flags column when no same-day odds_tracker.csv snapshot
# exists for its (player, market) — ranking fell back to edge_pct for that row.
NO_MARKET_FLAG = "NO-MKT-CHECK"

# Dynamic calibration rank weights — built from calibration.json at startup.
# Maps "MARKET_PICK" → float weight (e.g. "REB_OVER" → 1.24, "PTS_OVER" → 0.74).
# Empty until _build_calib_rank_weights() is called.
_CALIB_RANK_WEIGHTS: dict = {}

# Minimum samples before we trust a calibration hit-rate for weighting/filtering.
_CALIB_MIN_N = 20

# Today's book lean lookup — populated at startup from odds_tracker.csv.
# Key: (player_name_lower, market_label)  e.g. ("lebron james", "PTS")
# Value: {"book_lean": "OVER"|"UNDER"|"NEUTRAL", "lean_strength": float,
#          "devig_over": float, "vig_pct": float}
_TODAY_LEAN: dict = {}


def _load_today_lean(snap_date: date) -> dict:
    """Load today's odds snapshot from odds_tracker.csv into a fast lookup dict.
    If no snapshot exists for today, attempts to fetch one automatically."""
    tracker_path = oddstracker.TRACKER_FILE
    lean_map = {}

    # Auto-fetch if today's snapshot is missing
    needs_fetch = True
    if os.path.isfile(tracker_path) and os.path.getsize(tracker_path) > 0:
        try:
            df = pd.read_csv(tracker_path, usecols=["date", "player", "market"])
            if (df["date"] == snap_date.isoformat()).any():
                needs_fetch = False
        except (OSError, ValueError, KeyError):
            pass

    if needs_fetch:
        print("\n  No odds snapshot for today — fetching book lean data…")
        try:
            oddstracker.snapshot(snap_date)
        except (requests.exceptions.RequestException, json.JSONDecodeError, OSError) as e:
            print(f"  ⚠  Could not fetch odds snapshot: {e}")
            return lean_map

    # Load from CSV
    try:
        df = pd.read_csv(tracker_path)
        today = df[df["date"] == snap_date.isoformat()]
        for _, row in today.iterrows():
            key = (str(row["player"]).strip().lower(), str(row["market"]).strip())
            devig_over = float(row.get("devig_over", 0.5))
            lean_map[key] = {
                "book_lean": row.get("book_lean", "NEUTRAL"),
                "lean_strength": float(row.get("lean_strength", 0.0)),
                "devig_over": devig_over,
                "devig_under": float(row.get("devig_under", 1.0 - devig_over)),
                "vig_pct": float(row.get("vig_pct", 0.0)),
                "avg_over_odds": float(row.get("avg_over_odds", -110)),
                "avg_under_odds": float(row.get("avg_under_odds", -110)),
            }
        if lean_map:
            print(
                f"  Book lean loaded: {len(lean_map)} player-market lines for {snap_date.isoformat()}"
            )
    except (OSError, ValueError, KeyError) as e:
        print(f"  ⚠  Could not load odds_tracker.csv: {e}")

    return lean_map


def _get_lean(player: str, market: str) -> dict:
    """Look up today's book lean for a (player, market) pair.
    Returns empty dict when no data available."""
    return _TODAY_LEAN.get((player.strip().lower(), market.strip()), {})


def _build_calib_rank_weights(calib: dict) -> dict:
    """Build per-(market, direction) rank weights from calibration.json.

    weight = clamp(smoothed_rate / prior_rate, 0.40, 1.55)
    Only computed when n >= _CALIB_MIN_N; defaults to 1.0 otherwise.
    """
    weights = {}
    prior = calib.get("prior_hit_rate", 0.535)
    for key, v in calib.get("market_direction", {}).items():
        if v.get("n", 0) >= _CALIB_MIN_N:
            w = max(0.40, min(1.55, v["rate"] / prior))
            weights[key] = round(w, 3)
    return weights


def _p_market(player: str, market: str, pick: str):
    """De-vigged market probability (devig_power, via oddstracker.py) for the
    picked side. Returns None when no same-day odds_tracker.csv snapshot
    exists for this (player, market)."""
    lean = _get_lean(player, market)
    if not lean:
        return None
    devig_over = lean.get("devig_over", 0.5)
    return lean.get("devig_under", 1.0 - devig_over) if pick == "UNDER" else devig_over


def _decimal_odds(player: str, market: str, pick: str):
    """Decimal price for the picked side, converted from odds_tracker.csv's
    average American price. Returns None when there's no same-day snapshot
    or the price is unusable (american_to_decimal raises on junk book data,
    e.g. |odds| < 100)."""
    lean = _get_lean(player, market)
    if not lean:
        return None
    american = lean.get("avg_under_odds") if pick == "UNDER" else lean.get("avg_over_odds")
    try:
        return american_to_decimal(american)
    except (ValueError, TypeError):
        return None


def _market_edge_pct(player: str, market: str, pick: str, confidence: float):
    """True edge vs. the de-vigged market, in percentage points:
    model_probability − devig_market_probability (for the picked side).

    Returns None when no same-day odds_tracker.csv snapshot exists for this
    (player, market) — the caller should fall back to edge_pct.
    """
    p_mkt = _p_market(player, market, pick)
    if p_mkt is None:
        return None
    model_prob = confidence / 100.0
    return (model_prob - p_mkt) * 100.0


def _enrich_with_market_edge(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a MktEdgePct column (None where unavailable) and mark rows with
    no same-day odds_tracker match as NO-MKT-CHECK in Flags. Idempotent —
    safe to call on a frame that's already been enriched.
    """
    if df.empty:
        df = df.copy()
        df["MktEdgePct"] = pd.Series(dtype="float64")
        return df
    df = df.copy()
    mkt_edges, flags_out = [], []
    for _, r in df.iterrows():
        me = _market_edge_pct(str(r["Player"]), str(r["Market"]), str(r["Pick"]), float(r["Confidence"]))
        mkt_edges.append(me)
        f = str(r.get("Flags", "")).strip()
        if me is None and NO_MARKET_FLAG not in f:
            f = (f + " " + NO_MARKET_FLAG).strip()
        flags_out.append(f)
    df["MktEdgePct"] = mkt_edges
    df["Flags"] = flags_out
    return df


def _rank_score(
    edge_pct: float,
    confidence: float,
    market: str,
    pick: str = "OVER",
    line: float = 99.0,
    mkt_edge_pct: float = None,
) -> float:
    """Ranking score used for all top-N tables (not displayed to user).

    When a same-day odds_tracker.csv snapshot exists for this (player, market),
    ranking is driven by mkt_edge_pct — model_probability − devig_market_probability
    (the true edge vs. the market) — instead of edge_pct (deviation from the
    posted line). Falls back to edge_pct only when mkt_edge_pct is None (no
    market snapshot). edge_pct itself is unchanged and remains display-only.

    Low-line bets (≤ 1.5) get the ranking edge capped at 30 (points/pp).
    Dynamic calibration weight boosts historically-good markets and
    suppresses historically-bad ones."""
    static_w = MARKET_RANK_WEIGHTS.get(market, 1.0)
    dynamic_w = _CALIB_RANK_WEIGHTS.get(f"{market}_{pick}", 1.0)
    w = static_w * dynamic_w
    ranking_edge = mkt_edge_pct if mkt_edge_pct is not None else edge_pct
    effective_edge = min(ranking_edge, 30.0) if line <= 1.5 else ranking_edge
    return effective_edge * w


def _apply_market_cap(df: pd.DataFrame) -> pd.DataFrame:
    """After sorting, drop rows that exceed the per-market cap."""
    counts = {}
    keep = []
    for idx, row in df.iterrows():
        mkt = row["Market"]
        cap = MARKET_CAP.get(mkt, 999)
        counts[mkt] = counts.get(mkt, 0) + 1
        if counts[mkt] <= cap:
            keep.append(idx)
    return df.loc[keep].reset_index(drop=True)


# =============================================================================
# HELPERS
# =============================================================================
def _abbr(team_full_name: str) -> str:
    return props_model.TEAM_NAME_TO_ABBR.get(team_full_name, team_full_name[:3].upper())


def _match_odds_event(all_events, home_abbr, away_abbr):
    # Normalize ESPN short abbreviations (NY→NYK, GS→GSW, etc.) so they match
    # the standard abbreviations produced from Odds API full team names.
    home_norm = props_model.ESPN_TO_NBA_ABBR.get(home_abbr, home_abbr)
    away_norm = props_model.ESPN_TO_NBA_ABBR.get(away_abbr, away_abbr)
    for ev in all_events:
        if (
            _abbr(ev.get("home_team", "")) == home_norm
            and _abbr(ev.get("away_team", "")) == away_norm
        ):
            return ev
    return None


def _edge_floored(df: pd.DataFrame) -> pd.DataFrame:
    """Filter picks below edge threshold and ban markets with strong evidence of bias.

    v9 Fix 4: mean_margin is now an optional confirmation signal, not required.
    Previously, `entry.get("mean_margin") or 0.0` silently treated missing margin
    as 0.0 (no structural miss), making the filter unpredictable when projections
    hadn't flowed through yet. Now: primary signal is raw_rate < 0.46 on n >= 30;
    margin, when present, can *lift* a ban if it contradicts the hit-rate signal.
    """
    if df.empty:
        return df

    # Standard edge floor (per-direction)
    book_ok = (
        (df["book_count"] >= 3) if "book_count" in df.columns else pd.Series(True, index=df.index)
    )
    edge_ok = df.apply(
        lambda r: (
            r["Edge%"] >= props_model.UNDER_EDGE_PCT_FLOOR
            if r["Pick"] == "UNDER"
            else r["Edge%"] >= props_model.OVER_EDGE_PCT_FLOOR
        ),
        axis=1,
    )
    df = df[book_ok & edge_ok].reset_index(drop=True)
    if df.empty:
        return df

    # v9 Fix 4: market-level ban requires strong evidence on a sufficient sample.
    # Primary: raw_rate < 0.46 with n >= 30 (was _CALIB_MIN_N=20, raised for robustness).
    # Override: if mean_margin IS present and > -1.5, the miss rate may be noise — don't ban.
    # If mean_margin IS absent (None), rely on raw_rate alone (safer than defaulting to 0).
    calib = getattr(props_model, "CALIB", {}) or {}
    md_calib = calib.get("market_direction", {})
    bad_markets: set = set()
    for key, info in md_calib.items():
        n = info.get("n", 0)
        raw_rate = info.get("raw_rate", 0.5)
        if n < 30 or raw_rate >= 0.46:
            continue
        mean_margin = info.get("mean_margin")  # None when projections were missing
        if mean_margin is not None and mean_margin > -1.5:
            continue  # margin contradicts hit-rate — near-miss noise, don't ban
        parts = key.rsplit("_", 1)
        if len(parts) == 2:
            bad_markets.add(tuple(parts))

    if bad_markets:
        mask = df.apply(lambda r: (r["Market"], r["Pick"]) not in bad_markets, axis=1)
        df = df[mask].reset_index(drop=True)

    return df


def _pace_label(home_pace: float, away_pace: float, league_avg: float = 99.5) -> str:
    game_pace = (home_pace + away_pace) / 2.0
    if game_pace > league_avg + 1.5:
        return f"FAST ({game_pace:.1f})"
    if game_pace < league_avg - 1.5:
        return f"SLOW ({game_pace:.1f})"
    return f"AVG  ({game_pace:.1f})"


# =============================================================================
# SHARED DISPLAY HELPERS
# =============================================================================
_PROP_HDR = (
    f"  {'Player':<22} {'Mkt':<8} {'Line':>5} {'Proj':>5}   {'E%':>5}   {'Conf':>6}   {'Pick':<5}"
)
_PROP_SEP = f"  {'─' * 22} {'─' * 8} {'─' * 5} {'─' * 5}   {'─' * 5}   {'─' * 6}   {'─' * 5}"


def _conf_marker(conf: float) -> str:
    """Single-character confidence marker: ★ High ≥66%  · space ≥53%  * below breakeven."""
    if conf >= 66:
        return "★"
    if conf < BREAKEVEN_CONF:
        return "*"
    return " "


def _pick_detail_line(r) -> str:
    """Build the compact detail sub-line: range · book lean · key flags."""
    parts = []

    # Projected range
    lo, hi = r.get("Proj_Low"), r.get("Proj_High")
    if lo is not None and hi is not None:
        parts.append(f"Range {lo:.0f}–{hi:.0f}")

    # True edge vs. the de-vigged market (what actually drives the "RANKED"
    # table's sort when present — see that table's title; this line is shown
    # on every table this row appears in, so it states the value, not a claim
    # about which table's order it drove). Absence is signaled by the
    # NO-MKT-CHECK flag below instead.
    mkt_edge = r.get("MktEdgePct")
    if pd.notna(mkt_edge):
        parts.append(f"Mkt-edge {mkt_edge:+.1f}pp vs. devig market")

    # Book lean
    lean = _get_lean(str(r.get("Player", "")), str(r.get("Market", "")))
    if lean and lean.get("book_lean", "NEUTRAL") != "NEUTRAL":
        sym = "▲" if lean["book_lean"] == "OVER" else "▼"
        agrees = "✓" if lean["book_lean"] == r.get("Pick", "") else "✗"
        parts.append(f"Books {sym}{lean['book_lean']}({lean['lean_strength']:.2f}){agrees}")

    # Key flags — only the informative ones; skip noise like LOW-LINE / ⚠PO-ROLE
    show_prefixes = (
        "DvP",
        "TREND",
        "★STRONG",
        "HI-USG",
        "LO-USG",
        "BLW-",
        "USG+",
        "NO-PO",
        "LOW-PO",
        NO_MARKET_FLAG,
    )
    key_flags = [
        f for f in str(r.get("Flags", "")).split() if any(f.startswith(p) for p in show_prefixes)
    ]
    if key_flags:
        parts.append(" ".join(key_flags))

    return "  ▸ " + "   ·   ".join(parts) if parts else ""


# =============================================================================
# PRINT SECTIONS
# =============================================================================
def print_team_section(pred: dict):
    """Clean team model box — fixed width, all 10 factors shown."""
    b = pred["breakdown"]
    ha = pred["home_abbr"]
    aa = pred["away_abbr"]
    hp = pred.get("home_pace", 99.5)
    ap = pred.get("away_pace", 99.5)
    margin = pred["expected_margin"]
    fav = pred["favorite"]

    spread_str = (
        f"{ha} {pred['home_spread']:+.1f}" if margin > 0 else f"{aa} {pred['away_spread']:+.1f}"
    )
    conf_str = (
        "Strong ✓"
        if pred["fav_prob"] >= 0.68
        else "Lean"
        if pred["fav_prob"] >= 0.57
        else "Toss-up"
    )
    pace_str = _pace_label(hp, ap)
    IW = WIDTH - 4  # inner width

    print(f"  ┌{'─' * IW}┐")
    # Line 1: matchup, spread, O/U, pace
    l1 = (
        f"{aa} {pred['away_prob']:.1%}  vs  {ha} {pred['home_prob']:.1%}"
        f"   Sprd: {spread_str}   O/U: {pred['predicted_total']:.1f}   {pace_str}"
    )
    print(f"  │  {l1:<{IW - 3}}│")
    # Line 2: score, fav, confidence
    l2 = (
        f"Proj: {aa} {pred['away_score']:.0f} – {ha} {pred['home_score']:.0f}"
        f"   Fav: {fav} {pred['fav_prob']:.1%}   [{conf_str}]"
    )
    print(f"  │  {l2:<{IW - 3}}│")
    # Factor lines
    f1 = (
        f"NetRtg {b['net_rtg']:+.1f}  Inj {b['injuries']:+.1f}  "
        f"FF {b['four_factors']:+.1f}  Form {b['form']:+.1f}  Rest {b['rest']:+.1f}"
    )
    f2 = (
        f"HCA {b['hca']:+.1f}  Pace {b['pace']:+.1f}  "
        f"H2H {b['h2h']:+.1f}  Matchup {b.get('matchup', 0):+.1f}  "
        f"Clutch {b.get('clutch', 0):+.1f}"
    )
    print(f"  │  {f1:<{IW - 3}}│")
    print(f"  │  {f2:<{IW - 3}}│")
    print(f"  └{'─' * IW}┘")

    # Significant injuries
    sig_inj = [
        d
        for d in pred.get("h_inj_det", []) + pred.get("a_inj_det", [])
        if d.get("status") in ("Out", "Doubtful")
    ]
    if sig_inj:
        parts = [f"{d['player']} ({d['status']}, {d['adj']:+.1f})" for d in sig_inj[:3]]
        print(f"  ⚠  Injuries: {', '.join(parts)}")


def _devil_advocate_risks(r) -> list:
    """Return devil's advocate risk strings for a pick row."""
    risks = []
    flags = str(r.get("Flags", ""))
    calib_md = props_model.CALIB.get("market_direction", {})
    entry = calib_md.get(f"{r['Market']}_{r['Pick']}", {})
    lean = _get_lean(str(r.get("Player", "")), str(r.get("Market", "")))

    if lean and lean.get("book_lean", "NEUTRAL") != "NEUTRAL":
        if lean["book_lean"] != r.get("Pick", ""):
            risks.append(
                f"Books lean {lean['book_lean']} "
                f"({lean['devig_over']:.0%} OVER devig, strength {lean['lean_strength']:.2f}) — model DISAGREES"
            )
    if "TREND↓" in flags:
        risks.append("Recent form declining")
    if entry.get("near_miss_rate", 0) >= 0.30:
        risks.append(
            f"Misses within 1 unit {entry['near_miss_rate']:.0%} of the time — historically thin edge"
        )
    if "NO-PO" in flags:
        risks.append("No playoff sample — projection based on regular season only")
    elif "LOW-PO" in flags:
        risks.append("Limited playoff sample (1 game) — role uncertain")
    if "ROT-RISK" in flags:
        risks.append("Rotation risk — minutes likely drop in playoffs")
    if "⚠MINS-VOL" in flags:
        risks.append("High minute volatility — range is wide")
    if "ROAD-B2B" in flags:
        risks.append("Road back-to-back — fatigue factor")
    return risks


def _print_pick_row(r, rank: int = 0, show_rank: bool = False, show_verbose: bool = False):
    """Print one pick row + detail sub-line + risk bullets."""
    marker = _conf_marker(r["Confidence"])
    rank_col = f"{rank:<3} " if show_rank else ""
    team_col = f"{r.get('Team', ''):<5} " if show_rank else ""  # team only in summary tables

    print(
        f"  {rank_col}{r['Player']:<22} {team_col}{r['Market']:<8} "
        f"{r['Line']:>5.1f} {r['Projection']:>5.1f}"
        f"   {r['Edge%']:>4.1f}%   {marker}{r['Confidence']:>5.1f}%   {r['Pick']:<5}"
    )

    detail = _pick_detail_line(r)
    if detail:
        print(detail)

    for risk in _devil_advocate_risks(r):
        print(f"    ⚠ {risk}")

    if show_verbose:
        dets = r.get("details", {}) or {}
        meta = r.get("meta", {}) or {}
        for sc, d in dets.items():
            cmu = d.get("combined_mult", 1.0)
            cmu_u = d.get("combined_mult_uncapped", cmu)
            cap_note = f"(⚡cap from {cmu_u:.3f})" if abs(cmu_u - cmu) > 0.001 else ""
            print(
                f"    └ {sc}: {d.get('rate', 0):.4f}/min"
                f" × {d.get('min', 0):.1f}min"
                f" × H2H={d.get('series_h2h', 1):.3f}"
                f" × comb={cmu:.3f}{cap_note}"
                f"  [def={d.get('def_team', d.get('def', 1)):.3f}"
                f" dvp={d.get('dvp', 1):.3f}"
                f" ha={d.get('ha', 1):.3f}"
                f" usg={d.get('usage', 1):.3f}]"
                f" → {d.get('final', 0):.1f}"
            )
        pen = []
        bp = meta.get("bench_penalty", 0)
        vp = meta.get("mins_vol_penalty", 0)
        dp = r.get("direction_penalty", 0.0)
        uf = meta.get("usg_factor", 1.0)
        if bp > 0:
            pen.append(f"bench={bp:.1f}")
        if vp > 0:
            pen.append(f"vol={vp:.1f}")
        if dp > 0:
            pen.append(f"dir_pen={dp:.1f}")
        if uf != 1.0:
            pen.append(f"usg={uf:.3f}")
        if pen:
            print(f"    └ penalties: {', '.join(pen)}")


def print_props_tables(results: list) -> pd.DataFrame:
    """Print top-N props tables. Returns full props DataFrame."""
    if not results:
        print("  No props available.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    dfF = _edge_floored(df)
    dfF = _enrich_with_market_edge(dfF)
    N = props_model.TOP_BETS_PER_GAME

    dfF["_rscore"] = dfF.apply(
        lambda r: _rank_score(
            r["Edge%"], r["Confidence"], r["Market"], r["Pick"], r["Line"], r["MktEdgePct"]
        ),
        axis=1,
    )

    top_edge = _apply_market_cap(
        dfF.sort_values("_rscore", ascending=False)
        .drop_duplicates(subset=["Player"], keep="first")
        .head(N)
        .reset_index(drop=True)
    )
    top_conf = _apply_market_cap(
        dfF.sort_values("Confidence", ascending=False)
        .drop_duplicates(subset=["Player"], keep="first")
        .head(N)
        .reset_index(drop=True)
    )

    def _tbl(title, tdf, verbose=False):
        if tdf.empty:
            return
        print(f"\n  ── {title}")
        print(_PROP_HDR)
        print(_PROP_SEP)
        for rank, (_, r) in enumerate(tdf.iterrows()):
            _print_pick_row(r, rank=rank + 1, show_rank=False, show_verbose=verbose)
        print(_PROP_SEP)

    _tbl(f"TOP {N} — RANKED  mkt-edge vs. devig market, else Edge%", top_edge, verbose=VERBOSE)
    _tbl(f"TOP {N} BY CONFIDENCE  most likely to hit", top_conf)

    no_po = df[df["PO_Games"] == 0]
    bench = df[df["Is_Bench"]]
    notes = []
    if not no_po.empty:
        notes.append(f"{no_po['Player'].nunique()} player(s) with no playoff data")
    if not bench.empty:
        notes.append(f"{bench['Player'].nunique()} bench player(s)")
    if notes:
        print(f"  ⚠  {' · '.join(notes)}")
    print(f"  {len(df)} props analyzed   ★ = High ≥66%   * = below breakeven 53%")
    return df


def print_best_bets(pred: dict, props_df: pd.DataFrame):
    """Cross-reference team model signals with top props picks."""
    if props_df.empty:
        return

    margin = pred["expected_margin"]
    model_total = pred["predicted_total"]
    home_abbr = pred["home_abbr"]
    away_abbr = pred["away_abbr"]
    fav_abbr = home_abbr if margin > 0 else away_abbr
    dog_abbr = away_abbr if margin > 0 else home_abbr
    game_pace = (pred.get("home_pace", 99.5) + pred.get("away_pace", 99.5)) / 2.0
    home_form_edge = pred["breakdown"].get("form", 0.0)

    pf = props_df.copy()
    pf["_score"] = pf["Edge%"] * pf["Confidence"] / 100
    candidates = (
        _edge_floored(pf)
        .sort_values("_score", ascending=False)
        .drop_duplicates(subset=["Player", "Market"])
        .head(10)
    )

    rows = []
    for _, r in candidates.iterrows():
        pick, market, team, conf = r["Pick"], r["Market"], r.get("Team", ""), r["Confidence"]
        signals = []

        if any(m in market for m in ("PTS", "PRA", "REB+AST")):
            if model_total > 220 and pick == "OVER":
                signals.append(f"↑ high-total game ({model_total:.0f})")
            elif model_total < 208 and pick == "UNDER":
                signals.append(f"↓ low-total game ({model_total:.0f})")
        if game_pace > 101.0 and pick == "OVER" and any(m in market for m in ("PTS", "AST", "3PM")):
            signals.append(f"↑ fast pace ({game_pace:.1f})")
        if game_pace < 97.5 and pick == "UNDER" and "PTS" in market:
            signals.append(f"↓ slow pace ({game_pace:.1f})")
        if abs(margin) >= 7:
            if team == fav_abbr and pick == "OVER" and "PTS" in market:
                signals.append(f"↑ {fav_abbr} big fav ({margin:+.1f})")
            elif team == dog_abbr and pick == "UNDER":
                signals.append(f"↓ {dog_abbr} heavy dog ({margin:+.1f})")
        if abs(margin) < 4 and "AST" in market and pick == "OVER":
            signals.append("↑ close game")
        if abs(home_form_edge) >= 1.5:
            form_team = home_abbr if home_form_edge > 0 else away_abbr
            if team == form_team and pick == "OVER" and "PTS" in market:
                signals.append(f"↑ {form_team} hot form")

        if not signals and conf < BREAKEVEN_CONF:
            continue

        marker = "★" if signals else "·"
        be_tag = "  ⚠below breakeven" if conf < BREAKEVEN_CONF else ""
        signal_str = "  " + "  ·  ".join(signals) if signals else ""
        rows.append((marker, r, signal_str, be_tag))

    if not rows:
        return

    IW = WIDTH - 4
    print(f"\n  ┌─ BEST BETS {'─' * (IW - 10)}┐")
    for marker, r, signal_str, be_tag in rows:
        lean = _get_lean(str(r.get("Player", "")), str(r.get("Market", "")))
        lean_tag = ""
        if lean and lean.get("book_lean", "NEUTRAL") != "NEUTRAL":
            sym = "▲" if lean["book_lean"] == "OVER" else "▼"
            agr = "✓" if lean["book_lean"] == r["Pick"] else "✗"
            lean_tag = f"  Books {sym}{lean['book_lean']}({lean['lean_strength']:.2f}){agr}"
        conf_str = f"{'★' if r['Confidence'] >= 66 else ' '}{r['Confidence']:.0f}%"
        line1 = (
            f"  {marker} {r['Player']:<22} {r['Market']:<8} {r['Pick']:<5} "
            f"line {r['Line']:.1f}  E%={r['Edge%']:.0f}%  Conf={conf_str}"
        )
        print(line1)
        if signal_str or lean_tag or be_tag:
            print(f"      {(signal_str + lean_tag + be_tag).strip()}")
    print(f"  └{'─' * IW}┘")


# =============================================================================
# JSON EXPORT
# =============================================================================
def _pick_to_dict(row: pd.Series, game_key: str, pred) -> dict:
    """Convert a props DataFrame row + game context to a JSON-exportable dict."""
    meta = row.get("meta") or {}
    player, market, pick = str(row["Player"]), str(row["Market"]), str(row["Pick"])
    p_mkt = _p_market(player, market, pick)
    d = _decimal_odds(player, market, pick)
    p_model = float(row["Confidence"]) / 100.0
    total_penalty = float(row.get("total_penalty", row.get("direction_penalty", 0.0)))
    # Sized off p_model, not market_calibration.py's blended p_final: that
    # module's own holdout_check() currently fails to generalize out-of-
    # sample on the live market sample (~150 rows) -- using it to size real
    # stakes would be premature. Revisit once that check passes.
    stake_pct = props_model.compute_stake_pct(p_model, d, total_penalty)
    return {
        "game": game_key,
        "player": row["Player"],
        "market": row["Market"],
        "line": float(row["Line"]),
        "pick": row["Pick"],
        "projection": float(row["Projection"]),
        "confidence": float(row["Confidence"]),
        "edge_pct": float(row["Edge%"]),
        "flags": row["Flags"],
        "p_market": p_mkt,  # de-vigged market probability at prediction time, or None
        "decimal_odds": d,
        "stake_pct": stake_pct,  # fractional-Kelly stake recommendation, 0.0 if no odds/negative EV
        "po_count": int(row.get("PO_Games", 0)),
        "avg_min": float(meta.get("avg_min", 0)),
        "book_count": int(row.get("book_count", 0)),
        "usg_pct": float(meta.get("usg_pct", 0)),
        # Game-level context from team model (None if team model unavailable)
        "game_spread": float(pred["expected_margin"]) if pred else None,
        "game_total": float(pred["predicted_total"]) if pred else None,
        "game_pace": float((pred.get("home_pace", 99.5) + pred.get("away_pace", 99.5)) / 2)
        if pred
        else None,
        "fav_win_pct": float(pred["fav_prob"]) if pred else None,
    }


def export_game_picks_db(team_preds: list, game_date: date) -> None:
    """Upsert game winner/spread picks into predictions.db."""
    if not team_preds:
        return
    rows = []
    for r in team_preds:
        margin = r["expected_margin"]
        home_spread = r["home_spread"] if margin > 0 else r["away_spread"]
        fav = r["favorite"]
        rows.append(
            {
                "home_abbr": r["home_abbr"],
                "away_abbr": r["away_abbr"],
                "winner_pick": fav,
                "spread": round(float(home_spread), 1),
                "spread_pick": fav,
                "win_pct": round(float(r["fav_prob"]), 3),
                "total": round(float(r["predicted_total"]), 1),
            }
        )
    n = pred_db.upsert_game_predictions(rows, game_date)
    print(f"  Saved {n} game pick(s) → predictions.db")


def export_picks_db(all_picks: list, game_date: date) -> None:
    """Deduplicate and upsert prop picks into predictions.db."""
    if not all_picks:
        return
    seen: dict = {}
    for p in all_picks:
        key = (p["player"], p["market"])
        if key not in seen or p["confidence"] > seen[key]["confidence"]:
            seen[key] = p
    final = sorted(seen.values(), key=lambda x: x["confidence"], reverse=True)
    n = pred_db.upsert_prop_predictions(final, game_date)
    print(f"\n  Saved {n} pick(s) → predictions.db")


# =============================================================================
# MAIN
# =============================================================================
def _init_calibration() -> None:
    """Load calibration.json and populate global rank weights."""
    global _CALIB_RANK_WEIGHTS
    props_model._load_calibration()
    _CALIB_RANK_WEIGHTS = _build_calib_rank_weights(props_model.CALIB)
    if _CALIB_RANK_WEIGHTS:
        top3 = sorted(_CALIB_RANK_WEIGHTS.items(), key=lambda x: x[1], reverse=True)[:3]
        bot3 = sorted(_CALIB_RANK_WEIGHTS.items(), key=lambda x: x[1])[:3]
        print(
            f"  Calib weights loaded ({len(_CALIB_RANK_WEIGHTS)} markets) — "
            f"top: {', '.join(f'{k}={v:.2f}' for k, v in top3)}  "
            f"bottom: {', '.join(f'{k}={v:.2f}' for k, v in bot3)}"
        )


def _fetch_matchups_and_lean() -> tuple:
    """Return (matchups, game_date) and populate today's book-lean snapshot."""
    global _TODAY_LEAN
    print("\n  Fetching games from ESPN…")
    matchups = team_model.get_todays_games()
    if not matchups:
        print("  No games found.")
        return [], None
    game_date = matchups[0].get("game_date", date.today())
    _TODAY_LEAN = _load_today_lean(game_date)
    tag = "TODAY" if game_date == date.today() else game_date.strftime("%A %B %d")
    print(f"  {len(matchups)} game(s)  —  {tag}")
    for m in matchups:
        print(f"    {m['away_abbr']} @ {m['home_abbr']}")
    return matchups, game_date


def _fetch_injuries() -> tuple:
    """Return (team_injuries, props_injuries) and print significant absences."""
    print("\n  Fetching injury report…")
    team_injuries = team_model.get_espn_injuries()
    props_injuries = props_model.get_espn_injuries()
    inj_count = sum(len(v) for v in team_injuries.values())
    print(f"  {inj_count} player(s) on report")
    for tn, injs in team_injuries.items():
        sig = [i for i in injs if i.get("status") in ("Out", "Doubtful")]
        if sig:
            ab = team_model.NAME_TO_ESPN_ABBR.get(tn, "???")
            names = ", ".join(f"{i['player']} ({i['status']})" for i in sig[:4])
            print(f"    {ab}: {names}")
    return team_injuries, props_injuries


def _run_team_model(matchups: list, team_injuries: dict) -> tuple:
    """Run the 10-factor team model; return (team_preds, team_map)."""
    print("\n  Running team model (10 factors)…")
    team_preds = []
    try:
        team_preds = team_model.predict_games_data(SEASON, matchups, team_injuries)
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  ⚠  Team model error: {e} — continuing with props only.")
    return team_preds, {(p["away_abbr"], p["home_abbr"]): p for p in team_preds}


def _setup_odds_and_spreads(matchups: list, team_map: dict) -> list:
    """Fetch Odds API events, resolve spreads; return event list."""
    print("\n  Fetching Odds API events…")
    try:
        all_events = props_model.get_all_odds_events()
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"  ⚠  Odds API error: {e}")
        all_events = []
    if not all_events:
        print("  No events returned — props unavailable for all games.")
        return []

    print("  Resolving spreads…")
    for m in matchups:
        ha = m["home_abbr"]
        aa = m["away_abbr"]
        gk = f"{aa}@{ha}"
        odds_ev = _match_odds_event(all_events, ha, aa)
        pred = team_map.get((aa, ha))
        if odds_ev and gk not in props_model.GAME_SPREADS:
            props_model.GAME_SPREADS[gk] = props_model._auto_fetch_spread(
                odds_ev["id"], m["home_name"]
            )
        if pred:
            props_model.GAME_SPREADS[gk] = pred["expected_margin"]
            print(f"    {gk}: model spread {pred['expected_margin']:+.1f}")
    return all_events


def _process_single_game(
    m: dict, team_map: dict, all_events: list, props_injuries: dict, all_picks_export: list
) -> tuple:
    """Run team model section + props for one matchup; return (results, props_df)."""
    ha = m["home_abbr"]
    aa = m["away_abbr"]
    gk = f"{aa}@{ha}"
    pred = team_map.get((aa, ha))
    ev = _match_odds_event(all_events, ha, aa)
    sprd = props_model.GAME_SPREADS.get(gk, 0.0)

    tip_str = f"  │  {m.get('status', '')}" if m.get("status") else ""
    print(f"\n{'━' * WIDTH}")
    print(f"  {aa} @ {ha}{tip_str}")
    print(f"{'━' * WIDTH}")

    if pred:
        print_team_section(pred)
    else:
        print("  [Team model: could not resolve — check ESPN game data]")

    print()
    props_df = pd.DataFrame()
    results = []
    if ev:
        try:
            _hp = pred.get("home_pace", 99.5) if pred else 99.5
            _ap = pred.get("away_pace", 99.5) if pred else 99.5
            pace_factor = (_hp + _ap) / 2.0 / 99.5
            results = props_model.process_game(
                m["home_name"],
                m["away_name"],
                ha,
                aa,
                ev,
                props_injuries,
                spread=sprd,
                pace_factor=pace_factor,
            )
            if results:
                props_df = print_props_tables(results)
                if not props_df.empty:
                    dfF = _edge_floored(props_df)
                    dfF = _enrich_with_market_edge(dfF)
                    dfF["_rscore"] = dfF.apply(
                        lambda r: _rank_score(
                            r["Edge%"], r["Confidence"], r["Market"], r["Pick"], r["Line"],
                            r["MktEdgePct"],
                        ),
                        axis=1,
                    )
                    N = props_model.TOP_BETS_PER_GAME
                    top_e = _apply_market_cap(
                        dfF.sort_values("_rscore", ascending=False)
                        .drop_duplicates(subset=["Player"], keep="first")
                        .head(N)
                    )
                    top_c = _apply_market_cap(
                        dfF.sort_values("Confidence", ascending=False)
                        .drop_duplicates(subset=["Player"], keep="first")
                        .head(N)
                    )
                    for _, row in pd.concat([top_e, top_c]).iterrows():
                        all_picks_export.append(_pick_to_dict(row, gk, pred))
            else:
                print("  No props returned for this game.")
        except Exception:
            # intentionally broad: one bad game must not abort the rest of the batch
            logger.exception("Props processing failed for game %s", gk)
    else:
        print("  No Odds API event found — props unavailable for this game.")

    if pred and not props_df.empty:
        print_best_bets(pred, props_df)

    return results, props_df


def _print_global_summary(
    team_preds: list, all_props_results: list, all_picks_export: list, game_date
) -> None:
    """Print the end-of-run summary: game picks table + top props + exports."""
    print(f"\n{'═' * WIDTH}")
    print(f"  SUMMARY  —  {len(team_preds)} game(s)  ·  {len(all_props_results)} props analyzed")
    print(f"{'═' * WIDTH}")

    if team_preds:
        print("\n  ── GAME PICKS  sorted by confidence")
        print(
            f"  {'Matchup':<12}  {'Fav':<5}  {'Win%':>6}  {'Spread':>9}  "
            f"{'O/U':>6}  {'Pace':<6}  Result"
        )
        print(f"  {'─' * 12}  {'─' * 5}  {'─' * 6}  {'─' * 9}  {'─' * 6}  {'─' * 6}  {'─' * 14}")
        for r in sorted(team_preds, key=lambda x: x["fav_prob"], reverse=True):
            fav = r["favorite"]
            prob = r["fav_prob"]
            spv = abs(r["home_spread"] if r["expected_margin"] > 0 else r["away_spread"])
            conf = "Strong ✓" if prob >= 0.68 else ("Lean" if prob >= 0.57 else "Toss-up")
            gp = (r.get("home_pace", 99.5) + r.get("away_pace", 99.5)) / 2
            pace = "FAST" if gp > 101 else ("SLOW" if gp < 97.5 else "AVG ")
            b2b = (
                f"  ⚠ {r['away_abbr']} B2B"
                if r.get("away_rest") == "B2B"
                else (f"  ⚠ {r['home_abbr']} B2B" if r.get("home_rest") == "B2B" else "")
            )
            print(
                f"  {r['away_abbr']} @ {r['home_abbr']:<4}  "
                f"{fav:<5}  {prob:>5.1%}  {fav} -{spv:<5.1f}  "
                f"{r['predicted_total']:>6.1f}  {pace}  {conf}{b2b}"
            )
        print(f"  {'─' * WIDTH}")
        print("  Strong ✓ ≥68%  ·  Lean 57–68%  ·  Toss-up <57%")

    if all_props_results:
        all_df = pd.DataFrame(all_props_results)
        floored = _edge_floored(all_df)
        floored = _enrich_with_market_edge(floored)
        floored["_rscore"] = floored.apply(
            lambda r: _rank_score(
                r["Edge%"], r["Confidence"], r["Market"], r["Pick"], r["Line"], r["MktEdgePct"]
            ),
            axis=1,
        )
        top_conf = _apply_market_cap(
            floored.sort_values("Confidence", ascending=False)
            .drop_duplicates(subset=["Player", "Market"])
            .head(10)
        )
        top_edge = _apply_market_cap(
            floored.sort_values("_rscore", ascending=False)
            .drop_duplicates(subset=["Player", "Market"])
            .head(10)
        )

        sum_hdr = (
            f"  {'#':<3} {'Player':<22} {'Team':<5} {'Mkt':<8} "
            f"{'Line':>5} {'Proj':>5}   {'E%':>5}   {'Conf':>6}   {'Pick':<5}"
        )
        sum_sep = (
            f"  {'─' * 3} {'─' * 22} {'─' * 5} {'─' * 8} "
            f"{'─' * 5} {'─' * 5}   {'─' * 5}   {'─' * 6}   {'─' * 5}"
        )

        def _print_top(title, tdf):
            if tdf.empty:
                return
            print(f"\n  ── {title}")
            print(sum_hdr)
            print(sum_sep)
            for rank, (_, r) in enumerate(tdf.iterrows(), 1):
                _print_pick_row(r, rank=rank, show_rank=True)
            print(sum_sep)
            print("  ★ = High ≥66%   * = below breakeven 53%")

        _print_top("TOP 10 BY CONFIDENCE  most likely to hit", top_conf)
        _print_top("TOP 10 — RANKED  mkt-edge vs. devig market, else Edge%", top_edge)

    export_picks_db(all_picks_export, game_date)
    export_game_picks_db(team_preds, game_date)
    if all_picks_export:
        print("  Run calibrate.py after grading to keep the model improving.")

    if hasattr(props_model, "NULL_POS_PLAYERS") and props_model.NULL_POS_PLAYERS:
        print(
            f"\n  ⚠ {len(props_model.NULL_POS_PLAYERS)} player(s) had no position "
            f"(DvP=1.0 for these). Consider adding to MANUAL_POSITIONS in playerlinepredictor.py:"
        )
        for p in sorted(props_model.NULL_POS_PLAYERS):
            print(f"    {p}")

    print(f"\n{'═' * WIDTH}")
    print("  For informational purposes only. Not financial advice.")


def main():
    print("\n" + "═" * WIDTH)
    print("  NBA ANALYSIS  —  Team Model  ×  Player Props")
    if VERBOSE:
        print("  [ VERBOSE ]")
    print("═" * WIDTH)

    _init_calibration()

    matchups, game_date = _fetch_matchups_and_lean()
    if not matchups:
        return

    team_injuries, props_injuries = _fetch_injuries()
    team_preds, team_map = _run_team_model(matchups, team_injuries)
    all_events = _setup_odds_and_spreads(matchups, team_map)

    print("\n  Loading player data (stats, USG%, DvP rosters)…")
    props_model.get_team_season_stats()
    props_model.get_league_logs()
    props_model.get_all_player_usage()
    props_model.prefetch_rosters_for_games(all_events)

    all_props_results: list = []
    all_picks_export: list = []
    for m in matchups:
        results, _ = _process_single_game(m, team_map, all_events, props_injuries, all_picks_export)
        all_props_results.extend(results)

    _print_global_summary(team_preds, all_props_results, all_picks_export, game_date)
    print(f"{'═' * WIDTH}\n")


if __name__ == "__main__":
    main()
