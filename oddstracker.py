"""
NBA Odds Tracker
================
Snapshots bookmaker-implied probabilities for every player prop line each game
day, then compares them against actual results to learn which markets and lean
directions are profitable.

What it tracks per (player, market):
  - Consensus line and average over/under American odds across all books
  - Raw implied probability (before removing vig)
  - De-vigged probability (true implied probability with house edge removed)
  - Book lean: which side books are pricing more expensive (OVER / UNDER / NEUTRAL)
  - Vig percentage (how much juice the house is taking)
  - Lean strength: how far the line is from 50/50 (in de-vigged probability)

Usage:
    python3 oddstracker.py              # snapshot today's odds
    python3 oddstracker.py 2026-04-30   # snapshot a specific date
    python3 oddstracker.py --analyze    # show book lean accuracy from full history
    python3 oddstracker.py --analyze --update-calib  # also write to calibration.json

The tracker logs to odds_tracker.csv.  After checkresults.py runs for the same
date, it updates the 'result' column so accuracy can be measured.
"""

import csv
import json
import logging
import os
import sys
import time
import warnings
import requests
import pandas as pd
import urllib3
from datetime import date

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# CONFIG — import shared constants from playerlinepredictor to avoid duplication
# v7 Fix 4b: single source of truth for API key, base URL, and market lists
# ---------------------------------------------------------------------------
import playerlinepredictor as _props_model

ODDS_API_KEY  = _props_model.ODDS_API_KEY
BASE_URL      = _props_model.BASE_URL
SPORT         = _props_model.SPORT
MARKETS       = _props_model.MARKETS
MARKET_LABELS = _props_model.MARKET_LABELS

# Lean is "NEUTRAL" when de-vigged OVER probability is within this band of 50%.
NEUTRAL_BAND = 0.04   # |devig_over − 0.50| ≤ 4% → NEUTRAL

TRACKER_FILE  = os.path.join(_HERE, "odds_tracker.csv")
BET_LOG_FILE  = os.path.join(_HERE, "bet_log.csv")
CALIB_FILE    = os.path.join(_HERE, "calibration.json")

TRACKER_FIELDS = [
    "date", "game", "player", "market", "line",
    "books_count",
    "avg_over_odds", "avg_under_odds",
    "implied_over",  "implied_under",
    "devig_over",    "devig_under",
    "book_lean",     "lean_strength",
    "vig_pct",       "k_value",   # v7 Fix 5: power-method de-vig exponent
    "result",        # filled after games: OVER_HIT / UNDER_HIT / PUSH / DNP
]


# ===========================================================================
# MATH HELPERS
# ===========================================================================
def american_to_implied(price: float) -> float:
    """Convert American odds to raw implied probability (includes vig)."""
    if price >= 0:
        return 100.0 / (price + 100.0)
    else:
        return abs(price) / (abs(price) + 100.0)


def devig(over_price: float, under_price: float):
    """Remove vig using proportional method. Returns (devig_over, devig_under, vig_pct)."""
    raw_o = american_to_implied(over_price)
    raw_u = american_to_implied(under_price)
    total = raw_o + raw_u
    vig   = (total - 1.0) * 100.0
    dv_o  = raw_o / total
    dv_u  = raw_u / total
    return round(dv_o, 4), round(dv_u, 4), round(vig, 2)


def devig_power(over_price: float, under_price: float,
                max_iter: int = 50, tol: float = 1e-6):
    """Power-method de-vig using bisection search for exponent k.

    Finds k such that raw_over^k + raw_under^k = 1.0, which handles
    asymmetric lines more accurately than the proportional method.
    Returns (devig_over, devig_under, vig_pct, k).
    """
    raw_o = american_to_implied(over_price)
    raw_u = american_to_implied(under_price)
    total = raw_o + raw_u
    vig_pct = round((total - 1.0) * 100.0, 2)
    if total <= 1.0:
        return round(raw_o, 4), round(raw_u, 4), vig_pct, 1.0
    lo, hi = 0.5, 2.0
    k = 1.0
    for _ in range(max_iter):
        k = (lo + hi) / 2
        s = raw_o ** k + raw_u ** k
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = k
        else:
            hi = k
    return round(raw_o ** k, 4), round(raw_u ** k, 4), vig_pct, round(k, 4)


def book_lean(devig_over: float) -> tuple:
    """Return (lean_label, lean_strength) given de-vigged OVER probability."""
    strength = abs(devig_over - 0.50)
    if devig_over > 0.50 + NEUTRAL_BAND:
        return "OVER",  round(strength, 3)
    elif devig_over < 0.50 - NEUTRAL_BAND:
        return "UNDER", round(strength, 3)
    else:
        return "NEUTRAL", round(strength, 3)


# ===========================================================================
# ODDS API HELPERS
# ===========================================================================
def get_all_events() -> list:
    from datetime import datetime, timezone
    r = requests.get(
        f"{BASE_URL}/sports/{SPORT}/events",
        params={"apiKey": ODDS_API_KEY}, timeout=20, 
    )
    r.raise_for_status()
    events = r.json()
    now      = datetime.now(timezone.utc).isoformat()
    upcoming = [e for e in events if e.get("commence_time", "") > now]
    in_play  = len(events) - len(upcoming)
    if in_play:
        print(f"  Skipping {in_play} in-play game(s) — pre-game prop markets are closed.")
    return upcoming


def get_event_props(event_id: str) -> dict:
    """Fetch player prop odds for a single event. Tries wider regions if needed."""
    for regions in ["us", "us,us2,uk,eu,au"]:
        try:
            r = requests.get(
                f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds",
                params={
                    "apiKey":      ODDS_API_KEY,
                    "regions":     regions,
                    "markets":     ",".join(MARKETS),
                    "oddsFormat":  "american",
                    "dateFormat":  "iso",
                },
                timeout=25, 
            )
            r.raise_for_status()
            data = r.json()
            if data.get("bookmakers"):
                return data
        except Exception as e:
            print(f"    Warning: props fetch ({regions}): {e}")
    return {}


# ===========================================================================
# PARSE & AGGREGATE ODDS
# ===========================================================================
def parse_props_to_rows(props_json: dict, game_key: str, snap_date: str) -> list:
    """
    Parse raw Odds API response into per-(player, market) rows with averaged
    odds across all bookmakers, implied probabilities, and book lean.
    """
    bookmakers = props_json.get("bookmakers", [])
    if not bookmakers:
        return []

    # Collect all (bookmaker, player, market, side, line, price) tuples
    raw = []
    for bk in bookmakers:
        book = bk.get("title", "unknown")
        for mkt in bk.get("markets", []):
            mk   = mkt.get("key", "")
            label = MARKET_LABELS.get(mk, mk)
            over_row  = None
            under_row = None
            for oc in mkt.get("outcomes", []):
                side  = oc.get("name", "").lower()
                price = oc.get("price")
                point = oc.get("point")
                player = oc.get("description", "")
                if not player or price is None:
                    continue
                entry = {"book": book, "player": player, "market": label,
                         "line": point, "price": float(price)}
                if side == "over":
                    over_row  = entry
                elif side == "under":
                    under_row = entry
            # Only process when both sides are present from the same book
            if over_row and under_row and over_row["player"] == under_row["player"]:
                raw.append({
                    "book":    book,
                    "player":  over_row["player"],
                    "market":  over_row["market"],
                    "line":    over_row["line"],
                    "over_p":  over_row["price"],
                    "under_p": under_row["price"],
                })

    if not raw:
        return []

    df = pd.DataFrame(raw)
    rows = []
    for (player, market), grp in df.groupby(["player", "market"]):
        consensus_line  = float(grp["line"].median())
        avg_over_odds   = round(float(grp["over_p"].mean()), 1)
        avg_under_odds  = round(float(grp["under_p"].mean()), 1)
        n_books         = int(grp["book"].nunique())

        dv_over, dv_under, vig, k_val = devig_power(avg_over_odds, avg_under_odds)  # v7 Fix 5
        lean, strength                = book_lean(dv_over)

        rows.append({
            "date":         snap_date,
            "game":         game_key,
            "player":       player,
            "market":       market,
            "line":         consensus_line,
            "books_count":  n_books,
            "avg_over_odds":  avg_over_odds,
            "avg_under_odds": avg_under_odds,
            "implied_over":   round(american_to_implied(avg_over_odds), 4),
            "implied_under":  round(american_to_implied(avg_under_odds), 4),
            "devig_over":     dv_over,
            "devig_under":    dv_under,
            "book_lean":      lean,
            "lean_strength":  strength,
            "vig_pct":        vig,
            "k_value":        k_val,
            "result":         "",   # filled after game via update_results()
        })
    return rows


# ===========================================================================
# SNAPSHOT — fetch and save today's odds
# ===========================================================================
def snapshot(snap_date: date):
    """Fetch all player prop odds for snap_date and log to odds_tracker.csv."""
    print(f"\n  Fetching events for {snap_date.isoformat()}…")
    try:
        events = get_all_events()
    except Exception as e:
        print(f"  Error fetching events: {e}")
        return

    if not events:
        print("  No events found.")
        return

    print(f"  Found {len(events)} event(s).")
    all_rows = []

    for ev in events:
        home = ev.get("home_team", "?")
        away = ev.get("away_team", "?")
        gk   = f"{away[:3].upper()}@{home[:3].upper()}"
        print(f"  Fetching props: {gk}…")
        try:
            props = get_event_props(ev["id"])
            rows  = parse_props_to_rows(props, gk, snap_date.isoformat())
            print(f"    {len(rows)} player-market line(s) parsed")
            all_rows.extend(rows)
        except Exception as e:
            print(f"    Error: {e}")
        time.sleep(0.8)   # rate-limit courtesy pause

    if not all_rows:
        print("  No prop rows collected — lines may not be posted yet or have already closed.")
        print("  Tip: books post player props 2–6 hours before tip and close ~15 min before.")
        return

    _append_rows(all_rows, snap_date.isoformat())
    print(f"\n  Logged {len(all_rows)} prop line(s) → {os.path.basename(TRACKER_FILE)}")


def _append_rows(rows: list, snap_date: str):
    """Append rows to odds_tracker.csv; skip duplicates (same date/player/market)."""
    # Load existing keys to deduplicate
    existing_keys = set()
    if os.path.isfile(TRACKER_FILE) and os.path.getsize(TRACKER_FILE) > 0:
        try:
            existing = pd.read_csv(TRACKER_FILE, usecols=["date", "player", "market"])
            for _, r in existing.iterrows():
                if str(r["date"]) == snap_date:
                    existing_keys.add((r["player"], r["market"]))
        except (OSError, KeyError, pd.errors.ParserError) as exc:
            logger.warning("Could not read existing tracker rows: %s", exc)

    new_rows = [r for r in rows
                if (r["player"], r["market"]) not in existing_keys]
    if not new_rows:
        print("  All rows already logged for this date — nothing new added.")
        return

    file_exists = os.path.isfile(TRACKER_FILE) and os.path.getsize(TRACKER_FILE) > 0
    with open(TRACKER_FILE, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRACKER_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)


# ===========================================================================
# SCHEMA MIGRATION — add k_value column to old odds_tracker.csv rows
# ===========================================================================
def _migrate_tracker_csv():
    """One-time migration: insert 'k_value' column before 'result' if missing."""
    if not os.path.isfile(TRACKER_FILE):
        return
    with open(TRACKER_FILE) as fh:
        lines = fh.readlines()
    if not lines:
        return
    header_fields = [c.strip() for c in lines[0].strip().split(",")]
    if "k_value" in header_fields:
        return  # already migrated

    new_lines = []
    # Find position of 'result' column to insert k_value before it
    result_idx = header_fields.index("result") if "result" in header_fields else len(header_fields)
    for i, line in enumerate(lines):
        fields = line.rstrip("\n").split(",")
        if i == 0:
            fields = fields[:result_idx] + ["k_value"] + fields[result_idx:]
        elif len(fields) == len(header_fields):
            fields = fields[:result_idx] + [""] + fields[result_idx:]
        new_lines.append(",".join(fields) + "\n")

    with open(TRACKER_FILE, "w") as fh:
        fh.writelines(new_lines)
    print("  Migrated odds_tracker.csv → added 'k_value' column to existing rows.")


# ===========================================================================
# UPDATE RESULTS — fill in 'result' column after games finish
# ===========================================================================
def update_results():
    """Cross-reference odds_tracker.csv with bet_log.csv to fill result column.

    For each (date, player, market) in odds_tracker where result is blank,
    look for a matching row in bet_log and derive whether OVER or UNDER hit.
    bet_log tells us the actual value; we compare to the consensus line.
    """
    if not os.path.isfile(TRACKER_FILE):
        return
    if not os.path.isfile(BET_LOG_FILE):
        return

    _migrate_tracker_csv()
    tracker = pd.read_csv(TRACKER_FILE)
    betlog  = pd.read_csv(BET_LOG_FILE)

    if "actual" not in betlog.columns:
        return

    betlog["actual"] = pd.to_numeric(betlog["actual"], errors="coerce")

    updated = 0
    for idx, row in tracker.iterrows():
        if pd.notna(row["result"]) and str(row["result"]).strip():
            continue   # already filled

        match = betlog[
            (betlog["date"]   == str(row["date"])) &
            (betlog["player"] == str(row["player"])) &
            (betlog["market"] == str(row["market"]))
        ]
        if match.empty:
            continue

        actual = float(match.iloc[0]["actual"])
        line   = float(row["line"])

        if actual > line:
            tracker.at[idx, "result"] = "OVER_HIT"
        elif actual < line:
            tracker.at[idx, "result"] = "UNDER_HIT"
        else:
            tracker.at[idx, "result"] = "PUSH"
        updated += 1

    if updated > 0:
        tracker.to_csv(TRACKER_FILE, index=False)
        print(f"  Updated {updated} result(s) in odds_tracker.csv")


# ===========================================================================
# ANALYSIS — book lean accuracy and market profitability
# ===========================================================================
def analyze(update_calib: bool = False):
    """
    Analyze how accurate the book lean direction is per market.

    Outputs:
      - Overall book lean accuracy (OVER lean → did OVER hit?)
      - Per-market lean accuracy
      - Vig analysis (are high-vig markets harder to beat?)
      - Strong lean accuracy (lean_strength > 0.06 vs weak lean)
    Optionally writes book_lean section to calibration.json.
    """
    if not os.path.isfile(TRACKER_FILE):
        print("  odds_tracker.csv not found — run snapshot first.")
        return

    _migrate_tracker_csv()
    df = pd.read_csv(TRACKER_FILE)
    df = df[df["result"].isin(["OVER_HIT", "UNDER_HIT"])].copy()

    if df.empty:
        print("  No graded rows yet — run update_results() first.")
        return

    # Did the book lean direction actually win?
    def lean_hit(row):
        if row["book_lean"] == "NEUTRAL":
            return None
        return (row["book_lean"] == "OVER"  and row["result"] == "OVER_HIT") or \
               (row["book_lean"] == "UNDER" and row["result"] == "UNDER_HIT")

    df["lean_correct"] = df.apply(lean_hit, axis=1)
    # Did going AGAINST the book lean win (contrarian signal)?
    df["contra_correct"] = df["lean_correct"].map({True: False, False: True, None: None})

    non_neutral = df[df["book_lean"] != "NEUTRAL"]
    graded_n    = len(non_neutral)

    print("\n" + "=" * 65)
    print("  ODDS TRACKER — Book Lean Analysis")
    print("=" * 65)
    print(f"  Total graded non-neutral rows: {graded_n}")
    if graded_n == 0:
        print("  Not enough data yet.")
        return

    overall_acc  = non_neutral["lean_correct"].mean() * 100
    contra_acc   = non_neutral["contra_correct"].mean() * 100
    print(f"  Follow-the-book accuracy:   {overall_acc:.1f}%")
    print(f"  Fade-the-book (contrarian): {contra_acc:.1f}%")
    print(f"  ({'Follow' if overall_acc > contra_acc else 'Fade'} the book is more profitable)")

    # ── Per-market breakdown ────────────────────────────────────────────────
    print(f"\n  {'Market':<12} {'N':>4} {'FollowAcc':>10} {'FadeAcc':>9}  {'Best lean':>12}  Avg Vig")
    print(f"  {'─'*12} {'─'*4} {'─'*10} {'─'*9}  {'─'*12}  {'─'*7}")
    market_stats = {}
    for market, grp in non_neutral.groupby("market"):
        n      = len(grp)
        f_acc  = grp["lean_correct"].mean() * 100
        c_acc  = grp["contra_correct"].mean() * 100
        avg_vig= grp["vig_pct"].mean()
        best   = f"Follow ({f_acc:.0f}%)" if f_acc >= c_acc else f"Fade ({c_acc:.0f}%)"
        print(f"  {market:<12} {n:>4} {f_acc:>9.1f}% {c_acc:>8.1f}%  {best:>12}  {avg_vig:>5.1f}%")
        market_stats[market] = {
            "n": n,
            "follow_acc":  round(f_acc / 100, 3),
            "fade_acc":    round(c_acc / 100, 3),
            "strategy":    "follow" if f_acc >= c_acc else "fade",
            "avg_vig":     round(avg_vig, 2),
        }

    # ── Lean direction breakdown: OVER lean vs UNDER lean ──────────────────
    print(f"\n  {'Direction':<10} {'N':>4}  {'Acc':>6}  Avg Lean Strength")
    print(f"  {'─'*10} {'─'*4}  {'─'*6}  {'─'*18}")
    for lean_dir in ["OVER", "UNDER"]:
        sub = non_neutral[non_neutral["book_lean"] == lean_dir]
        if sub.empty:
            continue
        acc = sub["lean_correct"].mean() * 100
        avg_s = sub["lean_strength"].mean()
        print(f"  {lean_dir:<10} {len(sub):>4}  {acc:>5.1f}%  {avg_s:.3f}")

    # ── Lean strength breakdown ─────────────────────────────────────────────
    print(f"\n  {'Lean Strength':<18} {'N':>4}  {'Acc':>6}  Description")
    print(f"  {'─'*18} {'─'*4}  {'─'*6}  {'─'*25}")
    bins = [
        ("Weak  (<0.05)",  non_neutral[non_neutral["lean_strength"] <  0.05]),
        ("Mod   (0.05-0.10)", non_neutral[(non_neutral["lean_strength"] >= 0.05) &
                                          (non_neutral["lean_strength"] <  0.10)]),
        ("Strong (≥0.10)", non_neutral[non_neutral["lean_strength"] >= 0.10]),
    ]
    for label, sub in bins:
        if sub.empty:
            continue
        acc = sub["lean_correct"].mean() * 100
        print(f"  {label:<18} {len(sub):>4}  {acc:>5.1f}%  "
              f"{'Books confident' if 'Strong' in label else ''}")

    # ── Vig analysis ────────────────────────────────────────────────────────
    low_vig  = non_neutral[non_neutral["vig_pct"] < 5.0]
    high_vig = non_neutral[non_neutral["vig_pct"] >= 5.0]
    if not low_vig.empty and not high_vig.empty:
        print(f"\n  Vig <5%:  follow acc {low_vig['lean_correct'].mean()*100:.1f}%  "
              f"(n={len(low_vig)}) — more efficient, harder to beat")
        print(f"  Vig ≥5%:  follow acc {high_vig['lean_correct'].mean()*100:.1f}%  "
              f"(n={len(high_vig)}) — more juice, may signal sharp disagreement")

    print("=" * 65)

    # ── Optionally update calibration.json ──────────────────────────────────
    if update_calib:
        _write_to_calib(market_stats, non_neutral)


def _write_to_calib(market_stats: dict, df: pd.DataFrame):
    """Write book lean accuracy to calibration.json as a new 'book_lean' section."""
    calib = {}
    if os.path.isfile(CALIB_FILE):
        try:
            with open(CALIB_FILE) as fh:
                calib = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load calibration.json: %s", exc)

    # Per-(market, lean_direction) stats
    lean_detail = {}
    for (market, lean_dir), grp in df.groupby(["market", "book_lean"]):
        n       = len(grp)
        acc     = float(grp["lean_correct"].mean())
        avg_str = float(grp["lean_strength"].mean())
        key = f"{market}_{lean_dir}"
        lean_detail[key] = {
            "n":          n,
            "accuracy":   round(acc, 3),
            "avg_strength": round(avg_str, 3),
            # adj: positive = follow book is profitable, negative = fade is better
            "strategy_adj": round((acc - 0.5) * 100, 1),
        }

    calib["book_lean"] = {
        "generated":   date.today().isoformat(),
        "total_graded": int(len(df)),
        "by_market":   market_stats,
        "by_market_lean": lean_detail,
    }

    with open(CALIB_FILE, "w") as fh:
        json.dump(calib, fh, indent=2)
    print(f"\n  Written book_lean section → {os.path.basename(CALIB_FILE)}")


# ===========================================================================
# QUICK SUMMARY — show today's snapshot with lean labels
# ===========================================================================
def show_today(snap_date: date):
    """Print a readable table of today's odds snapshot."""
    if not os.path.isfile(TRACKER_FILE):
        print("  No odds_tracker.csv found.")
        return

    _migrate_tracker_csv()
    df = pd.read_csv(TRACKER_FILE)
    today = df[df["date"] == snap_date.isoformat()].copy()
    if today.empty:
        print(f"  No snapshot for {snap_date.isoformat()} — run without --show to fetch first.")
        return

    print(f"\n{'═'*76}")
    print(f"  ODDS SNAPSHOT — {snap_date.strftime('%A, %B %d %Y')}")
    print(f"{'═'*76}")
    print(f"  {'Player':<22} {'Mkt':<9} {'Line':>5} {'O-Odds':>7} {'U-Odds':>7} "
          f"{'Lean':<8} {'Strength':>8} {'Vig':>5}")
    print(f"  {'─'*22} {'─'*9} {'─'*5} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*5}")

    # Sort by game, then market, then lean strength
    today = today.sort_values(["game", "market", "lean_strength"], ascending=[True, True, False])
    current_game = None
    for _, r in today.iterrows():
        if r["game"] != current_game:
            current_game = r["game"]
            print(f"\n  ── {r['game']} ──")
        lean_marker = "▲" if r["book_lean"] == "OVER" else ("▼" if r["book_lean"] == "UNDER" else "─")
        print(f"  {r['player']:<22} {r['market']:<9} {r['line']:>5.1f} "
              f"{r['avg_over_odds']:>+7.0f} {r['avg_under_odds']:>+7.0f}  "
              f"{lean_marker} {r['book_lean']:<6} {r['lean_strength']:>8.3f} "
              f"{r['vig_pct']:>4.1f}%")

    print(f"\n  {len(today)} prop lines tracked.  ▲=OVER lean  ▼=UNDER lean  ─=NEUTRAL")
    print(f"{'═'*76}\n")


# ===========================================================================
# DISAGREEMENT ANALYSIS — model picks vs book lean
# ===========================================================================
def analyze_disagreements():
    """Compare model picks vs book lean direction — hit rate when model agrees vs disagrees."""
    if not os.path.isfile(TRACKER_FILE):
        print("  odds_tracker.csv not found — run snapshot first.")
        return
    if not os.path.isfile(BET_LOG_FILE):
        print("  bet_log.csv not found — run checkresults.py first.")
        return

    odds_df = pd.read_csv(TRACKER_FILE)
    bet_df  = pd.read_csv(BET_LOG_FILE)

    merged = bet_df.merge(
        odds_df[["date", "player", "market", "book_lean", "lean_strength"]],
        on=["date", "player", "market"],
        how="inner",
    )
    merged = merged[merged["result"].isin(["HIT", "MISS"])].copy()
    if merged.empty:
        print("  No overlapping graded rows between bet_log and odds_tracker.")
        return

    merged["agrees"] = (merged["pick"] == merged["book_lean"])
    merged["hit"]    = (merged["result"] == "HIT").astype(int)

    agreed    = merged[merged["agrees"] == True]
    disagreed = merged[merged["agrees"] == False]

    print("\n" + "=" * 65)
    print("  MODEL vs BOOK — Disagreement Analysis")
    print("=" * 65)
    print(f"  Overlapping graded bets: {len(merged)}")
    if not agreed.empty:
        print(f"  Model agrees  with book:  {len(agreed):>3}  "
              f"hit rate {agreed['hit'].mean()*100:.1f}%")
    if not disagreed.empty:
        print(f"  Model disagrees w/ book:  {len(disagreed):>3}  "
              f"hit rate {disagreed['hit'].mean()*100:.1f}%")

    print(f"\n  {'Market':<12} {'Agree N':>8} {'Agree%':>8} {'Disagree N':>11} {'Disagree%':>10}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*11} {'─'*10}")
    for market in sorted(merged["market"].unique()):
        sub   = merged[merged["market"] == market]
        a     = sub[sub["agrees"] == True]
        d     = sub[sub["agrees"] == False]
        a_pct = f"{a['hit'].mean()*100:.1f}%" if not a.empty else "—"
        d_pct = f"{d['hit'].mean()*100:.1f}%" if not d.empty else "—"
        print(f"  {market:<12} {len(a):>8} {a_pct:>8} {len(d):>11} {d_pct:>10}")

    print("=" * 65)


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    args = sys.argv[1:]

    # Parse date argument if provided
    snap_date = date.today()
    for a in args:
        try:
            snap_date = date.fromisoformat(a)
        except ValueError:
            pass

    update_calib = "--update-calib" in args

    if "--analyze" in args:
        update_results()
        analyze(update_calib=update_calib)
    elif "--disagree" in args:           # v7 Fix 7
        update_results()
        analyze_disagreements()
    elif "--show" in args:
        show_today(snap_date)
    else:
        snapshot(snap_date)
        update_results()
        show_today(snap_date)
