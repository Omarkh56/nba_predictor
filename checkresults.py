"""
NBA Props Results Checker
==========================
Automatically loads predictions from predictions_YYYY-MM-DD.json written by
nba_combined.py.  If no JSON file is found for the target date, falls back to
the manual PREDICTIONS list below.

Usage:
    python3 checkresults.py              # grades yesterday's games
    python3 checkresults.py 2026-04-17   # grades a specific date

Requirements:
    pip install nba_api pandas
"""

import csv
import json
import logging
import os
import sys
import time
import warnings
from datetime import date, timedelta

import pandas as pd
import requests.exceptions
from nba_api.stats.endpoints import boxscoretraditionalv2, leaguegamefinder
from nba_api.stats.static import players

import oddstracker
import predictions_db as pred_db

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))

# Date to check — pass as argument (YYYY-MM-DD) or defaults to yesterday
if len(sys.argv) > 1:
    CHECK_DATE = date.fromisoformat(sys.argv[1])
else:
    CHECK_DATE = date.today() - timedelta(days=1)


# =============================================================================
# AUTO-LOAD from predictions.db (written by nba_combined.py)
# Falls back to the manual PREDICTIONS list below if no rows exist.
# =============================================================================
def _load_predictions_db(target_date: date) -> list:
    rows = pred_db.load_prop_predictions(target_date)
    if rows:
        print(
            f"  Auto-loaded {len(rows)} prediction(s) from predictions.db "
            f"({target_date.isoformat()})"
        )
    return rows


# =============================================================================
# MANUAL FALLBACK — only used when no DB rows exist for CHECK_DATE
# =============================================================================
PREDICTIONS = []

# =============================================================================
# Resolve which predictions to use
# =============================================================================
_auto = _load_predictions_db(CHECK_DATE)
PREDICTIONS = _auto if _auto else PREDICTIONS

# Path to the running bet log (auto-created if missing)
LOG_FILE = os.path.join(_HERE, "bet_log.csv")
GAME_LOG_FILE = os.path.join(_HERE, "game_log.csv")

# =============================================================================
# MARKET → BOX SCORE COLUMNS
# =============================================================================
MARKET_TO_COLS = {
    "PTS": ["PTS"],
    "REB": ["REB"],
    "AST": ["AST"],
    "3PM": ["FG3M"],
    "PTS+REB": ["PTS", "REB"],
    "PTS+AST": ["PTS", "AST"],
    "REB+AST": ["REB", "AST"],
    "PRA": ["PTS", "REB", "AST"],
}


# =============================================================================
# GAME PREDICTIONS — load from predictions.db
#
# Schema stored by nba_combined.py:
#   home_abbr, away_abbr, winner_pick, spread (home-team line),
#   spread_pick, win_pct, total
#
# Spread convention:
#   spread = -7.5  →  home favored by 7.5; home covers if they win by 8+
#   spread = +4.5  →  away favored by 4.5; home covers if they lose by <5 or win
# =============================================================================
def _load_game_predictions_db(target_date: date) -> list:
    rows = pred_db.load_game_predictions(target_date)
    if rows:
        print(
            f"  Auto-loaded {len(rows)} game prediction(s) from predictions.db "
            f"({target_date.isoformat()})"
        )
    return rows


def build_game_results(check_date: date) -> dict:
    """Return {(home_abbr, away_abbr): result_dict} for all games on check_date.

    Uses leaguegamefinder (already called by get_games_on_date) — no extra
    box-score fetches needed; team scores are in the gamefinder response.
    """
    if _is_offseason(check_date):
        return {}

    date_str = check_date.strftime("%m/%d/%Y")
    try:
        gf = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=date_str,
            date_to_nullable=date_str,
            league_id_nullable="00",
        )
        df = gf.get_data_frames()[0]
    except (requests.exceptions.RequestException, json.JSONDecodeError, IndexError) as e:
        print(f"  Warning: could not reach stats.nba.com for game results — {e}")
        return {}
    if df.empty:
        return {}

    results = {}
    for game_id in df["GAME_ID"].unique():
        sub = df[df["GAME_ID"] == game_id]
        home_rows = sub[sub["MATCHUP"].str.contains(r"vs\.", na=False)]
        away_rows = sub[sub["MATCHUP"].str.contains("@", na=False)]
        if home_rows.empty or away_rows.empty:
            continue
        h = home_rows.iloc[0]
        a = away_rows.iloc[0]
        home_abbr = str(h["TEAM_ABBREVIATION"]).upper()
        away_abbr = str(a["TEAM_ABBREVIATION"]).upper()
        home_score = int(h["PTS"])
        away_score = int(a["PTS"])
        margin = home_score - away_score  # positive = home won by this much
        winner = home_abbr if home_score > away_score else away_abbr
        results[(home_abbr, away_abbr)] = {
            "game_id": game_id,
            "home": home_abbr,
            "away": away_abbr,
            "home_score": home_score,
            "away_score": away_score,
            "margin": margin,
            "winner": winner,
        }
    return results


def check_game_predictions(game_preds: list, game_results: dict) -> list:
    """Grade game winner and spread predictions against actual results."""
    out = []
    for pred in game_preds:
        home = pred["home_abbr"]
        away = pred["away_abbr"]

        gr = game_results.get((home, away)) or game_results.get((away, home))
        if gr is None:
            out.append(
                {
                    **pred,
                    "home_score": None,
                    "away_score": None,
                    "winner": None,
                    "winner_hit": "NOT FOUND",
                    "home_margin": None,
                    "spread_hit": "NOT FOUND",
                }
            )
            continue

        # Winner check
        winner_hit = "HIT ✓" if gr["winner"] == pred["winner_pick"] else "MISS ✗"

        # Spread check: home_margin = home_score - away_score
        #   home covers if home_margin > -spread
        #   push         if home_margin == -spread
        spread = pred["spread"]
        spread_hit = "N/A"
        if spread is not None:
            hm = gr["margin"]  # home_score - away_score
            if hm == -spread:
                spread_hit = "PUSH"
            elif pred["spread_pick"] == gr["home"]:
                spread_hit = "HIT ✓" if hm > -spread else "MISS ✗"
            elif pred["spread_pick"] == gr["away"]:
                spread_hit = "HIT ✓" if hm < -spread else "MISS ✗"

        out.append(
            {
                "home": gr["home"],
                "away": gr["away"],
                "home_score": gr["home_score"],
                "away_score": gr["away_score"],
                "winner": gr["winner"],
                "winner_pick": pred["winner_pick"],
                "winner_hit": winner_hit,
                "spread": spread,
                "spread_pick": pred["spread_pick"],
                "home_margin": gr["margin"],
                "spread_hit": spread_hit,
            }
        )
    return out


def print_game_results(results: list, check_date: date):
    if not results:
        return

    print()
    print("=" * 80)
    print(f"  GAME PREDICTIONS — {check_date.strftime('%A, %B %d %Y')}")
    print("=" * 80)
    print(
        f"  {'Game':<12} {'Score':<12} {'Winner':>7} {'Pick':>7} {'W-Res':<8}  "
        f"{'Spread':>7} {'S-Pick':>7} {'S-Res':<8}"
    )
    print(f"  {'─' * 12} {'─' * 12} {'─' * 7} {'─' * 7} {'─' * 8}  {'─' * 7} {'─' * 7} {'─' * 8}")

    winner_hits = 0
    winner_total = 0
    spread_hits = 0
    spread_total = 0

    for r in results:
        if r["home_score"] is None:
            game_str = f"{r['away_abbr']}@{r['home_abbr']}"
            score_str = "NOT FOUND"
            print(f"  {game_str:<12} {score_str:<12}")
            continue

        game_str = f"{r['away']}@{r['home']}"
        score_str = f"{r['away_score']}-{r['home_score']}"
        spread_str = f"{r['spread']:+.1f}" if r["spread"] is not None else "  N/A"
        spick_str = r["spread_pick"] if r["spread"] is not None else "  N/A"
        sres_str = r["spread_hit"]

        print(
            f"  {game_str:<12} {score_str:<12} {r['winner']:>7} {r['winner_pick']:>7} "
            f"{r['winner_hit']:<8}  {spread_str:>7} {spick_str:>7} {sres_str:<8}"
        )

        if "HIT" in r["winner_hit"] or "MISS" in r["winner_hit"]:
            winner_total += 1
            if "HIT" in r["winner_hit"]:
                winner_hits += 1

        if "HIT" in r["spread_hit"] or "MISS" in r["spread_hit"]:
            spread_total += 1
            if "HIT" in r["spread_hit"]:
                spread_hits += 1

    print(f"\n  {'─' * 60}")
    if winner_total > 0:
        print(
            f"  Winner picks:  {winner_hits}W – {winner_total - winner_hits}L  "
            f"({winner_hits / winner_total * 100:.1f}%)"
        )
    if spread_total > 0:
        print(
            f"  Spread picks:  {spread_hits}W – {spread_total - spread_hits}L  "
            f"({spread_hits / spread_total * 100:.1f}%)"
        )
    print("=" * 80)


def append_game_to_log(results: list, check_date: date):
    """Append graded game predictions to game_log.csv."""
    fieldnames = [
        "date",
        "game",
        "home",
        "away",
        "home_score",
        "away_score",
        "winner",
        "winner_pick",
        "winner_hit",
        "spread",
        "spread_pick",
        "home_margin",
        "spread_hit",
    ]

    file_exists = os.path.isfile(GAME_LOG_FILE) and os.path.getsize(GAME_LOG_FILE) > 0
    new_rows = []

    for r in results:
        if r["home_score"] is None:
            continue  # game not found — skip

        winner_out = (
            "HIT"
            if "HIT" in r["winner_hit"]
            else ("MISS" if "MISS" in r["winner_hit"] else r["winner_hit"])
        )
        spread_out = (
            "HIT"
            if "HIT" in r["spread_hit"]
            else ("MISS" if "MISS" in r["spread_hit"] else r["spread_hit"])
        )

        new_rows.append(
            {
                "date": check_date.isoformat(),
                "game": f"{r['away']}@{r['home']}",
                "home": r["home"],
                "away": r["away"],
                "home_score": r["home_score"],
                "away_score": r["away_score"],
                "winner": r["winner"],
                "winner_pick": r["winner_pick"],
                "winner_hit": winner_out,
                "spread": r["spread"] if r["spread"] is not None else "",
                "spread_pick": r["spread_pick"],
                "home_margin": r["home_margin"],
                "spread_hit": spread_out,
            }
        )

    if not new_rows:
        return

    with open(GAME_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)

    print(f"  Logged {len(new_rows)} game result(s) → {os.path.basename(GAME_LOG_FILE)}")


# =============================================================================
# FETCH ACTUAL STATS
# =============================================================================
def get_player_id(name):
    matches = players.find_players_by_full_name(name)
    if not matches:
        return None
    active = [p for p in matches if p.get("is_active")]
    return active[0]["id"] if active else matches[0]["id"]


def _is_offseason(check_date: date) -> bool:
    """NBA regular season runs roughly Oct–Apr; playoffs through mid-June.
    Returns True for the clear off-season window (mid-June → early October)
    so we can skip network calls when there are definitely no games."""
    return check_date.month in (7, 8, 9) or (check_date.month == 6 and check_date.day > 20)


def get_games_on_date(check_date):
    """Returns list of game_ids played on check_date. Returns [] on network error."""
    date_str = check_date.strftime("%m/%d/%Y")
    try:
        gf = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=date_str,
            date_to_nullable=date_str,
            league_id_nullable="00",
        )
        df = gf.get_data_frames()[0]
    except (requests.exceptions.RequestException, json.JSONDecodeError, IndexError) as e:
        print(f"  Warning: could not reach stats.nba.com — {e}")
        return []
    if df.empty:
        return []
    return df["GAME_ID"].unique().tolist()


def get_box_score(game_id):
    """Returns player stats DataFrame for a game."""
    time.sleep(0.8)
    bs = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
    return bs.get_data_frames()[0]  # player stats


def build_actual_stats(check_date):
    """
    Returns a dict: { player_id: {PTS, REB, AST, FG3M, ...} }
    for all players who played on check_date.
    """
    if _is_offseason(check_date):
        print(f"  Off-season ({check_date.isoformat()}) — no NBA games to fetch.")
        return {}

    print(f"Fetching games for {check_date.strftime('%A, %B %d %Y')}...")
    game_ids = get_games_on_date(check_date)
    if not game_ids:
        print("  No games found for that date.")
        return {}

    print(f"  Found {len(game_ids)} game(s). Fetching box scores...")
    all_rows = []
    for gid in game_ids:
        try:
            df = get_box_score(gid)
            all_rows.append(df)
        except Exception:
            # intentionally broad: one bad box-score must not abort the rest of the batch
            logger.exception("Box score fetch failed for game %s", gid)

    if not all_rows:
        return {}

    combined = pd.concat(all_rows, ignore_index=True)
    stats = {}
    for _, row in combined.iterrows():
        pid = row.get("PLAYER_ID")
        if pd.isna(pid):
            continue
        stats[int(pid)] = {
            "name": row.get("PLAYER_NAME", ""),
            "PTS": float(row.get("PTS", 0) or 0),
            "REB": float(row.get("REB", 0) or 0),
            "AST": float(row.get("AST", 0) or 0),
            "FG3M": float(row.get("FG3M", 0) or 0),
            "MIN": row.get("MIN", "0"),
        }
    print(f"  Loaded stats for {len(stats)} players.\n")
    return stats


# =============================================================================
# EVALUATE
# =============================================================================
def check_predictions(predictions, actual_stats):
    if not predictions:
        print("No predictions to check — fill in the PREDICTIONS list at the top of this file.")
        return

    results = []
    for pred in predictions:
        player = pred["player"]
        market = pred["market"].upper()
        line = float(pred["line"])
        pick = pred["pick"].upper()

        cols = MARKET_TO_COLS.get(market)
        if cols is None:
            print(f"  Unknown market '{market}' for {player} — skipping.")
            continue

        pid = get_player_id(player)
        if pid is None:
            results.append({**pred, "actual": None, "result": "PLAYER NOT FOUND"})
            continue

        player_data = actual_stats.get(pid)
        if player_data is None:
            results.append({**pred, "actual": None, "result": "DID NOT PLAY"})
            continue

        actual_val = sum(player_data.get(c, 0) for c in cols)

        if pick == "OVER":
            hit = actual_val > line
        else:  # UNDER
            hit = actual_val < line

        push = actual_val == line

        if push:
            outcome = "PUSH"
            margin = 0.0
        elif pick == "OVER":
            outcome = "HIT ✓" if hit else "MISS ✗"
            margin = round(actual_val - line, 1)  # +cushion / −missed by
        else:
            outcome = "HIT ✓" if hit else "MISS ✗"
            margin = round(line - actual_val, 1)  # +cushion / −missed by

        results.append(
            {
                "player": player,
                "market": market,
                "line": line,
                "pick": pick,
                "projection": pred.get(
                    "projection", ""
                ),  # v8 Fix: persist projection through grading pipeline
                "actual": round(actual_val, 1),
                "margin": margin,
                "result": outcome,
                "minutes": player_data.get("MIN", "?"),
            }
        )

    return results


def print_results(results):
    if not results:
        return

    hits = sum(1 for r in results if "HIT" in r["result"])
    misses = sum(1 for r in results if "MISS" in r["result"])
    pushes = sum(1 for r in results if "PUSH" in r["result"])
    dnp = sum(1 for r in results if r["actual"] is None)
    graded = hits + misses

    print("=" * 80)
    print(f"  RESULTS — {CHECK_DATE.strftime('%A, %B %d %Y')}")
    print("=" * 80)
    print(
        f"  {'Player':<22} {'Mkt':<9} {'Line':>5} {'Pick':<6} {'Actual':>6}  {'Margin':>7}  Result"
    )
    print(f"  {'─' * 22} {'─' * 9} {'─' * 5} {'─' * 6} {'─' * 6}  {'─' * 7}  {'─' * 10}")

    for r in results:
        actual_str = f"{r['actual']:>6.1f}" if r["actual"] is not None else "   N/A"
        if r.get("margin") is not None and r["actual"] is not None:
            m = r["margin"]
            margin_str = f"{m:>+7.1f}"
        else:
            margin_str = "       "
        print(
            f"  {r['player']:<22} {r['market']:<9} {r['line']:>5.1f} "
            f"{r['pick']:<6} {actual_str}  {margin_str}  {r['result']}"
        )

    print(f"\n  {'─' * 60}")
    if graded > 0:
        acc = hits / graded * 100
        print(f"  Record:   {hits}W – {misses}L" + (f" – {pushes}P" if pushes else ""))
        print(f"  Accuracy: {acc:.1f}%  ({graded} graded bets)")

    # Near-miss / cushion summary for graded bets with margin data
    graded_margins = [
        r
        for r in results
        if r.get("margin") is not None
        and r["actual"] is not None
        and ("HIT" in r["result"] or "MISS" in r["result"])
    ]
    if graded_margins:
        near_misses = [
            r for r in graded_margins if "MISS" in r["result"] and -1.0 <= r["margin"] < 0
        ]
        miss_list = [r for r in graded_margins if "MISS" in r["result"]]
        hit_list = [r for r in graded_margins if "HIT" in r["result"]]

        if near_misses:
            print("\n  Near-misses (within 1 unit of line):")
            for r in sorted(near_misses, key=lambda x: x["margin"], reverse=True):
                print(f"    {r['player']:<22} {r['market']:<9}  missed by {abs(r['margin']):.1f}")

        if miss_list:
            closest = max(miss_list, key=lambda x: x["margin"])
            print(
                f"\n  Closest miss:   {closest['player']} {closest['market']} "
                f"(missed by {abs(closest['margin']):.1f})"
            )
        if hit_list:
            biggest = max(hit_list, key=lambda x: x["margin"])
            print(
                f"  Biggest cushion: {biggest['player']} {biggest['market']} "
                f"(hit by {biggest['margin']:.1f})"
            )

    if dnp:
        print(f"\n  {dnp} prediction(s) could not be graded (DNP or not found)")
    print("=" * 80)


# =============================================================================
# BET LOG
# =============================================================================
def append_to_log(results, check_date):
    """Append graded bets to bet_log.csv for downstream calibration.

    Each row: date, player, market, line, pick, projection, actual, result
    'projection' is written only when the PREDICTIONS entry has a 'projection' key.
    PUSH and DNP rows are skipped — only HIT/MISS are logged.
    """
    # v8 Fix: surface projection coverage so silent drops are visible
    graded = [r for r in results if r.get("result") in ("HIT ✓", "MISS ✗", "PUSH", "HIT", "MISS")]
    with_proj = sum(1 for r in graded if r.get("projection") not in ("", None, 0))
    if graded:
        print(
            f"  Projection coverage: {with_proj}/{len(graded)} graded rows have projection values"
        )
        if with_proj == 0 and len(graded) > 0:
            print(
                "  ⚠ NO projections being written — check that check_predictions preserves the field"
            )

    fieldnames = [
        "date",
        "player",
        "market",
        "line",
        "pick",
        "projection",
        "actual",
        "margin",
        "result",
    ]

    file_exists = os.path.isfile(LOG_FILE) and os.path.getsize(LOG_FILE) > 0

    # Migrate old schema (no margin column) before appending new rows
    if file_exists:
        with open(LOG_FILE) as fh:
            header_fields = [c.strip() for c in fh.readline().strip().split(",")]
        if "margin" not in header_fields:
            with open(LOG_FILE) as fh:
                lines = fh.readlines()
            migrated = []
            for i, line in enumerate(lines):
                fields = line.rstrip("\n").split(",")
                if i == 0:
                    fields = fields[:-1] + ["margin", fields[-1]]
                elif len(fields) == 8:
                    fields = fields[:-1] + ["", fields[-1]]
                migrated.append(",".join(fields) + "\n")
            with open(LOG_FILE, "w") as fh:
                fh.writelines(migrated)

    new_rows = []
    for r in results:
        if r.get("actual") is None:
            continue  # DNP / not found
        outcome = r["result"]
        if "HIT" in outcome:
            outcome = "HIT"
        elif "MISS" in outcome:
            outcome = "MISS"
        else:
            continue  # PUSH — not useful for hit-rate learning

        new_rows.append(
            {
                "date": check_date.isoformat(),
                "player": r["player"],
                "market": r["market"],
                "line": r["line"],
                "pick": r["pick"],
                "projection": r.get("projection", ""),  # populated when pred has 'projection'
                "actual": r["actual"],
                "margin": r.get("margin", ""),
                "result": outcome,
            }
        )

    if not new_rows:
        return

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)

    print(f"\n  Logged {len(new_rows)} bet(s) → {os.path.basename(LOG_FILE)}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Off-season guard — skip everything if there are no games
    if _is_offseason(CHECK_DATE):
        print(f"\n  Off-season ({CHECK_DATE.isoformat()}) — NBA regular season resumes in October.")
        print("  No games to grade. Run this script again on a game day.")
        sys.exit(0)

    # ── Player prop predictions ───────────────────────────────────────────────
    if not PREDICTIONS:
        print(f"\nNo player prop predictions found for {CHECK_DATE.isoformat()}.")
        print("  Run nba_combined.py on game day — picks are saved to predictions.db.")
        print("Run nba_combined.py on game day to generate it automatically.")
        print("Or add picks manually to the PREDICTIONS list in this file.")
    else:
        actual = build_actual_stats(CHECK_DATE)
        results = check_predictions(PREDICTIONS, actual)
        print_results(results)
        if results:
            append_to_log(results, CHECK_DATE)
            oddstracker.update_results()  # fill result column in odds_tracker.csv
            print("  Run calibrate.py to update calibration.json with the new data.")

    # ── Game winner / spread predictions ─────────────────────────────────────
    game_preds = _load_game_predictions_db(CHECK_DATE)
    if game_preds:
        game_results = build_game_results(CHECK_DATE)
        graded_games = check_game_predictions(game_preds, game_results)
        print_game_results(graded_games, CHECK_DATE)
        if graded_games:
            append_game_to_log(graded_games, CHECK_DATE)
    else:
        print(f"\nNo game predictions found for {CHECK_DATE.isoformat()} in predictions.db.")
