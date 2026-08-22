"""
kalman_daily_update.py — O(1)-per-game incremental EKF/KF update.

Run after checkresults.py confirms yesterday's games so that the adaptive
filter absorbs each result before today's predictions.

Usage:
    python3 kalman_daily_update.py --date 2026-10-29

For each completed game on that date, this script:
  1. Loads tvp_state.json (current filter state)
  2. Calls predict_step()  (time update — adds Q to P)
  3. Calls update(x, y)    (measurement update — adjusts β using the result)
  4. Saves the new state back to tvp_state.json

The game features (x) are read from predictions.db (the saved pre-game
prediction row already has the feature vector embedded).  The outcome (y) is
read from the NBA API (or supplied via --results-json for offline use).

If a game date has no entries in predictions.db, the script exits cleanly
(nothing to update).
"""

import argparse
import json
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE       = Path(__file__).parent
DB_PATH     = _HERE / "predictions.db"
TVP_STATE   = _HERE / "tvp_state.json"
PARAMS_FILE = _HERE / "learned_params.json"

from kalman_filter import GameEKF, PlayerKF, load_tvp_state, save_tvp_state, PARAMS_FILE


# ── Feature extraction from a saved prediction row ──────────────────────────────

# These are the keys stored in the game_predictions table that map to GAME_FEATURES.
# The feature order must match train_model.GAME_FEATURES exactly.
_GAME_FEATURE_COLS = [
    "net_rtg_diff", "efg_diff", "tov_diff", "orb_diff", "ftr_diff",
    "form_diff", "home_b2b", "away_b2b", "rest_diff", "h2h_margin",
]


def _load_completed_games(target_date: date) -> list:
    """
    Return list of {home_abbr, away_abbr, features, home_won} dicts for games
    on target_date that have both a prediction and a known result in predictions.db.
    """
    import sqlite3
    if not DB_PATH.exists():
        return []

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT * FROM game_predictions WHERE date = ? AND actual_result IS NOT NULL",
            (target_date.isoformat(),),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        con.close()

    games = []
    for row in rows:
        d = dict(row)
        # actual_result stores "HOME" or "AWAY" (set by checkresults.py)
        home_won = 1 if d.get("actual_result") == "HOME" else 0
        # Feature vector — may be NULL for older rows
        feats = [d.get(col) for col in _GAME_FEATURE_COLS]
        if any(v is None for v in feats):
            continue
        games.append({
            "home_abbr": d["home_abbr"],
            "away_abbr": d["away_abbr"],
            "features":  np.array(feats, dtype=float),
            "home_won":  home_won,
        })
    return games


def _load_completed_games_from_json(path: str) -> list:
    """
    Offline alternative: load from a JSON file with structure:
    [{"home_abbr": "LAL", "away_abbr": "GSW", "home_won": 1,
      "net_rtg_diff": 2.3, "efg_diff": 0.01, ... (all 10 features)}]
    """
    with open(path) as fh:
        raw = json.load(fh)
    out = []
    for g in raw:
        feats = [g.get(col, 0.0) for col in _GAME_FEATURE_COLS]
        out.append({
            "home_abbr": g["home_abbr"],
            "away_abbr": g["away_abbr"],
            "features":  np.array(feats, dtype=float),
            "home_won":  int(g["home_won"]),
        })
    return out


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Incremental EKF update from yesterday's results")
    ap.add_argument(
        "--date", default=None,
        help="Game date ISO-8601 (default: yesterday)",
    )
    ap.add_argument(
        "--results-json", default=None, metavar="PATH",
        help="Offline: load game results from this JSON file instead of predictions.db",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be updated without writing tvp_state.json",
    )
    args = ap.parse_args()

    target_date = (
        datetime.strptime(args.date, "%Y-%m-%d").date()
        if args.date
        else date.today() - timedelta(days=1)
    )
    print(f"kalman_daily_update — processing {target_date}")

    # ── Load filter state ──────────────────────────────────────────────────────
    state = load_tvp_state(TVP_STATE)
    if state is None:
        sys.exit(
            "tvp_state.json not found — run kalman_validate.py first to"
            " produce the initial filter state."
        )

    ekf        = GameEKF.from_dict(state["game_model"])
    player_kf  = PlayerKF(stat_states=state.get("player_models", {}))
    rec        = state.get("recommendation", "keep_static")

    # ── Load game results ──────────────────────────────────────────────────────
    if args.results_json:
        games = _load_completed_games_from_json(args.results_json)
    else:
        games = _load_completed_games(target_date)

    if not games:
        print(f"  No completed games with saved features found for {target_date} — nothing to update.")
        return

    # ── Apply EKF updates ──────────────────────────────────────────────────────
    print(f"  Updating EKF with {len(games)} game(s) ...")
    for g in games:
        ekf.predict_step()
        pre_p = ekf.update(g["features"], g["home_won"])
        home  = g["home_abbr"]
        away  = g["away_abbr"]
        y     = g["home_won"]
        print(
            f"    {home} vs {away}: pre-update p(home)={pre_p:.3f}  "
            f"actual={'HOME' if y else 'AWAY'}"
        )

    print(f"  EKF n_updates: {ekf.n_updates}")

    if args.dry_run:
        print("  --dry-run: tvp_state.json not modified.")
        return

    save_tvp_state(
        game_ekf       = ekf,
        player_kf      = player_kf,
        last_updated   = target_date.isoformat(),
        recommendation = rec,
        path           = TVP_STATE,
    )
    print(f"  ✓ tvp_state.json updated (Q={ekf.q_scalar:.1e})")


if __name__ == "__main__":
    main()
