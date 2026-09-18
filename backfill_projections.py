"""
backfill_projections.py — Phase 1, Step 2: recover missing bet_log.csv
projections.

bet_log.csv has only 82 rows (2026-05-10..12) with a projection value —
earlier rows lost it before the logging fix landed. predictions_YYYY-MM-DD.json
files (the originally-assumed source) don't exist anywhere in this project;
that export was replaced by predictions.db a while back (see nba_combined.py's
history). predictions.db DOES have projections for the same window (923 rows,
2026-04-18..2026-05-09), so this backfills from there instead, matched on
(date, player, market) — predictions.db's actual primary key.

Writes to bet_log_backfilled.csv (a NEW file) and reports coverage. Does NOT
touch the original bet_log.csv — nothing downstream reads the backfilled file
until you decide to replace the original.

Usage:
    python3 backfill_projections.py
"""

import sqlite3
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent
LOG_FILE = _HERE / "bet_log.csv"
DB_FILE = _HERE / "predictions.db"
OUT_FILE = _HERE / "bet_log_backfilled.csv"


def load_projection_lookup() -> dict:
    """Return {(date, player, market): (projection, line)} from predictions.db."""
    if not DB_FILE.exists():
        return {}
    conn = sqlite3.connect(str(DB_FILE))
    cur = conn.execute(
        "SELECT date, player, market, projection, line FROM prop_predictions "
        "WHERE projection IS NOT NULL AND projection != 0"
    )
    lookup = {
        (row[0], row[1].strip(), row[2].strip()): (float(row[3]), row[4])
        for row in cur.fetchall()
    }
    conn.close()
    return lookup


def main() -> int:
    if not LOG_FILE.exists():
        print(f"bet_log.csv not found at {LOG_FILE}")
        return 1

    df = pd.read_csv(LOG_FILE)
    df["projection"] = pd.to_numeric(df["projection"], errors="coerce")
    before_have_proj = int((df["projection"].notna() & (df["projection"] != 0)).sum())

    lookup = load_projection_lookup()
    print(f"predictions.db: {len(lookup)} (date, player, market) rows with a usable projection")

    matched = 0
    line_mismatches = 0
    filled_projection = []
    for _, row in df.iterrows():
        has_proj = pd.notna(row["projection"]) and row["projection"] != 0
        if has_proj:
            filled_projection.append(row["projection"])
            continue
        key = (str(row["date"]).strip(), str(row["player"]).strip(), str(row["market"]).strip())
        hit = lookup.get(key)
        if hit is None:
            filled_projection.append(row["projection"])  # stays NaN
            continue
        proj, db_line = hit
        matched += 1
        try:
            if db_line is not None and abs(float(db_line) - float(row["line"])) > 1e-6:
                line_mismatches += 1
        except (TypeError, ValueError):
            pass
        filled_projection.append(proj)

    df["projection"] = filled_projection
    after_have_proj = int((df["projection"].notna() & (df["projection"] != 0)).sum())

    df.to_csv(OUT_FILE, index=False)

    print("=" * 72)
    print("  PROJECTION BACKFILL — bet_log.csv -> bet_log_backfilled.csv")
    print("=" * 72)
    print(f"  Total graded rows:                {len(df)}")
    print(f"  Had a projection before backfill:  {before_have_proj}")
    print(f"  Backfilled from predictions.db:    {matched}")
    print(f"  Have a projection after backfill:  {after_have_proj} "
          f"({after_have_proj / len(df):.1%} coverage)")
    print(f"  Still missing:                     {len(df) - after_have_proj}")
    if matched:
        print(f"  Line mismatches among backfilled rows: {line_mismatches}/{matched} "
              f"(predictions.db's line differs from bet_log's logged line — "
              f"the projection is still for that player/market, just check "
              f"before trusting projection-vs-line deltas on these rows)")
    print(f"\n  Wrote {OUT_FILE} — original bet_log.csv untouched.")
    print("  Review coverage above before replacing the original.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
