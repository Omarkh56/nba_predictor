"""
migrate_predictions.py — one-shot migration of existing JSON prediction
files into predictions.db.

Run once:
    python3 migrate_predictions.py

Reads all predictions_YYYY-MM-DD.json and game_predictions_YYYY-MM-DD.json
files in the project directory, writes them into predictions.db, verifies
row counts match, then deletes the JSON files.
"""

import glob
import json
import os
import sys
from datetime import date
from pathlib import Path

import predictions_db as db

_HERE = Path(__file__).parent


def migrate_props(json_files: list) -> int:
    total = 0
    for fpath in sorted(json_files):
        fname = os.path.basename(fpath)
        date_str = fname.replace("predictions_", "").replace(".json", "")
        try:
            d = date.fromisoformat(date_str)
        except ValueError:
            print(f"  skipping unrecognised filename: {fname}")
            continue
        with open(fpath) as fh:
            rows = json.load(fh)
        if not isinstance(rows, list):
            print(f"  skipping {fname}: unexpected format")
            continue
        written = db.upsert_prop_predictions(rows, d)
        print(f"  props  {date_str}: {written} row(s)")
        total += written
    return total


def migrate_games(json_files: list) -> int:
    total = 0
    for fpath in sorted(json_files):
        fname = os.path.basename(fpath)
        date_str = fname.replace("game_predictions_", "").replace(".json", "")
        try:
            d = date.fromisoformat(date_str)
        except ValueError:
            print(f"  skipping unrecognised filename: {fname}")
            continue
        with open(fpath) as fh:
            rows = json.load(fh)
        if not isinstance(rows, list):
            print(f"  skipping {fname}: unexpected format")
            continue
        written = db.upsert_game_predictions(rows, d)
        print(f"  games  {date_str}: {written} row(s)")
        total += written
    return total


def verify_and_delete(prop_files: list, game_files: list,
                       prop_total: int, game_total: int) -> bool:
    """Verify DB row counts match, then delete JSON files."""
    import sqlite3

    conn = sqlite3.connect(str(db.DB_PATH))
    db_props = conn.execute("SELECT COUNT(*) FROM prop_predictions").fetchone()[0]
    db_games = conn.execute("SELECT COUNT(*) FROM game_predictions").fetchone()[0]
    conn.close()

    print(f"\n  Verification:")
    print(f"    prop_predictions  — migrated {prop_total}, in DB {db_props}")
    print(f"    game_predictions  — migrated {game_total}, in DB {db_games}")

    # DB may have more rows if re-run; what matters is that DB >= migrated
    if db_props < prop_total or db_games < game_total:
        print("  ERROR: DB has fewer rows than migrated — aborting delete.")
        return False

    all_json = prop_files + game_files
    for fpath in all_json:
        os.remove(fpath)
        print(f"  deleted {os.path.basename(fpath)}")

    print(f"\n  Removed {len(all_json)} JSON file(s).")
    return True


def main():
    prop_files = sorted(glob.glob(str(_HERE / "predictions_*.json")))
    game_files = sorted(glob.glob(str(_HERE / "game_predictions_*.json")))

    print(f"Found {len(prop_files)} prop file(s), {len(game_files)} game file(s).")
    if not prop_files and not game_files:
        print("Nothing to migrate.")
        return

    print("\nMigrating to predictions.db …")
    prop_total = migrate_props(prop_files)
    game_total = migrate_games(game_files)

    ok = verify_and_delete(prop_files, game_files, prop_total, game_total)
    if ok:
        print("\nMigration complete.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
