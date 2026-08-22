"""
predictions_db.py — SQLite persistence for NBA predictions.

Replaces the per-day predictions_YYYY-MM-DD.json and
game_predictions_YYYY-MM-DD.json files with a single predictions.db
database.  Writing upserts on (date, player, market) / (date, home, away)
so re-running the same day overwrites cleanly.
"""

import sqlite3
from datetime import date
from pathlib import Path
from typing import List, Optional

DB_PATH = Path(__file__).parent / "predictions.db"

_CREATE_PROP = """
CREATE TABLE IF NOT EXISTS prop_predictions (
    date        TEXT NOT NULL,
    game        TEXT NOT NULL,
    player      TEXT NOT NULL,
    market      TEXT NOT NULL,
    line        REAL,
    pick        TEXT,
    projection  REAL,
    confidence  REAL,
    edge_pct    REAL,
    flags       TEXT,
    po_count    INTEGER,
    avg_min     REAL,
    book_count  INTEGER,
    usg_pct     REAL,
    game_spread REAL,
    game_total  REAL,
    game_pace   REAL,
    fav_win_pct REAL,
    PRIMARY KEY (date, player, market)
)
"""

_CREATE_GAME = """
CREATE TABLE IF NOT EXISTS game_predictions (
    date        TEXT NOT NULL,
    home_abbr   TEXT NOT NULL,
    away_abbr   TEXT NOT NULL,
    winner_pick TEXT,
    spread      REAL,
    spread_pick TEXT,
    win_pct     REAL,
    total       REAL,
    PRIMARY KEY (date, home_abbr, away_abbr)
)
"""


def _connect(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_PROP)
    conn.execute(_CREATE_GAME)
    conn.commit()
    return conn


def upsert_prop_predictions(rows: List[dict], game_date: date,
                             path: Path = DB_PATH) -> int:
    """Insert or replace prop pick rows for game_date. Returns row count written."""
    if not rows:
        return 0
    date_str = game_date.isoformat()
    with _connect(path) as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO prop_predictions
               (date, game, player, market, line, pick, projection, confidence,
                edge_pct, flags, po_count, avg_min, book_count, usg_pct,
                game_spread, game_total, game_pace, fav_win_pct)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    date_str,
                    r.get("game", ""),
                    r["player"],
                    r["market"],
                    r.get("line"),
                    r.get("pick"),
                    r.get("projection"),
                    r.get("confidence"),
                    r.get("edge_pct"),
                    r.get("flags", ""),
                    r.get("po_count"),
                    r.get("avg_min"),
                    r.get("book_count"),
                    r.get("usg_pct"),
                    r.get("game_spread"),
                    r.get("game_total"),
                    r.get("game_pace"),
                    r.get("fav_win_pct"),
                )
                for r in rows
            ],
        )
    return len(rows)


def upsert_game_predictions(rows: List[dict], game_date: date,
                             path: Path = DB_PATH) -> int:
    """Insert or replace game pick rows for game_date. Returns row count written."""
    if not rows:
        return 0
    date_str = game_date.isoformat()
    with _connect(path) as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO game_predictions
               (date, home_abbr, away_abbr, winner_pick, spread, spread_pick,
                win_pct, total)
               VALUES (?,?,?,?,?,?,?,?)""",
            [
                (
                    date_str,
                    r["home_abbr"],
                    r["away_abbr"],
                    r.get("winner_pick"),
                    r.get("spread"),
                    r.get("spread_pick"),
                    r.get("win_pct"),
                    r.get("total"),
                )
                for r in rows
            ],
        )
    return len(rows)


def load_prop_predictions(target_date: date,
                           path: Path = DB_PATH) -> List[dict]:
    """Return prop predictions for target_date (empty list if none)."""
    if not Path(path).exists():
        return []
    date_str = target_date.isoformat()
    with _connect(path) as conn:
        cur = conn.execute(
            "SELECT player, market, line, pick, projection FROM prop_predictions "
            "WHERE date = ? ORDER BY confidence DESC",
            (date_str,),
        )
        return [
            {
                "player":     row[0],
                "market":     row[1],
                "line":       float(row[2]) if row[2] is not None else 0.0,
                "pick":       row[3],
                "projection": float(row[4]) if row[4] is not None else 0.0,
            }
            for row in cur.fetchall()
        ]


def load_game_predictions(target_date: date,
                           path: Path = DB_PATH) -> List[dict]:
    """Return game predictions for target_date (empty list if none)."""
    if not Path(path).exists():
        return []
    date_str = target_date.isoformat()
    with _connect(path) as conn:
        cur = conn.execute(
            "SELECT home_abbr, away_abbr, winner_pick, spread, spread_pick "
            "FROM game_predictions WHERE date = ?",
            (date_str,),
        )
        return [
            {
                "home_abbr":   row[0],
                "away_abbr":   row[1],
                "winner_pick": row[2],
                "spread":      float(row[3]) if row[3] is not None else None,
                "spread_pick": row[4],
            }
            for row in cur.fetchall()
        ]


def available_dates(path: Path = DB_PATH) -> List[str]:
    """Return sorted list of all dates that have prop predictions."""
    if not Path(path).exists():
        return []
    with _connect(path) as conn:
        cur = conn.execute(
            "SELECT DISTINCT date FROM prop_predictions ORDER BY date"
        )
        return [row[0] for row in cur.fetchall()]
