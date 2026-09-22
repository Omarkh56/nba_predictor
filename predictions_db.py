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
from typing import List

DB_PATH = Path(__file__).parent / "predictions.db"

PROP_PROVENANCE_FIELDS = (
    "rs_raw_count",
    "rs_usable_count",
    "po_raw_count",
    "po_usable_count",
    "fetch_status",
    "source_last_game_date",
    "data_as_of_utc",
    "data_age_days",
    "active_season",
    "season_mode",
    "model_version",
    "book_spread",
    "model_margin",
    "residual_correction_version",
    "probability_calibration_version",
)

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
    p_model     REAL,
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
    p_market    REAL,
    decimal_odds REAL,
    stake_pct   REAL,
    rs_raw_count INTEGER,
    rs_usable_count INTEGER,
    po_raw_count INTEGER,
    po_usable_count INTEGER,
    fetch_status TEXT,
    source_last_game_date TEXT,
    data_as_of_utc TEXT,
    data_age_days INTEGER,
    active_season TEXT,
    season_mode TEXT,
    model_version TEXT,
    book_spread REAL,
    model_margin REAL,
    residual_correction_version TEXT,
    probability_calibration_version TEXT,
    PRIMARY KEY (date, player, market)
)
"""

# Columns added after the table's original release — CREATE TABLE IF NOT
# EXISTS is a no-op on an already-existing table, so a real DB created before
# a given column existed needs an explicit ALTER TABLE. Add new columns here.
_MIGRATIONS = [
    ("prop_predictions", "p_model", "REAL"),
    ("prop_predictions", "p_market", "REAL"),
    ("prop_predictions", "decimal_odds", "REAL"),
    ("prop_predictions", "stake_pct", "REAL"),
    ("prop_predictions", "rs_raw_count", "INTEGER"),
    ("prop_predictions", "rs_usable_count", "INTEGER"),
    ("prop_predictions", "po_raw_count", "INTEGER"),
    ("prop_predictions", "po_usable_count", "INTEGER"),
    ("prop_predictions", "fetch_status", "TEXT"),
    ("prop_predictions", "source_last_game_date", "TEXT"),
    ("prop_predictions", "data_as_of_utc", "TEXT"),
    ("prop_predictions", "data_age_days", "INTEGER"),
    ("prop_predictions", "active_season", "TEXT"),
    ("prop_predictions", "season_mode", "TEXT"),
    ("prop_predictions", "model_version", "TEXT"),
    ("prop_predictions", "book_spread", "REAL"),
    ("prop_predictions", "model_margin", "REAL"),
    ("prop_predictions", "residual_correction_version", "TEXT"),
    ("prop_predictions", "probability_calibration_version", "TEXT"),
    ("game_predictions", "home_prob", "REAL"),
    ("game_predictions", "p_market", "REAL"),
    ("game_predictions", "decimal_odds", "REAL"),
    ("game_predictions", "market_edge", "REAL"),
    ("game_predictions", "ev", "REAL"),
    ("game_predictions", "odds_event_id", "TEXT"),
    ("game_predictions", "odds_snapshot_at", "TEXT"),
    ("game_predictions", "odds_books_count", "INTEGER"),
]

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
    for table, col, coltype in _MIGRATIONS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
    conn.commit()
    return conn


def upsert_prop_predictions(rows: List[dict], game_date: date, path: Path = DB_PATH) -> int:
    """Insert or replace prop pick rows for game_date. Returns row count written."""
    if not rows:
        return 0
    date_str = game_date.isoformat()
    with _connect(path) as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO prop_predictions
               (date, game, player, market, line, pick, projection, confidence, p_model,
                edge_pct, flags, po_count, avg_min, book_count, usg_pct,
                game_spread, game_total, game_pace, fav_win_pct, p_market,
                decimal_odds, stake_pct, rs_raw_count, rs_usable_count,
                po_raw_count, po_usable_count, fetch_status,
                source_last_game_date, data_as_of_utc, data_age_days,
                active_season, season_mode, model_version, book_spread,
                model_margin, residual_correction_version, probability_calibration_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                    r.get("p_model"),
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
                    r.get("p_market"),
                    r.get("decimal_odds"),
                    r.get("stake_pct"),
                    r.get("rs_raw_count"),
                    r.get("rs_usable_count"),
                    r.get("po_raw_count"),
                    r.get("po_usable_count"),
                    r.get("fetch_status"),
                    r.get("source_last_game_date"),
                    r.get("data_as_of_utc"),
                    r.get("data_age_days"),
                    r.get("active_season"),
                    r.get("season_mode"),
                    r.get("model_version"),
                    r.get("book_spread"),
                    r.get("model_margin"),
                    r.get("residual_correction_version"),
                    r.get("probability_calibration_version"),
                )
                for r in rows
            ],
        )
    return len(rows)


def upsert_game_predictions(rows: List[dict], game_date: date, path: Path = DB_PATH) -> int:
    """Insert or replace game pick rows for game_date. Returns row count written."""
    if not rows:
        return 0
    date_str = game_date.isoformat()
    with _connect(path) as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO game_predictions
               (date, home_abbr, away_abbr, winner_pick, spread, spread_pick,
                win_pct, total, home_prob, p_market, decimal_odds, market_edge,
                ev, odds_event_id, odds_snapshot_at, odds_books_count)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                    r.get("home_prob"),
                    r.get("p_market"),
                    r.get("decimal_odds"),
                    r.get("market_edge"),
                    r.get("ev"),
                    r.get("odds_event_id"),
                    r.get("odds_snapshot_at"),
                    r.get("odds_books_count"),
                )
                for r in rows
            ],
        )
    return len(rows)


def load_prop_predictions(target_date: date, path: Path = DB_PATH) -> List[dict]:
    """Return prop predictions for target_date (empty list if none)."""
    if not Path(path).exists():
        return []
    date_str = target_date.isoformat()
    with _connect(path) as conn:
        cur = conn.execute(
            "SELECT player, market, line, pick, projection, flags, p_market, confidence, "
            "decimal_odds, stake_pct, rs_raw_count, rs_usable_count, po_raw_count, "
            "po_usable_count, fetch_status, source_last_game_date, data_as_of_utc, "
            "data_age_days, active_season, season_mode, model_version, book_spread, model_margin, p_model, "
            "residual_correction_version, probability_calibration_version "
            "FROM prop_predictions WHERE date = ? ORDER BY confidence DESC",
            (date_str,),
        )
        return [
            {
                "player": row[0],
                "market": row[1],
                "line": float(row[2]) if row[2] is not None else 0.0,
                "pick": row[3],
                "projection": float(row[4]) if row[4] is not None else 0.0,
                "flags": row[5] or "",
                "p_market": float(row[6]) if row[6] is not None else None,
                "confidence": float(row[7]) if row[7] is not None else None,
                "decimal_odds": float(row[8]) if row[8] is not None else None,
                "stake_pct": float(row[9]) if row[9] is not None else None,
                "rs_raw_count": int(row[10]) if row[10] is not None else None,
                "rs_usable_count": int(row[11]) if row[11] is not None else None,
                "po_raw_count": int(row[12]) if row[12] is not None else None,
                "po_usable_count": int(row[13]) if row[13] is not None else None,
                "fetch_status": row[14],
                "source_last_game_date": row[15],
                "data_as_of_utc": row[16],
                "data_age_days": int(row[17]) if row[17] is not None else None,
                "active_season": row[18],
                "season_mode": row[19],
                "model_version": row[20],
                "book_spread": float(row[21]) if row[21] is not None else None,
                "model_margin": float(row[22]) if row[22] is not None else None,
                "p_model": float(row[23]) if row[23] is not None else None,
                "residual_correction_version": row[24],
                "probability_calibration_version": row[25],
            }
            for row in cur.fetchall()
        ]


def load_all_prop_flags(path: Path = DB_PATH) -> dict:
    """Return {(date, player, market, pick): flags} for every stored prop
    prediction. Used to backfill the Flags column into bet_log.csv for
    already-graded rows (checkresults.py only started writing it forward
    from when this was added)."""
    if not Path(path).exists():
        return {}
    with _connect(path) as conn:
        cur = conn.execute("SELECT date, player, market, pick, flags FROM prop_predictions")
        return {
            (row[0], row[1].strip(), row[2].strip(), (row[3] or "").strip().upper()): row[4] or ""
            for row in cur.fetchall()
        }


def load_all_prop_model_confidence(path: Path = DB_PATH) -> dict:
    """Return stored selected-side p_model, falling back to historical confidence/100."""
    if not Path(path).exists():
        return {}
    with _connect(path) as conn:
        cur = conn.execute("SELECT date, player, market, pick, confidence, p_model FROM prop_predictions")
        return {
            (row[0], row[1].strip(), row[2].strip(), (row[3] or "").strip().upper()):
                (float(row[5]) if row[5] is not None else
                 float(row[4]) / 100.0 if row[4] is not None else None)
            for row in cur.fetchall()
        }


def load_all_prop_stakes(path: Path = DB_PATH) -> dict:
    """Return {(date, player, market, pick): (decimal_odds, stake_pct)} for
    every stored prop prediction. Used to carry the actual stake
    recommendation through into bet_log.csv at grading time — will backfill
    nothing for rows predicted before this was added (predictions.db never
    stored a stake before now), same asymmetry as p_market's coverage."""
    if not Path(path).exists():
        return {}
    with _connect(path) as conn:
        cur = conn.execute(
            "SELECT date, player, market, pick, decimal_odds, stake_pct FROM prop_predictions"
        )
        return {
            (row[0], row[1].strip(), row[2].strip(), (row[3] or "").strip().upper()):
                (
                    float(row[4]) if row[4] is not None else None,
                    float(row[5]) if row[5] is not None else None,
                )
            for row in cur.fetchall()
        }


def load_game_predictions(target_date: date, path: Path = DB_PATH) -> List[dict]:
    """Return game predictions for target_date (empty list if none)."""
    if not Path(path).exists():
        return []
    date_str = target_date.isoformat()
    with _connect(path) as conn:
        cur = conn.execute(
            "SELECT home_abbr, away_abbr, winner_pick, spread, spread_pick, "
            "home_prob, p_market, decimal_odds, market_edge, ev, odds_event_id, "
            "odds_snapshot_at, odds_books_count "
            "FROM game_predictions WHERE date = ?",
            (date_str,),
        )
        return [
            {
                "home_abbr": row[0],
                "away_abbr": row[1],
                "winner_pick": row[2],
                "spread": float(row[3]) if row[3] is not None else None,
                "spread_pick": row[4],
                "home_prob": float(row[5]) if row[5] is not None else None,
                "p_market": float(row[6]) if row[6] is not None else None,
                "decimal_odds": float(row[7]) if row[7] is not None else None,
                "market_edge": float(row[8]) if row[8] is not None else None,
                "ev": float(row[9]) if row[9] is not None else None,
                "odds_event_id": row[10],
                "odds_snapshot_at": row[11],
                "odds_books_count": row[12],
            }
            for row in cur.fetchall()
        ]


def available_dates(path: Path = DB_PATH) -> List[str]:
    """Return sorted list of all dates that have prop predictions."""
    if not Path(path).exists():
        return []
    with _connect(path) as conn:
        cur = conn.execute("SELECT DISTINCT date FROM prop_predictions ORDER BY date")
        return [row[0] for row in cur.fetchall()]
