"""
config.py — Shared season configuration (v9).

Season strings were hardcoded and drifting out of sync across the pipeline
(playerlinepredictor.py/nba_combined.py/newnbapredictor.py said "2026-27",
dvp.py said "2025-26", train_model.py's DEFAULT_SEASONS ended at "2024-25").
Every module that needs "the current season" or "the last N seasons to
train on" should import from here instead of hardcoding a string.

NBA_SEASON env var overrides current_season() for backtesting past seasons
without editing code, e.g.:
    NBA_SEASON=2023-24 python3 nba_combined.py
"""

import os
from datetime import date

# NBA seasons start in October and are named by their starting year, e.g.
# the 2026-27 season runs October 2026 -> June 2027.
SEASON_START_MONTH = 10


def current_season(d: date = None) -> str:
    """Return the season string for date `d` (default: today), e.g. "2026-27".

    October or later -> that calendar year starts the season.
    Before October -> the PREVIOUS calendar year started the season we're
    still in (a game in February 2027 is still the "2026-27" season).

    NBA_SEASON env var overrides this entirely when set (backtesting past
    seasons without editing code).
    """
    override = os.environ.get("NBA_SEASON")
    if override:
        return override
    d = d or date.today()
    start_year = d.year if d.month >= SEASON_START_MONTH else d.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _season_years(season: str) -> int:
    """Parse a season string's starting year, e.g. "2023-24" -> 2023."""
    return int(season.split("-")[0])


def _season_from_start_year(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def training_seasons(n: int = 5, d: date = None) -> list:
    """Return the last `n` COMPLETED seasons as a list, oldest first, e.g.
    training_seasons(5) called in the 2026-27 season -> ["2020-21", ...,
    "2024-25"] — the in-progress season itself is never included, since it
    has no final-season data to train on yet."""
    current = current_season(d)
    current_start = _season_years(current)
    last_completed_start = current_start - 1
    starts = range(last_completed_start - n + 1, last_completed_start + 1)
    return [_season_from_start_year(y) for y in starts]


def validation_season(d: date = None) -> str:
    """The most recent COMPLETED season — used as the walk-forward
    validation/test season, never the in-progress one."""
    return training_seasons(n=1, d=d)[0]
