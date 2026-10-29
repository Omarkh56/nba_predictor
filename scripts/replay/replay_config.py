"""Replay configuration (versioned).

Anything that changes WHICH observations are replayed, or HOW metrics are computed, must bump
REPLAY_CONFIG_VERSION so committed summaries from different configurations are never compared silently.
"""
from __future__ import annotations

import os

REPLAY_CONFIG_VERSION = "replay-cfg-1"
CACHE_SCHEMA_VERSION = "nba-cache-1"
DEFAULT_SEASON = "2025-26"

SEASON_TYPE_RS = "Regular Season"
SEASON_TYPE_PO = "Playoffs"
REGIME_LABEL = {SEASON_TYPE_RS: "regular", SEASON_TYPE_PO: "playoffs"}

# Displayed market label -> (odds-api market key, internal stat columns).
# NOTE: the internal box-score column is FG3M (three-point field goals MADE); the displayed/bookmaker
# market is "3PM" (player_threes). They are the same quantity: market "3PM" == stat column "FG3M".
MARKETS = [
    ("PTS", "player_points", ["PTS"]),
    ("REB", "player_rebounds", ["REB"]),
    ("AST", "player_assists", ["AST"]),
    ("3PM", "player_threes", ["FG3M"]),
    ("PTS+REB", "player_points_rebounds", ["PTS", "REB"]),
    ("PTS+AST", "player_points_assists", ["PTS", "AST"]),
    ("REB+AST", "player_rebounds_assists", ["REB", "AST"]),
    ("PRA", "player_points_rebounds_assists", ["PTS", "REB", "AST"]),
]
MARKET_ORDER = [m[0] for m in MARKETS]
INTERNAL_STAT_TO_DISPLAY = {"PTS": "PTS", "REB": "REB", "AST": "AST", "FG3M": "3PM"}

# Pre-game eligibility: at least ELIG_MIN_PRIOR_GAMES prior PLAYED games this season, with NO minutes floor.
# Low-minute players are a reporting stratum, not an exclusion (the live picks included many role players, and a
# minutes floor would hide exactly the role-collapse cases the minute-history rework targets). Uses all prior
# played games, independent of any model filter, so the population is identical across model versions and paired
# comparisons stay valid.
ELIGIBILITY_VERSION = "elig-2"
ELIG_MIN_PRIOR_GAMES = 10
ELIG_MIN_AVG_MINUTES_LAST10 = 0.0  # 0 = no floor; role strata below are reporting-only

# Pre-game role strata from the rolling average of the last 10 played games' minutes (reporting only).
LOW_MINUTES_THRESHOLD = {SEASON_TYPE_RS: 15.0, SEASON_TYPE_PO: 10.0}  # "former thresholds" of the old minute filters
CORE_MINUTES_THRESHOLD = 20.0

# Chronological split by observation count over distinct game dates, per regime.
SPLIT_FRACTIONS = (0.50, 0.25, 0.25)  # train / validation / final untouched test
SPLITS_FILE = os.path.join(os.path.dirname(__file__), "splits.json")

# Replay does not know the matchup pace the live pipeline derived from its team model, and has no historical
# injury reports or book spreads. These are held neutral (documented limitations, see README).
NEUTRAL_PACE_FACTOR = 1.0
NEUTRAL_SPREAD = 0.0

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260918

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "replay-cache")
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
