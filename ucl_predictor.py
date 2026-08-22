#!/usr/bin/env python3
"""
UEFA Champions League Match Prediction Engine
==============================================
Predicts match outcomes, win probabilities, predicted scores, and identifies
value betting opportunities for upcoming UCL fixtures.

10-Factor Model:
  ① xG Differential  ② Squad/Injuries  ③ Home/Venue  ④ Recent Form
  ⑤ Head-to-Head     ⑥ Tactical        ⑦ Stage       ⑧ Rest
  ⑨ Aggregate Context  ⑩ Coefficient & Pedigree

Usage:
    python3 ucl_predictor.py
    python3 ucl_predictor.py --verbose
    python3 ucl_predictor.py --date 2026-04-22
    python3 ucl_predictor.py --next 10 --no-csv

Requirements:
    pip install requests pandas numpy
    (scipy optional but recommended for Poisson accuracy)

API Keys:
    API-Football:  https://www.api-football.com/  (100 req/day free)
    The Odds API:  https://the-odds-api.com/       (500 req/month free)
"""

import sys
import os
import json
import math
import time
import csv
import warnings
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

warnings.filterwarnings("ignore")

# TLS NOTE: verify=False was previously monkey-patched here for macOS LibreSSL
# 2.8.3. Removed — install 'certifi' and keep it updated instead:
#   pip install --upgrade certifi
import urllib3

# Optional scipy for more accurate Poisson PMF
try:
    from scipy.stats import poisson as _scipy_poisson
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# =============================================================================
# CONFIGURATION — Fill in your API keys here
# =============================================================================
API_FOOTBALL_KEY = "2f441e72e8df843b9a9be62e473663a1"   # https://www.api-football.com/
ODDS_API_KEY     = "2fbb8e37aaf66ba7b9e5d92bdbafb0f5"        # https://the-odds-api.com/

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
ODDS_API_BASE     = "https://api.the-odds-api.com/v4"
ESPN_BASE         = "https://site.api.espn.com/apis/site/v2/sports/soccer/uefa.champions"

# Fix 2: Team ID map for ESPN fallback.
# When ESPN is the fixture source, team IDs are None and every downstream API-Football
# fetch silently returns nothing (stats, H2H, form, injuries all need numeric IDs).
# This mapping lets _parse_espn_fixtures populate real IDs for all current UCL clubs.
UCL_TEAM_IDS = {
    "arsenal": 42,              "liverpool": 40,
    "manchester city": 50,      "aston villa": 66,
    "celtic": 247,              "real madrid": 541,
    "barcelona": 529,           "atletico madrid": 530,
    "girona": 546,              "bayern munich": 157,
    "borussia dortmund": 165,   "bayer leverkusen": 168,
    "rb leipzig": 173,          "stuttgart": 172,
    "inter milan": 505,         "ac milan": 489,
    "juventus": 496,            "napoli": 492,
    "atalanta": 499,            "bologna": 502,
    "paris saint-germain": 85,  "lille": 79,
    "monaco": 91,               "brest": 106,
    "benfica": 211,             "porto": 212,
    "sporting cp": 228,         "psv eindhoven": 197,
    "feyenoord": 215,           "club brugge": 569,
    "red bull salzburg": 571,   "sturm graz": 237,
    "young boys": 788,          "shakhtar donetsk": 225,
    "dinamo zagreb": 431,       "slovan bratislava": 656,
    "sparta prague": 553,       "chelsea": 49,
    "manchester united": 33,    "tottenham": 47,
}

# Fix 4: Country/league lookup used when ESPN doesn't supply team country.
# Without this, factor ⑩ always returns 1.00× for both teams.
# Keys are lowercased normalized team names (matching output of _normalize_team_name).
TEAM_COUNTRY = {
    "arsenal": "England",         "liverpool": "England",
    "manchester city": "England", "manchester united": "England",
    "chelsea": "England",         "aston villa": "England",
    "tottenham": "England",       "celtic": "Scotland",
    "real madrid": "Spain",       "barcelona": "Spain",
    "atletico madrid": "Spain",   "girona": "Spain",
    "real sociedad": "Spain",     "villarreal": "Spain",
    "bayern munich": "Germany",   "borussia dortmund": "Germany",
    "bayer leverkusen": "Germany","rb leipzig": "Germany",
    "stuttgart": "Germany",       "eintracht frankfurt": "Germany",
    "inter milan": "Italy",       "ac milan": "Italy",
    "juventus": "Italy",          "napoli": "Italy",
    "atalanta": "Italy",          "bologna": "Italy",
    "lazio": "Italy",             "roma": "Italy",
    "paris saint-germain": "France","lille": "France",
    "monaco": "France",           "brest": "France",
    "olympique marseille": "France","lyon": "France",
    "benfica": "Portugal",        "porto": "Portugal",
    "sporting cp": "Portugal",    "braga": "Portugal",
    "psv eindhoven": "Netherlands","ajax": "Netherlands",
    "feyenoord": "Netherlands",   "club brugge": "Belgium",
    "anderlecht": "Belgium",      "red bull salzburg": "Austria",
    "sturm graz": "Austria",      "young boys": "Switzerland",
    "shakhtar donetsk": "Ukraine","dynamo kyiv": "Ukraine",
    "dinamo zagreb": "Croatia",   "slovan bratislava": "Slovakia",
    "sparta prague": "Czech Republic","galatasaray": "Turkey",
    "fenerbahce": "Turkey",       "celtic": "Scotland",
}

# Fix 6: Alias map for name normalisation — covers all major spelling variants
# across API-Football, ESPN, and The Odds API.
TEAM_ALIASES = {
    "psg": "paris saint-germain",
    "paris saint germain": "paris saint-germain",
    "paris sg": "paris saint-germain",
    "inter": "inter milan",
    "internazionale": "inter milan",
    "fc internazionale": "inter milan",
    "atletico": "atletico madrid",
    "atlético madrid": "atletico madrid",
    "atlético de madrid": "atletico madrid",
    "atl. madrid": "atletico madrid",
    "man city": "manchester city",
    "man utd": "manchester united",
    "man united": "manchester united",
    "rb salzburg": "red bull salzburg",
    "fc salzburg": "red bull salzburg",
    "fc red bull salzburg": "red bull salzburg",
    "spurs": "tottenham",
    "tottenham hotspur": "tottenham",
    "wolves": "wolverhampton",
    "dortmund": "borussia dortmund",
    "bvb": "borussia dortmund",
    "borussia dortmund 09": "borussia dortmund",
    "leverkusen": "bayer leverkusen",
    "bayer 04": "bayer leverkusen",
    "bayer 04 leverkusen": "bayer leverkusen",
    "fc barcelona": "barcelona",
    "barça": "barcelona",
    "barca": "barcelona",
    "ac milan": "ac milan",     # kept as-is; also in pedigree as "milan"
    "milan": "ac milan",
    "leipzig": "rb leipzig",
    "rasenballsport leipzig": "rb leipzig",
    "real": "real madrid",      # only used when standalone — not "Real Sociedad"
    "brugge": "club brugge",
    "sporting": "sporting cp",
    "sporting clube de portugal": "sporting cp",
    "young boys bern": "young boys",
    "bsc young boys": "young boys",
    "shakhtar": "shakhtar donetsk",
    "dynamo zagreb": "dinamo zagreb",
}







































# Competition IDs
UCL_LEAGUE_ID = 2       # UEFA Champions League in API-Football
UCL_SEASON    = 2025    # 2025 = 2025-26 season (fixture scheduling)
STATS_SEASON  = 2024    # Last UCL season accessible on API-Football free tier (2022-2025)

# Poisson model parameters (based on historical UCL averages)
MAX_GOALS          = 7      # Max goals in probability matrix
UCL_AVG_HOME_XG    = 1.40   # Historical UCL avg home expected goals/match
UCL_AVG_AWAY_XG    = 1.10   # Historical UCL avg away expected goals/match
HOME_ADVANTAGE_XG  = 0.25   # Base home advantage in xG

# Value bet detection threshold
VALUE_THRESHOLD = 0.05   # 5% edge required to flag as a value bet

# Rate limiting — API-Football free tier: ~10 requests/minute
API_FOOTBALL_SLEEP = 6.5  # seconds between API-Football calls (~9/min, safely under 10/min)
ODDS_API_SLEEP     = 0.5  # seconds between Odds API calls

# Cache settings (seconds TTL per category)
CACHE_DIR = Path(".ucl_cache")
CACHE_TTL = {
    "fixtures":    3600,   # 1 hour — fixtures update rarely
    "stats":       86400,  # 24 hours — season stats change slowly
    "h2h":         86400,  # 24 hours
    "injuries":    1800,   # 30 minutes — injury news breaks fast
    "lineups":     900,    # 15 minutes — posted close to kickoff
    "predictions": 3600,   # 1 hour
    "odds":        900,    # 15 minutes — odds move frequently
    "form":        3600,   # 1 hour
}

# UEFA country coefficient tiers — top-4 leagues get 1.02× xG multiplier in UCL
UEFA_TOP_LEAGUES = {"Spain", "England", "Germany", "Italy", "ES", "EN", "DE", "IT",
                    "Espana", "Inglaterra", "Alemania", "Italia"}

# Club UCL pedigree: finals appearances in last 10 years → xG bonus in knockout
UCL_PEDIGREE = {
    "real madrid": 0.03, "manchester city": 0.02, "bayern munich": 0.02,
    "chelsea": 0.01, "liverpool": 0.02, "paris saint germain": 0.01,
    "juventus": 0.01, "atletico madrid": 0.01, "borussia dortmund": 0.01,
    "inter milan": 0.02, "internazionale": 0.02, "milan": 0.01,
}

# Position weights for injury impact on xG
INJURY_POS_WEIGHTS = {
    "G":  0.15,  # Goalkeeper — biggest single defensive impact
    "CB": 0.10,  # Centre-back partnership critical in UCL
    "ST": 0.12,  # Striker — most direct attacking impact
    "CM": 0.08,  # Central midfielder — engine of play
    "AM": 0.07,  # Attacking midfielder
    "DM": 0.06,  # Defensive mid
    "FB": 0.05,  # Full-back / wing-back
    "W":  0.05,  # Winger
}

# Injury probability by status label
INJURY_STATUS_PROBS = {
    "out": 1.00, "missing": 1.00, "suspended": 1.00,
    "doubtful": 0.70, "unlikely": 0.85,
    "questionable": 0.40, "day-to-day": 0.40,
    "probable": 0.15, "slight chance": 0.15,
}


# =============================================================================
# CACHE — file-based JSON cache with per-category TTL
# =============================================================================
def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    # Sanitize key to valid filename
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(key))
    return CACHE_DIR / f"{safe}.json"

def cache_get(key: str, ttl_key: str = "fixtures"):
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        age  = time.time() - data.get("_ts", 0)
        if age < CACHE_TTL.get(ttl_key, 3600):
            return data.get("payload")
    except Exception:
        pass
    return None

def cache_set(key: str, payload):
    try:
        _cache_path(key).write_text(json.dumps({"_ts": time.time(), "payload": payload},
                                               default=str))
    except Exception:
        pass


# =============================================================================
# Fix 6: Team name normalisation — used everywhere names are compared.
# Strips legal suffixes, resolves known aliases, lowercases.
# =============================================================================
def _normalize_team_name(name: str) -> str:
    """
    Normalise a team name for consistent comparison across data sources.
    Steps: lowercase → strip common suffixes/prefixes → resolve TEAM_ALIASES.
    E.g. 'FC Barcelona' → 'barcelona', 'PSG' → 'paris saint-germain'
    """
    if not name:
        return ""
    n = name.lower().strip()
    # Remove common legal/country suffixes that vary between sources
    for sfx in ("fc", " cf", " sc", " ac", " afc", " bsc", " vfb", " rb"):
        if n.startswith(sfx + " "):
            n = n[len(sfx):].strip()
        if n.endswith(" " + sfx):
            n = n[:-len(sfx)].strip()
    # Resolve alias map (longest key wins to avoid partial clobbers)
    for alias, canonical in sorted(TEAM_ALIASES.items(), key=lambda x: -len(x[0])):
        if n == alias or n == alias.lower():
            return canonical
    return n


def _team_id_from_name(name: str):
    """
    Fix 2: Look up API-Football team ID from a display name using the
    UCL_TEAM_IDS mapping.  Tries exact normalized match first, then
    substring containment, to handle "Arsenal FC" → 42.
    Returns None if no match found.
    """
    norm = _normalize_team_name(name)
    if norm in UCL_TEAM_IDS:
        return UCL_TEAM_IDS[norm]
    # Substring check: "Bayern Munich Women" → still finds "bayern munich"
    for key, tid in UCL_TEAM_IDS.items():
        if key in norm or norm in key:
            return tid
    return None


# =============================================================================
# API CLIENTS — rate-limited, cached, with graceful error handling
# =============================================================================

def _api_football_get(endpoint: str, params: dict, cache_key: str,
                      ttl_key: str = "fixtures"):
    """Rate-limited, cached request to API-Football v3."""
    cached = cache_get(cache_key, ttl_key)
    if cached is not None:
        return cached

    if API_FOOTBALL_KEY in ("YOUR_API_FOOTBALL_KEY", "", None):
        return None   # Key not configured

    url     = f"{API_FOOTBALL_BASE}/{endpoint.lstrip('/')}"
    headers = {"x-apisports-key": API_FOOTBALL_KEY, "Accept": "application/json"}
    for attempt in range(3):
        try:
            time.sleep(API_FOOTBALL_SLEEP)   # Respect rate limit
            resp = requests.get(url, headers=headers, params=params, timeout=20, )
            if resp.status_code == 204:      # No content — valid empty response
                cache_set(cache_key, None)
                return None
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 15))
                print(f"  [API-Football] 429 rate limit — waiting {retry_after}s (attempt {attempt+1}/3)...")
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            data   = resp.json()
            errors = data.get("errors", {})
            if errors:
                print(f"  [API-Football] {endpoint} errors: {errors}")
                cache_set(cache_key, [])
                return []
            result = data.get("response")    # May be list or dict depending on endpoint
            cache_set(cache_key, result)
            return result
        except requests.exceptions.HTTPError as e:
            print(f"  [API-Football] HTTP error {endpoint}: {e}")
            return None
        except Exception as e:
            print(f"  [API-Football] {endpoint}: {e}")
            return None
    print(f"  [API-Football] {endpoint}: gave up after 3 rate-limit retries")
    return None


def _odds_api_get(endpoint: str, params: dict, cache_key: str):
    """Rate-limited, cached request to The Odds API."""
    cached = cache_get(cache_key, "odds")
    if cached is not None:
        return cached

    if ODDS_API_KEY in ("YOUR_ODDS_API_KEY", "", None):
        return None

    url              = f"{ODDS_API_BASE}/{endpoint.lstrip('/')}"
    params["apiKey"] = ODDS_API_KEY
    try:
        time.sleep(ODDS_API_SLEEP)
        resp = requests.get(url, params=params, timeout=20, )
        resp.raise_for_status()
        data = resp.json()
        cache_set(cache_key, data)
        return data
    except Exception as e:
        print(f"  [Odds API] {endpoint}: {e}")
        return None


def _espn_get(endpoint: str, params: dict = None, cache_key: str = "espn_sb"):
    """ESPN public API — no auth required, used as fallback."""
    cached = cache_get(cache_key, "fixtures")
    if cached is not None:
        return cached
    try:
        url  = f"{ESPN_BASE}/{endpoint.lstrip('/')}"
        resp = requests.get(url, params=params or {}, timeout=15, )
        resp.raise_for_status()
        data = resp.json()
        cache_set(cache_key, data)
        return data
    except Exception as e:
        print(f"  [ESPN] {endpoint}: {e}")
        return None


# =============================================================================
# DATA FETCHERS
# =============================================================================

def verify_api_football_key() -> bool:
    """
    Fix 3: Call /status to validate the API-Football key before any data fetches.
    Prints remaining daily request quota and subscription info.
    Returns True if the key is valid and has quota remaining, False otherwise.
    """
    if API_FOOTBALL_KEY in ("YOUR_API_FOOTBALL_KEY", "", None):
        return False
    try:
        resp = requests.get(
            f"{API_FOOTBALL_BASE}/status",
            headers={"x-apisports-key": API_FOOTBALL_KEY},
            timeout=10,
        )
        data = resp.json().get("response", {})
        sub  = data.get("subscription", {})
        req  = data.get("requests", {})
        plan = sub.get("plan", "unknown")
        used = req.get("current", "?")
        lim  = req.get("limit_day", "?")
        remaining = (lim - used) if isinstance(lim, int) and isinstance(used, int) else "?"
        print(f"  API-Football: plan={plan}, used={used}/{lim} today "
              f"({remaining} remaining)")
        # Free tier has 100 req/day; warn when running low
        if isinstance(remaining, int) and remaining < 15:
            print(f"  ⚠  Only {remaining} API-Football requests left today — "
                  f"use --clear-cache sparingly")
        return remaining != 0
    except Exception as e:
        print(f"  ⚠  API-Football key check failed: {e}")
        print("     Model will fall back to ESPN with reduced accuracy.")
        return False


def fetch_upcoming_fixtures(next_n: int = 20) -> list:
    """Fetch upcoming UCL fixtures; falls back to ESPN if API key missing."""
    data = _api_football_get(
        "fixtures",
        {"league": UCL_LEAGUE_ID, "season": UCL_SEASON, "next": next_n, "status": "NS"},
        cache_key=f"fixtures_upcoming_{UCL_SEASON}_{next_n}",
        ttl_key="fixtures",
    )
    if data:
        return data if isinstance(data, list) else []

    # ESPN fallback — provides basic fixture info without stats
    print("  Trying ESPN fallback for fixtures...")
    espn = _espn_get("scoreboard", cache_key="espn_scoreboard")
    return _parse_espn_fixtures(espn) if espn else []


def _parse_espn_fixtures(espn: dict) -> list:
    """Normalise ESPN scoreboard events into API-Football fixture shape."""
    out = []
    for ev in espn.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue
        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
        home_name = home.get("team", {}).get("displayName", "")
        away_name = away.get("team", {}).get("displayName", "")
        # Fix 2: resolve API-Football IDs from the UCL_TEAM_IDS mapping so that
        # downstream fetches (stats, H2H, form, injuries) work even on ESPN fixtures.
        home_id = _team_id_from_name(home_name)
        away_id = _team_id_from_name(away_name)
        out.append({
            "_source": "espn",
            "fixture": {
                "id":     ev.get("id"),
                "date":   ev.get("date", ""),
                "status": {"short": "NS"},
                "venue":  {"name": comp.get("venue", {}).get("fullName", "")},
            },
            "league": {"round": ev.get("name", "UEFA Champions League"), "id": UCL_LEAGUE_ID},
            "teams": {
                "home": {"id": home_id, "name": home_name},
                "away": {"id": away_id, "name": away_name},
            },
            "goals": {"home": None, "away": None},
        })
    return out


def fetch_team_stats(team_id: int) -> object:
    """Fetch team's UCL season statistics from API-Football.
    Uses STATS_SEASON (2024) — free tier blocks 2025."""
    return _api_football_get(
        "teams/statistics",
        {"league": UCL_LEAGUE_ID, "season": STATS_SEASON, "team": team_id},
        cache_key=f"tstats_{team_id}_{STATS_SEASON}",
        ttl_key="stats",
    )


def fetch_h2h(team1_id: int, team2_id: int) -> object:
    """Fetch historical head-to-head results between two teams.
    Omits 'last' and 'status' — both are paid-tier-only parameters on API-Football free."""
    key = f"h2h_{min(team1_id, team2_id)}_{max(team1_id, team2_id)}"
    return _api_football_get(
        "fixtures/headtohead",
        {"h2h": f"{team1_id}-{team2_id}", "season": STATS_SEASON},
        cache_key=key, ttl_key="h2h",
    )


def fetch_injuries(fixture_id: int) -> object:
    """Fetch injury report for a specific fixture."""
    return _api_football_get(
        "injuries", {"fixture": fixture_id},
        cache_key=f"inj_{fixture_id}", ttl_key="injuries",
    )


def fetch_api_prediction(fixture_id: int) -> object:
    """Fetch API-Football's own prediction (includes form/attack/defense comparison)."""
    return _api_football_get(
        "predictions", {"fixture": fixture_id},
        cache_key=f"pred_{fixture_id}", ttl_key="predictions",
    )


def fetch_recent_form(team_id: int, last: int = 6) -> object:
    """Fetch last N completed matches for a team across all competitions.
    Uses league+season instead of 'last' — free tier blocks the 'last' parameter.
    Returns only the most recent `last` fixtures, sorted descending by date."""
    raw = _api_football_get(
        "fixtures",
        {"league": UCL_LEAGUE_ID, "season": STATS_SEASON, "team": team_id, "status": "FT"},
        cache_key=f"form_{team_id}_last{last}", ttl_key="form",
    )
    if not raw:
        return raw
    # Sort by date descending and slice to `last` most recent games
    try:
        sorted_raw = sorted(
            raw,
            key=lambda f: f.get("fixture", {}).get("date", ""),
            reverse=True,
        )
        return sorted_raw[:last]
    except Exception:
        return raw[:last] if isinstance(raw, list) else raw


def fetch_ucl_odds() -> object:
    """Fetch all UCL match odds from The Odds API (h2h + totals)."""
    return _odds_api_get(
        "sports/soccer_uefa_champions_league/odds",
        {"regions": "us,uk,eu", "markets": "h2h,totals", "oddsFormat": "decimal"},
        cache_key="ucl_odds_all",
    )


# =============================================================================
# DATA PARSERS
# =============================================================================

def parse_team_stats(raw) -> dict:
    """
    Extract xG-proxy and per-game averages from API-Football team statistics.
    The API doesn't provide xG directly — we use goals × 0.85 as proxy
    (actual goals are noisier than xG; regression factor reduces overfit).
    """
    if not raw:
        return {}

    # /teams/statistics returns a single dict, not a list
    s = raw if isinstance(raw, dict) else (raw[0] if raw else {})

    fixtures = s.get("fixtures", {})
    goals    = s.get("goals", {})

    played_h = int(fixtures.get("played", {}).get("home", 0) or 0)
    played_a = int(fixtures.get("played", {}).get("away", 0) or 0)
    played_t = max(int(fixtures.get("played", {}).get("total", 1) or 1), 1)

    gf_h = float(goals.get("for", {}).get("total", {}).get("home", 0) or 0)
    gf_a = float(goals.get("for", {}).get("total", {}).get("away", 0) or 0)
    gf_t = float(goals.get("for", {}).get("total", {}).get("total", 0) or 0)
    ga_h = float(goals.get("against", {}).get("total", {}).get("home", 0) or 0)
    ga_a = float(goals.get("against", {}).get("total", {}).get("away", 0) or 0)
    ga_t = float(goals.get("against", {}).get("total", {}).get("total", 0) or 0)

    gf_pg = gf_t / played_t
    ga_pg = ga_t / played_t

    # Form string from API (e.g. "WWDLW") — fallback form indicator
    form_str = s.get("form", "") or ""

    return {
        "played_total":      played_t,
        "played_home":       played_h,
        "played_away":       played_a,
        "goals_for_pg":      round(gf_pg, 3),
        "goals_against_pg":  round(ga_pg, 3),
        "gf_home_pg":        round(gf_h / max(played_h, 1), 3),
        "ga_home_pg":        round(ga_h / max(played_h, 1), 3),
        "gf_away_pg":        round(gf_a / max(played_a, 1), 3),
        "ga_away_pg":        round(ga_a / max(played_a, 1), 3),
        # xG proxy: goals × 0.85 regression factor
        "xg_for_per90":      round(gf_pg * 0.85, 3),
        "xg_against_per90":  round(ga_pg * 0.85, 3),
        "form_str":          form_str[-6:] if form_str else "",
        "raw": s,
    }


def parse_api_prediction(raw) -> dict:
    """Extract form/attack/defense comparison percentages from API prediction."""
    if not raw:
        return {}
    p = (raw[0] if isinstance(raw, list) and raw else raw) or {}

    pred   = p.get("predictions", {})
    comp   = p.get("comparison", {})
    teams  = p.get("teams", {})

    return {
        "home_attack":  _pct(comp.get("att",  {}).get("home",  "50%")),
        "away_attack":  _pct(comp.get("att",  {}).get("away",  "50%")),
        "home_defense": _pct(comp.get("def",  {}).get("home",  "50%")),
        "away_defense": _pct(comp.get("def",  {}).get("away",  "50%")),
        "home_form":    _pct(comp.get("form", {}).get("home",  "50%")),
        "away_form":    _pct(comp.get("form", {}).get("away",  "50%")),
        "api_winner":   pred.get("winner", {}).get("name", ""),
        "api_home_pct": _pct(pred.get("percent", {}).get("home",  "0%")),
        "api_draw_pct": _pct(pred.get("percent", {}).get("draw",  "0%")),
        "api_away_pct": _pct(pred.get("percent", {}).get("away",  "0%")),
        # Last-5 form from prediction endpoint (aggregated stats)
        "home_last5":   teams.get("home", {}).get("last_5", {}),
        "away_last5":   teams.get("away", {}).get("last_5", {}),
    }


def parse_form_fixtures(form_data, team_id: int) -> dict:
    """
    Compute rolling xG-proxy differential over last N matches.
    UCL matches weighted 1.5×; domestic 1.0× (UCL tactics differ).
    """
    if not form_data or not team_id:
        return {"xg_diff": 0.0, "goal_diff_pg": 0.0, "gf_pg": 0.0, "ga_pg": 0.0, "games": 0}

    total_gf = total_ga = total_w = 0.0
    games = 0

    for fix in form_data:
        if fix.get("fixture", {}).get("status", {}).get("short") not in ("FT", "AET", "PEN"):
            continue
        teams  = fix.get("teams", {})
        gls    = fix.get("goals", {})
        lg_id  = fix.get("league", {}).get("id")

        home_id = teams.get("home", {}).get("id")
        away_id = teams.get("away", {}).get("id")
        if home_id == team_id:
            gf, ga = (gls.get("home") or 0), (gls.get("away") or 0)
        elif away_id == team_id:
            gf, ga = (gls.get("away") or 0), (gls.get("home") or 0)
        else:
            continue

        # UCL matches weighted 1.5× — tactical context matters more
        w = 1.5 if lg_id == UCL_LEAGUE_ID else 1.0
        total_gf += gf * w
        total_ga += ga * w
        total_w  += w
        games    += 1

    if total_w == 0:
        return {"xg_diff": 0.0, "goal_diff_pg": 0.0, "gf_pg": 0.0, "ga_pg": 0.0, "games": 0}

    gf_pg = total_gf / total_w
    ga_pg = total_ga / total_w
    gd_pg = gf_pg - ga_pg

    return {
        "xg_diff":     round(gd_pg * 0.85, 3),   # xG proxy
        "goal_diff_pg": round(gd_pg, 3),
        "gf_pg":        round(gf_pg, 3),
        "ga_pg":        round(ga_pg, 3),
        "games":        games,
    }


def _form_from_last5(last5: dict) -> dict:
    """
    Fix 5: Extract a form dict from the API-Football /predictions last_5 block.
    Used as a fallback when individual match logs aren't available (e.g. ESPN
    fixtures with IDs that haven't triggered a /fixtures fetch yet).
    last_5 structure: {"played": 5, "goals": {"for": {"average": "1.6"},
                                               "against": {"average": "0.8"}}}
    """
    if not last5 or not isinstance(last5, dict):
        return {"xg_diff": 0.0, "goal_diff_pg": 0.0, "gf_pg": 0.0, "ga_pg": 0.0, "games": 0}
    goals = last5.get("goals", {})
    # average goals-per-game over last 5 matches
    gf_avg = float((goals.get("for",     {}).get("average") or 0))
    ga_avg = float((goals.get("against", {}).get("average") or 0))
    played = int(last5.get("played", 5) or 5)
    gd     = gf_avg - ga_avg
    return {
        "xg_diff":      round(gd * 0.85, 3),   # xG proxy via regression
        "goal_diff_pg": round(gd, 3),
        "gf_pg":        round(gf_avg, 3),
        "ga_pg":        round(ga_avg, 3),
        "games":        played,
    }


def _pct(val) -> float:
    """Normalise percentage values to 0.0–1.0 float."""
    if val is None:
        return 0.5
    if isinstance(val, (int, float)):
        return float(val) / 100.0 if float(val) > 1.0 else float(val)
    try:
        return float(str(val).strip().rstrip("%")) / 100.0
    except (ValueError, AttributeError):
        return 0.5


# =============================================================================
# FIXTURE METADATA — stage detection + aggregate score parsing
# =============================================================================

def detect_stage(fixture: dict) -> dict:
    """
    Classify UCL round into group/knockout/final and derive modifiers.
    Modifiers are applied to expected goals totals.
    """
    round_name = (fixture.get("league", {}).get("round") or "").lower()
    s = {
        "is_group":       False, "is_league_phase": False,
        "is_knockout":    False, "is_first_leg":    False,
        "is_second_leg":  False, "is_final":        False,
        "round_name":     round_name,
        "total_modifier": 0.0,   # Applied to both teams' xG
        "stage_modifier": 0.0,   # Applied asymmetrically (home conservative in 1L)
    }

    if any(x in round_name for x in ("group", "league phase", "matchday")):
        s["is_group"] = s["is_league_phase"] = True
        # Group stage plays relatively open — no modifier
    elif "final" in round_name and not any(x in round_name for x in ("semi", "quarter")):
        s.update({"is_final": True, "is_knockout": True,
                  "total_modifier": -0.15})   # Finals are tight/tactical/low-scoring
    elif any(x in round_name for x in ("1st", "first", "leg 1")):
        s.update({"is_knockout": True, "is_first_leg": True,
                  "total_modifier": -0.08,    # Lower expected total in 1st legs
                  "stage_modifier": -0.08})   # Home team tends to be cautious
    elif any(x in round_name for x in ("2nd", "second", "leg 2")):
        s.update({"is_knockout": True, "is_second_leg": True,
                  "total_modifier": +0.10})   # More open — must-score scenarios
    elif any(x in round_name for x in ("round of 16", "quarter", "semi")):
        # Generic knockout reference (single-leg format or unspecified leg)
        s.update({"is_knockout": True, "total_modifier": -0.08})

    return s


def extract_aggregate(fixture: dict) -> tuple:
    """
    Try to extract first-leg aggregate score from fixture data.
    Returns (agg_home, agg_away) as ints, or (None, None) if unavailable.
    API-Football sometimes stores this in fixture notes or score history.
    """
    # Check aggregate fields that some API versions expose
    score = fixture.get("score", {})
    agg   = score.get("penalty") or score.get("aggregate")
    if agg:
        try:
            h = int(agg.get("home") or 0)
            a = int(agg.get("away") or 0)
            return h, a
        except (TypeError, ValueError):
            pass
    return None, None


def _compute_rest_days(form_raw, fixture_date_str: str) -> int:
    """Return days since team's last completed match before this fixture."""
    if not form_raw or not fixture_date_str:
        return 5  # Assume average rest if unknown
    try:
        fix_dt = datetime.fromisoformat(fixture_date_str.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        return 5

    most_recent = None
    for fix in (form_raw or []):
        status = fix.get("fixture", {}).get("status", {}).get("short", "")
        if status not in ("FT", "AET", "PEN"):
            continue
        d_str = fix.get("fixture", {}).get("date", "")
        try:
            match_dt = datetime.fromisoformat(d_str.replace("Z", "+00:00")).date()
            if most_recent is None or match_dt > most_recent:
                most_recent = match_dt
        except (ValueError, TypeError):
            pass
    if most_recent is None:
        return 5
    days = (fix_dt - most_recent).days
    return max(0, days)


# =============================================================================
# FACTOR CALCULATIONS — each returns (home_adj, away_adj, detail_dict)
# =============================================================================

def f1_xg_differential(home_stats: dict, away_stats: dict) -> tuple:
    """
    ① xG Differential — the foundation of the model.
    Uses attack vs defense matchup to set base xG for each team.
    Applies a small secondary adjustment for overall xG quality gap.
    """
    # Use xG proxy from season stats; fall back to UCL averages
    h_xgf = home_stats.get("xg_for_per90",    UCL_AVG_HOME_XG)
    h_xga = home_stats.get("xg_against_per90", UCL_AVG_AWAY_XG)
    a_xgf = away_stats.get("xg_for_per90",    UCL_AVG_AWAY_XG)
    a_xga = away_stats.get("xg_against_per90", UCL_AVG_HOME_XG)

    # Base xG = average of own attack and opponent defence concession rate
    home_base = max(0.20, (h_xgf + a_xga) / 2.0)
    away_base = max(0.20, (a_xgf + h_xga) / 2.0)

    # xG differential gap drives secondary adjustment
    h_diff   = h_xgf - h_xga
    a_diff   = a_xgf - a_xga
    diff_gap = h_diff - a_diff   # positive = home team statistically superior

    adj_h = diff_gap * 0.15      # Scale down — already embedded in base
    adj_a = -diff_gap * 0.15

    detail = {
        "home_xgf": round(h_xgf, 3), "home_xga": round(h_xga, 3),
        "away_xgf": round(a_xgf, 3), "away_xga": round(a_xga, 3),
        "home_base": round(home_base, 3), "away_base": round(away_base, 3),
        "h_diff": round(h_diff, 3), "a_diff": round(a_diff, 3),
        "adj_home": round(adj_h, 3), "adj_away": round(adj_a, 3),
    }
    return home_base, away_base, adj_h, adj_a, detail


def f2_injuries(injuries_data, home_id: int, away_id: int) -> tuple:
    """
    ② Squad Availability — sum of position-weighted injury probabilities.
    Positive penalty = xG reduction for the affected team.
    Position weights: GK 0.15, ST 0.12, CB 0.10, CM 0.08, etc.
    """
    h_pen = a_pen = 0.0
    h_missing = []
    a_missing = []

    if not injuries_data:
        return 0.0, 0.0, {"home_penalty": 0.0, "away_penalty": 0.0,
                          "home_missing": [], "away_missing": []}

    for inj in (injuries_data or []):
        player  = inj.get("player", {})
        team    = inj.get("team", {})
        reason  = (inj.get("reason") or inj.get("type") or "out").lower()
        team_id = team.get("id")
        name    = player.get("name", "Unknown")
        pos     = player.get("pos", player.get("type", ""))

        # Determine miss probability from status text
        miss_prob = 1.0
        for key, prob in INJURY_STATUS_PROBS.items():
            if key in reason:
                miss_prob = prob
                break

        pos_w   = _pos_weight(pos)
        penalty = pos_w * miss_prob

        if team_id == home_id:
            h_pen += penalty
            h_missing.append(f"{name} ({pos}, {miss_prob:.0%})")
        elif team_id == away_id:
            a_pen += penalty
            a_missing.append(f"{name} ({pos}, {miss_prob:.0%})")

    # Cap total penalty — a decimated squad still fields 11 players
    h_pen = min(h_pen, 0.60)
    a_pen = min(a_pen, 0.60)

    detail = {
        "home_penalty":  round(h_pen, 3), "away_penalty": round(a_pen, 3),
        "home_missing":  h_missing[:5],   "away_missing": a_missing[:5],
    }
    return h_pen, a_pen, detail   # Returned as positive penalties (subtracted later)


def _pos_weight(pos_str: str) -> float:
    """Map API position string to injury xG weight."""
    p = (pos_str or "").upper().strip()
    if p in ("G", "GK", "GKP"):       return INJURY_POS_WEIGHTS["G"]
    if p in ("CB", "DC", "RCB","LCB"):return INJURY_POS_WEIGHTS["CB"]
    if p in ("ST","CF","FW","SS"):     return INJURY_POS_WEIGHTS["ST"]
    if p in ("CM","MC","M"):           return INJURY_POS_WEIGHTS["CM"]
    if p in ("AM","CAM","OM"):         return INJURY_POS_WEIGHTS["AM"]
    if p in ("DM","CDM","DM","HB"):    return INJURY_POS_WEIGHTS["DM"]
    if p in ("LB","RB","WB","LWB","RWB","LM","RM"): return INJURY_POS_WEIGHTS["FB"]
    if p in ("LW","RW","WF"):          return INJURY_POS_WEIGHTS["W"]
    if "D" in p:  return INJURY_POS_WEIGHTS["CB"]
    if "M" in p:  return INJURY_POS_WEIGHTS["CM"]
    if "F" in p:  return INJURY_POS_WEIGHTS["ST"]
    return 0.06   # Default for unknown positions


def f3_home_venue(stage: dict, agg_home=None, agg_away=None) -> tuple:
    """
    ③ Home/Away & Venue Factor.
    UCL home advantage ≈ 0.25 xG. Modified for finals (neutral) and second
    legs based on aggregate score context.
    """
    if stage["is_final"]:
        # Neutral ground — no home advantage
        return 0.0, 0.0, {"home_adj": 0.0, "away_adj": 0.0,
                           "reason": "Neutral venue (Final)"}

    if stage["is_second_leg"] and agg_home is not None and agg_away is not None:
        diff = agg_home - agg_away   # positive = home team (tonight) leads aggregate
        if diff == 0:
            # Level aggregate — both teams must push → symmetric boost, still home edge
            h_adj = HOME_ADVANTAGE_XG * 0.70
            a_adj = 0.10
            reason = f"Level on aggregate — both push, slight home edge"
        elif diff > 0:
            # Tonight's home team leads aggregate — can play conservatively
            h_adj = HOME_ADVANTAGE_XG * 0.85 - 0.05
            a_adj = 0.10   # Away must score → attacks more
            reason = f"Home leads agg {agg_home}-{agg_away} → away team must attack"
        else:
            # Tonight's home team trails — must score
            h_adj = HOME_ADVANTAGE_XG + 0.10
            a_adj = -0.05
            reason = f"Home trails agg {agg_home}-{agg_away} → must-score boost"
    elif stage["is_first_leg"]:
        # First leg: home side usually cautious — don't over-expose early
        h_adj = HOME_ADVANTAGE_XG * 0.70   # ~0.175 instead of 0.25
        a_adj = 0.0
        reason = "1st leg — home team plays cautiously"
    else:
        # Standard UCL home advantage (54% home win rate)
        h_adj = HOME_ADVANTAGE_XG
        a_adj = 0.0
        reason = "Standard UCL home advantage"

    detail = {"home_adj": round(h_adj, 3), "away_adj": round(a_adj, 3), "reason": reason}
    return h_adj, a_adj, detail


def f4_form(home_form: dict, away_form: dict,
            home_season: dict, away_season: dict) -> tuple:
    """
    ④ Recent Form — last 6 matches vs season baseline.
    form_adj = (recent_xg_diff − season_xg_diff) × 0.20
    Bayesian shrinkage applied: few games → regression toward zero.
    """
    # Season baseline xG differentials
    h_s_diff = (home_season.get("xg_for_per90", UCL_AVG_HOME_XG) -
                home_season.get("xg_against_per90", UCL_AVG_AWAY_XG))
    a_s_diff = (away_season.get("xg_for_per90", UCL_AVG_AWAY_XG) -
                away_season.get("xg_against_per90", UCL_AVG_HOME_XG))

    # Recent form differentials (default to season avg if no form data)
    h_f_diff = home_form.get("xg_diff", h_s_diff)
    a_f_diff = away_form.get("xg_diff", a_s_diff)

    # How much does recent form deviate from season average?
    h_raw_adj = (h_f_diff - h_s_diff) * 0.20
    a_raw_adj = (a_f_diff - a_s_diff) * 0.20

    # Bayesian shrinkage: n_games / (n_games + 8) — small samples regress to 0
    h_n      = max(home_form.get("games", 0), 0)
    a_n      = max(away_form.get("games", 0), 0)
    h_shrink = h_n / (h_n + 8)
    a_shrink = a_n / (a_n + 8)

    h_adj = max(-0.20, min(h_raw_adj * h_shrink, 0.20))
    a_adj = max(-0.20, min(a_raw_adj * a_shrink, 0.20))

    detail = {
        "home_season_diff": round(h_s_diff, 3), "away_season_diff": round(a_s_diff, 3),
        "home_form_diff":   round(h_f_diff, 3), "away_form_diff":   round(a_f_diff, 3),
        "home_shrinkage":   round(h_shrink, 3), "away_shrinkage":   round(a_shrink, 3),
        "home_adj":         round(h_adj, 3),    "away_adj":         round(a_adj, 3),
        "home_games":       h_n,                "away_games":       a_n,
    }
    return h_adj, a_adj, detail


def f5_h2h(h2h_data, home_team_id: int) -> tuple:
    """
    ⑤ Head-to-Head History — exponential decay + Bayesian shrinkage.
    UCL meetings weighted 2× vs domestic H2H.
    h2h_adj = weighted_margin × (n / (n + 8)) × 0.10
    """
    if not h2h_data:
        return 0.0, 0.0, {"adj": 0.0, "n_games": 0, "note": "No H2H data"}

    total_margin = 0.0
    total_weight = 0.0
    n_games      = 0

    for i, fix in enumerate(h2h_data[:10]):
        score    = fix.get("score", {}).get("fulltime", {})
        teams    = fix.get("teams", {})
        home_id  = teams.get("home", {}).get("id")
        h_goals  = score.get("home")
        a_goals  = score.get("away")
        if h_goals is None or a_goals is None:
            continue

        # Flip goal margin if perspective is reversed
        margin = (int(h_goals) - int(a_goals)) if home_id == home_team_id else \
                 (int(a_goals) - int(h_goals))

        # UCL match = 2× weight (same competition context as tonight)
        lg_id    = fix.get("league", {}).get("id")
        ucl_mult = 2.0 if lg_id == UCL_LEAGUE_ID else 1.0

        # Exponential decay: 0.5^(i/3) — most recent game = 1.0
        w = (0.5 ** (i / 3.0)) * ucl_mult
        total_margin += margin * w
        total_weight += w
        n_games      += 1

    if total_weight == 0 or n_games == 0:
        return 0.0, 0.0, {"adj": 0.0, "n_games": 0}

    h2h_margin = total_margin / total_weight

    # Bayesian shrinkage: small samples regress toward 0
    shrinkage = n_games / (n_games + 8)
    h2h_adj   = max(-0.15, min(h2h_margin * shrinkage * 0.10, 0.15))

    detail = {
        "weighted_margin": round(h2h_margin, 3),
        "n_games":         n_games,
        "shrinkage":       round(shrinkage, 3),
        "adj":             round(h2h_adj, 3),
    }
    return h2h_adj, -h2h_adj, detail


def f6_tactical(home_stats: dict, away_stats: dict, api_comp: dict) -> tuple:
    """
    ⑥ Tactical Matchup — possession, pressing, attack/defense ratings.
    Counter-attacking vs high-possession team: counter-attacker gets +0.08 xG.
    """
    h_adj = a_adj = 0.0
    reasons = []

    # API comparison percentages (0.0–1.0 scale)
    h_att = api_comp.get("home_attack",  0.50)
    a_att = api_comp.get("away_attack",  0.50)
    h_def = api_comp.get("home_defense", 0.50)
    a_def = api_comp.get("away_defense", 0.50)

    # Attack edge over opponent's defence
    h_att_edge = (h_att - a_def) * 0.10
    a_att_edge = (a_att - h_def) * 0.10
    h_adj += h_att_edge
    a_adj += a_att_edge

    # Possession differential (from goals stats proxy: low goals often = high poss)
    # API-Football doesn't always expose possession % in /teams/statistics
    # Use xG data as a proxy: high xG for + high xG against → attacking style
    h_xgf = home_stats.get("xg_for_per90", 1.3)
    a_xgf = away_stats.get("xg_for_per90", 1.3)
    h_xga = home_stats.get("xg_against_per90", 1.0)
    a_xga = away_stats.get("xg_against_per90", 1.0)

    # Teams with significantly lower xGA tend to be defensive/possession-based
    h_is_possession = h_xga < 0.80 and h_xgf > 1.20
    a_is_possession = a_xga < 0.80 and a_xgf > 1.20
    h_is_counter    = h_xga > 1.30 and h_xgf > 1.10
    a_is_counter    = a_xga > 1.30 and a_xgf > 1.10

    # UCL historically favors low-block counter vs high possession teams
    if h_is_possession and a_is_counter:
        a_adj += 0.08
        reasons.append("Away counter-attacks vs home possession (+0.08)")
    elif a_is_possession and h_is_counter:
        h_adj += 0.08
        reasons.append("Home counter-attacks vs away possession (+0.08)")

    h_adj = max(-0.15, min(h_adj, 0.15))
    a_adj = max(-0.15, min(a_adj, 0.15))

    detail = {
        "home_adj": round(h_adj, 3), "away_adj": round(a_adj, 3),
        "h_att_edge": round(h_att_edge, 3), "a_att_edge": round(a_att_edge, 3),
        "reasons": reasons,
    }
    return h_adj, a_adj, detail


def f7_stage(stage: dict, home_xg: float, away_xg: float) -> tuple:
    """
    ⑦ Competition Stage Context — knockout matches are tighter.
    Applies total_modifier (both teams) and asymmetric stage_modifier.
    """
    total_mod = stage.get("total_modifier", 0.0)
    stage_mod = stage.get("stage_modifier", 0.0)

    avg_xg = (home_xg + away_xg) / 2.0

    # Total modifier reduces/increases expected goal total symmetrically
    base_adj = avg_xg * total_mod / 2.0
    # Stage modifier applies asymmetrically (e.g. home cautious in 1st leg)
    h_adj = base_adj + stage_mod / 2.0
    a_adj = base_adj - stage_mod / 2.0

    h_adj = max(-0.25, min(h_adj, 0.25))
    a_adj = max(-0.25, min(a_adj, 0.25))

    detail = {
        "stage":      stage.get("round_name", ""),
        "total_mod":  round(total_mod, 3),
        "stage_mod":  round(stage_mod, 3),
        "home_adj":   round(h_adj, 3),
        "away_adj":   round(a_adj, 3),
    }
    return h_adj, a_adj, detail


def f8_rest(home_days: int, away_days: int) -> tuple:
    """
    ⑧ Rest & Fixture Congestion.
    3+ extra rest days = +0.08 xG advantage; ≤2 days since last match = -0.05.
    """
    h_adj = a_adj = 0.0
    diff  = home_days - away_days

    if diff >= 3:
        h_adj, a_adj = +0.08, -0.04
    elif diff <= -3:
        h_adj, a_adj = -0.04, +0.08
    elif diff > 0:
        h_adj = 0.03 * (diff / 3.0)
    elif diff < 0:
        a_adj = 0.03 * (abs(diff) / 3.0)

    # Acute fatigue: played 3 days ago or less
    if home_days <= 2:
        h_adj -= 0.05
    if away_days <= 2:
        a_adj -= 0.05

    detail = {
        "home_rest": home_days, "away_rest": away_days,
        "diff": diff,
        "home_adj": round(h_adj, 3), "away_adj": round(a_adj, 3),
    }
    return h_adj, a_adj, detail


def f9_aggregate_context(stage: dict, agg_home=None, agg_away=None) -> tuple:
    """
    ⑨ Aggregate Score Context for two-leg ties (no away goals rule since 2021).
    Trailing team pushes higher xG but leaves gaps; leading team sits deeper.
    """
    if not stage["is_second_leg"] or agg_home is None or agg_away is None:
        return 0.0, 0.0, {"home_adj": 0.0, "away_adj": 0.0, "note": "N/A"}

    diff  = agg_home - agg_away   # positive = tonight's home team leads aggregate
    h_adj = a_adj = 0.0
    note  = ""

    if diff == 0:
        h_adj, a_adj = 0.05, 0.05
        note = "Level — both teams push forward"
    elif diff > 0:
        h_adj = -0.05   # Home (leading) can sit back
        a_adj = +0.08   # Away (trailing) must attack
        note  = f"Home leads agg by {diff} — away team attacks"
    else:
        h_adj = +0.08   # Home (trailing) must score
        a_adj = -0.05
        note  = f"Home trails agg by {abs(diff)} — home team attacks"

    detail = {
        "home_adj": round(h_adj, 3), "away_adj": round(a_adj, 3),
        "agg_home": agg_home, "agg_away": agg_away, "note": note,
    }
    return h_adj, a_adj, detail


def f10_coefficient(home_name: str, away_name: str,
                    home_country: str, away_country: str,
                    stage: dict) -> tuple:
    """
    ⑩ UEFA Country Coefficient & Club UCL Pedigree.
    Top-4 leagues: 1.02× multiplier on xG.
    Knockout stage pedigree: up to +0.03 additive bonus.
    """
    # Country coefficient tier check (handles full names and abbreviations)
    h_mult = 1.02 if any(c in home_country for c in UEFA_TOP_LEAGUES) else 1.00
    a_mult = 1.02 if any(c in away_country for c in UEFA_TOP_LEAGUES) else 1.00

    h_ped = a_ped = 0.0
    if stage["is_knockout"]:
        # Fix 6: normalise both sides so "FC Barcelona" → "barcelona" matches pedigree key
        hn_norm = _normalize_team_name(home_name)
        an_norm = _normalize_team_name(away_name)
        for club, bonus in UCL_PEDIGREE.items():
            club_norm = _normalize_team_name(club)
            if club_norm in hn_norm or hn_norm in club_norm:
                h_ped = bonus
            if club_norm in an_norm or an_norm in club_norm:
                a_ped = bonus

    detail = {
        "home_league_mult": h_mult, "away_league_mult": a_mult,
        "home_pedigree":    h_ped,  "away_pedigree":    a_ped,
        "home_country":     home_country, "away_country": away_country,
    }
    return h_mult, a_mult, h_ped, a_ped, detail


# =============================================================================
# POISSON PROBABILITY MODEL
# =============================================================================

def _poisson_pmf(k: int, lam: float) -> float:
    """Poisson PMF using scipy if available, otherwise manual calculation."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    if HAS_SCIPY:
        return float(_scipy_poisson.pmf(k, lam))
    try:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except (OverflowError, ValueError):
        return 0.0


def build_score_matrix(lam_home: float, lam_away: float,
                       max_g: int = MAX_GOALS) -> list:
    """
    Build probability matrix P[h][a] = P(home scores h) × P(away scores a).
    Size: (max_g+1) × (max_g+1).
    """
    matrix = []
    for h in range(max_g + 1):
        ph = _poisson_pmf(h, lam_home)
        row = []
        for a in range(max_g + 1):
            row.append(ph * _poisson_pmf(a, lam_away))
        matrix.append(row)
    return matrix


def compute_outcome_probs(matrix: list) -> dict:
    """Sum score matrix into P(home win), P(draw), P(away win)."""
    home_win = draw = away_win = 0.0
    top_scores = []

    for h, row in enumerate(matrix):
        for a, p in enumerate(row):
            top_scores.append((p, h, a))
            if h > a:    home_win += p
            elif h == a: draw     += p
            else:        away_win += p

    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw     /= total
        away_win /= total

    top_scores.sort(reverse=True)
    return {
        "home_win":   round(home_win, 4),
        "draw":       round(draw,     4),
        "away_win":   round(away_win, 4),
        "top_scores": [(f"{h}-{a}", round(p * 100, 1))
                       for p, h, a in top_scores[:5]],
    }


def compute_ou_prob(matrix: list, threshold: float = 2.5) -> float:
    """P(total goals > threshold) for over/under markets."""
    over = 0.0
    for h, row in enumerate(matrix):
        for a, p in enumerate(row):
            if h + a > threshold:
                over += p
    return round(over, 4)


def compute_btts_prob(matrix: list) -> float:
    """P(both teams score ≥ 1 goal) = P(home ≥ 1 AND away ≥ 1)."""
    btts = sum(p for h, row in enumerate(matrix)
                 for a, p in enumerate(row) if h >= 1 and a >= 1)
    return round(btts, 4)


def compute_et_advancement(lam_home: float, lam_away: float,
                            agg_home: int, agg_away: int) -> dict:
    """
    For knockout 2nd legs: compute P(advancing) for each team after 90 mins.
    ET occurs when aggregate is level; in ET we apply a 53/47 home advantage split.
    Then penalties are approximately 50/50.
    """
    matrix  = build_score_matrix(lam_home, lam_away)
    h_adv = a_adv = et_prob = 0.0

    for h, row in enumerate(matrix):
        for a, p in enumerate(row):
            new_h = agg_home + h
            new_a = agg_away + a
            if new_h > new_a:     h_adv  += p
            elif new_h < new_a:   a_adv  += p
            else:                 et_prob += p   # Aggregate level → extra time

    # In ET (30 min): slight home advantage persists
    et_h_win = et_prob * 0.53
    et_a_win = et_prob * 0.47

    # Penalties: approximately 50/50 for both teams from ET level
    # Simplified: whoever reaches ET from ET level goes to pens, modeled as 50/50
    h_total = h_adv + et_h_win
    a_total = a_adv + et_a_win
    denom   = h_total + a_total

    return {
        "home_advance": round(h_total / denom, 4) if denom > 0 else 0.5,
        "away_advance": round(a_total / denom, 4) if denom > 0 else 0.5,
        "et_prob":      round(et_prob, 4),
    }


# =============================================================================
# VALUE DETECTION — compare model probabilities to market implied probabilities
# =============================================================================

def find_matching_odds_event(ucl_odds, home_name: str, away_name: str) -> object:
    """
    Fix 7: Match a fixture to its Odds API entry.
    Requires BOTH home AND away teams to match (or the reverse fixture order),
    preventing false matches from teams that share common words like 'Real' or 'Sporting'.
    """
    if not ucl_odds or not isinstance(ucl_odds, list):
        return None
    for ev in ucl_odds:
        ev_hn = ev.get("home_team", "")
        ev_an = ev.get("away_team", "")
        # Both teams must match — avoids e.g. "Sporting CP" matching "Sporting Kansas City"
        if _names_match(ev_hn, home_name) and _names_match(ev_an, away_name):
            return ev
        # Also try reversed fixture order (Odds API sometimes lists them differently)
        if _names_match(ev_hn, away_name) and _names_match(ev_an, home_name):
            return ev
    return None


# Words that appear in many club names and must NOT count as unique identifiers.
# Two teams sharing only these words are NOT a match.
_FOOTBALL_STOP_WORDS = {
    "club", "football", "united", "city", "real", "sporting", "athletic",
    "olympic", "olympique", "stade", "dynamo", "dinamo", "union", "sport",
    "cf", "fc", "sc", "ac", "afc", "bsc", "rb", "vfb", "1.",
}

def _names_match(api_name: str, our_name: str) -> bool:
    """
    Fix 7: Strict team name comparison.
    Two names match if either:
    (a) their normalised forms are identical or one contains the other as a substring, OR
    (b) they share 2+ significant words (>3 chars, not in the football stop-word list).
    This prevents 'Real Madrid' ↔ 'Real Sociedad' false positives while still
    handling minor spelling differences like 'Inter' ↔ 'Inter Milan'.
    """
    a = _normalize_team_name(api_name)
    b = _normalize_team_name(our_name)
    if not a or not b:
        return False
    # (a) direct normalised match or containment
    if a == b or a in b or b in a:
        return True
    # (b) count significant shared words
    a_words = {w for w in a.split() if len(w) > 3 and w not in _FOOTBALL_STOP_WORDS}
    b_words = {w for w in b.split() if len(w) > 3 and w not in _FOOTBALL_STOP_WORDS}
    return len(a_words & b_words) >= 2


def extract_market_implied_probs(odds_event: dict) -> dict:
    """Extract vig-adjusted implied probabilities from Odds API bookmaker data."""
    if not odds_event:
        return {}

    h2h_h = h2h_d = h2h_a = []
    over25 = under25 = []
    home_nm = odds_event.get("home_team", "").lower()
    away_nm = odds_event.get("away_team", "").lower()

    for bk in (odds_event.get("bookmakers") or []):
        for mkt in (bk.get("markets") or []):
            key = mkt.get("key", "")
            for oc in (mkt.get("outcomes") or []):
                nm    = (oc.get("name") or "").lower()
                price = oc.get("price")
                if not price or price <= 1.0:
                    continue
                if key == "h2h":
                    if any(w in nm for w in home_nm.split()[:2]):   h2h_h.append(price)
                    elif nm == "draw":                                h2h_d.append(price)
                    elif any(w in nm for w in away_nm.split()[:2]): h2h_a.append(price)
                elif key == "totals":
                    pt = oc.get("point", 0)
                    if abs(pt - 2.5) < 0.01:
                        if nm == "over":   over25.append(price)
                        elif nm == "under": under25.append(price)

    def _avg_ip(prices):
        if not prices:
            return None
        avg = sum(prices) / len(prices)
        return round(1.0 / avg, 4)

    return {
        "home":    _avg_ip(h2h_h),
        "draw":    _avg_ip(h2h_d),
        "away":    _avg_ip(h2h_a),
        "over25":  _avg_ip(over25),
        "under25": _avg_ip(under25),
    }


def detect_value_bets(model_probs: dict, market_probs: dict) -> list:
    """
    Compare model probabilities to implied market probabilities.
    Flag bets where edge > VALUE_THRESHOLD (5%).
    """
    if not market_probs:
        return []

    checks = [
        ("Home Win",        model_probs.get("home_win", 0),          market_probs.get("home")),
        ("Draw",            model_probs.get("draw", 0),               market_probs.get("draw")),
        ("Away Win",        model_probs.get("away_win", 0),           market_probs.get("away")),
        ("Over 2.5 Goals",  model_probs.get("over25", 0),             market_probs.get("over25")),
        ("Under 2.5 Goals", 1 - model_probs.get("over25", 0),         market_probs.get("under25")),
        ("BTTS Yes",        model_probs.get("btts", 0),               None),
    ]

    out = []
    for name, m_p, mkt_ip in checks:
        if mkt_ip is None or mkt_ip <= 0 or m_p <= 0:
            continue
        edge = m_p - mkt_ip
        out.append({
            "market":    name,
            "model_p":   round(m_p * 100, 1),
            "market_ip": round(mkt_ip * 100, 1),
            "edge":      round(edge * 100, 1),
            "value":     edge >= VALUE_THRESHOLD,
        })

    return sorted(out, key=lambda x: x["edge"], reverse=True)


# =============================================================================
# FULL MATCH PREDICTION ORCHESTRATOR
# =============================================================================

def predict_match(fixture: dict, ucl_odds, verbose: bool = False) -> object:
    """
    Run all 10 factors for a single fixture, combine into Poisson model,
    detect value bets, and return the full prediction dict.
    """
    fix_data  = fixture.get("fixture", {})
    teams     = fixture.get("teams", {})
    league    = fixture.get("league", {})

    home_team = teams.get("home", {})
    away_team = teams.get("away", {})
    home_name = home_team.get("name", "Unknown")
    away_name = away_team.get("name", "Unknown")
    home_id   = home_team.get("id")
    away_id   = away_team.get("id")
    fix_id    = fix_data.get("id")
    fix_date  = fix_data.get("date", "")
    venue_nm  = fix_data.get("venue", {}).get("name", "")

    stage         = detect_stage(fixture)
    agg_home, agg_away = extract_aggregate(fixture)

    if verbose:
        print(f"\n  [VERBOSE] Processing {home_name} vs {away_name} (id={fix_id})")
        print(f"           Stage: {stage['round_name']} | "
              f"Knockout={stage['is_knockout']} | 2nd_leg={stage['is_second_leg']}")

    # ── Fetch all data (cached) ──────────────────────────────────────────────
    home_raw_stats = fetch_team_stats(home_id) if home_id else None
    away_raw_stats = fetch_team_stats(away_id) if away_id else None
    h2h_data       = fetch_h2h(home_id, away_id) if home_id and away_id else None
    injuries_data  = fetch_injuries(fix_id) if fix_id else None
    api_pred_raw   = fetch_api_prediction(fix_id) if fix_id else None
    home_form_raw  = fetch_recent_form(home_id) if home_id else None
    away_form_raw  = fetch_recent_form(away_id) if away_id else None

    # ── Parse ───────────────────────────────────────────────────────────────
    home_stats = parse_team_stats(home_raw_stats) if home_raw_stats else {}
    away_stats = parse_team_stats(away_raw_stats) if away_raw_stats else {}
    api_pred   = parse_api_prediction(api_pred_raw) if api_pred_raw else {}
    home_form  = parse_form_fixtures(home_form_raw, home_id) if home_form_raw and home_id else {}
    away_form  = parse_form_fixtures(away_form_raw, away_id) if away_form_raw and away_id else {}

    # Fix 5: form logs may be empty when team IDs weren't available at fetch time,
    # but the /predictions endpoint already bundles last-5 aggregated stats.
    # Fall back to those so f4_form gets a real signal instead of zero.
    if home_form.get("games", 0) == 0 and api_pred.get("home_last5"):
        home_form = _form_from_last5(api_pred["home_last5"])
        if verbose: print("  [VERBOSE] Home form: using last_5 from predictions endpoint")
    if away_form.get("games", 0) == 0 and api_pred.get("away_last5"):
        away_form = _form_from_last5(api_pred["away_last5"])
        if verbose: print("  [VERBOSE] Away form: using last_5 from predictions endpoint")

    # Graceful defaults if API returns no data
    if not home_stats:
        home_stats = {"xg_for_per90": UCL_AVG_HOME_XG, "xg_against_per90": UCL_AVG_AWAY_XG}
        if verbose: print("  [VERBOSE] No home stats — using UCL averages")
    if not away_stats:
        away_stats = {"xg_for_per90": UCL_AVG_AWAY_XG, "xg_against_per90": UCL_AVG_HOME_XG}
        if verbose: print("  [VERBOSE] No away stats — using UCL averages")

    home_rest = _compute_rest_days(home_form_raw, fix_date)
    away_rest = _compute_rest_days(away_form_raw, fix_date)

    # Fix 4: ESPN doesn't supply country, so factor ⑩ coefficient was always 1.00×.
    # Look up country from TEAM_COUNTRY using the normalised team name as key.
    home_country = home_team.get("country", "")
    away_country = away_team.get("country", "")
    if not home_country:
        home_country = TEAM_COUNTRY.get(_normalize_team_name(home_name), "")
    if not away_country:
        away_country = TEAM_COUNTRY.get(_normalize_team_name(away_name), "")

    # ── Apply 10 Factors ────────────────────────────────────────────────────

    # ① xG Differential — base xG and quality-gap adjustment
    home_base, away_base, f1_h, f1_a, det1 = f1_xg_differential(home_stats, away_stats)

    # ② Injuries — xG penalties (positive = loss for that team)
    inj_h_pen, inj_a_pen, det2 = f2_injuries(injuries_data, home_id, away_id)
    f2_h = -inj_h_pen   # Convert to additive adjustment (negative = hurts home)
    f2_a = -inj_a_pen

    # ③ Home/Venue advantage
    f3_h, f3_a, det3 = f3_home_venue(stage, agg_home, agg_away)

    # ④ Recent form vs season baseline
    f4_h, f4_a, det4 = f4_form(home_form, away_form, home_stats, away_stats)

    # ⑤ Head-to-head history
    f5_h, f5_a, det5 = (f5_h2h(h2h_data, home_id) if home_id
                        else (0.0, 0.0, {"n_games": 0, "note": "No team ID"}))

    # ⑥ Tactical matchup
    f6_h, f6_a, det6 = f6_tactical(home_stats, away_stats, api_pred)

    # ⑦ Competition stage context
    f7_h, f7_a, det7 = f7_stage(stage, home_base, away_base)

    # ⑧ Rest and fixture congestion
    f8_h, f8_a, det8 = f8_rest(home_rest, away_rest)

    # ⑨ Aggregate score context (2nd legs only)
    f9_h, f9_a, det9 = f9_aggregate_context(stage, agg_home, agg_away)

    # ⑩ Country coefficient and club pedigree
    coeff_h, coeff_a, ped_h, ped_a, det10 = f10_coefficient(
        home_name, away_name, home_country, away_country, stage)

    # ── Combine factors into final expected goals ────────────────────────────
    # Base xG (from factor ①) plus additive adjustments from factors ②–⑨
    raw_lam_h = (home_base + f1_h + f2_h + f3_h + f4_h +
                 f5_h + f6_h + f7_h + f8_h + f9_h)
    raw_lam_a = (away_base + f1_a + f2_a + f3_a + f4_a +
                 f5_a + f6_a + f7_a + f8_a + f9_a)

    # ⑩ Apply coefficient multiplier and pedigree bonus on top
    lam_home = max(0.20, raw_lam_h * coeff_h + ped_h)
    lam_away = max(0.20, raw_lam_a * coeff_a + ped_a)

    # ── Poisson model ────────────────────────────────────────────────────────
    matrix   = build_score_matrix(lam_home, lam_away)
    probs    = compute_outcome_probs(matrix)
    probs["over25"] = compute_ou_prob(matrix, 2.5)
    probs["over35"] = compute_ou_prob(matrix, 3.5)
    probs["btts"]   = compute_btts_prob(matrix)

    # Most likely scoreline from matrix
    top  = probs["top_scores"]
    best_score = top[0][0] if top else "1-1"
    bs_parts   = best_score.split("-")
    pred_h_gls = int(bs_parts[0]) if bs_parts[0].isdigit() else 1
    pred_a_gls = int(bs_parts[1]) if len(bs_parts) > 1 and bs_parts[1].isdigit() else 1

    # Advancement probability for knockout second legs
    advance_probs = None
    if stage["is_second_leg"] and agg_home is not None and agg_away is not None:
        advance_probs = compute_et_advancement(lam_home, lam_away, agg_home, agg_away)

    # ── Value detection via odds ──────────────────────────────────────────────
    odds_event    = find_matching_odds_event(ucl_odds, home_name, away_name)
    market_probs  = extract_market_implied_probs(odds_event) if odds_event else {}
    value_bets    = detect_value_bets(probs, market_probs)

    # ── Factor breakdown for display ─────────────────────────────────────────
    factors = [
        {"n": "①", "name": "xG Differential",    "h": f1_h, "a": f1_a, "det": det1},
        {"n": "②", "name": "Injuries",            "h": f2_h, "a": f2_a, "det": det2},
        {"n": "③", "name": "Home/Venue",           "h": f3_h, "a": f3_a, "det": det3},
        {"n": "④", "name": "Form (L6)",             "h": f4_h, "a": f4_a, "det": det4},
        {"n": "⑤", "name": "H2H",                  "h": f5_h, "a": f5_a, "det": det5},
        {"n": "⑥", "name": "Tactical",              "h": f6_h, "a": f6_a, "det": det6},
        {"n": "⑦", "name": f"Stage",               "h": f7_h, "a": f7_a, "det": det7},
        {"n": "⑧", "name": "Rest",                 "h": f8_h, "a": f8_a, "det": det8},
        {"n": "⑨", "name": "Aggregate Context",    "h": f9_h, "a": f9_a, "det": det9},
        {"n": "⑩", "name": "Coefficient/Pedigree", "h": 0.0,  "a": 0.0,  "det": det10,
                             "note": (f"×{coeff_h:.2f}/{coeff_a:.2f} mult  "
                                      f"+{ped_h:.2f}/{ped_a:.2f} ped")},
    ]

    return {
        "fixture_id":    fix_id,
        "home_name":     home_name,
        "away_name":     away_name,
        "home_id":       home_id,
        "away_id":       away_id,
        "fixture_date":  fix_date,
        "venue":         venue_nm,
        "round":         league.get("round", ""),
        "stage":         stage,
        "lam_home":      round(lam_home, 3),
        "lam_away":      round(lam_away, 3),
        "home_base":     round(home_base, 3),
        "away_base":     round(away_base, 3),
        "probs":         probs,
        "pred_h":        pred_h_gls,
        "pred_a":        pred_a_gls,
        "home_rest":     home_rest,
        "away_rest":     away_rest,
        "advance_probs": advance_probs,
        "agg_home":      agg_home,
        "agg_away":      agg_away,
        "home_stats":    home_stats,
        "away_stats":    away_stats,
        "home_form":     home_form,
        "away_form":     away_form,
        "factors":       factors,
        "value_bets":    value_bets,
        "market_probs":  market_probs,
        "inj_home":      det2.get("home_missing", []),
        "inj_away":      det2.get("away_missing", []),
        "det10":         det10,
    }


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def _fmt_date(date_str: str) -> tuple:
    """Parse ISO date string to (day_str, time_str) in local timezone."""
    try:
        dt    = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        local = dt.astimezone()
        return local.strftime("%a %b %-d"), local.strftime("%-I:%M %p")
    except Exception:
        return date_str[:10], ""


def _sign(val: float) -> str:
    return f"+{val:.2f}" if val >= 0 else f"{val:.2f}"


def _form_str(form: dict) -> str:
    gd = form.get("goal_diff_pg", form.get("xg_diff", 0.0)) or 0.0
    return f"+{gd:.1f}" if gd >= 0 else f"{gd:.1f}"


def print_prediction(pred: dict, verbose: bool = False):
    """Pretty-print a single match prediction in the specified format."""
    hn  = pred["home_name"]
    an  = pred["away_name"]
    p   = pred["probs"]
    day, tip = _fmt_date(pred.get("fixture_date", ""))
    rnd = pred.get("round", "UCL")
    sep = "━" * 65

    agg_str = ""
    if pred["agg_home"] is not None:
        agg_str = f"  │  1st Leg: {pred['agg_home']}-{pred['agg_away']}"

    print(f"\n{sep}")
    print(f"  {hn} vs {an}  │  {day}  │  {tip}")
    print(f"  {rnd}{agg_str}")
    if pred.get("venue"):
        print(f"  {pred['venue']}")
    print(sep)

    # ── Team stats table ────────────────────────────────────────────────────
    h_wpct = f"{p['home_win']*100:.1f}%"
    a_wpct = f"{p['away_win']*100:.1f}%"
    h_xg   = f"{pred['lam_home']:.2f}"
    a_xg   = f"{pred['lam_away']:.2f}"
    h_off  = f"{pred['home_stats'].get('xg_for_per90',  0):.2f}"
    h_def  = f"{pred['home_stats'].get('xg_against_per90', 0):.2f}"
    a_off  = f"{pred['away_stats'].get('xg_for_per90',  0):.2f}"
    a_def  = f"{pred['away_stats'].get('xg_against_per90', 0):.2f}"
    h_frm  = _form_str(pred.get("home_form", {}))
    a_frm  = _form_str(pred.get("away_form", {}))
    h_rst  = f"{pred['home_rest']}d" if pred["home_rest"] < 30 else "N/A"
    a_rst  = f"{pred['away_rest']}d" if pred["away_rest"] < 30 else "N/A"

    header = f"  {'Team':<22} {'Win%':>6}  {'xG':>5}  {'Off':>5}  {'Def':>5}  {'Form':>5}  {'Rest':>5}"
    sep2   = f"  {'─'*22} {'─'*6}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}"
    print(f"\n{header}")
    print(sep2)
    print(f"  {hn:<22} {h_wpct:>6}  {h_xg:>5}  {h_off:>5}  {h_def:>5}  {h_frm:>5}  {h_rst:>5}")
    print(f"  {an:<22} {a_wpct:>6}  {a_xg:>5}  {a_off:>5}  {a_def:>5}  {a_frm:>5}  {a_rst:>5}")
    print(f"  {'Draw':<22} {p['draw']*100:.1f}%")

    # ── Predicted outcome ──────────────────────────────────────────────────
    total = pred["lam_home"] + pred["lam_away"]
    over  = p.get("over25", 0)
    btts  = p.get("btts", 0)
    top3  = p.get("top_scores", [])[:3]
    top3_str = ",  ".join(f"{s} ({pct:.1f}%)" for s, pct in top3)

    print(f"\n  Predicted Score:  {hn} {pred['pred_h']} - {pred['pred_a']} {an}")
    print(f"  Total:            O/U 2.5  ({over*100:.1f}% over, {total:.2f} xG)  "
          f"│  BTTS {btts*100:.1f}%")
    print(f"  Most Likely:      {top3_str}")

    # Advancement probability (2nd leg knockouts)
    adv = pred.get("advance_probs")
    if adv:
        print(f"\n  Aggregate Advancement:")
        print(f"    {hn:<22}  {adv['home_advance']*100:.1f}% advance")
        print(f"    {an:<22}  {adv['away_advance']*100:.1f}% advance")
        print(f"    Extra Time probability: {adv['et_prob']*100:.1f}%")

    # ── Factor Breakdown ────────────────────────────────────────────────────
    print(f"\n  Factor Breakdown:")
    for f in pred["factors"]:
        h_val = f.get("h", 0.0)
        a_val = f.get("a", 0.0)
        note  = f.get("note", "")
        det   = f.get("det", {})

        if f["n"] == "⑩":
            print(f"    {f['n']} {f['name']:<22}  {note}")
        else:
            net = h_val - a_val   # Net adjustment in home team's favour
            # Build a short context note from the detail dict
            ctx = _factor_context(f["n"], det, hn, an)
            print(f"    {f['n']} {f['name']:<22}  {_sign(net)} xG"
                  + (f"  ({hn[:12]}: {_sign(h_val)} / {an[:12]}: {_sign(a_val)})" if verbose
                     else (f"  {ctx}" if ctx else "")))

        if verbose:
            _print_verbose_detail(f["n"], det, hn, an)

    # Injury summary
    inj_h = pred.get("inj_home", [])
    inj_a = pred.get("inj_away", [])
    if inj_h or inj_a:
        print(f"\n  Injury Report:")
        if inj_h:
            print(f"    {hn[:18]:} ({len(inj_h)} flagged): {', '.join(inj_h[:3])}")
        if inj_a:
            print(f"    {an[:18]:} ({len(inj_a)} flagged): {', '.join(inj_a[:3])}")

    # ── Value Bets ──────────────────────────────────────────────────────────
    print(f"\n  Value Bets:")
    vbets = pred.get("value_bets", [])
    if vbets:
        for vb in vbets:
            tag  = "  ✓ VALUE" if vb["value"] else ""
            esign = "+" if vb["edge"] >= 0 else ""
            print(f"    {vb['market']:<20}  Model: {vb['model_p']:>5.1f}%  "
                  f"Market: {vb['market_ip']:>5.1f}%  Edge: {esign}{vb['edge']:.1f}%{tag}")
    else:
        print("    No odds data available — set ODDS_API_KEY for value detection")


def _factor_context(factor_n: str, det: dict, hn: str, an: str) -> str:
    """Build a concise context note for each factor in the breakdown."""
    if factor_n == "②":   # Injuries
        parts = []
        hm = det.get("home_missing", [])
        am = det.get("away_missing", [])
        if hm: parts.append(f"{hn[:8]} missing {hm[0].split(' (')[0]}")
        if am: parts.append(f"{an[:8]} missing {am[0].split(' (')[0]}")
        return " | ".join(parts) if parts else "No significant injuries"
    if factor_n == "③":
        return det.get("reason", "")
    if factor_n == "④":
        hg = det.get("home_games", 0)
        ag = det.get("away_games", 0)
        return f"n={hg}/{ag} matches"
    if factor_n == "⑤":
        n = det.get("n_games", 0)
        return f"n={n} H2H meetings" if n > 0 else det.get("note", "")
    if factor_n == "⑦":
        stage = det.get("stage", "")
        return stage[:30] if stage else ""
    if factor_n == "⑧":
        hr = det.get("home_rest", "?")
        ar = det.get("away_rest", "?")
        return f"{hr}d vs {ar}d"
    if factor_n == "⑨":
        return det.get("note", "")
    return ""


def _print_verbose_detail(factor_n: str, det: dict, hn: str, an: str):
    """Print verbose calculation breakdown for a single factor."""
    if not det:
        return
    skip = {"home_missing", "away_missing", "reasons"}
    for k, v in det.items():
        if k in skip:
            continue
        if isinstance(v, float):
            print(f"         {k}: {v:.4f}")
        elif isinstance(v, int):
            print(f"         {k}: {v}")
        elif isinstance(v, str) and v:
            print(f"         {k}: {v}")
    if factor_n == "②":
        hm = det.get("home_missing", [])
        am = det.get("away_missing", [])
        if hm: print(f"         {hn[:10]} out: {', '.join(hm)}")
        if am: print(f"         {an[:10]} out: {', '.join(am)}")
    if factor_n == "⑥":
        for r in det.get("reasons", []):
            print(f"         {r}")


# =============================================================================
# CSV EXPORT
# =============================================================================

def export_csv(predictions: list, filename: str = None):
    """Export all predictions to a timestamped CSV file."""
    if not filename:
        filename = f"ucl_predictions_{date.today().strftime('%Y%m%d')}.csv"
    if not predictions:
        return

    rows = []
    for pred in predictions:
        p   = pred["probs"]
        adv = pred.get("advance_probs") or {}
        top = (p.get("top_scores") or [("N/A", 0)])[0]
        top_vbet = next((v for v in pred.get("value_bets", []) if v["value"]), None)

        rows.append({
            "fixture_id":      pred.get("fixture_id", ""),
            "date":            (pred.get("fixture_date") or "")[:10],
            "home_team":       pred["home_name"],
            "away_team":       pred["away_name"],
            "round":           pred.get("round", ""),
            "venue":           pred.get("venue", ""),
            "home_win_pct":    round(p["home_win"] * 100, 1),
            "draw_pct":        round(p["draw"]     * 100, 1),
            "away_win_pct":    round(p["away_win"] * 100, 1),
            "home_xg":         pred["lam_home"],
            "away_xg":         pred["lam_away"],
            "pred_score":      f"{pred['pred_h']}-{pred['pred_a']}",
            "total_xg":        round(pred["lam_home"] + pred["lam_away"], 2),
            "over25_pct":      round(p.get("over25", 0) * 100, 1),
            "over35_pct":      round(p.get("over35", 0) * 100, 1),
            "btts_pct":        round(p.get("btts", 0) * 100, 1),
            "most_likely_score": top[0],
            "ml_score_prob":   top[1],
            "home_advance_pct": round(adv.get("home_advance", 0) * 100, 1) if adv else "",
            "away_advance_pct": round(adv.get("away_advance", 0) * 100, 1) if adv else "",
            "et_prob_pct":     round(adv.get("et_prob", 0) * 100, 1) if adv else "",
            "home_rest_days":  pred["home_rest"],
            "away_rest_days":  pred["away_rest"],
            "top_value_bet":   top_vbet["market"] if top_vbet else "",
            "top_value_edge":  top_vbet["edge"]   if top_vbet else "",
            "injuries_home":   "; ".join(pred.get("inj_home", [])),
            "injuries_away":   "; ".join(pred.get("inj_away", [])),
        })

    try:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\n  Exported {len(rows)} prediction(s) to: {filename}")
    except OSError as e:
        print(f"\n  CSV export failed: {e}")


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_summary(predictions: list):
    """Print a compact summary table of all match predictions."""
    if len(predictions) < 2:
        return
    print(f"\n{'='*72}")
    print(f"  PREDICTIONS SUMMARY — {len(predictions)} matches")
    hdr = (f"  {'Match':<33}  {'Score':>5}  "
           f"{'H Win':>6}  {'Draw':>5}  {'A Win':>6}  {'O2.5':>5}")
    print(hdr)
    print(f"  {'─'*33}  {'─'*5}  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*5}")
    for pred in predictions:
        match_str = f"{pred['home_name'][:15]} v {pred['away_name'][:14]}"
        score_str = f"{pred['pred_h']}-{pred['pred_a']}"
        p = pred["probs"]
        vb_count  = sum(1 for v in pred.get("value_bets", []) if v["value"])
        vb_tag    = f"  [{vb_count}✓]" if vb_count > 0 else ""
        print(f"  {match_str:<33}  {score_str:>5}  "
              f"{p['home_win']*100:>5.1f}%  {p['draw']*100:>4.1f}%  "
              f"{p['away_win']*100:>5.1f}%  {p.get('over25',0)*100:>4.1f}%{vb_tag}")
    print(f"{'='*72}")


# =============================================================================
# MAIN
# =============================================================================

def _parse_fixture_date(fixture: dict) -> date:
    d = fixture.get("fixture", {}).get("date", "")
    try:
        return datetime.fromisoformat(d.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        return date.today()


def main():
    parser = argparse.ArgumentParser(
        description="UCL Match Prediction Engine — 10-Factor Poisson Model")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print full per-factor calculation breakdown")
    parser.add_argument("--date",     type=str,
                        help="Filter to fixtures on this date (YYYY-MM-DD)")
    parser.add_argument("--next",     type=int, default=20,
                        help="Number of upcoming fixtures to fetch (default: 20)")
    parser.add_argument("--no-csv",   action="store_true",
                        help="Skip CSV export")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete cached API responses and start fresh")
    args = parser.parse_args()

    if args.clear_cache and CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print("  Cache cleared.")

    print("\n" + "=" * 65)
    print("  UEFA CHAMPIONS LEAGUE — MATCH PREDICTION ENGINE")
    print("  10-Factor Poisson Model  |  xG · Injuries · Form · H2H")
    print("  Tactical · Stage · Rest · Coefficient · Value Bets")
    if args.verbose:
        print("  [VERBOSE MODE ON]")
    if not HAS_SCIPY:
        print("  [tip] pip install scipy for more precise Poisson probabilities")
    print("=" * 65)

    # Fix 3: validate API-Football key before any data fetches and show quota status.
    # This tells the user immediately whether real stats will flow into the model.
    api_key_ok  = API_FOOTBALL_KEY not in ("YOUR_API_FOOTBALL_KEY", "", None)
    odds_key_ok = ODDS_API_KEY     not in ("YOUR_ODDS_API_KEY",     "", None)
    if api_key_ok:
        print("\n  Checking API-Football subscription...")
        api_key_ok = verify_api_football_key()   # prints quota; returns False if dead key
        if not api_key_ok:
            print("     Falling back to ESPN — team IDs resolved via UCL_TEAM_IDS map")
    else:
        print("\n  ⚠  API_FOOTBALL_KEY not set — using ESPN fallback + UCL_TEAM_IDS lookup")
        print("     Get a free key at https://www.api-football.com/")
        print("     Set API_FOOTBALL_KEY at the top of this file.")
    if not odds_key_ok:
        print("  ⚠  ODDS_API_KEY not set — value detection disabled")
        print("     Get a free key at https://the-odds-api.com/")
    print()

    # ── Fetch upcoming fixtures ──────────────────────────────────────────────
    print("  Fetching upcoming UCL fixtures...")
    fixtures = fetch_upcoming_fixtures(next_n=args.next)
    if not fixtures:
        print("  No upcoming fixtures found.")
        print("  Check your API key or try --clear-cache to refresh.")
        return

    # Filter by date if requested
    if args.date:
        try:
            filter_date = datetime.strptime(args.date, "%Y-%m-%d").date()
            fixtures = [f for f in fixtures if _parse_fixture_date(f) == filter_date]
            if not fixtures:
                print(f"  No fixtures found on {args.date}.")
                return
            print(f"  Filtered to {len(fixtures)} fixture(s) on {args.date}")
        except ValueError:
            print(f"  Invalid date format '{args.date}' — use YYYY-MM-DD")
            return

    print(f"  Found {len(fixtures)} upcoming fixture(s)")

    # ── Pre-fetch UCL odds once (saves API quota vs per-fixture calls) ───────
    ucl_odds = None
    if odds_key_ok:
        print("  Fetching UCL odds...")
        ucl_odds = fetch_ucl_odds()
        if ucl_odds:
            print(f"  {len(ucl_odds) if isinstance(ucl_odds, list) else '?'} events with odds loaded")

    # ── Process each fixture ─────────────────────────────────────────────────
    predictions = []
    n = len(fixtures)
    for i, fixture in enumerate(fixtures):
        hn = fixture.get("teams", {}).get("home", {}).get("name", "?")
        an = fixture.get("teams", {}).get("away", {}).get("name", "?")
        print(f"\n  [{i+1}/{n}] {hn} vs {an}...")

        try:
            pred = predict_match(fixture, ucl_odds, verbose=args.verbose)
            if pred:
                predictions.append(pred)
                print_prediction(pred, verbose=args.verbose)
        except Exception as exc:
            print(f"  Error processing {hn} vs {an}: {exc}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # ── Summary ──────────────────────────────────────────────────────────────
    if predictions:
        print_summary(predictions)
        if not args.no_csv:
            export_csv(predictions)

    print("\n  ⚠  Not financial advice. For entertainment purposes only.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
