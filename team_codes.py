"""team_codes.py — one team-code vocabulary for the whole pipeline.

Every source spells teams differently:

  * NBA Stats API (nba_api, box scores, bet/game logs) uses the canonical
    3-letter codes: NYK, GSW, SAS, NOP, UTA, WAS, ...
  * ESPN (scoreboard, injuries) uses short codes for six teams:
    NY, GS, SA, NO, UTAH, WSH.
  * The Odds API uses full names ("Los Angeles Clippers"); ESPN display names
    say "LA Clippers".

Before this module the pipeline kept three partial maps (the main one mapped
"UTH" instead of ESPN's "UTAH") and the odds tracker cut keys from full names,
so the Lakers and Clippers both became "LOS". Anything that compares, joins or
stores teams should call normalize_team() and keep the canonical NBA code.

Slate dates use US Eastern time, matching the NBA and ESPN calendars: a
10:30 pm ET tip is 02:30 UTC the next day but belongs to the Eastern date.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

SLATE_TZ = ZoneInfo("America/New_York")

TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

NBA_CODES = frozenset(TEAM_NAME_TO_ABBR.values())

# Other spellings of full names seen in ESPN / nba_api payloads.
NAME_ALIASES = {
    "LA Clippers": "LAC",
    "LA Lakers": "LAL",
}

# Short codes from ESPN (plus a few other common sources) -> NBA Stats codes.
ESPN_TO_NBA_ABBR = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "WSH": "WAS",
    "UTAH": "UTA",  # ESPN's code for the Jazz
    "UTH": "UTA",   # the old map's spelling, kept for compatibility
    "PHO": "PHX",
    "BRK": "BKN",
    "CHO": "CHA",
}

_LOWER_NAMES = {name.lower(): code for name, code in {**TEAM_NAME_TO_ABBR, **NAME_ALIASES}.items()}


def normalize_team(value) -> str:
    """Return the NBA Stats code for a team code or name, or "" when unknown."""
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text:
        return ""
    upper = text.upper()
    if upper in NBA_CODES:
        return upper
    if upper in ESPN_TO_NBA_ABBR:
        return ESPN_TO_NBA_ABBR[upper]
    return _LOWER_NAMES.get(text.lower(), "")


def game_key(away, home) -> str:
    """Canonical "AWAY@HOME" key; an unknown team keeps its raw text."""
    away_code = normalize_team(away) or str(away).strip().upper()
    home_code = normalize_team(home) or str(home).strip().upper()
    return f"{away_code}@{home_code}"


def _parse_utc(value) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def slate_date_of(commence_time) -> date | None:
    """US Eastern calendar date of an ISO commence time, or None if unparseable."""
    parsed = _parse_utc(commence_time)
    return parsed.astimezone(SLATE_TZ).date() if parsed else None


def event_slate_date(event) -> date | None:
    return slate_date_of((event or {}).get("commence_time"))


def events_for_date(events, slate_date: date) -> list:
    """Keep only events whose tip-off falls on slate_date (US Eastern)."""
    return [ev for ev in events or [] if event_slate_date(ev) == slate_date]


def today_slate_date(now: datetime | None = None) -> date:
    """Today's date on the NBA (US Eastern) calendar."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(SLATE_TZ).date()
