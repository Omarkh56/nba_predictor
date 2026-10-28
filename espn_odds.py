"""Free DraftKings odds from ESPN's public (unofficial, keyless) API.

The Odds API charges about 8 credits per game for this project's 8 prop markets, and the
account ran out on Oct 25 2026. ESPN's core API carries DraftKings' lines for the same 8
markets in one request per game, and its scoreboard carries each game's spread, total and
moneyline. This module reshapes them like The Odds API's responses, so the existing
parsers read them unchanged; source="espn_draftkings" marks where they came from.

ESPN lists each prop's two prices without saying which is the over. The first one is: 157
of 159 pairs matched The Odds API's labelled DraftKings prices for Oct 25 2026 that way
(the other 90 had moved between the two snapshots). A pair whose implied probabilities
don't sum like an over/under with a normal margin (1.00-1.15) is dropped.

NBA_ESPN_ODDS=0 turns all of this off (the suite does, so tests never reach ESPN).
"""

from __future__ import annotations

import os
import time
from datetime import date

import requests

import raw_archive
import team_codes

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team}/roster"
PROPS_URL = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{event}"
             "/competitions/{event}/odds/{provider}/propBets")
ATHLETE_URL = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/athletes/{athlete}"
DRAFTKINGS = "100"  # ESPN's provider id
SOURCE = "espn_draftkings"
BOOK = {"key": "draftkings", "title": "DraftKings"}
PROP_TYPES = {  # ESPN prop type -> The Odds API market key (playerlinepredictor.MARKETS)
    "Total Points": "player_points",
    "Total Rebounds": "player_rebounds",
    "Total Assists": "player_assists",
    "Total 3-Point Field Goals": "player_threes",
    "Total Points and Rebounds": "player_points_rebounds",
    "Total Points and Assists": "player_points_assists",
    "Total Assists and Rebounds": "player_rebounds_assists",
    "Total Points, Rebounds, and Assists": "player_points_rebounds_assists",
}
PAIR_MARGIN = (1.0, 1.15)  # over + under implied probability, margin included
# A run asks for the same game twice: the odds snapshot (entry prices), then the props
# model (lines), up to ~40 minutes apart. Both must see the same quotes, as with
# odds_api.CACHE_SECONDS.
CACHE_SECONDS = 3600
_cache: dict = {}


def enabled() -> bool:
    return os.environ.get("NBA_ESPN_ODDS", "1").strip().lower() not in ("0", "false", "off", "no")


def _get(url: str, params: dict, endpoint: str):
    """One archived GET, cached for CACHE_SECONDS; failures raise RequestException."""
    key = (url, tuple(sorted(params.items())))
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
        return hit[1]
    archived = {"url": url, **params}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        raw_archive.save("espn", endpoint, archived, None, status="fetch_failed",
                         http_status=raw_archive.http_status_of(exc), error=type(exc).__name__)
        raise requests.exceptions.RequestException(
            f"ESPN {endpoint} unavailable ({type(exc).__name__})") from None
    raw_archive.save("espn", endpoint, archived, data, http_status=resp.status_code)
    _cache[key] = (time.monotonic(), data)
    return data


def _american(value):
    """An American price from ESPN's text ("-110", "+150", "EVEN"), or None."""
    text = str(value or "").strip().upper()
    if text in ("EVEN", "EV"):
        return 100.0
    try:
        price = float(text)
    except ValueError:
        return None
    return price if abs(price) >= 100 else None


def _implied(price: float) -> float:
    return -price / (-price + 100.0) if price < 0 else 100.0 / (price + 100.0)


def _game(away: str, home: str, day: date):
    """ESPN's scoreboard entry for away@home on the slate date, or None."""
    want = (team_codes.normalize_team(away), team_codes.normalize_team(home))
    data = _get(SCOREBOARD_URL, {"dates": day.strftime("%Y%m%d")}, "scoreboard")
    for event in data.get("events", []):
        competition = (event.get("competitions") or [{}])[0]
        sides = {t.get("homeAway"): t.get("team") or {} for t in competition.get("competitors", [])}
        if (team_codes.normalize_team(sides.get("away", {}).get("abbreviation", "")),
                team_codes.normalize_team(sides.get("home", {}).get("abbreviation", ""))) == want:
            return {"id": str(event["id"]), "competition": competition,
                    "team_ids": [sides["home"].get("id"), sides["away"].get("id")]}
    return None


def _event_teams(event: dict) -> tuple:
    """(away, home, slate date) of an Odds API-style event: team names or codes, ISO tip."""
    return event.get("away_team", ""), event.get("home_team", ""), team_codes.slate_date_of(
        event.get("commence_time"))


def game_lines(away: str, home: str, day: date):
    """DraftKings' {home_spread (negative = home favoured), total, home_moneyline,
    away_moneyline} from ESPN's scoreboard, or None when ESPN has no line for the game."""
    game = _game(away, home, day)
    odds = next((o for o in (game or {}).get("competition", {}).get("odds") or []
                 if str((o.get("provider") or {}).get("id")) == DRAFTKINGS), None)
    if not odds:
        return None
    spread = odds.get("spread")
    home_spread = None
    if isinstance(spread, (int, float)):
        # Sign from who is favoured, not the field's convention: never invert a favourite.
        favourite = (odds.get("homeTeamOdds") or {}).get("favorite")
        home_spread = -abs(float(spread)) if favourite else abs(float(spread))
    moneyline = odds.get("moneyline") or {}
    return {"home_spread": home_spread,
            "total": float(odds["overUnder"]) if isinstance(odds.get("overUnder"), (int, float)) else None,
            "home_moneyline": _american(((moneyline.get("home") or {}).get("close") or {}).get("odds")),
            "away_moneyline": _american(((moneyline.get("away") or {}).get("close") or {}).get("odds"))}


def game_lines_for_event(event: dict):
    away, home, day = _event_teams(event)
    return game_lines(away, home, day) if day else None


def h2h_for_event(event: dict) -> dict:
    """DraftKings' moneyline for an Odds API event, shaped like its h2h response; {} if none."""
    lines = game_lines_for_event(event)
    if not lines or lines["home_moneyline"] is None or lines["away_moneyline"] is None:
        return {}
    home, away = event["home_team"], event["away_team"]
    return {"id": event["id"], "home_team": home, "away_team": away, "source": SOURCE,
            "bookmakers": [{**BOOK, "markets": [{"key": "h2h", "outcomes": [
                {"name": home, "price": lines["home_moneyline"]},
                {"name": away, "price": lines["away_moneyline"]}]}]}]}


def _athlete_names(game: dict) -> dict:
    names = {}
    for team in game["team_ids"]:
        if team:
            roster = _get(ROSTER_URL.format(team=team), {}, "roster")
            names.update({str(a["id"]): a.get("displayName") for a in roster.get("athletes", [])
                          if a.get("id")})
    return names


def props(away: str, home: str, day: date) -> dict:
    """DraftKings' player props for away@home, shaped like The Odds API's event odds;
    {} when ESPN has the game but no lines yet."""
    game = _game(away, home, day)
    if game is None:
        return {}
    url = PROPS_URL.format(event=game["id"], provider=DRAFTKINGS)
    items, page, pages = [], 1, 1
    while page <= pages:
        data = _get(url, {"limit": 1000, "page": page}, "prop_bets")
        items += data.get("items", [])
        pages, page = int(data.get("pageCount") or 1), page + 1
    quotes = {}
    for item in items:
        market = PROP_TYPES.get((item.get("type") or {}).get("name"))
        ref = (item.get("athlete") or {}).get("$ref", "")
        line = ((item.get("current") or {}).get("target") or {}).get("value")
        price = _american(((item.get("odds") or {}).get("american") or {}).get("value"))
        if market and "/athletes/" in ref and isinstance(line, (int, float)) and price is not None:
            athlete = ref.split("/athletes/")[1].split("?")[0]
            quotes.setdefault((athlete, market, float(line)), []).append(price)
    if not quotes:
        return {}
    names = _athlete_names(game)
    markets = {}
    for (athlete, market, line), prices in quotes.items():
        if len(prices) != 2 or not PAIR_MARGIN[0] < _implied(prices[0]) + _implied(prices[1]) < PAIR_MARGIN[1]:
            continue
        if athlete not in names:  # not on either roster today (a trade): ask once
            names[athlete] = _get(ATHLETE_URL.format(athlete=athlete), {}, "athlete").get("displayName")
        if not names[athlete]:
            continue
        over, under = prices  # ESPN lists the over first (see the module docstring)
        markets.setdefault(market, []).extend([
            {"name": "Over", "description": names[athlete], "price": over, "point": line},
            {"name": "Under", "description": names[athlete], "price": under, "point": line}])
    if not markets:
        return {}
    return {"source": SOURCE, "bookmakers": [{**BOOK, "markets": [
        {"key": key, "outcomes": outcomes} for key, outcomes in markets.items()]}]}


def props_for_event(event: dict) -> dict:
    away, home, day = _event_teams(event)
    return props(away, home, day) if day else {}


def odds_api_or_espn(fetch, event) -> dict:
    """fetch() (The Odds API's props for `event`), or ESPN's DraftKings props when The Odds
    API refuses or can't be reached: out of credits (401), no key, already bought today."""
    try:
        return fetch()
    except requests.exceptions.RequestException as exc:
        if not event or not enabled():
            raise
        print(f"    Odds API props unavailable ({exc}); using ESPN's free DraftKings props")
        return props_for_event(event)
