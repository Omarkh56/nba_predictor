"""Pre-game NBA moneyline snapshots, separate from player-prop odds.

Run ``python3 game_odds_tracker.py [YYYY-MM-DD]``. Dates use NBA's Eastern
calendar; snapshot_at records when the quote was observed in UTC. Prices are
averaged in decimal payout space, then converted to American odds, so books
straddling +100/-100 cannot average into an invalid price near zero. Fair
probabilities, vig and k are averaged after de-vigging each paired book.
"""

import argparse
import csv
import json
import logging
import math
import os
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

import oddstracker
from odds_api import get_json
from roi_analysis import american_to_decimal

logger = logging.getLogger(__name__)
TRACKER_FILE = os.path.join(os.path.dirname(__file__), "game_odds_tracker.csv")
MAX_SNAPSHOT_AGE = timedelta(minutes=30)
TRACKER_FIELDS = [
    "date", "snapshot_at", "event_id", "commence_time", "game", "market",
    "home_abbr", "away_abbr", "home_team", "away_team", "books_count",
    "avg_home_odds", "avg_away_odds", "home_decimal_odds", "away_decimal_odds",
    "implied_home", "implied_away", "devig_home", "devig_away",
    "book_lean", "lean_strength", "vig_pct", "k_value",
]


def normalize_team(team: str) -> str:
    """Use the existing full-name/ESPN aliases; never truncate LA teams."""
    model = oddstracker._props_model
    if not isinstance(team, str):
        return ""
    abbr = model.TEAM_NAME_TO_ABBR.get(team, team.upper())
    abbr = model.ESPN_TO_NBA_ABBR.get(abbr, abbr)
    return abbr if abbr in model.TEAM_NAME_TO_ABBR.values() else ""


def _timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("odds timestamps must be strings")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("odds timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def _game_date(event: dict) -> date:
    return _timestamp(event["commence_time"]).astimezone(ZoneInfo("America/New_York")).date()


def get_event_game_odds(event_id: str) -> dict:
    """Fetch paired h2h prices using the same event endpoint as props/spreads."""
    for regions in ("us", "us,us2,uk,eu,au"):
        data = get_json(
            f"{oddstracker.BASE_URL}/sports/{oddstracker.SPORT}/events/{event_id}/odds",
            params={"apiKey": oddstracker.ODDS_API_KEY, "regions": regions,
                    "markets": "h2h", "oddsFormat": "american", "dateFormat": "iso"},
            timeout=25,
        )
        if any(m.get("key") == "h2h" and m.get("outcomes")
               for book in data.get("bookmakers", []) for m in book.get("markets", [])):
            return data
    return {}


def _american_from_decimal(decimal: float) -> float:
    profit = decimal - 1.0
    return round(100.0 * profit if profit >= 1.0 else -100.0 / profit, 4)


def parse_game_odds_to_row(data: dict, event: dict, snapshot_at: datetime) -> dict:
    """Count only books quoting both named teams; outcome order is irrelevant."""
    home, away = event.get("home_team", ""), event.get("away_team", "")
    ha, aa = normalize_team(home), normalize_team(away)
    if not ha or not aa or ha == aa or not event.get("id"):
        return {}
    # Reject an API response for another event or reversed home/away metadata.
    if any(data.get(k, event.get(k)) != event.get(k)
           for k in ("id", "home_team", "away_team")):
        return {}
    paired = {}
    for book in data.get("bookmakers", []):
        book_key = book.get("key") or book.get("title")
        if not book_key:
            continue
        for market in book.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = market.get("outcomes", [])
            if len(outcomes) != 2 or {o.get("name") for o in outcomes} != {home, away}:
                continue
            try:
                prices = {o["name"]: float(o["price"]) for o in outcomes}
                if any(not math.isfinite(p) or abs(p) < 100 for p in prices.values()):
                    continue
                # Keep coherent two-way bookmaker pairs before making a consensus.
                ph, pa, vig, k = oddstracker.devig_power(prices[home], prices[away])
                if not 0 <= vig <= 25 or not (0 < ph < 1 and 0 < pa < 1) or abs(ph + pa - 1) > 0.001:
                    continue
                paired[book_key] = (american_to_decimal(prices[home]),
                                    american_to_decimal(prices[away]), ph, pa, vig, k,
                                    oddstracker.american_to_implied(prices[home]),
                                    oddstracker.american_to_implied(prices[away]))
            except (ValueError, TypeError, KeyError):
                continue
    if not paired:
        return {}
    hd = sum(p[0] for p in paired.values()) / len(paired)
    ad = sum(p[1] for p in paired.values()) / len(paired)
    hp, ap = _american_from_decimal(hd), _american_from_decimal(ad)
    ph, pa, vig, k, raw_home, raw_away = (
        sum(p[i] for p in paired.values()) / len(paired) for i in range(2, 8))
    lean, strength = oddstracker.book_lean(ph)
    return {
        "date": _game_date(event).isoformat(), "snapshot_at": snapshot_at.isoformat(),
        "event_id": event["id"], "commence_time": event["commence_time"],
        "game": f"{aa}@{ha}", "market": "h2h", "home_abbr": ha, "away_abbr": aa,
        "home_team": home, "away_team": away, "books_count": len(paired),
        "avg_home_odds": hp, "avg_away_odds": ap,
        "home_decimal_odds": hd, "away_decimal_odds": ad,
        "implied_home": raw_home, "implied_away": raw_away,
        "devig_home": ph, "devig_away": pa,
        "book_lean": {"OVER": "HOME", "UNDER": "AWAY", "NEUTRAL": "NEUTRAL"}[lean],
        "lean_strength": strength, "vig_pct": vig, "k_value": k,
    }


def usable_snapshot(row: dict, game_date: date, now: datetime) -> bool:
    """Require a recent, internally consistent quote for an unstarted game."""
    try:
        if not isinstance(game_date, date) or any(
            not isinstance(row[key], str) or not row[key]
            for key in ("date", "snapshot_at", "event_id", "commence_time", "game", "market",
                        "home_abbr", "away_abbr", "home_team", "away_team")
        ):
            return False
        observed, start = _timestamp(row["snapshot_at"]), _timestamp(row["commence_time"])
        ph, pa = float(row["devig_home"]), float(row["devig_away"])
        hd, ad = float(row["home_decimal_odds"]), float(row["away_decimal_odds"])
        values = [ph, pa, hd, ad, float(row["vig_pct"]), float(row["k_value"])]
        return (
            row["market"] == "h2h" and row["date"] == game_date.isoformat()
            and _game_date(row) == game_date and bool(row["event_id"])
            and normalize_team(row["home_team"]) == row["home_abbr"]
            and normalize_team(row["away_team"]) == row["away_abbr"]
            and bool(row["home_abbr"]) and row["home_abbr"] != row["away_abbr"]
            and row["game"] == f'{row["away_abbr"]}@{row["home_abbr"]}'
            and int(row["books_count"]) > 0 and all(math.isfinite(v) for v in values)
            and 0 < ph < 1 and 0 < pa < 1 and abs(ph + pa - 1) <= 0.001
            and hd > 1 and ad > 1 and 0 <= float(row["vig_pct"]) <= 25
            and 1 <= float(row["k_value"]) <= 2
            and abs(american_to_decimal(float(row["avg_home_odds"])) - hd) < 0.0001
            and abs(american_to_decimal(float(row["avg_away_odds"])) - ad) < 0.0001
            and start > now and observed < start
            and timedelta(0) <= now - observed <= MAX_SNAPSHOT_AGE
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def snapshot(snap_date: date = None, *, events: list = None, path: str = None,
             now: datetime = None) -> list:
    """Append one row per event per observation; repeat runs preserve movement."""
    snap_date = snap_date or date.today()
    now = now or datetime.now(timezone.utc)
    events = oddstracker.get_all_events() if events is None else events
    rows = []
    for event in events:
        try:
            if _game_date(event) != snap_date or _timestamp(event["commence_time"]) <= now:
                continue
            data = get_event_game_odds(event["id"])
            row = parse_game_odds_to_row(data, event, now)
            if row and usable_snapshot(row, snap_date, now):
                rows.append(row)
        except (requests.exceptions.RequestException, json.JSONDecodeError,
                KeyError, TypeError, ValueError) as exc:
            # Request exception strings can contain the API key in their URL.
            logger.warning("Game odds fetch failed for event %s (%s)",
                           event.get("id", "unknown"), type(exc).__name__)
    if rows:
        destination = path or TRACKER_FILE
        header = not os.path.isfile(destination) or os.path.getsize(destination) == 0
        with open(destination, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=TRACKER_FIELDS)
            if header:
                writer.writeheader()
            writer.writerows(rows)
    print(f"  Game moneylines: {len(rows)} snapshot(s) for {snap_date.isoformat()}")
    for row in rows:
        print(f'    {row["game"]}: home {row["avg_home_odds"]:+.0f}, '
              f'away {row["avg_away_odds"]:+.0f}; devig '
              f'{row["devig_home"]:.1%}/{row["devig_away"]:.1%} '
              f'({row["books_count"]} books, vig {row["vig_pct"]:.2f}%)')
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", nargs="?", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()
    try:
        snapshot(args.date)
    except (requests.exceptions.RequestException, json.JSONDecodeError, OSError) as exc:
        print(f"  Game odds snapshot failed ({type(exc).__name__})")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
