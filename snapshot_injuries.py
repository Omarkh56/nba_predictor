"""snapshot_injuries.py — archive one ESPN injury report and flatten it to CSV.

ESPN only serves the current injury report, so history exists only if we keep
it. Each run fetches the report once, saves the raw response to the raw archive
(raw_archive.py), then appends one row per listed player to
data/injury_snapshots.csv. Every run appends a status line to run_log.jsonl and
exits with run_log.EXIT_CODES (0 ok, 2 fetch_failed).

Usage:
    python3 snapshot_injuries.py               # today's US Eastern slate
    python3 snapshot_injuries.py 2026-10-21    # label the snapshot with this slate date

The report is not date-specific: DATE only labels which slate a snapshot was
taken for. Run it several times a day (see the README cron example) so status
changes before tip-off are captured.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import requests

import raw_archive
import team_codes
from run_log import RunLog

# Same endpoint and headers as newnbapredictor.py (kept separate so this archival
# job doesn't depend on the team model importing cleanly). ESPN's CDN rejects
# browser-like User-Agents from scripts, so send only Accept.
ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
ESPN_HEADERS = {"Accept": "application/json"}

CSV_PATH = Path(__file__).with_name("data") / "injury_snapshots.csv"
FIELDS = [
    "snapshot_at_utc", "slate_date", "team", "player", "espn_player_id", "status",
    "position", "status_type", "fantasy_status", "injury_type", "injury_location",
    "injury_detail", "injury_side", "return_date", "short_comment", "long_comment",
    "report_updated_at", "espn_injury_id", "run_id",
]
_ID_IN_URL = re.compile(r"/id/(\d+)|/full/(\d+)\.png")


def _espn_player_id(athlete: dict) -> str:
    """ESPN's athlete id, from the profile link (or headshot) since the record has no id field."""
    urls = [link.get("href") or "" for link in athlete.get("links") or []]
    urls.append((athlete.get("headshot") or {}).get("href") or "")
    for url in urls:
        match = _ID_IN_URL.search(url)
        if match:
            return match.group(1) or match.group(2)
    return ""


def parse_rows(payload: dict, *, snapshot_at: str, slate_date: date, run_id: str) -> list:
    """One flat row per listed player. Raises ValueError if the report is malformed."""
    teams = payload.get("injuries") if isinstance(payload, dict) else None
    if not isinstance(teams, list):
        raise ValueError("ESPN injury report has no 'injuries' list")
    rows = []
    for team_entry in teams:
        team_name = team_entry.get("displayName", "")
        for inj in team_entry.get("injuries") or []:
            athlete = inj.get("athlete") or {}
            espn_team = (athlete.get("team") or {}).get("abbreviation", "")
            details = inj.get("details") or {}
            rows.append({
                "snapshot_at_utc": snapshot_at,
                "slate_date": slate_date.isoformat(),
                "team": (team_codes.normalize_team(espn_team)
                         or team_codes.normalize_team(team_name) or espn_team or team_name),
                "player": athlete.get("displayName", ""),
                "espn_player_id": _espn_player_id(athlete),
                "status": inj.get("status", ""),
                "position": (athlete.get("position") or {}).get("abbreviation", ""),
                "status_type": (inj.get("type") or {}).get("description", ""),
                "fantasy_status": (details.get("fantasyStatus") or {}).get("abbreviation", ""),
                "injury_type": details.get("type", ""),
                "injury_location": details.get("location", ""),
                "injury_detail": details.get("detail", ""),
                "injury_side": details.get("side", ""),
                "return_date": details.get("returnDate", ""),
                "short_comment": inj.get("shortComment", ""),
                "long_comment": inj.get("longComment", ""),
                "report_updated_at": inj.get("date", ""),
                "espn_injury_id": inj.get("id", ""),
                "run_id": run_id,
            })
    return rows


def append_rows(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def main(argv=None, *, csv_path=None) -> int:
    parser = argparse.ArgumentParser(description="Archive the ESPN injury report and flatten it to CSV.")
    parser.add_argument("date", nargs="?", type=date.fromisoformat, default=None,
                        help="slate date YYYY-MM-DD on the US Eastern calendar (default: today)")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    slate_date = args.date or team_codes.today_slate_date()
    csv_path = Path(csv_path) if csv_path else CSV_PATH
    run = RunLog("snapshot_injuries", slate_date=slate_date)

    snapshot_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        response = requests.get(ESPN_INJURIES_URL, headers=ESPN_HEADERS, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        raw_archive.save("espn", "injuries", {}, None, run_id=run.run_id, status="fetch_failed",
                         http_status=raw_archive.http_status_of(exc), error=type(exc).__name__)
        print(f"  ESPN injuries error: {exc}")
        run.stage("espn_injuries", "fetch_failed", required=True, error=type(exc).__name__)
        run.finish()
        return run.exit_code()
    raw_archive.save("espn", "injuries", {}, payload, run_id=run.run_id,
                     http_status=raw_archive.http_status_of(response))

    try:
        rows = parse_rows(payload, snapshot_at=snapshot_at, slate_date=slate_date, run_id=run.run_id)
    except (ValueError, AttributeError, TypeError) as exc:
        print(f"  ESPN injury report could not be parsed: {exc}")
        run.stage("espn_injuries", "fetch_failed", required=True, error=f"malformed: {exc}")
        run.finish()
        return run.exit_code()
    run.stage("espn_injuries", "ok", required=True, players=len(rows),
              teams=len({r["team"] for r in rows}))

    try:
        append_rows(rows, csv_path)
    except OSError as exc:
        # Not a fetch, but the shared statuses have no other failure: the snapshot is lost.
        print(f"  Could not write {csv_path}: {exc}")
        run.stage("injury_csv", "fetch_failed", required=True, error=f"{type(exc).__name__}: {exc}")
    else:
        run.stage("injury_csv", "ok", required=True, rows=len(rows))
        print(f"  {len(rows)} injury rows for {slate_date.isoformat()} -> {csv_path}")
    run.finish()
    return run.exit_code()


if __name__ == "__main__":
    sys.exit(main())
