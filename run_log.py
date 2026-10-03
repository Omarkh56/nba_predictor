"""run_log.py — one JSON line per pipeline run, with a status for every stage.

A failed fetch must never look like an empty day. Each stage reports one of:

  ok                  the source answered and returned data
  no_games            the source answered: nothing scheduled or played that date
  market_unavailable  the source answered: no lines posted for this event (yet)
  fetch_failed        the request failed or the data isn't final: unknown, not empty
  skipped             not attempted (nothing to do, or an earlier stage failed)

A run's overall status is fetch_failed when a required stage failed, partial
when only optional stages failed, no_games when the schedule was empty, and ok
otherwise. nba_combined.py and checkresults.py exit with EXIT_CODES[status] so a
scheduler can tell a failed run from a quiet day.

Records go to run_log.jsonl next to this file (gitignored); NBA_RUN_LOG
overrides the path.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

STATUSES = ("ok", "no_games", "market_unavailable", "fetch_failed", "skipped")
EXIT_CODES = {"ok": 0, "no_games": 0, "partial": 3, "fetch_failed": 2}
RUN_LOG_FILE = Path(__file__).with_name("run_log.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_path() -> Path:
    override = os.environ.get("NBA_RUN_LOG")
    return Path(override) if override else RUN_LOG_FILE


class RunLog:
    """Collect stage statuses for one run and append them as one JSON line."""

    def __init__(self, command: str, slate_date=None, path=None, run_id: str | None = None):
        self.path = Path(path) if path else default_path()
        if isinstance(slate_date, date):
            slate_date = slate_date.isoformat()
        self.record = {
            "run_id": run_id or uuid.uuid4().hex[:12],
            "command": command,
            "slate_date": slate_date,
            "started_at": _now_iso(),
            "stages": [],
        }
        self._written = False

    @property
    def run_id(self) -> str:
        return self.record["run_id"]

    def set_slate_date(self, slate_date) -> None:
        if isinstance(slate_date, date):
            slate_date = slate_date.isoformat()
        self.record["slate_date"] = slate_date

    def stage(self, name: str, status: str, *, required: bool = False, **detail) -> dict:
        if status not in STATUSES:
            raise ValueError(f"unknown stage status {status!r}; expected one of {STATUSES}")
        entry = {"stage": name, "status": status, "required": bool(required)}
        entry.update({key: value for key, value in detail.items() if value is not None})
        self.record["stages"].append(entry)
        return entry

    def overall_status(self) -> str:
        stages = self.record["stages"]
        if any(s["status"] == "fetch_failed" and s["required"] for s in stages):
            return "fetch_failed"
        if any(s["status"] == "fetch_failed" for s in stages):
            return "partial"
        required = [s for s in stages if s["required"]]
        if required and all(s["status"] in ("no_games", "skipped") for s in required) and any(
            s["status"] == "no_games" for s in required
        ):
            return "no_games"
        return "ok"

    def finish(self, **summary) -> dict:
        """Append the record once and return it (later calls are no-ops)."""
        if self._written:
            return self.record
        self.record["finished_at"] = _now_iso()
        self.record["status"] = self.overall_status()
        self.record.update({key: value for key, value in summary.items() if value is not None})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(self.record, default=str) + "\n")
        self._written = True
        return self.record

    def exit_code(self) -> int:
        return EXIT_CODES[self.record.get("status") or self.overall_status()]


def read_runs(path=None) -> list:
    """Return every logged run, oldest first (blank or corrupt lines skipped)."""
    path = Path(path) if path else default_path()
    if not path.is_file():
        return []
    runs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return runs
