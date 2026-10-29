"""Deterministic, idempotent CSV storage for graded player-prop bets.

The public log identity is ``date + player + market + line + pick``.  Display
values stay human-readable in the CSV, while keys are normalized so harmless
differences such as accents, whitespace, or ``2.5`` versus ``2.500`` do not
create duplicate bets.
"""
from __future__ import annotations

import csv
import os
import re
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from finance import FINANCE_FIELDS, ensure_finance_fields


BET_LOG_FIELDS = [
    "run_id",
    "model_version",
    "date",
    "player",
    "market",
    "line",
    "pick",
    "projection",
    "actual",
    "margin",
    "result",
    "flags",
    "p_model",
    "p_market",
    "decimal_odds",
    "stake_pct",
    "realized_profit",
    *FINANCE_FIELDS,
]
KEY_FIELDS = ("date", "player", "market", "line", "pick")


class BetLogConflict(ValueError):
    """Two rows with one identity key contain incompatible populated values."""


@dataclass
class UpsertStats:
    before_rows: int
    incoming_rows: int
    after_rows: int
    duplicate_groups: int
    duplicates_removed: int
    inserted: int
    updated: int
    unchanged: int
    conflicts: int = 0

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


def _text(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() in {"nan", "none", "null"} else s


def _normal_player(value) -> str:
    s = _text(value).replace("’", "'").replace("`", "'")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().casefold()


def _normal_date(value) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    s = _text(value)
    try:
        return date.fromisoformat(s[:10]).isoformat()
    except ValueError as exc:
        raise ValueError("invalid bet-log date %r" % value) from exc


def _normal_market(value) -> str:
    s = re.sub(r"\s+", "", _text(value).upper())
    return "3PM" if s == "FG3M" else s


def _normal_line(value) -> str:
    try:
        d = Decimal(_text(value))
    except InvalidOperation as exc:
        raise ValueError("invalid bet-log line %r" % value) from exc
    if not d.is_finite():
        raise ValueError("invalid bet-log line %r" % value)
    rendered = format(d.normalize(), "f")
    return "0" if rendered in {"-0", ""} else rendered


def canonical_bet_key(row: Mapping[str, object]) -> Tuple[str, str, str, str, str]:
    """Return the normalized five-column identity for one bet-log row."""
    pick = _text(row.get("pick")).upper()
    if pick not in {"OVER", "UNDER"}:
        raise ValueError("invalid bet-log pick %r" % row.get("pick"))
    key = (
        _normal_date(row.get("date")),
        _normal_player(row.get("player")),
        _normal_market(row.get("market")),
        _normal_line(row.get("line")),
        pick,
    )
    if not key[1] or not key[2]:
        raise ValueError("bet-log identity contains an empty player or market")
    return key


def _semantically_equal(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        return Decimal(a) == Decimal(b)
    except InvalidOperation:
        return False


def _field_order(rows: Sequence[Mapping[str, object]], requested: Sequence[str]) -> List[str]:
    out: List[str] = []
    for field in requested:
        if field not in out:
            out.append(field)
    for row in rows:
        for field in row:
            if field not in out:
                out.append(field)
    return out


def _merge_row(
    current: MutableMapping[str, str], incoming: Mapping[str, object], key: Tuple[str, ...]
) -> bool:
    """Fill empty fields in ``current``. Return whether anything changed.

    Conflicting populated values abort the whole write, so an API retry or a
    second grading run can never silently rewrite a previously graded bet.
    """
    changed = False
    for field, raw in incoming.items():
        if field in KEY_FIELDS:
            continue
        new = _text(raw)
        old = _text(current.get(field))
        if not old and new:
            current[field] = new
            changed = True
        elif old and new and not _semantically_equal(old, new):
            raise BetLogConflict(
                "conflicting %s for %s: existing=%r incoming=%r" % (field, key, old, new)
            )
    return changed


def read_bet_rows(path: os.PathLike) -> Tuple[List[str], List[Dict[str, str]]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return list(BET_LOG_FIELDS), []
    with p.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return list(BET_LOG_FIELDS), []
        return list(reader.fieldnames), [dict(r) for r in reader]


def upsert_bet_rows(
    existing: Sequence[Mapping[str, object]], incoming: Iterable[Mapping[str, object]]
) -> Tuple[List[Dict[str, str]], UpsertStats]:
    """Deduplicate existing rows and merge incoming rows without writing a file."""
    additions = [ensure_finance_fields(r) for r in incoming]
    ordered: Dict[Tuple[str, ...], Dict[str, str]] = {}
    counts: Dict[Tuple[str, ...], int] = {}
    duplicate_groups = 0
    updated = 0
    inserted = 0
    unchanged = 0

    for raw in existing:
        row = {k: _text(v) for k, v in ensure_finance_fields(raw).items()}
        key = canonical_bet_key(row)
        counts[key] = counts.get(key, 0) + 1
        if key not in ordered:
            ordered[key] = row
        else:
            if counts[key] == 2:
                duplicate_groups += 1
            _merge_row(ordered[key], row, key)

    existing_unique = len(ordered)
    touched = set()
    for raw in additions:
        row = {k: _text(v) for k, v in raw.items()}
        key = canonical_bet_key(row)
        if key not in ordered:
            ordered[key] = row
            inserted += 1
        else:
            changed = _merge_row(ordered[key], row, key)
            if key not in touched:
                if changed:
                    updated += 1
                else:
                    unchanged += 1
        touched.add(key)

    rows = list(ordered.values())
    stats = UpsertStats(
        before_rows=len(existing),
        incoming_rows=len(additions),
        after_rows=len(rows),
        duplicate_groups=duplicate_groups,
        duplicates_removed=len(existing) - existing_unique,
        inserted=inserted,
        updated=updated,
        unchanged=unchanged,
    )
    return rows, stats


def write_bet_rows(path: os.PathLike, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    """Atomically replace ``path`` with the supplied rows."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({f: _text(row.get(f)) for f in fieldnames})
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def upsert_bet_log(path: os.PathLike, incoming: Iterable[Mapping[str, object]]) -> UpsertStats:
    """Idempotently merge ``incoming`` into a CSV, deduplicating old rows too."""
    old_fields, existing = read_bet_rows(path)
    additions = [dict(r) for r in incoming]
    rows, stats = upsert_bet_rows(existing, additions)
    fields = _field_order([*existing, *additions], [*BET_LOG_FIELDS, *old_fields])
    write_bet_rows(path, fields, rows)
    return stats


def dedupe_bet_log(source: os.PathLike, destination: os.PathLike) -> UpsertStats:
    """Build a deduplicated destination from ``source`` without modifying the source."""
    fields, existing = read_bet_rows(source)
    rows, stats = upsert_bet_rows(existing, [])
    write_bet_rows(destination, _field_order(existing, [*BET_LOG_FIELDS, *fields]), rows)
    return stats
