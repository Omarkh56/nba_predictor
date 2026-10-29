"""Closing-line joins for timestamped bookmaker-level prop snapshots."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from finance import american_to_decimal, decimal_to_implied, valid_decimal_odds, valid_probability


def _name(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip().casefold()


def _text(value) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp(value):
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(parsed) else parsed


def _blank_clv(reason: str) -> dict:
    return {
        "closing_line": "",
        "closing_decimal_odds": "",
        "line_clv": "",
        "implied_probability_clv": "",
        "raw_implied_probability_clv": "",
        "closing_snapshot_at": "",
        "clv_unmatched_reason": reason,
    }


def closing_for_bet(bet: Mapping[str, object], snapshots: pd.DataFrame) -> dict:
    """Find the final comparable quote strictly before game start."""
    if snapshots.empty:
        return _blank_clv("no_closing_snapshot")
    game_start = _timestamp(bet.get("game_start_at"))
    if game_start is None:
        return _blank_clv("missing_game_start_timestamp")

    required = {"snapshot_at", "event_id", "player", "market", "line", "side", "american_odds"}
    if not required <= set(snapshots.columns):
        return _blank_clv("no_closing_snapshot")

    frame = snapshots.copy()
    if "snapshot_kind" in frame:
        # Old untagged quotes retain their historical behavior. New entry
        # observations cannot stand in for a scheduled closing observation.
        frame = frame[frame["snapshot_kind"].fillna("").astype(str) != "entry"].copy()
    frame["_player"] = frame["player"].map(_name)
    frame["_market"] = frame["market"].astype(str).str.strip().str.upper()
    frame["_side"] = frame["side"].astype(str).str.strip().str.upper()
    frame["_snapshot"] = pd.to_datetime(frame["snapshot_at"], utc=True, errors="coerce")
    target = frame[
        (frame["_player"] == _name(bet.get("player")))
        & (frame["_market"] == _text(bet.get("market")).upper())
        & (frame["_side"] == _text(bet.get("pick")).upper())
    ].copy()
    bet_date = _text(bet.get("date"))[:10]
    if bet_date and "date" in target:
        target = target[target["date"].astype(str).str[:10] == bet_date]
    if target.empty:
        return _blank_clv("no_closing_snapshot")

    event_id = _text(bet.get("odds_event_id"))
    if event_id:
        target = target[target["event_id"].astype(str).str.strip() == event_id]
        if target.empty:
            return _blank_clv("no_closing_snapshot")
    else:
        events = {value for value in target["event_id"].astype(str).str.strip() if value}
        if len(events) > 1:
            return _blank_clv("ambiguous_event_player_match")

    before_start = target[target["_snapshot"].notna() & (target["_snapshot"] < game_start)]
    if before_start.empty:
        if target["_snapshot"].notna().any():
            return _blank_clv("snapshot_after_game_start")
        return _blank_clv("no_closing_snapshot")

    latest = before_start["_snapshot"].max()
    closing = before_start[before_start["_snapshot"] == latest].copy()
    book = _text(bet.get("odds_book"))
    if book:
        book_mask = pd.Series(False, index=closing.index)
        for field in ("book_key", "book_title"):
            if field in closing:
                book_mask |= closing[field].astype(str).str.strip().str.casefold() == book.casefold()
        closing = closing[book_mask]
        if closing.empty:
            return _blank_clv("no_closing_snapshot")

    numeric_lines = pd.to_numeric(closing["line"], errors="coerce").dropna()
    entry_line = _number(bet.get("line"))
    if numeric_lines.empty or entry_line is None:
        return _blank_clv("no_closing_snapshot")
    closing_line = float(numeric_lines.median())
    direction = _text(bet.get("pick")).upper()
    line_clv = closing_line - entry_line if direction == "OVER" else entry_line - closing_line

    same_line = closing[(pd.to_numeric(closing["line"], errors="coerce") - entry_line).abs() < 1e-9]
    base = {
        "closing_line": round(closing_line, 4),
        "closing_decimal_odds": "",
        "line_clv": round(line_clv, 4),
        "implied_probability_clv": "",
        "raw_implied_probability_clv": "",
        "closing_snapshot_at": latest.isoformat(),
        "clv_unmatched_reason": "",
    }
    if same_line.empty:
        base["clv_unmatched_reason"] = "different_line_with_no_same_line_comparison"
        return base

    prices = pd.to_numeric(same_line["american_odds"], errors="coerce").dropna()
    if prices.empty:
        base["clv_unmatched_reason"] = "no_closing_snapshot"
        return base
    try:
        # Average the books through implied probability: a plain mean of -110 and +104 is -3.
        implied = [decimal_to_implied(american_to_decimal(price)) for price in prices]
        closing_decimal = len(implied) / sum(implied)
    except (TypeError, ValueError):
        base["clv_unmatched_reason"] = "no_closing_snapshot"
        return base
    base["closing_decimal_odds"] = round(closing_decimal, 8)

    entry_decimal = bet.get("entry_decimal_odds")
    if not valid_decimal_odds(entry_decimal):
        entry_decimal = bet.get("decimal_odds")
    entry_raw = _number(bet.get("market_implied_probability"))
    if entry_raw is None and valid_decimal_odds(entry_decimal):
        entry_raw = decimal_to_implied(entry_decimal)
    closing_raw = decimal_to_implied(closing_decimal)
    if entry_raw is not None:
        base["raw_implied_probability_clv"] = round(closing_raw - entry_raw, 8)

    devig = (
        pd.to_numeric(same_line["devig_probability"], errors="coerce").dropna()
        if "devig_probability" in same_line else pd.Series(dtype=float)
    )
    entry_devig = bet.get("p_market")
    if not devig.empty and valid_probability(entry_devig):
        base["implied_probability_clv"] = round(float(devig.mean()) - float(entry_devig), 8)
    else:
        base["clv_unmatched_reason"] = "missing_opposing_side"
    return base


def apply_closing_clv(rows: Iterable[Mapping[str, object]], snapshots: pd.DataFrame) -> list[dict]:
    output = []
    for raw in rows:
        row = dict(raw)
        # Never rewrite an already joined closing record.
        if row.get("closing_snapshot_at") not in (None, ""):
            output.append(row)
            continue
        row.update(closing_for_bet(row, snapshots))
        output.append(row)
    return output


def clv_summary(rows: Iterable[Mapping[str, object]]) -> dict:
    rows = list(rows)
    line_values = [_number(row.get("line_clv")) for row in rows]
    price_values = [_number(row.get("implied_probability_clv")) for row in rows]
    raw_values = [_number(row.get("raw_implied_probability_clv")) for row in rows]
    line_values = [value for value in line_values if value is not None]
    price_values = [value for value in price_values if value is not None]
    raw_values = [value for value in raw_values if value is not None]
    reasons = Counter(
        _text(row.get("clv_unmatched_reason")) for row in rows
        if _text(row.get("clv_unmatched_reason"))
    )
    attempted = len(rows)
    return {
        "attempted_rows": attempted,
        "closing_snapshot_rows": sum(bool(_text(row.get("closing_snapshot_at"))) for row in rows),
        "line_clv_rows": len(line_values),
        "price_clv_rows": len(price_values),
        "raw_price_clv_rows": len(raw_values),
        "line_clv_coverage": len(line_values) / attempted if attempted else 0.0,
        "price_clv_coverage": len(price_values) / attempted if attempted else 0.0,
        "average_line_clv": sum(line_values) / len(line_values) if line_values else None,
        "average_implied_probability_clv": sum(price_values) / len(price_values) if price_values else None,
        "average_raw_implied_probability_clv": sum(raw_values) / len(raw_values) if raw_values else None,
        "unmatched_reasons": dict(sorted(reasons.items())),
    }


def write_clv_reports(rows: Iterable[Mapping[str, object]], reports_dir: Path) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = clv_summary(rows)
    (reports_dir / "clv_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# CLV summary", "",
        f"- Attempted rows: {summary['attempted_rows']}",
        f"- Closing snapshots matched: {summary['closing_snapshot_rows']}",
        f"- Directional line CLV rows: {summary['line_clv_rows']}",
        f"- Same-line de-vig price CLV rows: {summary['price_clv_rows']}",
        f"- Same-line raw price CLV rows: {summary['raw_price_clv_rows']}",
        f"- Directional line CLV coverage: {summary['line_clv_coverage'] * 100:.2f}%",
        f"- Same-line de-vig price CLV coverage: {summary['price_clv_coverage'] * 100:.2f}%",
        f"- Average line CLV: {summary['average_line_clv']}",
        f"- Average de-vig implied-probability CLV: {summary['average_implied_probability_clv']}",
        "", "## Unmatched reasons", "",
    ]
    if summary["unmatched_reasons"]:
        lines.extend(f"- {reason}: {count}" for reason, count in summary["unmatched_reasons"].items())
    else:
        lines.append("- None")
    (reports_dir / "clv_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
