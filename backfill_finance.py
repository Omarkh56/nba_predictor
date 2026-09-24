#!/usr/bin/env python3
"""Safely recover exact-line historical entry prices into a new bet log."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

from bet_log_utils import BET_LOG_FIELDS, read_bet_rows, write_bet_rows
from finance import (
    american_to_decimal,
    american_to_implied,
    ensure_finance_fields,
)


HERE = Path(__file__).resolve().parent
DEFAULT_BETS = HERE / "bet_log.csv"
DEFAULT_ODDS = HERE / "odds_tracker.csv"
DEFAULT_OUTPUT = HERE / "bet_log_finance_backfilled.csv"
DEFAULT_REPORTS = HERE / "reports"


def _name(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip().casefold()


def _market(value) -> str:
    return re.sub(r"\s+", "", str(value or "").upper())


def _line(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _value(row, field):
    value = row.get(field)
    return "" if value is None or (not isinstance(value, str) and pd.isna(value)) else value


def _odds_index(odds: pd.DataFrame) -> tuple[dict, set]:
    exact = {}
    loose = set()
    for _, row in odds.iterrows():
        day = str(row.get("date") or "")[:10]
        player = _name(row.get("player"))
        market = _market(row.get("market"))
        line = _line(row.get("line"))
        if not day or not player or not market or line is None:
            continue
        loose.add((day, player, market))
        exact.setdefault((day, player, market, line), []).append(row)
    return exact, loose


def backfill_rows(bets: list[dict], odds: pd.DataFrame) -> tuple[list[dict], dict]:
    exact, loose = _odds_index(odds)
    output = []
    reasons = Counter()
    recovered = 0
    ambiguous = 0
    legacy_without_exact = 0
    for raw in bets:
        had_explicit_entry_decimal = raw.get("entry_decimal_odds") not in (None, "")
        row = ensure_finance_fields(raw)
        # Phase-1 compatibility normally aliases legacy decimal_odds. During
        # audit backfill that legacy value is not accepted as a verified entry
        # price until its exact line is matched below.
        if not had_explicit_entry_decimal:
            row["entry_decimal_odds"] = ""
        day = str(row.get("date") or "")[:10]
        loose_key = (day, _name(row.get("player")), _market(row.get("market")))
        line = _line(row.get("line"))
        candidates = exact.get((*loose_key, line), []) if line is not None else []
        if not candidates:
            reason = "no_exact_line_match" if loose_key in loose else "no_player_market_match"
            reasons[reason] += 1
            if row.get("decimal_odds") not in (None, ""):
                legacy_without_exact += 1
            output.append(row)
            continue

        # Repeated identical rows are safe; conflicting rows are ambiguous.
        signatures = {
            (
                _value(candidate, "avg_over_odds"),
                _value(candidate, "avg_under_odds"),
                _value(candidate, "devig_over"),
                _value(candidate, "devig_under"),
                _value(candidate, "vig_pct"),
            )
            for candidate in candidates
        }
        if len(signatures) != 1:
            reasons["ambiguous_exact_line_match"] += 1
            ambiguous += 1
            output.append(row)
            continue

        quote = candidates[0]
        pick = str(row.get("pick") or "").strip().upper()
        selected_field = "avg_under_odds" if pick == "UNDER" else "avg_over_odds"
        devig_field = "devig_under" if pick == "UNDER" else "devig_over"
        american = _value(quote, selected_field)
        try:
            decimal = american_to_decimal(american)
            raw_implied = american_to_implied(american)
        except (TypeError, ValueError):
            reasons["invalid_selected_odds"] += 1
            output.append(row)
            continue

        row["entry_american_odds"] = american
        row["entry_decimal_odds"] = round(decimal, 8)
        row["market_implied_probability"] = round(raw_implied, 8)
        devig = _line(_value(quote, devig_field))
        if row.get("p_market") in (None, "") and devig is not None and 0 < devig < 1:
            row["p_market"] = devig
        vig = _line(_value(quote, "vig_pct"))
        if vig is not None:
            row["vig_pct"] = vig
        row["odds_source"] = "legacy_odds_tracker_exact_line_consensus"
        recovered += 1
        output.append(row)

    report = {
        "input_rows": len(bets),
        "rows_with_entry_odds_recovered": recovered,
        "rows_with_closing_odds_recovered": 0,
        "rows_with_line_clv": 0,
        "rows_with_price_clv": 0,
        "rows_remaining_without_recovered_entry_odds": len(bets) - recovered,
        "ambiguous_matches": ambiguous,
        "legacy_decimal_odds_without_exact_line_match": legacy_without_exact,
        "stakes_backfilled": 0,
        "profits_backfilled": 0,
        "timestamps_backfilled": 0,
        "unmatched_reasons": dict(sorted(reasons.items())),
    }
    return output, report


def _report_markdown(report: dict) -> str:
    lines = [
        "# Historical finance backfill", "",
        "Only actual stored consensus quotes matching date, normalized player, exact market, "
        "and exact line were used. Stakes, profits, closing prices, CLV, and timestamps were "
        "not reconstructed.", "",
    ]
    for key, value in report.items():
        if key != "unmatched_reasons":
            lines.append(f"- {key.replace('_', ' ').capitalize()}: {value}")
    lines.extend(["", "## Unmatched reasons", ""])
    lines.extend(
        [f"- {reason}: {count}" for reason, count in report["unmatched_reasons"].items()]
        or ["- None"]
    )
    return "\n".join(lines) + "\n"


def run(source: Path = DEFAULT_BETS, odds_path: Path = DEFAULT_ODDS,
        destination: Path = DEFAULT_OUTPUT, reports_dir: Path = DEFAULT_REPORTS) -> dict:
    if source.resolve() == destination.resolve():
        raise ValueError("destination must differ from bet_log.csv; review is required before replacement")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    old_fields, bets = read_bet_rows(source)
    odds = pd.read_csv(odds_path)
    output, report = backfill_rows(bets, odds)
    fields = list(dict.fromkeys([*BET_LOG_FIELDS, *old_fields, *[key for row in output for key in row]]))
    write_bet_rows(destination, fields, output)
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("source bet log changed during backfill")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "finance_backfill_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports_dir / "finance_backfill_summary.md").write_text(
        _report_markdown(report), encoding="utf-8"
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bets", type=Path, default=DEFAULT_BETS)
    parser.add_argument("--odds", type=Path, default=DEFAULT_ODDS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args(argv)
    report = run(args.bets, args.odds, args.out, args.reports_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote review file → {args.out}")
    print("Original bet_log.csv was not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
