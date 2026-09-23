"""Shared finance schema and safe odds/identity helpers.

This module deliberately contains no assumed prices. Missing values stay
missing so downstream ROI, bankroll and CLV calculations can exclude them.
"""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Mapping


# Existing public columns (line, p_model, p_market, decimal_odds, stake_pct,
# result, actual and realized_profit) remain in bet_log.csv. These additions
# complete the record without renaming or removing the old fields.
FINANCE_FIELDS = [
    "bet_id",
    "logged_at",
    "game_start_at",
    "entry_american_odds",
    "entry_decimal_odds",
    "market_implied_probability",
    "vig_pct",
    "model_edge",
    "expected_value",
    "kelly_fraction",
    "stake_skip_reason",
    "bankroll_before",
    "stake_amount",
    "closing_line",
    "closing_decimal_odds",
    "line_clv",
    "implied_probability_clv",
    "raw_implied_probability_clv",
    "odds_book",
    "odds_source",
    "odds_event_id",
    "odds_snapshot_at",
    "closing_snapshot_at",
    "clv_unmatched_reason",
]


def _finite(value) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("value must be finite")
    return number


def american_to_decimal(american) -> float:
    """Convert valid American odds to decimal odds.

    American prices whose absolute value is below 100 are rejected. They are
    usually corrupted spread/line values rather than prices.
    """
    price = _finite(american)
    if abs(price) < 100:
        raise ValueError("not a valid American price: %s" % price)
    return 1.0 + price / 100.0 if price > 0 else 1.0 + 100.0 / abs(price)


def decimal_to_implied(decimal_odds) -> float:
    price = _finite(decimal_odds)
    if price <= 1.0:
        raise ValueError("decimal odds must be greater than 1")
    return 1.0 / price


def american_to_implied(american) -> float:
    return decimal_to_implied(american_to_decimal(american))


def valid_probability(value) -> bool:
    try:
        p = _finite(value)
    except (TypeError, ValueError):
        return False
    return 0.0 < p < 1.0


def valid_decimal_odds(value) -> bool:
    try:
        return _finite(value) > 1.0
    except (TypeError, ValueError):
        return False


def _identity_text(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip().casefold()


def _identity_line(value) -> str:
    try:
        line = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError("invalid bet line %r" % value) from exc
    if not line.is_finite():
        raise ValueError("invalid bet line %r" % value)
    rendered = format(line.normalize(), "f")
    return "0" if rendered in ("", "-0") else rendered


def make_bet_id(date_value, player, market, line, pick) -> str:
    """Stable public ID for the existing five-column bet identity."""
    identity = "|".join((
        str(date_value)[:10],
        _identity_text(player),
        re.sub(r"\s+", "", str(market or "").upper()),
        _identity_line(line),
        str(pick or "").strip().upper(),
    ))
    return "bet_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def ensure_finance_fields(row: Mapping[str, object]) -> dict:
    """Return a backward-compatible row with every finance column present."""
    out = dict(row)
    for field in FINANCE_FIELDS:
        out.setdefault(field, "")
    if not out.get("bet_id") and all(out.get(k) not in (None, "") for k in
                                      ("date", "player", "market", "line", "pick")):
        out["bet_id"] = make_bet_id(out["date"], out["player"], out["market"],
                                    out["line"], out["pick"])
    # Keep the legacy public decimal_odds column as an alias going forward.
    if not out.get("entry_decimal_odds") and out.get("decimal_odds") not in (None, ""):
        out["entry_decimal_odds"] = out["decimal_odds"]
    if not out.get("decimal_odds") and out.get("entry_decimal_odds") not in (None, ""):
        out["decimal_odds"] = out["entry_decimal_odds"]
    return out
