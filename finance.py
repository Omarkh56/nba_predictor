"""Shared finance schema and safe odds/identity helpers.

This module deliberately contains no assumed prices. Missing values stay
missing so downstream ROI, bankroll and CLV calculations can exclude them.
"""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Mapping, Optional


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


STAKE_SKIP_MISSING_ODDS = "missing_or_invalid_odds"
STAKE_SKIP_MISSING_PROBABILITY = "missing_or_invalid_model_probability"
STAKE_SKIP_STALE_DATA = "stale_required_data"
STAKE_SKIP_NONPOSITIVE_EV = "nonpositive_expected_value"
STAKE_SKIP_NONPOSITIVE_KELLY = "nonpositive_full_kelly"
STAKE_SKIP_INSUFFICIENT_EDGE = "insufficient_model_edge"
STAKE_SKIP_EDGE_NOT_ABOVE_VIG = "edge_not_above_market_vig"


@dataclass(frozen=True)
class StakeDecision:
    """Auditable result of the staking policy for one priced pick."""

    stake_pct: float
    full_kelly: Optional[float]
    expected_value: Optional[float]
    model_edge: Optional[float]
    market_implied_probability: Optional[float]
    stake_skip_reason: str


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


def _optional_probability(value) -> Optional[float]:
    return _finite(value) if valid_probability(value) else None


def _parse_utc_timestamp(value) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def odds_data_is_fresh(snapshot_at, as_of, game_start_at=None,
                       max_age_minutes: float = 60.0) -> bool:
    """Return whether a timestamped odds record is safe for live staking.

    A date-only or missing timestamp cannot prove freshness. A quote observed
    after the decision time or at/after game start is also unusable.
    """
    snapshot = _parse_utc_timestamp(snapshot_at)
    decision_time = _parse_utc_timestamp(as_of)
    game_start = _parse_utc_timestamp(game_start_at)
    try:
        max_age = _finite(max_age_minutes)
    except (TypeError, ValueError):
        return False
    if snapshot is None or decision_time is None or max_age < 0:
        return False
    if snapshot > decision_time:
        return False
    if (decision_time - snapshot).total_seconds() > max_age * 60.0:
        return False
    if game_start is not None and snapshot >= game_start:
        return False
    return True


def recommend_stake(p_model, decimal_odds, *, devig_market_probability=None,
                    vig_pct=None, min_edge: float = 0.02,
                    kelly_multiplier: float = 0.25,
                    max_stake_pct: float = 0.05,
                    data_is_fresh: bool = True) -> StakeDecision:
    """Apply the production staking gates and fractional-Kelly formula.

    When a valid two-sided de-vig probability is unavailable, model edge is
    measured against the selected side's raw implied probability. ``vig_pct``
    must remain missing in that case; the explicit edge and positive-EV gates
    still apply.
    """
    if not valid_decimal_odds(decimal_odds):
        return StakeDecision(0.0, None, None, None, None, STAKE_SKIP_MISSING_ODDS)
    price = _finite(decimal_odds)
    raw_implied = decimal_to_implied(price)

    if not valid_probability(p_model):
        return StakeDecision(
            0.0, None, None, None, raw_implied, STAKE_SKIP_MISSING_PROBABILITY
        )
    probability = _finite(p_model)
    devig_probability = _optional_probability(devig_market_probability)
    edge_reference = devig_probability if devig_probability is not None else raw_implied
    model_edge = probability - edge_reference
    expected_value = probability * price - 1.0
    b = price - 1.0
    q = 1.0 - probability
    full_kelly = (b * probability - q) / b

    if not data_is_fresh:
        return StakeDecision(
            0.0, full_kelly, expected_value, model_edge, raw_implied,
            STAKE_SKIP_STALE_DATA,
        )
    if expected_value <= 0:
        return StakeDecision(
            0.0, full_kelly, expected_value, model_edge, raw_implied,
            STAKE_SKIP_NONPOSITIVE_EV,
        )
    if full_kelly <= 0:
        return StakeDecision(
            0.0, full_kelly, expected_value, model_edge, raw_implied,
            STAKE_SKIP_NONPOSITIVE_KELLY,
        )

    try:
        required_edge = max(0.0, _finite(min_edge))
    except (TypeError, ValueError):
        required_edge = 0.0
    if model_edge <= required_edge:
        return StakeDecision(
            0.0, full_kelly, expected_value, model_edge, raw_implied,
            STAKE_SKIP_INSUFFICIENT_EDGE,
        )

    # A vig comparison is valid only with a paired, de-vigged price. The
    # tracker records vig as percentage points (4.5 means 4.5%).
    if devig_probability is not None and vig_pct not in (None, ""):
        try:
            vig = _finite(vig_pct)
            vig_fraction = vig / 100.0
        except (TypeError, ValueError):
            vig_fraction = None
        if vig_fraction is not None and vig_fraction >= 0 and model_edge <= vig_fraction:
            return StakeDecision(
                0.0, full_kelly, expected_value, model_edge, raw_implied,
                STAKE_SKIP_EDGE_NOT_ABOVE_VIG,
            )

    multiplier = max(0.0, _finite(kelly_multiplier))
    cap = max(0.0, _finite(max_stake_pct))
    stake_pct = max(0.0, min(multiplier * full_kelly, cap))
    return StakeDecision(
        stake_pct, full_kelly, expected_value, model_edge, raw_implied, ""
    )


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
