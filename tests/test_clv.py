"""Closing-line joins reject weak or post-start matches."""

import pandas as pd
import pytest

from clv import closing_for_bet


START = "2026-05-01T20:00:00+00:00"


def bet(**updates):
    row = {
        "date": "2026-05-01",
        "player": "Nikola Jokic",
        "market": "PTS",
        "pick": "OVER",
        "line": 25.5,
        "odds_event_id": "event-1",
        "game_start_at": START,
        "entry_decimal_odds": 2.0,
        "market_implied_probability": 0.5,
        "p_market": 0.52,
        "odds_book": "",
    }
    row.update(updates)
    return row


def quote(snapshot_at, *, event="event-1", line=25.5, side="OVER", price=-120,
          devig=0.58, book="Book A"):
    return {
        "snapshot_at": snapshot_at,
        "date": "2026-05-01",
        "event_id": event,
        "player": "Nikola Jokić",
        "market": "PTS",
        "line": line,
        "side": side,
        "american_odds": price,
        "devig_probability": devig,
        "book_key": book.lower().replace(" ", ""),
        "book_title": book,
    }


def test_uses_final_valid_pregame_snapshot_and_computes_separate_clv():
    snapshots = pd.DataFrame([
        quote("2026-05-01T18:00:00+00:00", price=-110, devig=0.54),
        quote("2026-05-01T19:45:00+00:00", line=26.5, book="Book A"),
        quote("2026-05-01T19:45:00+00:00", line=26.5, book="Book B"),
        quote("2026-05-01T19:45:00+00:00", line=25.5, price=-120, devig=0.58,
              book="Book C"),
        quote("2026-05-01T20:01:00+00:00", line=27.5, price=-150, devig=0.65),
    ])
    result = closing_for_bet(bet(), snapshots)
    assert result["closing_snapshot_at"].startswith("2026-05-01T19:45:00")
    assert result["closing_line"] == pytest.approx(26.5)
    assert result["line_clv"] == pytest.approx(1.0)
    assert result["implied_probability_clv"] == pytest.approx(0.06)
    assert result["raw_implied_probability_clv"] == pytest.approx(0.54545455 - 0.5)
    assert result["clv_unmatched_reason"] == ""


def test_under_line_clv_has_opposite_direction():
    snapshots = pd.DataFrame([
        quote("2026-05-01T19:00:00+00:00", line=24.5, side="UNDER", devig=0.55)
    ])
    result = closing_for_bet(bet(pick="UNDER", p_market=0.50), snapshots)
    assert result["line_clv"] == pytest.approx(1.0)


def test_post_start_snapshot_is_rejected():
    snapshots = pd.DataFrame([quote("2026-05-01T20:01:00+00:00")])
    result = closing_for_bet(bet(), snapshots)
    assert result["closing_line"] == ""
    assert result["clv_unmatched_reason"] == "snapshot_after_game_start"


def test_ambiguous_event_without_entry_event_id_is_rejected():
    snapshots = pd.DataFrame([
        quote("2026-05-01T19:00:00+00:00", event="event-1"),
        quote("2026-05-01T19:00:00+00:00", event="event-2"),
    ])
    result = closing_for_bet(bet(odds_event_id=""), snapshots)
    assert result["clv_unmatched_reason"] == "ambiguous_event_player_match"


def test_different_line_keeps_directional_clv_but_not_price_clv():
    snapshots = pd.DataFrame([
        quote("2026-05-01T19:00:00+00:00", line=26.5),
    ])
    result = closing_for_bet(bet(), snapshots)
    assert result["line_clv"] == pytest.approx(1.0)
    assert result["closing_decimal_odds"] == ""
    assert result["clv_unmatched_reason"] == "different_line_with_no_same_line_comparison"


def test_missing_paired_side_preserves_raw_price_clv_only():
    snapshots = pd.DataFrame([
        quote("2026-05-01T19:00:00+00:00", devig=None),
    ])
    result = closing_for_bet(bet(), snapshots)
    assert result["raw_implied_probability_clv"] != ""
    assert result["implied_probability_clv"] == ""
    assert result["clv_unmatched_reason"] == "missing_opposing_side"


def test_closing_price_averages_books_through_probability_not_american_odds():
    # Two books either side of even money: a plain mean of -110 and +104 is -3, not a price.
    frame = pd.DataFrame([quote("2026-05-01T19:50:00+00:00", price=-110, book="Book A"),
                          quote("2026-05-01T19:50:00+00:00", price=104, book="Book B")])
    result = closing_for_bet(bet(), frame)
    implied = (110 / 210 + 100 / 204) / 2
    assert result["closing_decimal_odds"] == pytest.approx(1 / implied, abs=1e-6)
