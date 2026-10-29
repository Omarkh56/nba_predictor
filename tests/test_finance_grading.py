"""Profit, bankroll, ROI, and drawdown behavior for recorded real bets."""

import pandas as pd
import pytest

from finance import grade_financial_rows, settlement_profit
from roi_analysis import build_finance_summary, maximum_drawdown


@pytest.mark.parametrize(
    "result,expected",
    [("HIT", 9.0), ("MISS", -10.0), ("PUSH", 0.0), ("VOID", 0.0)],
)
def test_settlement_profit_for_all_settled_outcomes(result, expected):
    assert settlement_profit(result, 10.0, 1.9) == pytest.approx(expected)


def test_settlement_profit_requires_real_odds_and_stake():
    assert settlement_profit("HIT", None, 1.9) is None
    assert settlement_profit("HIT", 10.0, None) is None
    assert settlement_profit("HIT", 10.0, 1.0) is None


def test_same_day_bets_use_start_of_day_bankroll():
    rows = [
        {"date": "2026-05-01", "result": "HIT", "stake_pct": 0.10,
         "decimal_odds": 2.0},
        {"date": "2026-05-01", "result": "MISS", "stake_pct": 0.10,
         "decimal_odds": 2.0},
        {"date": "2026-05-02", "result": "HIT", "stake_pct": 0.10,
         "decimal_odds": 2.0},
    ]
    graded, stats = grade_financial_rows(rows, starting_bankroll=100.0)
    assert graded[0]["bankroll_before"] == pytest.approx(100.0)
    assert graded[1]["bankroll_before"] == pytest.approx(100.0)
    assert graded[0]["stake_amount"] == pytest.approx(10.0)
    assert graded[1]["stake_amount"] == pytest.approx(10.0)
    assert graded[2]["bankroll_before"] == pytest.approx(100.0)
    assert stats.ending_bankroll == pytest.approx(110.0)


def test_repeated_finance_grading_is_idempotent():
    rows = [{"date": "2026-05-01", "result": "HIT", "stake_pct": 0.02,
             "entry_decimal_odds": 1.9}]
    first, _ = grade_financial_rows(rows, starting_bankroll=100.0)
    second, stats = grade_financial_rows(first, starting_bankroll=999.0)
    assert second == first
    assert stats.stake_amounts_added == 0
    assert stats.profits_added == 0
    assert stats.starting_bankroll == pytest.approx(100.0)


def test_roi_excludes_rows_without_recorded_stake_amount():
    bets = pd.DataFrame([
        {"date": "2026-05-01", "market": "PTS", "pick": "OVER", "result": "HIT",
         "entry_decimal_odds": 2.0, "stake_amount": 10.0, "realized_profit": 10.0,
         "bankroll_before": 100.0, "stake_pct": 0.10, "model_edge": 0.08},
        {"date": "2026-05-01", "market": "REB", "pick": "UNDER", "result": "HIT",
         "entry_decimal_odds": 2.0, "stake_amount": "", "realized_profit": ""},
    ])
    report = build_finance_summary(bets)
    assert report["finance_eligible_rows"] == 1
    assert report["finance_excluded_rows"] == 1
    assert report["overall"]["total_amount_staked"] == pytest.approx(10.0)
    assert report["overall"]["roi"] == pytest.approx(1.0)


def test_maximum_drawdown_uses_peak_to_trough_loss():
    amount, fraction = maximum_drawdown([100.0, 110.0, 90.0, 105.0])
    assert amount == pytest.approx(20.0)
    assert fraction == pytest.approx(20.0 / 110.0)
