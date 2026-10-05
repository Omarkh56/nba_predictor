"""Chronological real-odds staking comparisons."""

import pytest

from staking_backtest import (
    POLICIES,
    build_report,
    chronological_splits,
    simulate_policy,
)


def row(day="2026-05-01", result="HIT", **updates):
    value = {
        "date": day, "market": "PTS", "pick": "OVER", "result": result,
        "p_model": 0.60, "p_market": 0.50, "vig_pct": 4.0,
        "entry_decimal_odds": 2.0,
        "odds_source": "legacy_odds_tracker_exact_line_consensus",
    }
    value.update(updates)
    return value


def test_same_day_policy_stakes_use_opening_bankroll():
    metrics = simulate_policy(
        [row(result="HIT"), row(result="MISS")],
        POLICIES["quarter_kelly_cap_5pct"],
        starting_bankroll=100.0,
    )
    assert metrics["sample_size"] == 2
    assert metrics["total_staked"] == pytest.approx(10.0)
    assert metrics["total_profit_units"] == pytest.approx(0.0)
    assert metrics["ending_bankroll"] == pytest.approx(100.0)
    assert metrics["largest_stake"] == pytest.approx(5.0)


def test_flat_policy_uses_one_percent_after_same_finance_gates():
    metrics = simulate_policy([row()], POLICIES["flat_1pct"], starting_bankroll=100.0)
    assert metrics["sample_size"] == 1
    assert metrics["total_staked"] == pytest.approx(1.0)
    assert metrics["total_profit_units"] == pytest.approx(1.0)


def test_edge_not_above_vig_skips_every_policy():
    metrics = simulate_policy(
        [row(p_model=0.54, p_market=0.51, vig_pct=4.0)],
        POLICIES["flat_1pct"],
    )
    assert metrics["sample_size"] == 0
    assert metrics["skip_reasons"] == {"edge_not_above_market_vig": 1}


def test_chronological_split_uses_disjoint_whole_dates():
    rows = [row(day=f"2026-05-{day:02d}") for day in range(1, 11)]
    splits = chronological_splits(rows)
    dates = {name: {item["date"] for item in values} for name, values in splits.items()}
    assert dates["train"] == {f"2026-05-{day:02d}" for day in range(1, 7)}
    assert dates["validation"] == {"2026-05-07", "2026-05-08"}
    assert dates["test"] == {"2026-05-09", "2026-05-10"}
    assert dates["train"].isdisjoint(dates["validation"] | dates["test"])


def test_small_real_odds_sample_is_reported_inconclusive():
    rows = [row(day=f"2026-05-{day:02d}") for day in range(1, 11)]
    report = build_report(rows, {"rows_with_entry_odds_recovered": len(rows)})
    assert report["real_odds_sample_size"] == 10
    assert report["evidence_status"].startswith("inconclusive")
    assert report["production_policy"]["kelly_multiplier"] == pytest.approx(0.25)
    assert report["production_policy"]["maximum_stake_pct"] == pytest.approx(0.05)
