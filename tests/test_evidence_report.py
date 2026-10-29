"""Evidence math, price provenance, exclusions and read-only paper reporting."""
import json
from datetime import date, timedelta

import pandas as pd
import pytest

import evidence_report as evidence
import staking_backtest as staking
from bet_log_utils import BET_LOG_FIELDS, write_bet_rows
from run_log import RunLog


def bet(player="P1", day="2026-10-01", **updates):
    return {"date": day, "player": player, "market": "PTS", "line": 25.5, "pick": "OVER",
            "result": "HIT", "entry_decimal_odds": 2.0, "odds_source": "the_odds_api",
            "odds_snapshot_at": "2026-10-01T18:00:00+00:00", "odds_event_id": "event",
            "logged_at": "2026-10-01T18:01:00+00:00", "run_id": "entry", "model_version": "v1",
            "p_model": .65, "p_market": .50, "vig_pct": 4.0, **updates}


def test_wilson_known_values_and_extreme_rates():
    assert evidence.wilson_interval(5, 10) == pytest.approx([.2365930905, .7634069095])
    assert evidence.wilson_interval(0, 10) == pytest.approx([0.0, .2775327999])
    assert evidence.wilson_interval(10, 10) == pytest.approx([.7224672001, 1.0])
    assert evidence.wilson_interval(0, 0) is None
    with pytest.raises(ValueError):
        evidence.wilson_interval(11, 10)


def test_actual_prices_determine_paper_roi_and_break_even():
    rows = [bet(day="2026-10-01", entry_decimal_odds=3.0),
            bet(player="P2", day="2026-10-02", result="MISS", entry_decimal_odds=1.5)]
    result = evidence.summarize(rows)
    assert result["win_rate"] == .5
    assert result["paper_roi"] == .5  # (+2 - 1)/2 paper units
    assert result["break_even_win_rate_at_actual_prices"] == pytest.approx(.5)  # mean(1/3, 1/1.5)
    assert result["paper_roi_bootstrap_95"] == pytest.approx([-1.0, 2.0])
    assert evidence.bootstrap_interval([("one-day", 1.0), ("one-day", -1.0)]) is None
    assert evidence.bootstrap_interval([("day1", 1.0), ("day2", 1.0)]) == [1.0, 1.0]


def test_clv_requires_joined_closing_snapshot_and_has_wilson_positive_rate():
    result = evidence.summarize([
        bet(closing_snapshot_at="2026-10-01T18:50:00Z", implied_probability_clv=.02, line_clv=1),
        bet(player="P2", day="2026-10-02", closing_snapshot_at="2026-10-02T18:50:00Z",
            implied_probability_clv=-.01, line_clv=-1),
        bet(player="P3", implied_probability_clv=.99),  # orphan numeric CLV is not evidence
    ])
    clv = result["clv"]["implied_probability_clv"]
    assert clv["rows"] == 2 and clv["mean"] == pytest.approx(.005)
    assert clv["positive_rate"] == .5
    assert clv["positive_rate_wilson_95"] == evidence.wilson_interval(1, 2)
    assert clv["mean_bootstrap_95"] == pytest.approx([-.01, .02])


def test_missing_entry_reasons_and_missing_closing_are_separate():
    runs = [
        {"run_id": "empty", "slate_date": "2026-10-01",
         "stages": [{"stage": "props:UTA@DEN", "status": "market_unavailable"}]},
        {"run_id": "failed", "slate_date": "2026-10-01",
         "stages": [{"stage": "odds_events", "status": "fetch_failed", "required": True}]},
        {"run_id": "mixed", "slate_date": "2026-10-01",
         "stages": [{"stage": "props:UTA@DEN", "status": "market_unavailable"},
                    {"stage": "props:GSW@LAC", "status": "fetch_failed"}]},
    ]
    bets = [bet(player="empty", entry_decimal_odds=None, run_id="empty"),
            bet(player="failed", entry_decimal_odds=None, run_id="failed"),
            bet(player="mixed", entry_decimal_odds=None, run_id="mixed"),
            bet(player="joined", clv_unmatched_reason="no_closing_snapshot")]
    report = evidence.build_report(bets, runs+runs, as_of=date(2026, 10, 5))
    assert report["missing_price_reasons"] == {"market_unavailable": 1, "fetch_failed": 1,
                                              "ambiguous_run_price_status": 1}
    assert report["no_closing_snapshot_rows"] == 1
    assert report["unavailable_prop_event_stages"] == {"market_unavailable": 2, "fetch_failed": 1}
    assert report["verified_rows_needed"] == 199
    assert evidence.missing_price_reason(bet(game="GSW @ LAC"), {"entry": runs[-1]}) == "fetch_failed"


def test_numeric_legacy_alias_or_unsourced_entry_is_not_verified():
    assert not evidence.verified_price(bet(entry_decimal_odds=None, decimal_odds=1.909))
    assert not evidence.verified_price(bet(odds_source=None))
    assert not evidence.verified_price(bet(odds_snapshot_at="invalid"))
    assert evidence.verified_price(bet(odds_source="legacy_odds_tracker_exact_line_consensus",
                                       odds_snapshot_at=None, odds_event_id=None))


def test_weekly_window_versions_groups_and_deduplication():
    rows = [bet(day="2026-09-28"), bet(player="P2", day="2026-10-01", pick="UNDER", model_version="v2"),
            bet(player="P3", day="2026-10-06"), bet(player="P4", day="2026-10-03", market="REB")]
    report = evidence.build_report(rows+[rows[0]], [], as_of=date(2026, 10, 5))
    assert report["unique_bet_rows"] == 3
    assert report["weekly"]["overall"]["verified_price_rows"] == 2
    assert report["model_version_counts"] == {"v1": 2, "v2": 1}
    assert {r["market"] for r in report["by_market"]} == {"PTS", "REB"}
    assert {r["pick"] for r in report["by_direction"]} == {"OVER", "UNDER"}


def test_cli_writes_only_reports_and_is_deterministic(tmp_path):
    bets = tmp_path/"bet_log.csv"
    rows = [bet(), bet(player="P2", day="2026-10-02", result="MISS")]
    write_bet_rows(bets, list(dict.fromkeys([*BET_LOG_FIELDS, *rows[0]])), rows)
    runs = tmp_path/"runs.jsonl"
    log = RunLog("predict", "2026-10-01", path=runs, run_id="entry")
    log.stage("schedule", "ok", required=True)
    log.finish()
    original_bets, original_runs = bets.read_bytes(), runs.read_bytes()
    out = tmp_path/"reports"
    args = ["--bets", str(bets), "--runs", str(runs), "--reports-dir", str(out), "--as-of", "2026-10-05"]
    assert evidence.main(args) == 0
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert evidence.main(args) == 0
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}
    assert bets.read_bytes() == original_bets and runs.read_bytes() == original_runs
    report = json.loads((out/"evidence_weekly.json").read_text())
    assert report["paper_only"] is True and report["verified_rows_needed"] == 198
    assert "Wilson 95%" in (out/"evidence_weekly.md").read_text()


@pytest.mark.parametrize("count", [0, 199, 200])
def test_evidence_threshold_reports_exact_remaining_rows(count):
    rows = [bet(player=f"P{n}") for n in range(count)]
    report = evidence.build_report(rows, [], as_of=date(2026, 10, 5))
    assert report["verified_rows_needed"] == max(0, 200-count)
    assert report["evidence_status"] == ("inconclusive" if count < 200 else "minimum reached; reporting only")


@pytest.mark.parametrize("count,passed", [(199, False), (200, False), (250, True)])
def test_staking_recommendation_requires_total_and_both_holdouts(count, passed):
    start = date(2025, 1, 1)
    rows = [bet(player=f"P{n}", day=(start+timedelta(days=n)).isoformat()) for n in range(count)]
    report = staking.build_report(rows, {})
    assert report["recommendation_gate_passed"] is passed
    assert (report["validation_best_quarter_kelly_policy"] is not None) is passed
    assert report["verified_rows_needed"] == max(0, 200-count)
    assert report["production_policy"]["kelly_multiplier"] == .25
    assert report["production_policy"]["maximum_stake_pct"] == .05
    if count == 200:
        assert "below" not in report["evidence_status"]  # holdouts, not total, failed


def test_staking_preserves_frozen_price_and_excludes_unverified_explicit_odds():
    original = bet(entry_decimal_odds=3.0)
    quotes = pd.DataFrame([{"date": "2026-10-01", "player": "P1", "market": "PTS", "line": 25.5,
                            "avg_over_odds": -110, "avg_under_odds": -110,
                            "devig_over": .5, "devig_under": .5, "vig_pct": 4.0}])
    rows, _ = staking.verified_rows([original], quotes)
    assert rows[0]["entry_decimal_odds"] == 3.0
    assert staking.verified_rows([bet(odds_source=None)], pd.DataFrame())[0] == []
    report = staking.build_report([bet(player=f"P{n}", odds_source=None) for n in range(250)], {})
    assert report["real_odds_sample_size"] == 0
    assert report["validation_best_quarter_kelly_policy"] is None
