#!/usr/bin/env python3
"""Chronological staking-policy comparison using verified real entry odds only."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd

from backfill_finance import backfill_rows
from bet_log_utils import read_bet_rows
from finance import normalize_result, recommend_stake, valid_decimal_odds, valid_probability
from roi_analysis import maximum_drawdown


HERE = Path(__file__).resolve().parent
DEFAULT_BETS = HERE / "bet_log.csv"
DEFAULT_ODDS = HERE / "odds_tracker.csv"
DEFAULT_REPORTS = HERE / "reports"
MIN_CONCLUSIVE_REAL_ODDS = 200

POLICIES = {
    "flat_1pct": {"kind": "flat", "stake_pct": 0.01},
    "quarter_kelly_cap_1pct": {"kind": "kelly", "multiplier": 0.25, "cap": 0.01},
    "quarter_kelly_cap_2pct": {"kind": "kelly", "multiplier": 0.25, "cap": 0.02},
    "quarter_kelly_cap_3pct": {"kind": "kelly", "multiplier": 0.25, "cap": 0.03},
    "quarter_kelly_cap_5pct": {"kind": "kelly", "multiplier": 0.25, "cap": 0.05},
    "fractional_035_cap_5pct_comparison": {"kind": "kelly", "multiplier": 0.35, "cap": 0.05},
}


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def verified_rows(bets: list[dict], odds: pd.DataFrame) -> tuple[list[dict], dict]:
    backfilled, backfill_report = backfill_rows(bets, odds)
    rows = []
    for row in backfilled:
        price = _number(row.get("entry_decimal_odds"))
        probability = _number(row.get("p_model"))
        result = normalize_result(row.get("result"))
        if not valid_decimal_odds(price) or not valid_probability(probability):
            continue
        if result not in {"HIT", "MISS", "PUSH", "VOID"}:
            continue
        rows.append({**row, "entry_decimal_odds": price, "p_model": probability, "result": result})
    return rows, backfill_report


def chronological_splits(rows: list[dict]) -> dict[str, list[dict]]:
    dates = sorted({str(row.get("date"))[:10] for row in rows})
    if len(dates) < 3:
        return {"train": rows, "validation": [], "test": []}
    train_end = max(1, int(len(dates) * 0.60))
    validation_end = max(train_end + 1, int(len(dates) * 0.80))
    validation_end = min(validation_end, len(dates) - 1)
    buckets = {
        "train": set(dates[:train_end]),
        "validation": set(dates[train_end:validation_end]),
        "test": set(dates[validation_end:]),
    }
    return {
        name: [row for row in rows if str(row.get("date"))[:10] in allowed]
        for name, allowed in buckets.items()
    }


def _gate(row: dict, multiplier: float = 0.25, cap: float = 0.05):
    return recommend_stake(
        row.get("p_model"),
        row.get("entry_decimal_odds"),
        devig_market_probability=row.get("p_market"),
        vig_pct=row.get("vig_pct"),
        min_edge=0.02,
        kelly_multiplier=multiplier,
        max_stake_pct=cap,
        # Historical consensus rows lack timestamps. They are admitted here
        # only because exact-line stored prices were independently verified;
        # the live policy still requires timestamp freshness.
        data_is_fresh=True,
    )


def _worst_losing_streak(records: list[dict]) -> int:
    worst = current = 0
    for record in records:
        if record["result"] == "MISS":
            current += 1
            worst = max(worst, current)
        elif record["result"] == "HIT":
            current = 0
    return worst


def _breakdown(records: list[dict], fields: list[str]) -> list[dict]:
    groups = {}
    for record in records:
        key = tuple(str(record.get(field) or "UNKNOWN") for field in fields)
        groups.setdefault(key, []).append(record)
    output = []
    for key, group in sorted(groups.items()):
        staked = sum(record["stake_amount"] for record in group)
        profit = sum(record["profit"] for record in group)
        hits = sum(record["result"] == "HIT" for record in group)
        decisions = sum(record["result"] in {"HIT", "MISS"} for record in group)
        equity = [100.0]
        for record in group:
            equity.append(equity[-1] + record["profit"])
        drawdown, drawdown_pct = maximum_drawdown(equity)
        stakes = [record["stake_amount"] for record in group]
        returns = [record["return_on_opening_bankroll"] for record in group]
        output.append({
            **dict(zip(fields, key)),
            "n": len(group),
            "total_staked": staked,
            "profit_units": profit,
            "roi": profit / staked if staked else None,
            "hit_rate": hits / decisions if decisions else None,
            "maximum_drawdown_units": drawdown,
            "maximum_drawdown_pct": drawdown_pct,
            "volatility_of_bankroll_returns": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
            "worst_losing_streak": _worst_losing_streak(group),
            "median_stake": statistics.median(stakes) if stakes else None,
            "largest_stake": max(stakes) if stakes else None,
        })
    return output


def simulate_policy(rows: list[dict], policy: dict, starting_bankroll: float = 100.0) -> dict:
    by_day = {}
    skip_reasons = Counter()
    for row in rows:
        by_day.setdefault(str(row.get("date"))[:10], []).append(row)

    bankroll = starting_bankroll
    equity = [bankroll]
    records = []
    for day in sorted(by_day):
        day_open = bankroll
        day_profit = 0.0
        for row in by_day[day]:
            multiplier = policy.get("multiplier", 0.25)
            cap = policy.get("cap", 0.05)
            decision = _gate(row, multiplier=multiplier, cap=cap)
            if decision.stake_skip_reason:
                skip_reasons[decision.stake_skip_reason] += 1
                continue
            stake_pct = (
                policy["stake_pct"] if policy["kind"] == "flat"
                else decision.stake_pct
            )
            if stake_pct <= 0:
                skip_reasons["zero_policy_stake"] += 1
                continue
            stake = day_open * stake_pct
            price = float(row["entry_decimal_odds"])
            result = row["result"]
            profit = stake * (price - 1.0) if result == "HIT" else -stake if result == "MISS" else 0.0
            day_profit += profit
            records.append({
                "date": day,
                "market": row.get("market"),
                "pick": row.get("pick"),
                "result": result,
                "stake_pct": stake_pct,
                "stake_amount": stake,
                "profit": profit,
                "return_on_opening_bankroll": profit / day_open if day_open else 0.0,
            })
        bankroll = day_open + day_profit
        equity.append(bankroll)

    staked = sum(record["stake_amount"] for record in records)
    profit = sum(record["profit"] for record in records)
    stakes = [record["stake_amount"] for record in records]
    returns = [record["return_on_opening_bankroll"] for record in records]
    drawdown, drawdown_pct = maximum_drawdown(equity)
    return {
        "available_real_odds_rows": len(rows),
        "sample_size": len(records),
        "total_staked": staked,
        "total_profit_units": profit,
        "roi": profit / staked if staked else None,
        "ending_bankroll": bankroll,
        "maximum_drawdown_units": drawdown,
        "maximum_drawdown_pct": drawdown_pct,
        "volatility_of_bankroll_returns": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
        "worst_losing_streak": _worst_losing_streak(records),
        "median_stake": statistics.median(stakes) if stakes else None,
        "largest_stake": max(stakes) if stakes else None,
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "by_market": _breakdown(records, ["market"]),
        "by_direction": _breakdown(records, ["pick"]),
    }


def build_report(rows: list[dict], backfill_report: dict) -> dict:
    splits = chronological_splits(rows)
    split_report = {}
    for split, split_rows in splits.items():
        dates = sorted({str(row.get("date"))[:10] for row in split_rows})
        split_report[split] = {
            "date_start": dates[0] if dates else None,
            "date_end": dates[-1] if dates else None,
            "real_odds_rows": len(split_rows),
            "policies": {
                name: simulate_policy(split_rows, policy)
                for name, policy in POLICIES.items()
            },
        }
    overall = {name: simulate_policy(rows, policy) for name, policy in POLICIES.items()}

    validation = split_report["validation"]["policies"]
    quarter_candidates = {
        name: metrics for name, metrics in validation.items()
        if name.startswith("quarter_kelly") and metrics["roi"] is not None
    }
    validation_best = max(
        quarter_candidates,
        key=lambda name: quarter_candidates[name]["roi"],
        default=None,
    )
    conclusive = (
        len(rows) >= MIN_CONCLUSIVE_REAL_ODDS
        and split_report["validation"]["real_odds_rows"] >= 50
        and split_report["test"]["real_odds_rows"] >= 50
    )
    conclusion = (
        "conclusive" if conclusive else
        f"inconclusive: {len(rows)} verified real-odds rows is below the "
        f"{MIN_CONCLUSIVE_REAL_ODDS}-row minimum and chronological holdouts are thin"
    )
    return {
        "real_odds_sample_size": len(rows),
        "source_backfill": backfill_report,
        "split_method": "whole dates in chronological 60/20/20 order",
        "minimum_conclusive_real_odds_rows": MIN_CONCLUSIVE_REAL_ODDS,
        "evidence_status": conclusion,
        "validation_best_quarter_kelly_policy": validation_best,
        "production_policy": {
            "kelly_multiplier": 0.25,
            "maximum_stake_pct": 0.05,
            "decision": "unchanged because the real-odds policy comparison is inconclusive",
        },
        "splits": split_report,
        "overall_diagnostic": overall,
    }


def report_markdown(report: dict) -> str:
    lines = [
        "# Chronological staking-policy backtest", "",
        f"- Verified real-odds rows: {report['real_odds_sample_size']}",
        f"- Split: {report['split_method']}",
        f"- Evidence: {report['evidence_status']}",
        "- Production Kelly: 0.25 with a 5% cap (unchanged)",
        "- No missing prices were filled and no assumed -110 odds were used.", "",
    ]
    for split, detail in report["splits"].items():
        lines.extend([
            f"## {split.title()} ({detail['date_start']} to {detail['date_end']})", "",
            "| Policy | Real-odds rows | Bets | Profit | ROI | Max DD | Volatility | Median stake | Largest stake | Losing streak |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name, metrics in detail["policies"].items():
            roi = "n/a" if metrics["roi"] is None else f"{metrics['roi'] * 100:.2f}%"
            median = "n/a" if metrics["median_stake"] is None else f"{metrics['median_stake']:.4f}"
            largest = "n/a" if metrics["largest_stake"] is None else f"{metrics['largest_stake']:.4f}"
            lines.append(
                f"| {name} | {metrics['available_real_odds_rows']} | {metrics['sample_size']} | "
                f"{metrics['total_profit_units']:+.4f} | {roi} | "
                f"{metrics['maximum_drawdown_units']:.4f} | "
                f"{metrics['volatility_of_bankroll_returns']:.4f} | {median} | {largest} | "
                f"{metrics['worst_losing_streak']} |"
            )
        lines.append("")
    production = report["overall_diagnostic"]["quarter_kelly_cap_5pct"]
    for title, key, field in (
        ("Production-policy result by market", "by_market", "market"),
        ("Production-policy result by direction", "by_direction", "pick"),
    ):
        lines.extend([
            f"## {title}", "",
            "| Group | Bets | Profit | ROI | Max DD |", "|---|---:|---:|---:|---:|",
        ])
        for row in production[key]:
            roi = "n/a" if row["roi"] is None else f"{row['roi'] * 100:.2f}%"
            lines.append(
                f"| {row[field]} | {row['n']} | {row['profit_units']:+.4f} | {roi} | "
                f"{row['maximum_drawdown_units']:.4f} |"
            )
        lines.append("")
    lines.extend([
        "## Limits", "",
        "The verified prices cover a small, playoff-heavy period. The validation and test "
        "samples are too small to support a profitability or cap-selection conclusion. "
        "Results are implementation diagnostics, not evidence of a profitable strategy.", "",
    ])
    return "\n".join(lines)


def run(bet_path: Path = DEFAULT_BETS, odds_path: Path = DEFAULT_ODDS,
        reports_dir: Path = DEFAULT_REPORTS) -> dict:
    _, bets = read_bet_rows(bet_path)
    rows, backfill_report = verified_rows(bets, pd.read_csv(odds_path))
    report = build_report(rows, backfill_report)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "staking_policy_backtest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports_dir / "staking_policy_backtest.md").write_text(
        report_markdown(report) + "\n", encoding="utf-8"
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bets", type=Path, default=DEFAULT_BETS)
    parser.add_argument("--odds", type=Path, default=DEFAULT_ODDS)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args(argv)
    report = run(args.bets, args.odds, args.reports_dir)
    print(report_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
