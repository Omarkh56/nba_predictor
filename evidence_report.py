#!/usr/bin/env python3
"""Weekly paper evidence from recorded entries and run status; no price imputation."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np

from bet_log_utils import canonical_bet_key, read_bet_rows
from finance import normalize_result, valid_decimal_odds, verified_entry_price
from run_log import read_runs
from staking_backtest import MIN_CONCLUSIVE_REAL_ODDS
from team_codes import today_slate_date

HERE = Path(__file__).resolve().parent


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def wilson_interval(successes: int, trials: int) -> list[float] | None:
    """Two-sided 95% Wilson score interval for a binary proportion."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    if trials == 0:
        return None
    z = NormalDist().inv_cdf(.975)
    p = successes / trials
    denominator = 1 + z*z/trials
    center = (p + z*z/(2*trials)) / denominator
    radius = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / denominator
    return [max(0.0, center-radius), min(1.0, center+radius)]


def verified_price(row: dict) -> bool:
    """Accept explicit sourced entry prices, never a legacy decimal alias alone."""
    return verified_entry_price(row)


def bootstrap_interval(pairs: list[tuple[str, float]], draws=2000) -> list[float] | None:
    """Percentile interval for a mean, resampling whole dates with fixed seed.

    Dates retain all correlated bets and their original payout/CLV values.
    One date cannot supply an independent-date uncertainty estimate.
    """
    grouped = defaultdict(list)
    for day, value in pairs:
        grouped[day].append(value)
    if len(grouped) < 2:
        return None
    days = sorted(grouped)
    sums = np.array([sum(grouped[d]) for d in days])
    counts = np.array([len(grouped[d]) for d in days])
    rng = np.random.default_rng(20261005)
    means = []
    for _ in range(draws):
        sample = rng.integers(0, len(days), size=len(days))
        means.append(float(sums[sample].sum()/counts[sample].sum()))
    return [float(v) for v in np.quantile(means, [.025, .975])]


def summarize(rows: list[dict]) -> dict:
    decisions = [r for r in rows if normalize_result(r.get("result")) in {"HIT", "MISS"}]
    hits = sum(normalize_result(r.get("result")) == "HIT" for r in decisions)
    paper = []
    for row in rows:
        result = normalize_result(row.get("result"))
        if result in {"HIT", "MISS", "PUSH"}:
            profit = float(row["entry_decimal_odds"])-1 if result == "HIT" else -1 if result == "MISS" else 0
            paper.append((str(row.get("date"))[:10], profit))
    clv = {}
    for field in ("line_clv", "implied_probability_clv", "raw_implied_probability_clv"):
        values = [(str(r.get("date"))[:10], number(r.get(field))) for r in rows
                  if r.get("closing_snapshot_at")]
        values = [(d, v) for d, v in values if v is not None]
        positive = sum(v > 0 for _, v in values)
        clv[field] = {"rows": len(values), "mean": sum(v for _, v in values)/len(values) if values else None,
                      "mean_bootstrap_95": bootstrap_interval(values),
                      "positive_rate": positive/len(values) if values else None,
                      "positive_rate_wilson_95": wilson_interval(positive, len(values))}
    return {
        "verified_price_rows": len(rows), "decisions": len(decisions), "hits": hits,
        "misses": len(decisions)-hits, "win_rate": hits/len(decisions) if decisions else None,
        "win_rate_wilson_95": wilson_interval(hits, len(decisions)),
        "paper_roi": sum(v for _, v in paper)/len(paper) if paper else None,
        "paper_roi_bootstrap_95": bootstrap_interval(paper), "paper_units_staked": len(paper),
        "break_even_win_rate_at_actual_prices": (
            sum(1/float(r["entry_decimal_odds"]) for r in decisions)/len(decisions) if decisions else None),
        "clv": clv,
    }


def group_summaries(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(f) or "UNKNOWN").strip().upper() for f in fields)
        groups[key].append(row)
    return [{**dict(zip(fields, key)), **summarize(values)} for key, values in sorted(groups.items())]


def missing_price_reason(row: dict, runs: dict) -> str:
    explicit = str(row.get("price_unmatched_reason") or "")
    if explicit in {"market_unavailable", "fetch_failed"}:
        return explicit
    run = runs.get(str(row.get("run_id") or ""), {})
    stages = run.get("stages", [])
    if any((s.get("required") or s.get("stage") in {"odds_events", "schedule"})
           and s.get("status") == "fetch_failed" for s in stages):
        return "fetch_failed"
    game = str(row.get("game") or "").replace(" ", "")
    relevant = [s for s in stages if str(s.get("stage", "")).startswith("props:")
                and (not game or s["stage"] == "props:"+game)]
    statuses = {s.get("status") for s in relevant if s.get("status") in {"market_unavailable", "fetch_failed"}}
    if len(statuses) == 1:
        return statuses.pop()
    if len(statuses) > 1:
        return "ambiguous_run_price_status"
    return "unverified_price" if valid_decimal_odds(row.get("entry_decimal_odds")) else "unknown_price_reason"


def build_report(bets: list[dict], runs: list[dict], *, as_of: date | None = None) -> dict:
    as_of = as_of or today_slate_date()
    week_start = as_of-timedelta(days=6)
    # Count identities, not duplicate CSV rows or repeated run-log deliveries.
    unique = {}
    invalid = 0
    for row in bets:
        try:
            key = canonical_bet_key(row)
        except ValueError:
            invalid += 1
            continue
        if key[0] <= as_of.isoformat():
            unique.setdefault(key, row)
    rows = list(unique.values())
    run_map = {str(r.get("run_id")): r for r in runs if r.get("run_id")
               and (not r.get("slate_date") or str(r["slate_date"]) <= as_of.isoformat())}
    verified = [r for r in rows if verified_price(r)]
    settled = [r for r in verified if normalize_result(r.get("result")) in {"HIT", "MISS", "PUSH", "VOID"}]
    weekly = [r for r in settled if week_start.isoformat() <= str(r.get("date"))[:10]]
    missing = []
    for row in rows:
        if not verified_price(row):
            missing.append({"date": row.get("date"), "player": row.get("player"), "market": row.get("market"),
                            "run_id": row.get("run_id"), "reason": missing_price_reason(row, run_map)})
    reasons = Counter(r["reason"] for r in missing)
    closing_missing = sum(r.get("clv_unmatched_reason") == "no_closing_snapshot" for r in rows)
    stage_counts = Counter()
    for run in run_map.values():
        for stage in run.get("stages", []):
            if str(stage.get("stage", "")).startswith("props:") and stage.get("status") in {"market_unavailable", "fetch_failed"}:
                stage_counts[stage["status"]] += 1
    shortfall = max(0, MIN_CONCLUSIVE_REAL_ODDS-len(settled))
    return {
        "as_of": as_of.isoformat(), "week_start": week_start.isoformat(), "paper_only": True,
        "method": "one paper unit per sourced entry; actual payout prices; no assumed odds",
        "interval_methods": {"win_rate": "Wilson 95%", "positive_clv_rate": "Wilson 95%",
                             "roi_and_mean_clv": "2000-draw whole-date percentile bootstrap 95%, fixed seed"},
        "unique_bet_rows": len(rows), "invalid_identity_rows_excluded": invalid,
        "verified_price_rows": len(verified), "verified_settled_rows": len(settled),
        "minimum_verified_rows": MIN_CONCLUSIVE_REAL_ODDS, "verified_rows_needed": shortfall,
        "evidence_status": "minimum reached; reporting only" if not shortfall else "inconclusive",
        "prospective_verified_rows": sum(bool(r.get("run_id") and r.get("logged_at")) for r in settled),
        "model_version_counts": dict(sorted(Counter(str(r.get("model_version") or "UNKNOWN") for r in rows).items())),
        "verified_model_version_counts": dict(sorted(Counter(str(r.get("model_version") or "UNKNOWN") for r in settled).items())),
        "overall": summarize(settled), "by_market": group_summaries(settled, ("market",)),
        "by_direction": group_summaries(settled, ("pick",)),
        "by_market_direction": group_summaries(settled, ("market", "pick")),
        "weekly": {"overall": summarize(weekly), "by_market_direction": group_summaries(weekly, ("market", "pick"))},
        "missing_price_rows": missing, "missing_price_reasons": dict(sorted(reasons.items())),
        "no_closing_snapshot_rows": closing_missing,
        "unavailable_prop_event_stages": dict(sorted(stage_counts.items())),
        "limits": ["Wilson intervals assume independent binary trials; same-game bets can be correlated.",
                   "Bootstrap intervals need at least two dates and are descriptive with small samples.",
                   "Break-even is the mean of each actual price's 1/decimal_odds threshold.",
                   "Missing closing quotes can overlap missing entry prices; event-stage counts are not pick counts.",
                   "Historical exact-line prices are separate from prospective run provenance; no production stakes change."],
    }


def markdown(report: dict) -> str:
    def fmt(value):
        return "n/a" if value is None else f"{value:.4f}"
    def ci(values):
        return "n/a" if values is None else "["+", ".join(fmt(v) for v in values)+"]"
    lines = ["# Weekly paper evidence", "", f"As of {report['as_of']} (Eastern slate dates).",
             f"Week: {report['week_start']} through {report['as_of']}.", "",
             f"Verified settled rows: {report['verified_settled_rows']}; prospective: {report['prospective_verified_rows']}.",
             f"Evidence: {report['evidence_status']}; {report['verified_rows_needed']} more verified rows needed for {report['minimum_verified_rows']}.",
             "Paper only: one unit per priced pick. No policy or production stake changes.", ""]
    for title, records in [("Cumulative by market and direction", report["by_market_direction"]),
                           ("This week by market and direction", report["weekly"]["by_market_direction"])]:
        lines += ["## "+title, ""]
        for r in records:
            lines += [f"### {r['market']} {r['pick']} ({r['verified_price_rows']} priced rows)",
                      f"- Win rate: {fmt(r['win_rate'])}; Wilson 95%: {ci(r['win_rate_wilson_95'])}",
                      f"- Paper ROI: {fmt(r['paper_roi'])}; date-bootstrap 95%: {ci(r['paper_roi_bootstrap_95'])}",
                      f"- Break-even at actual prices: {fmt(r['break_even_win_rate_at_actual_prices'])}"]
            for field, m in r["clv"].items():
                lines += [f"- {field}: mean {fmt(m['mean'])} ({m['rows']} rows); bootstrap 95%: {ci(m['mean_bootstrap_95'])}; "
                          f"positive rate {fmt(m['positive_rate'])}, Wilson 95%: {ci(m['positive_rate_wilson_95'])}"]
            lines.append("")
        if not records:
            lines += ["No verified settled prices in this period.", ""]
    lines += ["## Model versions", "", *[f"- {version}: {count} rows" for version, count in report["model_version_counts"].items()], "",
              "## Missing prices and closing quotes", "", *[f"- {reason}: {count} bets" for reason, count in report["missing_price_reasons"].items()],
              f"No closing snapshot: {report['no_closing_snapshot_rows']} bet rows.",
              "Unavailable prop events (event counts, not picks): "+json.dumps(report["unavailable_prop_event_stages"], sort_keys=True), ""]
    lines += ["## Limits", "", *["- "+s for s in report["limits"]], "",
              "[Wilson method reference (NIST)](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)", ""]
    return "\n".join(lines)


def run(bet_path=HERE/"bet_log.csv", run_path=HERE/"run_log.jsonl", reports_dir=HERE/"reports", *, as_of=None):
    _, bets = read_bet_rows(bet_path)
    report = build_report(bets, read_runs(run_path), as_of=as_of)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir/"evidence_weekly.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    (reports_dir/"evidence_weekly.md").write_text(markdown(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bets", type=Path, default=HERE/"bet_log.csv")
    parser.add_argument("--runs", type=Path, default=HERE/"run_log.jsonl")
    parser.add_argument("--reports-dir", type=Path, default=HERE/"reports")
    parser.add_argument("--as-of", type=date.fromisoformat, default=today_slate_date())
    args = parser.parse_args(argv)
    report = run(args.bets, args.runs, args.reports_dir, as_of=args.as_of)
    print(f"Paper evidence: {report['verified_settled_rows']} verified settled rows; "
          f"{report['verified_rows_needed']} more needed for the minimum.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
