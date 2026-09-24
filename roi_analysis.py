#!/usr/bin/env python3
"""Finance reporting for recorded entry odds, stakes, and settled profit.

The production report reads finance fields already captured in bet_log.csv.
It never substitutes -110 or uses a loose historical odds join. Older join
helpers remain available only for explicitly labeled diagnostics.

Usage:
    python3 roi_analysis.py
    python3 roi_analysis.py --summary
    python3 roi_analysis.py --bets path/to/bet_log.csv --reports-dir path/to/reports
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

from finance import american_to_decimal, normalize_result, valid_decimal_odds

_HERE = Path(__file__).resolve().parent
DEFAULT_BET_LOG = _HERE / "bet_log.csv"
DEFAULT_ODDS = _HERE / "odds_tracker.csv"
DEFAULT_OUT = _HERE / "roi_per_bet.csv"

JOIN_KEYS = ["date", "player", "market"]


def realized_profit(won: bool, decimal_odds: float) -> float:
    """Profit on a $1 stake."""
    return (decimal_odds - 1.0) if won else -1.0


# ---------------------------------------------------------------------------
# Load + join
# ---------------------------------------------------------------------------
def _normalize_pick(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _normalize_result(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def load_joined(
    bet_path: Path, odds_path: Path
) -> tuple[pd.DataFrame, dict]:
    """Left-join odds onto graded bets on (date, player, market).

    Returns (joined_graded_with_odds, stats_dict).
    """
    bets = pd.read_csv(bet_path)
    odds = pd.read_csv(odds_path)

    required_bets = {"date", "player", "market", "pick", "result"}
    required_odds = {"date", "player", "market", "avg_over_odds", "avg_under_odds"}
    missing_b = required_bets - set(bets.columns)
    missing_o = required_odds - set(odds.columns)
    if missing_b:
        raise SystemExit(f"bet_log missing columns: {sorted(missing_b)}")
    if missing_o:
        raise SystemExit(f"odds_tracker missing columns: {sorted(missing_o)}")

    bets = bets.copy()
    odds = odds.copy()
    bets["date"] = bets["date"].astype(str).str.strip()
    odds["date"] = odds["date"].astype(str).str.strip()
    bets["player"] = bets["player"].astype(str).str.strip()
    odds["player"] = odds["player"].astype(str).str.strip()
    bets["market"] = bets["market"].astype(str).str.strip()
    odds["market"] = odds["market"].astype(str).str.strip()
    bets["pick"] = _normalize_pick(bets["pick"])
    bets["result"] = _normalize_result(bets["result"])

    # Graded rows only
    graded = bets[bets["result"].isin(["HIT", "MISS"])].copy()

    # Odds side: one row per join key (tracker already unique; keep first if not)
    odds_cols = JOIN_KEYS + ["avg_over_odds", "avg_under_odds", "books_count", "vig_pct"]
    odds_cols = [c for c in odds_cols if c in odds.columns]
    odds_u = odds[odds_cols].drop_duplicates(subset=JOIN_KEYS, keep="first")

    merged = graded.merge(odds_u, on=JOIN_KEYS, how="left", indicator=True)
    matched = merged[merged["_merge"] == "both"].copy()
    unmatched = merged[merged["_merge"] == "left_only"]

    stats = {
        "bets_total": len(bets),
        "bets_graded": len(graded),
        "odds_rows": len(odds),
        "matched": len(matched),
        "unmatched_graded": len(unmatched),
        "match_rate": (len(matched) / len(graded)) if len(graded) else 0.0,
    }
    return matched.drop(columns=["_merge"]), stats


def attach_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Select the American price for the pick direction and compute profit."""
    out = df.copy()

    def _price(row) -> float:
        pick = row["pick"]
        if pick == "OVER":
            return row["avg_over_odds"]
        if pick == "UNDER":
            return row["avg_under_odds"]
        return float("nan")

    out["american_odds"] = out.apply(_price, axis=1)
    out["won"] = out["result"] == "HIT"

    decimals = []
    profits = []
    for _, row in out.iterrows():
        try:
            dec = american_to_decimal(row["american_odds"])
            profits.append(realized_profit(bool(row["won"]), dec))
            decimals.append(dec)
        except (ValueError, TypeError):
            decimals.append(float("nan"))
            profits.append(float("nan"))

    out["decimal_odds"] = decimals
    out["profit_per_1"] = profits
    # Drop rows that matched the join but still lack a usable price
    usable = out.dropna(subset=["profit_per_1", "decimal_odds"]).copy()
    return usable


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _roi_table(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Aggregate hit rate + ROI. group_cols=None → single overall row."""
    if group_cols is None:
        grouped = [("ALL", df)]
        records = []
        sub = df
        n = len(sub)
        hits = int(sub["won"].sum())
        profit = float(sub["profit_per_1"].sum())
        records.append(
            {
                "group": "ALL",
                "n": n,
                "hits": hits,
                "hit_rate": hits / n if n else float("nan"),
                "profit": profit,
                "roi": profit / n if n else float("nan"),
                "avg_decimal": float(sub["decimal_odds"].mean()) if n else float("nan"),
            }
        )
        return pd.DataFrame(records)

    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(sub)
        hits = int(sub["won"].sum())
        profit = float(sub["profit_per_1"].sum())
        rec = {col: val for col, val in zip(group_cols, keys)}
        rec.update(
            {
                "n": n,
                "hits": hits,
                "hit_rate": hits / n if n else float("nan"),
                "profit": profit,
                "roi": profit / n if n else float("nan"),
                "avg_decimal": float(sub["decimal_odds"].mean()) if n else float("nan"),
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def flag_mismatches(by_md: pd.DataFrame) -> pd.DataFrame:
    """Hit rate above/below 50% disagreeing with ROI sign."""
    df = by_md.copy()
    df["hr_positive"] = df["hit_rate"] > 0.5
    df["roi_positive"] = df["roi"] > 0
    df["mismatch"] = df["hr_positive"] != df["roi_positive"]
    # Label the kind of divergence
    def _label(r) -> str:
        if not r["mismatch"]:
            return ""
        if r["hr_positive"] and not r["roi_positive"]:
            return "HIT_RATE_UP_ROI_DOWN"
        return "HIT_RATE_DOWN_ROI_UP"

    df["flag"] = df.apply(_label, axis=1)
    return df[df["mismatch"]].copy()


def cumulative_roi(df: pd.DataFrame) -> pd.DataFrame:
    """Running cumulative profit and ROI ordered by date (then original order)."""
    ordered = df.sort_values(["date"]).reset_index(drop=True)
    ordered["bet_number"] = range(1, len(ordered) + 1)
    ordered["cum_profit"] = ordered["profit_per_1"].cumsum()
    ordered["cum_roi"] = ordered["cum_profit"] / ordered["bet_number"]
    ordered["cum_hit_rate"] = ordered["won"].cumsum() / ordered["bet_number"]
    return ordered


# ---------------------------------------------------------------------------
# Production finance reporting from the recorded bet log
# ---------------------------------------------------------------------------
def maximum_drawdown(equity_values) -> tuple[float, float | None]:
    """Return maximum peak-to-trough loss in units and as a peak fraction."""
    values = [float(v) for v in equity_values if v is not None and math.isfinite(float(v))]
    if not values:
        return 0.0, None
    peak = values[0]
    max_loss = 0.0
    max_fraction = 0.0 if peak > 0 else None
    for value in values:
        peak = max(peak, value)
        loss = peak - value
        max_loss = max(max_loss, loss)
        if peak > 0:
            fraction = loss / peak
            max_fraction = max(max_fraction or 0.0, fraction)
    return max_loss, max_fraction


def _number_series(df: pd.DataFrame, field: str) -> pd.Series:
    if field not in df:
        return pd.Series(float("nan"), index=df.index, dtype=float)
    return pd.to_numeric(df[field], errors="coerce")


def _clean_json(value):
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, (float, int)) and not math.isfinite(float(value)):
        return None
    return value


def _prepare_recorded_finance(bets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = bets.copy()
    frame["result_normalized"] = frame.get("result", pd.Series(index=frame.index, dtype=str)).map(
        normalize_result
    )
    entry = _number_series(frame, "entry_decimal_odds")
    legacy = _number_series(frame, "decimal_odds")
    frame["finance_decimal_odds"] = entry.where(entry > 1.0, legacy)
    frame["finance_stake_amount"] = _number_series(frame, "stake_amount")
    frame["finance_profit"] = _number_series(frame, "realized_profit")
    frame["finance_model_edge"] = _number_series(frame, "model_edge")
    frame["finance_stake_pct"] = _number_series(frame, "stake_pct")
    frame["finance_p_model"] = _number_series(frame, "p_model")
    valid_result = frame["result_normalized"].isin(["HIT", "MISS", "PUSH", "VOID"])
    eligible_mask = (
        valid_result
        & (frame["finance_decimal_odds"] > 1.0)
        & (frame["finance_stake_amount"] > 0)
        & frame["finance_profit"].notna()
    )
    return frame, frame[eligible_mask].copy()


def _bankroll_curve(finance_rows: pd.DataFrame) -> tuple[list[dict], float, float | None, float | None]:
    if finance_rows.empty:
        return [], 0.0, None, None
    ordered = finance_rows.sort_values(["date", "logged_at"] if "logged_at" in finance_rows else ["date"])
    starts = _number_series(ordered, "bankroll_before").dropna()
    opening = float(starts.iloc[0]) if not starts.empty else None
    running = opening if opening is not None else 0.0
    curve = []
    equity = [running]
    for day, group in ordered.groupby(ordered["date"].astype(str), sort=True):
        profit = float(group["finance_profit"].sum())
        staked = float(group["finance_stake_amount"].sum())
        before = running if opening is not None else None
        running += profit
        equity.append(running)
        curve.append({
            "date": day,
            "bankroll_before": before,
            "amount_staked": staked,
            "profit": profit,
            "bankroll_after": running if opening is not None else None,
            "cumulative_profit": running - opening if opening is not None else running,
        })
    drawdown, drawdown_pct = maximum_drawdown(equity)
    ending = running if opening is not None else None
    return curve, drawdown, drawdown_pct, ending


def _summary_record(group: pd.DataFrame) -> dict:
    n = len(group)
    decisions = group[group["result_normalized"].isin(["HIT", "MISS"])]
    hits = int((decisions["result_normalized"] == "HIT").sum())
    total_staked = float(group["finance_stake_amount"].sum())
    profit = float(group["finance_profit"].sum())
    curve, drawdown, drawdown_pct, ending = _bankroll_curve(group)
    return {
        "n": n,
        "hits": hits,
        "misses": int((decisions["result_normalized"] == "MISS").sum()),
        "pushes": int((group["result_normalized"] == "PUSH").sum()),
        "voids": int((group["result_normalized"] == "VOID").sum()),
        "win_rate": hits / len(decisions) if len(decisions) else None,
        "total_amount_staked": total_staked,
        "total_profit": profit,
        "roi": profit / total_staked if total_staked > 0 else None,
        "average_decimal_odds": float(group["finance_decimal_odds"].mean()) if n else None,
        "average_model_edge": float(group["finance_model_edge"].mean())
        if group["finance_model_edge"].notna().any() else None,
        "average_stake_pct": float(group["finance_stake_pct"].mean())
        if group["finance_stake_pct"].notna().any() else None,
        "maximum_drawdown": drawdown,
        "maximum_drawdown_pct": drawdown_pct,
        "ending_bankroll": ending,
        "bankroll_curve": curve,
    }


def _group_records(frame: pd.DataFrame, field: str) -> list[dict]:
    if frame.empty:
        return []
    records = []
    for value, group in frame.groupby(field, dropna=False, sort=True):
        label = "UNKNOWN" if pd.isna(value) or str(value).strip() == "" else str(value)
        summary = _summary_record(group)
        summary.pop("bankroll_curve", None)
        records.append({field: label, **summary})
    return records


def build_finance_summary(bets: pd.DataFrame) -> dict:
    """Build real-odds/stake finance metrics without filling missing prices."""
    frame, eligible = _prepare_recorded_finance(bets)
    results = frame[frame["result_normalized"].isin(["HIT", "MISS", "PUSH", "VOID"])]
    decisions = results[results["result_normalized"].isin(["HIT", "MISS"])]

    eligible["season_mode_group"] = eligible.get(
        "season_mode", pd.Series("UNKNOWN", index=eligible.index)
    ).fillna("UNKNOWN").replace("", "UNKNOWN")
    eligible["probability_bucket"] = pd.cut(
        eligible["finance_p_model"],
        bins=[-math.inf, 0.55, 0.60, 0.65, math.inf],
        labels=["<55%", "55-60%", "60-65%", "65%+"],
    ).astype(object).where(eligible["finance_p_model"].notna(), "UNKNOWN")
    eligible["edge_bucket"] = pd.cut(
        eligible["finance_model_edge"],
        bins=[-math.inf, 0.02, 0.05, 0.10, math.inf],
        labels=["<=2%", "2-5%", "5-10%", "10%+"],
    ).astype(object).where(eligible["finance_model_edge"].notna(), "UNKNOWN")
    source = eligible.get("odds_book", pd.Series("", index=eligible.index)).fillna("").astype(str).str.strip()
    fallback = eligible.get("odds_source", pd.Series("", index=eligible.index)).fillna("").astype(str).str.strip()
    eligible["source_or_book"] = source.where(source != "", fallback).replace("", "UNKNOWN")
    eligible["month"] = eligible.get("date", pd.Series("", index=eligible.index)).astype(str).str[:7]

    overall = _summary_record(eligible)
    report = {
        "scope": "real logged bets with recorded entry odds and positive stake amounts",
        "logged_rows": len(frame),
        "settled_rows": len(results),
        "hit_miss_rows": len(decisions),
        "hits": int((decisions["result_normalized"] == "HIT").sum()),
        "misses": int((decisions["result_normalized"] == "MISS").sum()),
        "hit_rate_all_logged": float((decisions["result_normalized"] == "HIT").mean())
        if len(decisions) else None,
        "finance_eligible_rows": len(eligible),
        "finance_excluded_rows": len(results) - len(eligible),
        "overall": overall,
        "by_market": _group_records(eligible, "market"),
        "by_direction": _group_records(eligible, "pick"),
        "by_season_mode": _group_records(eligible, "season_mode_group"),
        "by_probability_bucket": _group_records(eligible, "probability_bucket"),
        "by_edge_bucket": _group_records(eligible, "edge_bucket"),
        "by_source_or_book": _group_records(eligible, "source_or_book"),
        "by_month": _group_records(eligible, "month"),
    }
    return _clean_json(report)


def finance_summary_markdown(report: dict) -> str:
    overall = report["overall"]
    roi = "n/a" if overall["roi"] is None else f"{overall['roi'] * 100:.2f}%"
    win_rate = "n/a" if overall["win_rate"] is None else f"{overall['win_rate'] * 100:.2f}%"
    ending = "n/a" if overall["ending_bankroll"] is None else f"{overall['ending_bankroll']:.2f}"
    lines = [
        "# Finance summary",
        "",
        "Real-money-compatible metrics include only logged bets with recorded entry odds, "
        "a positive stake amount, and a settled profit. Missing finance data is never filled "
        "with assumed odds.",
        "",
        f"- Settled logged rows: {report['settled_rows']}",
        f"- Finance-eligible rows: {report['finance_eligible_rows']}",
        f"- Excluded for missing/invalid finance data: {report['finance_excluded_rows']}",
        f"- Total staked: {overall['total_amount_staked']:.4f}",
        f"- Total profit: {overall['total_profit']:+.4f}",
        f"- ROI: {roi}",
        f"- Win rate: {win_rate}",
        f"- Maximum drawdown: {overall['maximum_drawdown']:.4f}",
        f"- Ending bankroll: {ending}",
        "",
        "## Breakdowns",
        "",
    ]
    for title, key, label in (
        ("Market", "by_market", "market"),
        ("Direction", "by_direction", "pick"),
        ("Season", "by_season_mode", "season_mode_group"),
        ("Probability bucket", "by_probability_bucket", "probability_bucket"),
        ("Edge bucket", "by_edge_bucket", "edge_bucket"),
        ("Odds source/book", "by_source_or_book", "source_or_book"),
        ("Month", "by_month", "month"),
    ):
        lines.extend([f"### {title}", "", "| Group | Bets | Staked | Profit | ROI |", "|---|---:|---:|---:|---:|"])
        for row in report[key]:
            row_roi = "n/a" if row["roi"] is None else f"{row['roi'] * 100:.2f}%"
            lines.append(
                f"| {row[label]} | {row['n']} | {row['total_amount_staked']:.4f} | "
                f"{row['total_profit']:+.4f} | {row_roi} |"
            )
        if not report[key]:
            lines.append("| No eligible bets | 0 | 0 | 0 | n/a |")
        lines.append("")
    return "\n".join(lines)


def write_finance_reports(bet_path: Path = DEFAULT_BET_LOG,
                          reports_dir: Path | None = None) -> dict:
    reports_dir = reports_dir or (_HERE / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    bets = pd.read_csv(bet_path) if bet_path.exists() and bet_path.stat().st_size else pd.DataFrame()
    report = build_finance_summary(bets)
    (reports_dir / "finance_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports_dir / "finance_summary.md").write_text(
        finance_summary_markdown(report) + "\n", encoding="utf-8"
    )
    curve = pd.DataFrame(report["overall"].get("bankroll_curve", []))
    curve.to_csv(reports_dir / "bankroll_curve.csv", index=False)
    return report


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def _pct(x: float) -> str:
    return f"{100.0 * x:+.1f}%" if pd.notna(x) else "  n/a"


def _roi(x: float) -> str:
    return f"{100.0 * x:+.1f}%" if pd.notna(x) else "  n/a"


def print_summary(
    stats: dict,
    overall: pd.DataFrame,
    by_market: pd.DataFrame,
    by_md: pd.DataFrame,
    mismatches: pd.DataFrame,
    cum: pd.DataFrame,
) -> None:
    print("=" * 72)
    print("  ROI ANALYSIS — bet_log ⋈ odds_tracker  on (date, player, market)")
    print("=" * 72)
    print(
        f"  Graded bets: {stats['bets_graded']:>5}   "
        f"Matched with odds: {stats['matched']:>5}   "
        f"Match rate: {100 * stats['match_rate']:.1f}%"
    )
    print(
        f"  Unmatched graded (no odds row): {stats['unmatched_graded']}  "
        f"— excluded from ROI"
    )
    if overall.empty:
        print("\n  No matched graded bets with usable odds. Nothing to report.")
        return

    o = overall.iloc[0]
    print("\n── Overall (matched bets only) ─────────────────────────────────")
    print(
        f"  n={int(o['n'])}  hits={int(o['hits'])}  "
        f"hit_rate={_pct(o['hit_rate'])}  "
        f"profit=${o['profit']:+.2f} / ${o['n']:.0f}  "
        f"ROI={_roi(o['roi'])}  "
        f"avg_dec={o['avg_decimal']:.3f}"
    )

    print("\n── ROI by market ────────────────────────────────────────────────")
    print(f"  {'market':<12} {'n':>5} {'hit%':>8} {'profit':>10} {'ROI':>8}")
    for _, r in by_market.iterrows():
        print(
            f"  {str(r['market']):<12} {int(r['n']):>5} "
            f"{_pct(r['hit_rate']):>8} {r['profit']:>+10.2f} {_roi(r['roi']):>8}"
        )

    print("\n── ROI by (market, pick) ─────────────────────────────────────────")
    print(
        f"  {'market':<12} {'pick':<6} {'n':>5} {'hit%':>8} "
        f"{'profit':>10} {'ROI':>8}"
    )
    for _, r in by_md.iterrows():
        print(
            f"  {str(r['market']):<12} {str(r['pick']):<6} {int(r['n']):>5} "
            f"{_pct(r['hit_rate']):>8} {r['profit']:>+10.2f} {_roi(r['roi']):>8}"
        )

    print("\n── Hit-rate vs ROI mismatches ───────────────────────────────────")
    print("  (hit rate > 50% but ROI < 0, or hit rate ≤ 50% but ROI > 0)")
    if mismatches.empty:
        print("  None — hit rate and ROI agree on sign for every market/direction.")
    else:
        print(
            f"  {'market':<12} {'pick':<6} {'n':>5} {'hit%':>8} "
            f"{'ROI':>8}  flag"
        )
        for _, r in mismatches.iterrows():
            print(
                f"  {str(r['market']):<12} {str(r['pick']):<6} {int(r['n']):>5} "
                f"{_pct(r['hit_rate']):>8} {_roi(r['roi']):>8}  {r['flag']}"
            )

    print("\n── Cumulative ROI over time (matched bets) ──────────────────────")
    # Print a compact checkpoint every ~10% of the series, plus first/last
    n = len(cum)
    checkpoints = sorted(
        set([0, n - 1] + [int(i * (n - 1) / 10) for i in range(1, 10)])
    )
    print(f"  {'date':<12} {'#':>5} {'cum_profit':>12} {'cum_ROI':>9} {'cum_hit%':>9}")
    for i in checkpoints:
        r = cum.iloc[i]
        print(
            f"  {str(r['date']):<12} {int(r['bet_number']):>5} "
            f"{r['cum_profit']:>+12.2f} {_roi(r['cum_roi']):>9} "
            f"{_pct(r['cum_hit_rate']):>9}"
        )
    print("=" * 72)


def export_per_bet(df: pd.DataFrame, path: Path) -> None:
    cols = [
        c
        for c in [
            "date",
            "player",
            "market",
            "line",
            "pick",
            "result",
            "american_odds",
            "decimal_odds",
            "won",
            "profit_per_1",
            "projection",
            "actual",
            "margin",
            "books_count",
            "vig_pct",
            "bet_number",
            "cum_profit",
            "cum_roi",
            "cum_hit_rate",
        ]
        if c in df.columns
    ]
    df[cols].to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bets", type=Path, default=DEFAULT_BET_LOG, help="bet_log.csv path")
    p.add_argument(
        "--reports-dir",
        type=Path,
        default=_HERE / "reports",
        help="directory for finance_summary.json/.md and bankroll_curve.csv",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="print the summary table (default behavior; kept for parity with calibrate.py)",
    )
    args = p.parse_args(argv)

    if not args.bets.exists():
        print(f"bet_log not found: {args.bets}", file=sys.stderr)
        return 1
    report = write_finance_reports(args.bets, args.reports_dir)
    overall = report["overall"]
    roi = "n/a" if overall["roi"] is None else f"{overall['roi'] * 100:.2f}%"
    print(
        f"Finance report: {report['finance_eligible_rows']} eligible / "
        f"{report['settled_rows']} settled rows; profit={overall['total_profit']:+.4f}; ROI={roi}"
    )
    print(f"Wrote reports → {args.reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
