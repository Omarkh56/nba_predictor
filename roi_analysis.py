#!/usr/bin/env python3
"""
roi_analysis.py — Realized dollar ROI from bet_log × odds_tracker
=================================================================
bet_log.csv and odds_tracker.csv share (date, player, market) but are never
joined elsewhere — calibrate.py only tracks hit rate. This script closes that
gap: attach the book price that matched each pick, convert American → decimal,
and compute realized profit per $1 stake.

Profit convention (unit stake = $1):
  win  → decimal_odds − 1
  loss → −1

Usage:
    python3 roi_analysis.py
    python3 roi_analysis.py --summary
    python3 roi_analysis.py --bets path/to/bet_log.csv --odds path/to/odds_tracker.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
DEFAULT_BET_LOG = _HERE / "bet_log.csv"
DEFAULT_ODDS = _HERE / "odds_tracker.csv"
DEFAULT_OUT = _HERE / "roi_per_bet.csv"

JOIN_KEYS = ["date", "player", "market"]


# ---------------------------------------------------------------------------
# Odds math
# ---------------------------------------------------------------------------
def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds. Raises ValueError if unusable.

    Valid American prices satisfy abs(odds) >= 100 (−100 / +100 = even money).
    Values like −2 or −3.5 are not American odds (data errors in odds_tracker)
    and would explode into absurd decimal prices (e.g. −2 → 51.0).
    """
    if pd.isna(american):
        raise ValueError("missing American odds")
    american = float(american)
    if abs(american) < 100:
        raise ValueError(f"not a valid American price: {american}")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


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
    p.add_argument("--odds", type=Path, default=DEFAULT_ODDS, help="odds_tracker.csv path")
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="per-bet realized profit CSV (default: roi_per_bet.csv)",
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
    if not args.odds.exists():
        print(f"odds_tracker not found: {args.odds}", file=sys.stderr)
        return 1

    matched, stats = load_joined(args.bets, args.odds)
    priced = attach_prices(matched)
    dropped_bad_odds = len(matched) - len(priced)
    if dropped_bad_odds:
        print(f"  ⚠  Dropped {dropped_bad_odds} matched row(s) with unusable odds")

    if priced.empty:
        print_summary(stats, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        return 0

    overall = _roi_table(priced)
    by_market = _roi_table(priced, ["market"])
    by_md = _roi_table(priced, ["market", "pick"])
    mismatches = flag_mismatches(by_md)
    cum = cumulative_roi(priced)

    print_summary(stats, overall, by_market, by_md, mismatches, cum)
    export_per_bet(cum, args.out)
    print(f"\n  Wrote per-bet realized profit → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
