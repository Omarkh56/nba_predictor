"""
diagnose.py — answers two questions:
  1. Is DvP actually helping your predictions?
  2. Why does PTS lose money?

Run from the project directory:
    python3 diagnose.py

Requires:
  - bet_log.csv with at least: date, player, market, line, pick, projection, actual, result
  - The 'projection' column needs values (not empty) for full analysis
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

LOG_PATH = Path(__file__).parent / "bet_log.csv"

if not LOG_PATH.exists():
    print(f"ERROR: bet_log.csv not found at {LOG_PATH}")
    sys.exit(1)

df = pd.read_csv(LOG_PATH)
graded = df[df["result"].isin(["HIT", "MISS"])].copy()

if graded.empty:
    print("No graded bets in log.")
    sys.exit(0)

graded["hit"]        = (graded["result"] == "HIT").astype(int)
graded["line"]       = pd.to_numeric(graded["line"], errors="coerce")
graded["actual"]     = pd.to_numeric(graded["actual"], errors="coerce")
graded["projection"] = pd.to_numeric(graded["projection"], errors="coerce")

has_proj = graded["projection"].notna().sum() > 0
print(f"\n{'='*70}")
print(f"  DIAGNOSTIC ANALYSIS — {len(graded)} graded bets")
print(f"  Projection data: {graded['projection'].notna().sum()}/{len(graded)} rows")
print(f"{'='*70}\n")

# =============================================================================
# PART 1: Why is PTS broken?
# =============================================================================
print("─" * 70)
print("  PART 1: PTS PROJECTION ANALYSIS")
print("─" * 70)

pts = graded[graded["market"] == "PTS"].copy()
if pts.empty:
    print("  No PTS bets in log.")
else:
    print(f"\n  Total PTS bets: {len(pts)}")
    print(f"  PTS_OVER:  {(pts['pick']=='OVER').sum()} bets, "
          f"hit rate {pts[pts['pick']=='OVER']['hit'].mean()*100:.1f}%")
    print(f"  PTS_UNDER: {(pts['pick']=='UNDER').sum()} bets, "
          f"hit rate {pts[pts['pick']=='UNDER']['hit'].mean()*100:.1f}%")

    if has_proj and pts["projection"].notna().sum() > 0:
        pts_with_proj = pts.dropna(subset=["projection", "actual", "line"])
        if not pts_with_proj.empty:
            # Bias: signed error of projection vs actual
            pts_with_proj["proj_error"]    = pts_with_proj["projection"] - pts_with_proj["actual"]
            pts_with_proj["proj_vs_line"]  = pts_with_proj["projection"] - pts_with_proj["line"]
            pts_with_proj["actual_vs_line"] = pts_with_proj["actual"] - pts_with_proj["line"]

            print(f"\n  PTS projection bias:")
            print(f"    Mean (projection − actual):  {pts_with_proj['proj_error'].mean():+.2f}  "
                  f"(positive = projecting too high)")
            print(f"    Median:                       {pts_with_proj['proj_error'].median():+.2f}")
            print(f"    Std dev:                      {pts_with_proj['proj_error'].std():.2f}")

            # Bin by projection size to see where bias is worst
            print(f"\n  Bias by projection range:")
            pts_with_proj["proj_bin"] = pd.cut(
                pts_with_proj["projection"],
                bins=[0, 10, 15, 20, 25, 30, 50],
                labels=["<10", "10-15", "15-20", "20-25", "25-30", "30+"],
            )
            bias_by_bin = pts_with_proj.groupby("proj_bin", observed=True).agg(
                n=("projection", "size"),
                mean_bias=("proj_error", "mean"),
                hit_rate=("hit", "mean"),
            ).round(3)
            print(bias_by_bin.to_string())

            # Bin by edge size — are large edges worse than small edges?
            print(f"\n  Hit rate by edge size (projection vs line gap):")
            pts_with_proj["edge_abs"] = pts_with_proj["proj_vs_line"].abs()
            pts_with_proj["edge_bin"] = pd.cut(
                pts_with_proj["edge_abs"],
                bins=[0, 1, 2, 3, 5, 100],
                labels=["<1", "1-2", "2-3", "3-5", "5+"],
            )
            edge_summary = pts_with_proj.groupby("edge_bin", observed=True).agg(
                n=("edge_abs", "size"),
                hit_rate=("hit", "mean"),
            ).round(3)
            print(edge_summary.to_string())
            print(f"\n  Interpretation: if hit rate DROPS as edge grows, your model is")
            print(f"  most wrong when it's most confident. That's a calibration failure.\n")

            # Worst single-game misses for PTS_OVER
            pts_over_miss = pts_with_proj[(pts_with_proj["pick"] == "OVER") & (pts_with_proj["hit"] == 0)]
            if not pts_over_miss.empty:
                worst = pts_over_miss.nlargest(10, "proj_error")[
                    ["date", "player", "line", "projection", "actual", "proj_error"]
                ]
                print(f"  Worst PTS_OVER misses (projection overshoot):")
                print(worst.to_string(index=False))
        else:
            print(f"\n  ⚠ No PTS rows have both projection and actual values.")
    else:
        print(f"\n  ⚠ No projection data in log — can't compute bias.")
        print(f"  Make sure nba_combined.py is writing projections to predictions JSON,")
        print(f"  and checkresults.py is propagating them to bet_log.csv.")

# =============================================================================
# PART 2: Is DvP doing anything?
# =============================================================================
print("\n" + "─" * 70)
print("  PART 2: DOES DVP HELP?")
print("─" * 70)

# DvP is embedded in the projection (we don't log dvp_factor directly).
# Proxy: compute (projection / line) ratio. If DvP is working, picks where
# the projection diverges most from the line should be the most accurate.
# If projection/line is flat across hit/miss, DvP (and other multipliers)
# aren't adding signal.

if has_proj:
    proj_df = graded.dropna(subset=["projection", "line", "actual"]).copy()
    proj_df["proj_ratio"]   = proj_df["projection"] / proj_df["line"]
    proj_df["actual_ratio"] = proj_df["actual"] / proj_df["line"]

    print(f"\n  For each market, do players whose PROJECTION is far from the LINE")
    print(f"  actually outperform/underperform the line in the SAME direction?")
    print(f"  (This is the signal DvP and other contextual factors should produce.)\n")

    for market in sorted(proj_df["market"].unique()):
        sub = proj_df[proj_df["market"] == market]
        if len(sub) < 30:
            continue
        # Correlation between projection deviation and actual deviation
        corr = sub["proj_ratio"].corr(sub["actual_ratio"])
        # Hit rate for confident picks (proj_ratio far from 1.0)
        confident = sub[(sub["proj_ratio"] - 1.0).abs() > 0.10]
        unconfident = sub[(sub["proj_ratio"] - 1.0).abs() <= 0.10]

        print(f"  {market:<9} n={len(sub):>4}  proj-vs-actual corr: {corr:+.3f}  "
              f"confident hit: {confident['hit'].mean()*100 if not confident.empty else 0:.1f}% (n={len(confident)})  "
              f"low-edge hit: {unconfident['hit'].mean()*100 if not unconfident.empty else 0:.1f}% (n={len(unconfident)})")

    print(f"\n  Interpretation:")
    print(f"    - Correlation > +0.30: projection has real signal")
    print(f"    - Correlation between -0.10 and +0.10: model is essentially random")
    print(f"    - Confident hit rate should be HIGHER than low-edge hit rate")
    print(f"      If they're similar, your contextual multipliers (incl. DvP) aren't helping.\n")

# =============================================================================
# PART 3: Player-level concentration of PTS misses
# =============================================================================
print("─" * 70)
print("  PART 3: WHICH PLAYERS ARE DRIVING THE PTS LOSSES?")
print("─" * 70)

pts_miss = graded[(graded["market"] == "PTS") & (graded["result"] == "MISS")]
if not pts_miss.empty:
    top_miss = pts_miss["player"].value_counts().head(15)
    print(f"\n  Top players by PTS miss count:")
    for player, count in top_miss.items():
        # Get their record
        all_pts = graded[(graded["player"] == player) & (graded["market"] == "PTS")]
        hits   = (all_pts["result"] == "HIT").sum()
        misses = (all_pts["result"] == "MISS").sum()
        rate   = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
        print(f"    {player:<25} {hits}W-{misses}L  ({rate:.0f}%)")
    print(f"\n  If a handful of players dominate the misses, your model has a")
    print(f"  player-specific bias — possibly minutes overestimation, role")
    print(f"  misclassification, or stale rate data for those names.")

print("\n" + "=" * 70)
print("  DIAGNOSTIC COMPLETE")
print("=" * 70 + "\n")