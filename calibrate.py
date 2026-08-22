"""
calibrate.py — NBA Props Calibration Engine
=============================================
Reads bet_log.csv (appended by checkresults.py after each grading run) and
produces calibration.json, which playerlinepredictor.py loads at startup to
apply data-driven confidence and projection adjustments.

Two types of calibration are computed:

  1. Market-direction hit rates
     For each (market, direction) pair (e.g. "3PM_OVER", "PTS_UNDER") we
     compute a shrinkage-smoothed hit rate and convert it to a confidence
     point-penalty adjustment relative to the population baseline.
     → Applied in process_game() as direction_penalty correction.

  2. Player-market projection bias
     For each (player, market) pair we compute the mean signed residual
     (actual − projection) with shrinkage toward 0.
     → Applied in project_prop() as a projection nudge.
     (Requires "projection" column in bet_log.csv, which checkresults.py
     writes when the PREDICTIONS dict includes a "projection" key.)

Usage:
    python3 calibrate.py            # compute and save calibration.json
    python3 calibrate.py --summary  # also print a detailed table
"""

import json
import sys
import os
import math
import pandas as pd
from pathlib import Path
from datetime import date

# ---------------------------------------------------------------------------
# Paths (relative to this script's directory)
# ---------------------------------------------------------------------------
_HERE      = Path(__file__).parent
LOG_FILE   = _HERE / "bet_log.csv"
CALIB_FILE = _HERE / "calibration.json"

# ---------------------------------------------------------------------------
# Calibration hyper-parameters
# ---------------------------------------------------------------------------
DEFAULT_PRIOR            = 0.50   # v7 Fix 2: uninformative prior when data < threshold
MIN_BETS_FOR_EMPIRICAL_PRIOR = 50  # use empirical rate as prior once we have this many bets
LAPLACE_K                = 1.5    # Laplace pseudo-counts for smoothing (per side)
SHRINKAGE_N              = 12     # sample size for ~50% trust in sample rate vs prior
MIN_BETS_TO_ADJ          = 4      # need at least this many graded bets for market-direction adj
PLAYER_MARKET_MIN_BETS   = 2      # v9 Fix 6: was 3; shrinkage estimator handles small-sample noise

# ---------------------------------------------------------------------------


def _laplace_smooth(hits, n, prior, k=LAPLACE_K):
    """Smooth hit rate toward prior using Laplace / additive smoothing."""
    return (hits + k * prior) / (n + k)


def _shrink_toward_prior(sample_rate, n, prior, shrink_n=SHRINKAGE_N):
    """Shrinkage estimator: blend sample rate with prior based on sample size."""
    w = n / (n + shrink_n)
    return w * sample_rate + (1 - w) * prior


def _migrate_log_csv():
    """One-time migration: add 'margin' column to bet_log.csv if missing.

    Old schema (8 cols): date,player,market,line,pick,projection,actual,result
    New schema (9 cols): date,player,market,line,pick,projection,actual,margin,result

    Reads the file line by line so it tolerates rows with either column count.
    """
    with open(LOG_FILE) as fh:
        lines = fh.readlines()
    if not lines:
        return
    header_fields = [c.strip() for c in lines[0].strip().split(",")]
    if "margin" in header_fields:
        return  # already migrated

    new_lines = []
    for i, line in enumerate(lines):
        raw = line.rstrip("\n")
        fields = raw.split(",")
        if i == 0:
            # Insert 'margin' before 'result' (last column)
            fields = fields[:-1] + ["margin", fields[-1]]
        elif len(fields) == 8:
            # Old data row — pad with empty margin before result
            fields = fields[:-1] + ["", fields[-1]]
        # 9-field rows are already correct
        new_lines.append(",".join(fields) + "\n")

    with open(LOG_FILE, "w") as fh:
        fh.writelines(new_lines)
    print("  Migrated bet_log.csv → added 'margin' column to existing rows.")


def calibrate(verbose=False):
    if not LOG_FILE.exists():
        print(f"bet_log.csv not found at {LOG_FILE}")
        print("Run checkresults.py to start building the log.")
        return

    _migrate_log_csv()
    df = pd.read_csv(LOG_FILE)
    graded = df[df["result"].isin(["HIT", "MISS"])].copy()

    if graded.empty:
        print("No graded bets in bet_log.csv yet — run checkresults.py first.")
        return

    graded["hit"] = (graded["result"] == "HIT").astype(int)
    total_bets = len(graded)
    total_hits = graded["hit"].sum()
    empirical_rate = total_hits / total_bets

    print(f"  bet_log: {total_bets} graded bets  "
          f"({total_hits}W – {total_bets - total_hits}L  "
          f"{empirical_rate*100:.1f}%)")

    # v7 Fix 2: self-updating prior — use empirical rate once we have enough data
    prior        = empirical_rate if total_bets >= MIN_BETS_FOR_EMPIRICAL_PRIOR else DEFAULT_PRIOR
    prior_source = "empirical"    if total_bets >= MIN_BETS_FOR_EMPIRICAL_PRIOR else "default"
    print(f"  Prior hit rate: {prior:.3f} (source: {prior_source})")

    # -----------------------------------------------------------------------
    # 1. Market-direction calibration
    # -----------------------------------------------------------------------
    market_dir = {}
    md_groups = graded.groupby(["market", "pick"])

    for (market, pick), grp in md_groups:
        n    = len(grp)
        hits = int(grp["hit"].sum())

        smoothed   = _laplace_smooth(hits, n, prior)
        final_rate = _shrink_toward_prior(smoothed, n, prior)

        # Confidence adjustment in percentage points:
        # Positive = historically hits more often → reduce direction_penalty
        # Negative = historically hits less often → increase direction_penalty
        conf_adj = (final_rate - prior) * 100.0

        # Only apply adjustments when there is enough data to be meaningful
        if n < MIN_BETS_TO_ADJ:
            conf_adj = 0.0

        key = f"{market}_{pick}"
        market_dir[key] = {
            "n":        n,
            "hits":     hits,
            "raw_rate": round(hits / n, 3),
            "rate":     round(float(final_rate), 3),
            "conf_adj": round(float(conf_adj), 1),
        }

    # -----------------------------------------------------------------------
    # 1b. Margin analysis (requires "margin" column added by checkresults.py)
    # -----------------------------------------------------------------------
    has_margin = ("margin" in graded.columns and
                  pd.to_numeric(graded["margin"], errors="coerce").notna().any())

    margin_stats = {}   # key → {mean_margin, near_miss_rate, n}
    if has_margin:
        mg = graded.copy()
        mg["margin_num"] = pd.to_numeric(mg["margin"], errors="coerce")
        mg_valid = mg.dropna(subset=["margin_num"])

        for (market, pick), grp in mg_valid.groupby(["market", "pick"]):
            n = len(grp)
            mean_m = float(grp["margin_num"].mean())
            # Near-miss rate: misses that were within 1 unit (margin in [-1, 0))
            miss_grp = grp[grp["result"] == "MISS"]
            near_m   = (miss_grp["margin_num"].between(-1.0, -0.001).sum()
                        if not miss_grp.empty else 0)
            total_m  = len(miss_grp)
            near_miss_rate = round(near_m / total_m, 3) if total_m > 0 else 0.0
            key = f"{market}_{pick}"
            margin_stats[key] = {
                "n":             n,
                "mean_margin":   round(mean_m, 2),
                "near_miss_rate": near_miss_rate,
            }
        # Merge mean_margin into market_dir entries
        for key, ms in margin_stats.items():
            if key in market_dir:
                market_dir[key]["mean_margin"]    = ms["mean_margin"]
                market_dir[key]["near_miss_rate"] = ms["near_miss_rate"]

    # -----------------------------------------------------------------------
    # 2. Player-market projection bias
    # -----------------------------------------------------------------------
    player_market = {}
    has_proj = ("projection" in graded.columns and
                graded["projection"].notna().any())

    # v7 Fix 3b: diagnostic output to debug empty player_market
    proj_count   = graded["projection"].notna().sum() if "projection" in graded.columns else 0
    proj_nonzero = int((pd.to_numeric(graded.get("projection", pd.Series(dtype=float)),
                                      errors="coerce") > 0).sum())
    print(f"  Projection column: {proj_count} non-null rows, {proj_nonzero} non-zero")
    if has_proj:
        pm_groups_size = (graded.dropna(subset=["projection"])
                                 .groupby(["player", "market"]).size())
        multi_sample = int((pm_groups_size >= PLAYER_MARKET_MIN_BETS).sum())
        print(f"  Player-market groups with ≥{PLAYER_MARKET_MIN_BETS} bets: {multi_sample}")

    if has_proj:
        proj_df = graded.dropna(subset=["projection", "actual"]).copy()
        proj_df["residual"] = proj_df["actual"].astype(float) - proj_df["projection"].astype(float)

        pm_groups = proj_df.groupby(["player", "market"])
        for (player, market), grp in pm_groups:
            n          = len(grp)
            mean_res   = float(grp["residual"].mean())
            # Shrinkage toward 0 — trust grows with more observations
            w          = n / (n + SHRINKAGE_N)
            adj        = w * mean_res if n >= PLAYER_MARKET_MIN_BETS else 0.0  # v7 Fix 3c
            key = f"{player}|{market}"
            player_market[key] = {
                "n":             n,
                "mean_residual": round(mean_res, 2),
                "adj":           round(float(adj), 2),
            }

    # -----------------------------------------------------------------------
    # Build + write calibration.json
    # -----------------------------------------------------------------------
    calib = {
        "generated":        date.today().isoformat(),
        "total_bets":       total_bets,
        "overall_hit_rate": round(empirical_rate, 3),
        "prior_hit_rate":   round(prior, 3),    # v7 Fix 2: reflects actual prior used
        "prior_source":     prior_source,
        "market_direction": market_dir,
        "player_market":    player_market,
    }

    with open(CALIB_FILE, "w") as f:
        json.dump(calib, f, indent=2)

    print(f"  Saved → {CALIB_FILE}")
    print(f"  {len(market_dir)} market-direction adjustments, "
          f"{len(player_market)} player-market biases")

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    if verbose or "--summary" in sys.argv:
        print()
        print("  Market-direction calibration:")
        print(f"  {'Key':<22} {'N':>4} {'Hits':>5} {'Raw%':>6} {'Adj%':>6}  Interpretation")
        print(f"  {'─'*22} {'─'*4} {'─'*5} {'─'*6} {'─'*6}  {'─'*35}")
        for key, v in sorted(market_dir.items(),
                              key=lambda x: x[1]["conf_adj"], reverse=True):
            raw_pct = v["raw_rate"] * 100
            adj_pp  = v["conf_adj"]
            sign    = "+" if adj_pp >= 0 else ""
            note    = ("boost conf" if adj_pp > 2
                       else "penalise conf" if adj_pp < -2
                       else "near-neutral")
            n_note  = " (insuff)" if v["n"] < MIN_BETS_TO_ADJ else ""
            print(f"  {key:<22} {v['n']:>4} {v['hits']:>5} {raw_pct:>5.1f}% "
                  f"{sign}{adj_pp:>4.1f}pp  {note}{n_note}")

        if margin_stats:
            print()
            print("  Margin analysis (positive = cushion, negative = missed by):")
            print(f"  {'Key':<22} {'N':>4} {'AvgMargin':>10} {'NearMiss%':>10}  Note")
            print(f"  {'─'*22} {'─'*4} {'─'*10} {'─'*10}  {'─'*25}")
            for key, ms in sorted(margin_stats.items(),
                                   key=lambda x: x[1]["mean_margin"]):
                nm_pct = ms["near_miss_rate"] * 100
                note = ""
                if ms["mean_margin"] < -1.0:
                    note = "proj running high"
                elif ms["mean_margin"] > 1.0:
                    note = "proj running low"
                if ms["near_miss_rate"] >= 0.30:
                    note += (" / " if note else "") + "freq near-miss"
                print(f"  {key:<22} {ms['n']:>4} {ms['mean_margin']:>+9.2f}  "
                      f"{nm_pct:>8.1f}%  {note}")

        if player_market:
            print()
            print("  Player-market projection bias:")
            print(f"  {'Key':<35} {'N':>4} {'MeanRes':>8} {'Adj':>6}")
            print(f"  {'─'*35} {'─'*4} {'─'*8} {'─'*6}")
            for key, v in sorted(player_market.items(),
                                  key=lambda x: abs(x[1]["adj"]), reverse=True):
                sign = "+" if v["adj"] >= 0 else ""
                print(f"  {key:<35} {v['n']:>4} {v['mean_residual']:>+7.2f}  "
                      f"{sign}{v['adj']:>5.2f}")


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  NBA Props Calibration Engine")
    print("=" * 55)
    calibrate(verbose="--summary" in sys.argv)
    print("=" * 55 + "\n")
