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
from datetime import date
from pathlib import Path

import pandas as pd

import fit_flag_penalties as ffp

# ---------------------------------------------------------------------------
# Paths (relative to this script's directory)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
LOG_FILE = _HERE / "bet_log.csv"
CALIB_FILE = _HERE / "calibration.json"

# ---------------------------------------------------------------------------
# Calibration hyper-parameters
# ---------------------------------------------------------------------------
DEFAULT_PRIOR = 0.50  # v7 Fix 2: uninformative prior when data < threshold
MIN_BETS_FOR_EMPIRICAL_PRIOR = 50  # use empirical rate as prior once we have this many bets
LAPLACE_K = 1.5  # Laplace pseudo-counts for smoothing (per side)
SHRINKAGE_N = 12  # sample size for ~50% trust in sample rate vs prior
MIN_BETS_TO_ADJ = 4  # need at least this many graded bets for market-direction adj
PLAYER_MARKET_MIN_BETS = 2  # v9 Fix 6: was 3; shrinkage estimator handles small-sample noise

# Flag-penalty fit (fit_flag_penalties.py) shrinkage. Larger than SHRINKAGE_N
# because these coefficients come from a ~30-feature logistic regression
# (correlated, higher-variance) rather than a simple per-group hit rate —
# needs more data before we let it move production penalties very far.
FLAG_SHRINK_N = 40

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


def _calibrate_market_direction(graded: "pd.DataFrame", prior: float) -> dict:
    """Compute per-(market, pick) smoothed hit rates and confidence adjustments."""
    market_dir = {}
    for (market, pick), grp in graded.groupby(["market", "pick"]):
        n = len(grp)
        hits = int(grp["hit"].sum())
        smoothed = _laplace_smooth(hits, n, prior)
        final_rate = _shrink_toward_prior(smoothed, n, prior)
        conf_adj = (final_rate - prior) * 100.0 if n >= MIN_BETS_TO_ADJ else 0.0
        market_dir[f"{market}_{pick}"] = {
            "n": n,
            "hits": hits,
            "raw_rate": round(hits / n, 3),
            "rate": round(float(final_rate), 3),
            "conf_adj": round(float(conf_adj), 1),
        }
    return market_dir


def _calibrate_margins(graded: "pd.DataFrame") -> dict:
    """Compute per-(market, pick) margin statistics if the column exists."""
    has_margin = (
        "margin" in graded.columns
        and pd.to_numeric(graded["margin"], errors="coerce").notna().any()
    )
    if not has_margin:
        return {}

    mg = graded.copy()
    mg["margin_num"] = pd.to_numeric(mg["margin"], errors="coerce")
    mg_valid = mg.dropna(subset=["margin_num"])
    margin_stats = {}
    for (market, pick), grp in mg_valid.groupby(["market", "pick"]):
        n = len(grp)
        mean_m = float(grp["margin_num"].mean())
        miss_grp = grp[grp["result"] == "MISS"]
        near_m = miss_grp["margin_num"].between(-1.0, -0.001).sum() if not miss_grp.empty else 0
        total_m = len(miss_grp)
        near_miss_rate = round(near_m / total_m, 3) if total_m > 0 else 0.0
        margin_stats[f"{market}_{pick}"] = {
            "n": n,
            "mean_margin": round(mean_m, 2),
            "near_miss_rate": near_miss_rate,
        }
    return margin_stats


def _calibrate_player_market(graded: "pd.DataFrame") -> dict:
    """Compute per-player per-market projection bias from residuals."""
    has_proj = "projection" in graded.columns and graded["projection"].notna().any()
    proj_count = graded["projection"].notna().sum() if "projection" in graded.columns else 0
    proj_nonzero = int(
        (pd.to_numeric(graded.get("projection", pd.Series(dtype=float)), errors="coerce") > 0).sum()
    )
    print(f"  Projection column: {proj_count} non-null rows, {proj_nonzero} non-zero")
    if has_proj:
        pm_size = graded.dropna(subset=["projection"]).groupby(["player", "market"]).size()
        print(
            f"  Player-market groups with ≥{PLAYER_MARKET_MIN_BETS} bets: "
            f"{int((pm_size >= PLAYER_MARKET_MIN_BETS).sum())}"
        )

    if not has_proj:
        return {}

    proj_df = graded.dropna(subset=["projection", "actual"]).copy()
    proj_df["residual"] = proj_df["actual"].astype(float) - proj_df["projection"].astype(float)
    player_market = {}
    for (player, market), grp in proj_df.groupby(["player", "market"]):
        n = len(grp)
        mean_res = float(grp["residual"].mean())
        w = n / (n + SHRINKAGE_N)
        adj = w * mean_res if n >= PLAYER_MARKET_MIN_BETS else 0.0
        player_market[f"{player}|{market}"] = {
            "n": n,
            "mean_residual": round(mean_res, 2),
            "adj": round(float(adj), 2),
        }
    return player_market


def _calibrate_flag_penalties(graded: "pd.DataFrame") -> dict:
    """Fit fit_flag_penalties.py's logistic regression on the graded log and
    shrink each hand-tuned-backed flag's fitted confidence-point effect
    toward its current hand-tuned constant, weighted by sample size.

    Stored 'pp' is confidence points GAINED by the picked side when the flag
    is present (positive = should raise confidence, negative = lower it) —
    the same sign convention playerlinepredictor.py's _flag_penalty() reads.
    Flags with no hand-tuned constant (informational-only today) are still
    fit and stored for visibility, with hand_tuned_pp = null.
    """
    df = graded.copy()
    if "flags" not in df.columns:
        df["flags"] = ""
    df["flags"] = df["flags"].fillna("")
    df["market"] = df["market"].astype(str).str.strip()
    df["pick"] = df["pick"].astype(str).str.strip().str.upper()

    if len(df) < 30:
        return {}

    X, feat_names = ffp.build_features(df)
    y = df["hit"].values
    base_rate = float(y.mean())
    clf = ffp.fit_logistic(X, y)
    coefs = dict(zip(feat_names, clf.coef_[0]))

    out = {}
    for name, (matcher, hand_pp, source) in ffp.PENALTY_MAP.items():
        mask = [matcher(f, r) for f, (_, r) in zip(df["flags"], df.iterrows())]
        n = int(sum(mask))
        fitted_pp = ffp.coef_to_pp(coefs.get(name, 0.0), base_rate)
        if hand_pp is None:
            shrunk_pp = fitted_pp if n >= MIN_BETS_TO_ADJ else 0.0
        else:
            w = n / (n + FLAG_SHRINK_N)
            shrunk_pp = w * fitted_pp + (1 - w) * hand_pp
        out[name] = {
            "n": n,
            "fitted_pp": round(fitted_pp, 2),
            "hand_tuned_pp": hand_pp,
            "pp": round(float(shrunk_pp), 2),
            "source": source,
        }
    return out


def _print_flag_penalty_summary(flag_penalties: dict) -> None:
    if not flag_penalties:
        return
    print()
    print("  Flag-penalty fit (shrunk toward hand-tuned constant by sample size):")
    print(f"  {'Flag':<18} {'N':>4} {'Hand pp':>9} {'Fitted pp':>10} {'Shrunk pp':>10}  Note")
    print(f"  {'─' * 18} {'─' * 4} {'─' * 9} {'─' * 10} {'─' * 10}  {'─' * 25}")
    for name, v in sorted(flag_penalties.items(), key=lambda x: -x[1]["n"]):
        hand = f"{v['hand_tuned_pp']:+.1f}" if v["hand_tuned_pp"] is not None else "  n/a"
        note = "no hand-tuned constant" if v["hand_tuned_pp"] is None else ""
        print(
            f"  {name:<18} {v['n']:>4} {hand:>9} {v['fitted_pp']:>+10.2f} {v['pp']:>+10.2f}  {note}"
        )


def _print_calibration_summary(market_dir: dict, margin_stats: dict, player_market: dict) -> None:
    """Print verbose calibration tables to stdout."""
    print()
    print("  Market-direction calibration:")
    print(f"  {'Key':<22} {'N':>4} {'Hits':>5} {'Raw%':>6} {'Adj%':>6}  Interpretation")
    print(f"  {'─' * 22} {'─' * 4} {'─' * 5} {'─' * 6} {'─' * 6}  {'─' * 35}")
    for key, v in sorted(market_dir.items(), key=lambda x: x[1]["conf_adj"], reverse=True):
        raw_pct = v["raw_rate"] * 100
        adj_pp = v["conf_adj"]
        sign = "+" if adj_pp >= 0 else ""
        note = "boost conf" if adj_pp > 2 else "penalise conf" if adj_pp < -2 else "near-neutral"
        n_note = " (insuff)" if v["n"] < MIN_BETS_TO_ADJ else ""
        print(
            f"  {key:<22} {v['n']:>4} {v['hits']:>5} {raw_pct:>5.1f}% "
            f"{sign}{adj_pp:>4.1f}pp  {note}{n_note}"
        )

    if margin_stats:
        print()
        print("  Margin analysis (positive = cushion, negative = missed by):")
        print(f"  {'Key':<22} {'N':>4} {'AvgMargin':>10} {'NearMiss%':>10}  Note")
        print(f"  {'─' * 22} {'─' * 4} {'─' * 10} {'─' * 10}  {'─' * 25}")
        for key, ms in sorted(margin_stats.items(), key=lambda x: x[1]["mean_margin"]):
            nm_pct = ms["near_miss_rate"] * 100
            note = ""
            if ms["mean_margin"] < -1.0:
                note = "proj running high"
            elif ms["mean_margin"] > 1.0:
                note = "proj running low"
            if ms["near_miss_rate"] >= 0.30:
                note += (" / " if note else "") + "freq near-miss"
            print(f"  {key:<22} {ms['n']:>4} {ms['mean_margin']:>+9.2f}  {nm_pct:>8.1f}%  {note}")

    if player_market:
        print()
        print("  Player-market projection bias:")
        print(f"  {'Key':<35} {'N':>4} {'MeanRes':>8} {'Adj':>6}")
        print(f"  {'─' * 35} {'─' * 4} {'─' * 8} {'─' * 6}")
        for key, v in sorted(player_market.items(), key=lambda x: abs(x[1]["adj"]), reverse=True):
            sign = "+" if v["adj"] >= 0 else ""
            print(f"  {key:<35} {v['n']:>4} {v['mean_residual']:>+7.2f}  {sign}{v['adj']:>5.2f}")


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

    print(
        f"  bet_log: {total_bets} graded bets  "
        f"({total_hits}W – {total_bets - total_hits}L  "
        f"{empirical_rate * 100:.1f}%)"
    )

    prior = empirical_rate if total_bets >= MIN_BETS_FOR_EMPIRICAL_PRIOR else DEFAULT_PRIOR
    prior_source = "empirical" if total_bets >= MIN_BETS_FOR_EMPIRICAL_PRIOR else "default"
    print(f"  Prior hit rate: {prior:.3f} (source: {prior_source})")

    market_dir = _calibrate_market_direction(graded, prior)
    margin_stats = _calibrate_margins(graded)
    player_market = _calibrate_player_market(graded)
    flag_penalties = _calibrate_flag_penalties(graded)

    # Merge margin stats into market_dir entries
    for key, ms in margin_stats.items():
        if key in market_dir:
            market_dir[key]["mean_margin"] = ms["mean_margin"]
            market_dir[key]["near_miss_rate"] = ms["near_miss_rate"]

    calib = {
        "generated": date.today().isoformat(),
        "total_bets": total_bets,
        "overall_hit_rate": round(empirical_rate, 3),
        "prior_hit_rate": round(prior, 3),
        "prior_source": prior_source,
        "market_direction": market_dir,
        "player_market": player_market,
        "flag_penalties": flag_penalties,
    }

    with open(CALIB_FILE, "w") as f:
        json.dump(calib, f, indent=2)

    print(f"  Saved → {CALIB_FILE}")
    print(
        f"  {len(market_dir)} market-direction adjustments, "
        f"{len(player_market)} player-market biases, "
        f"{len(flag_penalties)} flag-penalty fits"
    )

    if verbose or "--summary" in sys.argv:
        _print_calibration_summary(market_dir, margin_stats, player_market)
        _print_flag_penalty_summary(flag_penalties)


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  NBA Props Calibration Engine")
    print("=" * 55)
    calibrate(verbose="--summary" in sys.argv)
    print("=" * 55 + "\n")
