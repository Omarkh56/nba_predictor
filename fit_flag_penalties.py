"""
fit_flag_penalties.py — Data-driven check on playerlinepredictor.py's
hand-tuned confidence penalties.

playerlinepredictor.py applies a series of hand-picked point penalties to
prop confidence (UNDER_CONF_PENALTY, STAR_UNDER_PENALTY, book-count
penalties, bench/mins-vol/road-B2B penalties, playoff-no-PO, low-line,
suspicious-edge%, declining-trend, GTD/Probable — see PENALTY_MAP below)
that were tuned by hand against postmortems, never fit from data.

checkresults.py now logs each graded pick's risk Flags (via
predictions_db.prop_predictions.flags) into bet_log.csv alongside market
and pick direction, which were already there. This script:

  1. Parses those flags (plus market/pick/line/projection) into boolean
     predictors matching each hand-tuned penalty's actual trigger condition.
  2. Fits a logistic regression: hit ~ market + pick + flags.
  3. Converts each flag's fitted log-odds coefficient into an equivalent
     confidence-point delta, via the standard derivative approximation
     Δp ≈ β · p̄(1-p̄) at the sample hit rate — the same units the hand-tuned
     constants are already expressed in (points off confidence).
  4. Reports every flag next to its current hand-tuned constant (if any)
     and flags meaningful disagreements.
  5. Writes flag_penalties.json — a small, regenerable lookup table in the
     same spirit as calibration.json, so calibrate.py/train_model.py can
     recompute it as more graded bets accumulate instead of the constants
     staying fixed forever.

Usage:
    python3 fit_flag_penalties.py
    python3 fit_flag_penalties.py --min-n 15     # suppress low-sample flags
    python3 fit_flag_penalties.py --disagree-pp 3.0   # meaningful-margin threshold
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_HERE = Path(__file__).parent
LOG_FILE = _HERE / "bet_log.csv"
OUT_FILE = _HERE / "flag_penalties.json"

MIN_BETS_PER_MARKET = 20  # markets below this are pooled into an "OTHER" bucket

# ---------------------------------------------------------------------------
# Every hand-tuned penalty in playerlinepredictor.py, its exact trigger, and
# the boolean feature that reproduces that trigger from bet_log.csv +
# predictions.db's Flags string. `constant_pp` is None for flags that are
# purely informational today (shown in Flags/detail line but never actually
# subtracted from confidence) — those are still fit and reported, just with
# no "hand-tuned" value to compare against.
# ---------------------------------------------------------------------------
def _has(token: str):
    return lambda flags, row: token in flags


def _has_prefix(prefix: str):
    return lambda flags, row: any(t.startswith(prefix) for t in flags.split())


PENALTY_MAP = {
    # feature name -> (matcher(flags_str, row) -> bool, hand-tuned pp, source)
    "UNDER_PICK": (
        lambda flags, row: row["pick"] == "UNDER",
        -2.0, "UNDER_CONF_PENALTY (applies to every UNDER pick)",
    ),
    "STAR_UNDER": (
        _has("★BLW-BUF"),
        -5.0, "STAR_UNDER_PENALTY (UNDER + proj >= STAR_PROJ_THRESHOLD=28)",
    ),
    "LOW_LINE": (
        _has("LOW-LINE"),
        -5.0, "LOW_LINE_CONF_PENALTY (line <= 1.5)",
    ),
    "NO_PO": (
        _has("NO-PO"),
        -8.0, "PLAYOFF_NO_PO_PENALTY (playoff season, po_raw_count == 0)",
    ),
    "PTS_REB_OVER": (
        lambda flags, row: row["market"] == "PTS+REB" and row["pick"] == "OVER",
        -5.0, "PTS_REB_OVER_PENALTY",
    ),
    "ONE_BOOK": (
        _has("1-BOOK"),
        -4.0, "book_penalty (book_count == 1)",
    ),
    "TWO_BOOK": (
        _has("2-BOOK"),
        -2.0, "book_penalty (book_count == 2)",
    ),
    "ROAD_B2B": (
        _has("ROAD-B2B"),
        -3.0, "road_b2b_penalty (fixed 3.0; underlying road_b2b is boolean)",
    ),
    "SUSPICIOUS_EDGE": (
        _has_prefix("⚠SUSP-"),
        -20.0, "SUSPICIOUS_EDGE_PCT dock (edge_pct > 40)",
    ),
    "TREND_DOWN_OVER": (
        lambda flags, row: "TREND↓" in flags and row["pick"] == "OVER",
        -3.0, "inline trend penalty (declining trend + OVER only)",
    ),
    "GTD": (
        _has("⚠GTD"),
        -5.0, "_self_injury_check (Questionable/Day-To-Day)",
    ),
    "PROBABLE": (
        _has("PROB"),
        -1.0, "_self_injury_check (Probable)",
    ),
    # Continuous in the current code (not a single constant) — still fit as
    # a binary "flag present" predictor so we can see whether the flag
    # correlates with misses at all, just with no fixed pp to compare to.
    "BENCH": (
        _has("⚠BENCH"),
        None, "bench_penalty — continuous: max(0, (20 - avg_min) * 1.5), not a fixed constant",
    ),
    "MINS_VOL": (
        _has("⚠MINS-VOL"),
        None, "mins_vol_penalty — continuous: min(8, (CV-0.20)*25), not a fixed constant",
    ),
    # Flags with no confidence effect in the code today — included so the
    # regression can show whether they're informative even though the model
    # currently ignores them.
    "LOW_PO": (_has("LOW-PO"), None, "no fixed constant today — informational only"),
    "PO_ROLE": (_has("⚠PO-ROLE"), None, "no fixed constant today — overlaps NO-PO"),
    "ROT_RISK": (_has("ROT-RISK"), None, "no fixed constant today — informational only"),
    "TREND_DOWN_UNDER": (
        lambda flags, row: "TREND↓" in flags and row["pick"] == "UNDER",
        None, "declining trend currently only penalized for OVER picks",
    ),
    "TREND_UP": (_has("TREND↑"), None, "no fixed constant today — informational only"),
    "STRONG": (_has("★STRONG"), None, "no fixed constant today — informational only"),
    "HI_USG": (_has("HI-USG"), None, "no fixed constant today — informational only"),
    "LO_USG": (_has("LO-USG"), None, "no fixed constant today — informational only"),
    "DVP_HIGH": (_has_prefix("DvP+"), None, "no fixed constant today — informational only"),
    "DVP_LOW": (_has_prefix("DvP-"), None, "no fixed constant today — informational only"),
}


# ---------------------------------------------------------------------------
# Data loading / feature building
# ---------------------------------------------------------------------------
def load_graded(path: Path = LOG_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["flags"] = df["flags"].fillna("")
    df["market"] = df["market"].astype(str).str.strip()
    df["pick"] = df["pick"].astype(str).str.strip().str.upper()
    df["result"] = df["result"].astype(str).str.strip().str.upper()
    df = df[df["result"].isin(("HIT", "MISS"))].copy()
    df["hit"] = (df["result"] == "HIT").astype(int)
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (X, feature_names) — market/pick dummies + one boolean column
    per PENALTY_MAP entry."""
    feat = pd.DataFrame(index=df.index)

    market_counts = df["market"].value_counts()
    small_markets = set(market_counts[market_counts < MIN_BETS_PER_MARKET].index)
    market_grouped = df["market"].where(~df["market"].isin(small_markets), "OTHER")
    market_dummies = pd.get_dummies(market_grouped, prefix="mkt", drop_first=True)

    flag_cols = {}
    for name, (matcher, _pp, _src) in PENALTY_MAP.items():
        flag_cols[name] = [matcher(f, row) for f, (_, row) in zip(df["flags"], df.iterrows())]
    flag_df = pd.DataFrame(flag_cols, index=df.index).astype(int)

    feat = pd.concat([market_dummies.astype(int), flag_df], axis=1)
    return feat, list(feat.columns)


# ---------------------------------------------------------------------------
# Fit + convert to confidence points
# ---------------------------------------------------------------------------
def fit_logistic(X: pd.DataFrame, y: np.ndarray) -> LogisticRegression:
    # C=1.0 matches train_model.py's game-model logistic fit for consistency.
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    clf.fit(X.values, y)
    return clf


def coef_to_pp(coef: float, base_rate: float) -> float:
    """Convert a log-odds coefficient to an equivalent confidence-point
    delta via the derivative of the sigmoid at the sample base rate:
    d(p)/d(logit) = p(1-p). This is the same 0-100 'points off confidence'
    scale the hand-tuned constants are already written in."""
    return coef * base_rate * (1 - base_rate) * 100.0


def bootstrap_pp(X: pd.DataFrame, y: np.ndarray, feat_names: list[str],
                  n_boot: int = 200, seed: int = 42) -> dict[str, tuple[float, float]]:
    """Resample rows with replacement, refit, recompute each flag's fitted_pp
    n_boot times. Returns {feature: (p5, p95)} — an approximate 90% CI.
    With ~30 correlated features on ~1200 rows, point estimates alone are
    not trustworthy; this is what actually justifies calling a disagreement
    'meaningful' rather than noise from an underdetermined fit."""
    rng = np.random.default_rng(seed)
    n = len(y)
    boot_pp = {name: [] for name in feat_names}
    Xv = X.values
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb, yb = Xv[idx], y[idx]
        if len(np.unique(yb)) < 2:
            continue
        base_rate_b = float(yb.mean())
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
        clf.fit(Xb, yb)
        for name, coef in zip(feat_names, clf.coef_[0]):
            boot_pp[name].append(coef_to_pp(coef, base_rate_b))
    return {
        name: (float(np.percentile(vals, 5)), float(np.percentile(vals, 95)))
        if vals else (float("nan"), float("nan"))
        for name, vals in boot_pp.items()
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(df, feat_names, coefs, base_rate, min_n, ci) -> list[dict]:
    rows = []
    for name in feat_names:
        if name not in PENALTY_MAP:
            continue  # market dummies aren't penalty flags — skip in the comparison report
        matcher, hand_pp, source = PENALTY_MAP[name]
        mask = [matcher(f, r) for f, (_, r) in zip(df["flags"], df.iterrows())]
        n_flag = int(sum(mask))
        raw_hit_rate = float(df.loc[mask, "hit"].mean()) if n_flag else float("nan")
        fitted_pp = coef_to_pp(coefs.get(name, 0.0), base_rate)
        ci_lo, ci_hi = ci.get(name, (float("nan"), float("nan")))
        disagreement = None if hand_pp is None else round(fitted_pp - hand_pp, 2)
        # A disagreement is only trustworthy if the hand-tuned value falls
        # outside the bootstrap 90% CI of the fitted value — not just a big
        # point-estimate gap, which with ~30 correlated features on ~1200
        # rows can easily be noise.
        significant = (
            hand_pp is not None
            and not np.isnan(ci_lo)
            and (hand_pp < ci_lo or hand_pp > ci_hi)
        )
        rows.append({
            "flag": name,
            "source": source,
            "n": n_flag,
            "reliable": n_flag >= min_n,
            "raw_hit_rate": round(raw_hit_rate, 4) if n_flag else None,
            "hand_tuned_pp": hand_pp,
            "fitted_pp": round(fitted_pp, 2),
            "ci90": (round(ci_lo, 2), round(ci_hi, 2)),
            "disagreement_pp": disagreement,
            "significant": significant,
        })
    return rows


def print_report(rows: list[dict], base_rate: float, n_total: int) -> None:
    print("=" * 108)
    print("  FLAG PENALTY FIT — hand-tuned constants vs. logistic regression on bet_log.csv")
    print("=" * 108)
    print(f"  {n_total} graded bets   ·   baseline hit rate {base_rate:.1%}   ·   "
          f"90% CI from 200 bootstrap resamples")
    print(
        f"  {'Flag':<18} {'n':>5} {'raw hit%':>9} {'hand pp':>9} {'fitted pp':>10} "
        f"{'90% CI (pp)':>16}  reliable  note"
    )
    print(f"  {'-'*18} {'-'*5} {'-'*9} {'-'*9} {'-'*10} {'-'*16}  {'-'*8}  {'-'*30}")
    rows_sorted = sorted(
        rows, key=lambda r: (r["disagreement_pp"] is None, -(abs(r["disagreement_pp"] or 0)))
    )
    flagged = []
    for r in rows_sorted:
        hand = f"{r['hand_tuned_pp']:+.1f}" if r["hand_tuned_pp"] is not None else "  n/a"
        raw = f"{r['raw_hit_rate']:.1%}" if r["raw_hit_rate"] is not None else "   n/a"
        rel = "yes" if r["reliable"] else "LOW-N"
        ci_lo, ci_hi = r["ci90"]
        ci_str = f"[{ci_lo:+.1f}, {ci_hi:+.1f}]" if not np.isnan(ci_lo) else "        n/a"
        note = ""
        if r["reliable"] and r["significant"]:
            note = "⚠ DISAGREES"
            flagged.append(r)
        print(
            f"  {r['flag']:<18} {r['n']:>5} {raw:>9} {hand:>9} {r['fitted_pp']:>+10.2f} "
            f"{ci_str:>16}  {rel:>8}  {note}"
        )
    print("=" * 108)
    print(f"  {len(flagged)} flag(s) disagree with their hand-tuned constant: reliable sample "
          f"AND the hand-tuned value falls outside the fitted 90% CI.")
    for r in flagged:
        ci_lo, ci_hi = r["ci90"]
        print(f"    · {r['flag']}: hand={r['hand_tuned_pp']:+.1f}pp  fitted={r['fitted_pp']:+.2f}pp"
              f"  90% CI [{ci_lo:+.1f}, {ci_hi:+.1f}]"
              f"  (n={r['n']}, raw hit rate {r['raw_hit_rate']:.1%})  — {r['source']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-n", type=int, default=15, help="min sample size to trust a flag's fit")
    ap.add_argument("--n-boot", type=int, default=200, help="bootstrap resamples for the 90% CI")
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    args = ap.parse_args()

    if not LOG_FILE.exists():
        print(f"bet_log.csv not found at {LOG_FILE}")
        return 1

    df = load_graded(LOG_FILE)
    if len(df) < 30:
        print(f"Only {len(df)} graded bets — too few to fit reliably.")
        return 1

    X, feat_names = build_features(df)
    y = df["hit"].values
    base_rate = float(y.mean())

    clf = fit_logistic(X, y)
    coefs = dict(zip(feat_names, clf.coef_[0]))

    print(f"  Bootstrapping {args.n_boot} resamples for confidence intervals...")
    ci = bootstrap_pp(X, y, feat_names, n_boot=args.n_boot)

    rows = build_report(df, feat_names, coefs, base_rate, args.min_n, ci)
    print_report(rows, base_rate, len(df))

    out = {
        "generated_from": "bet_log.csv",
        "n_graded_bets": len(df),
        "baseline_hit_rate": round(base_rate, 4),
        "method": "LogisticRegression(C=1.0) on market dummies + flag booleans; "
                  "pp = coef * base_rate * (1-base_rate) * 100; "
                  f"90% CI from {args.n_boot} bootstrap resamples",
        "flags": {
            r["flag"]: {
                "fitted_pp": r["fitted_pp"],
                "hand_tuned_pp": r["hand_tuned_pp"],
                "ci90_pp": list(r["ci90"]),
                "n": r["n"],
                "reliable": r["reliable"],
                "significant": r["significant"],
                "raw_hit_rate": r["raw_hit_rate"],
                "source": r["source"],
            }
            for r in rows
        },
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
