"""
market_calibration.py — Market Calibration Weighting
=======================================================
Checks whether the sportsbook market's own de-vigged probability (p_market,
from oddstracker.py's devig_power) is itself well-calibrated against
realized outcomes — and where it isn't, turns the gap into a shrinkage-
corrected adjustment, then blends it with the model's own p_model by each
source's tracked reliability (inverse mean-squared-error weighting), rather
than a fixed blend ratio.

Requires bet_log.csv to have p_model/p_market columns (checkresults.py's
_migrate_market_calib_columns() backfills these; see that function's
docstring for coverage caveats — p_market coverage is much thinner than
p_model's, since it depends on odds_tracker.csv history).

Pipeline (mirrors the steps in the "Market Calibration Weighting" spec):
  3. reliability_table()      — bucket by p_market/p_model, compute the gap
  4. shrink_gap()              — n/(n+k) shrinkage toward zero (no correction)
  5. corrected_p_market()      — p_market + shrunk bucket adjustment, clipped
  6. blend_weights()/p_final() — inverse-MSE trust weighting of the two sources
  7. edge_and_ev()              — the connection point into edge/EV (Kelly/
                                   portfolio sizing do not exist in this
                                   codebase yet — this is as far as "feed into
                                   the edge/EV layer" goes today)
  8. holdout_check()            — out-of-sample check that the correction
                                   isn't just fitting bucket noise

Usage:
    python3 market_calibration.py              # fit + write calibration.json
    python3 market_calibration.py --summary    # print full tables
    python3 market_calibration.py --holdout    # step 8's out-of-sample check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from roi_analysis import american_to_decimal

_HERE = Path(__file__).parent
LOG_FILE = _HERE / "bet_log.csv"
CALIB_FILE = _HERE / "calibration.json"

BUCKET_WIDTH = 0.05  # 5-point reliability bands
MIN_N_BUCKET = 4  # matches calibrate.py's MIN_BETS_TO_ADJ — below this, no adjustment at all
SHRINK_N = 30  # shrinkage constant for bucket-gap correction (see shrink_gap())
MIN_N_BLEND = 20  # min per-(market,pick) n before trusting a per-market MSE over the pooled one

# p_market comes from noisy book quotes (a single-book snapshot with a bad
# price can devig to a near-0/near-1 probability) so it gets a tight sanity
# filter. p_model is already bounded to [0.5, ~1.0] by evaluate_prop()'s own
# construction -- no equivalent "junk data" risk -- so it only needs the
# generic non-null / in-[0,1] check.
VALID_P_RANGE = {"p_market": (0.03, 0.97), "p_model": (0.0, 1.0)}
P_CLIP = (0.02, 0.98)  # corrected probabilities are clipped to stay inside this range


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_graded(path: Path = LOG_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["market"] = df["market"].astype(str).str.strip()
    df["pick"] = df["pick"].astype(str).str.strip().str.upper()
    df["result"] = df["result"].astype(str).str.strip().str.upper()
    df = df[df["result"].isin(("HIT", "MISS"))].copy()
    df["hit"] = (df["result"] == "HIT").astype(int)
    for col in ("p_model", "p_market"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def _valid(df: pd.DataFrame, p_col: str) -> pd.DataFrame:
    """Rows with a usable probability in that column (not null, not a
    degenerate devig from bad book data)."""
    lo, hi = VALID_P_RANGE.get(p_col, (0.0, 1.0))
    ok = df[p_col].notna() & (df[p_col] >= lo) & (df[p_col] <= hi)
    return df[ok].copy()


# ---------------------------------------------------------------------------
# Step 3 — reliability curve
# ---------------------------------------------------------------------------
def bucket_of(p: float, width: float = BUCKET_WIDTH) -> str:
    # + a small epsilon before flooring: 0.60 / 0.05 == 11.999999999999998 in
    # float64, which would floor to 11 (bucket "55-60") instead of the
    # intended 12 ("60-65") right at a band boundary.
    lo = np.floor(p / width + 1e-9) * width
    hi = lo + width
    return f"{lo * 100:.0f}-{hi * 100:.0f}"


def reliability_table(df: pd.DataFrame, p_col: str, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Bucket by p_col into 5pp bands (and optionally by group_cols too,
    e.g. ['market', 'pick']), compute n / avg_p / hit_rate / raw_gap per cell.
    raw_gap = hit_rate - avg_p: positive means the source UNDER-states the
    true probability in that bucket (should be pushed up), negative means it
    OVER-states it (should be pushed down)."""
    d = _valid(df, p_col).copy()
    d["_bucket"] = d[p_col].apply(bucket_of)
    keys = ["_bucket"] + (group_cols or [])
    rows = []
    for key, grp in d.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        n = len(grp)
        avg_p = float(grp[p_col].mean())
        hit_rate = float(grp["hit"].mean())
        rec = dict(zip(keys, key))
        rec.update({"n": n, "avg_p": round(avg_p, 4), "hit_rate": round(hit_rate, 4),
                    "raw_gap": round(hit_rate - avg_p, 4)})
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values("n", ascending=False).reset_index(drop=True) if not out.empty else out


# ---------------------------------------------------------------------------
# Step 4 — shrinkage
# ---------------------------------------------------------------------------
def shrink_gap(raw_gap: float, n: int, k: float = SHRINK_N, min_n: int = MIN_N_BUCKET) -> float:
    """n/(n+k) shrinkage toward zero gap (= 'trust the market/model fully').
    Same form as calibrate.py's _shrink_toward_prior / dvp.py's shrinkage —
    below min_n, don't apply any correction at all (too little evidence)."""
    if n < min_n:
        return 0.0
    w = n / (n + k)
    return float(w * raw_gap)


def build_bucket_adjustments(df: pd.DataFrame, p_col: str) -> dict:
    """Pooled-by-band adjustments (step 3's primary output): {bucket: {...}}."""
    table = reliability_table(df, p_col)
    out = {}
    for _, r in table.iterrows():
        out[r["_bucket"]] = {
            "n": int(r["n"]),
            "avg_p": r["avg_p"],
            "hit_rate": r["hit_rate"],
            "raw_gap": r["raw_gap"],
            "adj": round(shrink_gap(r["raw_gap"], r["n"]), 4),
        }
    return out


def build_bucket_adjustments_by_market(df: pd.DataFrame, p_col: str) -> dict:
    """Per-(market, pick, bucket) adjustments — step 3's 'separately by
    market type' breakdown. Almost every cell here will be thin (see the
    module docstring); shrink_gap()'s min_n/k floor is what keeps a n=2 cell
    from doing anything. Keyed 'MARKET_PICK|bucket' to match the flat-string
    convention calibration.json's market_direction already uses."""
    table = reliability_table(df, p_col, group_cols=["market", "pick"])
    out = {}
    for _, r in table.iterrows():
        key = f"{r['market']}_{r['pick']}|{r['_bucket']}"
        out[key] = {
            "n": int(r["n"]),
            "avg_p": r["avg_p"],
            "hit_rate": r["hit_rate"],
            "raw_gap": r["raw_gap"],
            "adj": round(shrink_gap(r["raw_gap"], r["n"]), 4),
        }
    return out


# ---------------------------------------------------------------------------
# Step 5 — corrected p_market
# ---------------------------------------------------------------------------
def corrected_p_market(p_market: float, bucket_adjustments: dict, clip: tuple = P_CLIP) -> float:
    """p_market + the pooled bucket's shrunk adjustment, clipped to a valid
    probability range. Falls back to p_market unchanged if its bucket has no
    (or insufficient) data."""
    b = bucket_of(p_market)
    adj = bucket_adjustments.get(b, {}).get("adj", 0.0)
    lo, hi = clip
    return float(np.clip(p_market + adj, lo, hi))


# ---------------------------------------------------------------------------
# Step 6 — track each source's reliability, blend by trust
# ---------------------------------------------------------------------------
def _mse_by_market(df: pd.DataFrame, p_col: str, pooled_fallback: float) -> dict:
    """Per-(market,pick) mean squared error of p_col vs realized outcome —
    this IS the 'tracked reliability' each source's blend weight is built
    from: lower MSE = that source's probabilities have been closer to what
    actually happened for that market, so it should be trusted more there.
    Falls back to the pooled (all-markets) MSE when a group has too few
    observations to trust its own MSE (MIN_N_BLEND)."""
    d = _valid(df, p_col).copy()
    d["sq_err"] = (d[p_col] - d["hit"]) ** 2
    out = {}
    for (market, pick), grp in d.groupby(["market", "pick"]):
        n = len(grp)
        mse = float(grp["sq_err"].mean()) if n >= MIN_N_BLEND else pooled_fallback
        out[f"{market}_{pick}"] = {"n": n, "mse": round(mse, 5)}
    return out


def build_model_market_reliability(df: pd.DataFrame, bucket_adj_market: dict) -> dict:
    """Step 6's per-source tracked error, used for the inverse-MSE blend
    weights. p_market here is the CORRECTED value (applying the pooled
    bucket adjustment) so the market's tracked MSE reflects the number
    that will actually be blended, not the raw one."""
    d = df.copy()
    d["p_market_corrected"] = d["p_market"].apply(
        lambda p: corrected_p_market(p, bucket_adj_market) if pd.notna(p) else np.nan
    )
    pooled_model = float(((_valid(d, "p_model")["p_model"] - _valid(d, "p_model")["hit"]) ** 2).mean())
    pooled_market = float(
        ((_valid(d, "p_market_corrected")["p_market_corrected"]
          - _valid(d, "p_market_corrected")["hit"]) ** 2).mean()
    ) if not _valid(d, "p_market_corrected").empty else float("nan")

    model_mse = _mse_by_market(d, "p_model", pooled_model)
    market_mse = _mse_by_market(d, "p_market_corrected", pooled_market)
    return {
        "pooled_model_mse": round(pooled_model, 5),
        "pooled_market_mse": round(pooled_market, 5) if not np.isnan(pooled_market) else None,
        "model_mse_by_market": model_mse,
        "market_mse_by_market": market_mse,
    }


def blend_weights(market_key: str, reliability: dict) -> tuple[float, float]:
    """Inverse-MSE weights (w_model, w_market) for one market_pick key."""
    model_mse = reliability["model_mse_by_market"].get(
        market_key, {"mse": reliability["pooled_model_mse"]}
    )["mse"]
    pooled_market = reliability["pooled_market_mse"]
    market_mse = reliability["market_mse_by_market"].get(
        market_key, {"mse": pooled_market if pooled_market is not None else model_mse}
    )["mse"]
    w_model = 1.0 / max(model_mse, 1e-6)
    w_market = 1.0 / max(market_mse, 1e-6)
    return w_model, w_market


def p_final(p_model: float, p_market: float, market_key: str, reliability: dict,
            bucket_adj_market: dict) -> tuple[float, float, float]:
    """Step 5+6 combined: corrects p_market, then blends with p_model by
    each source's tracked reliability for this market. Returns
    (p_final, p_market_corrected, w_market_share) — w_market_share is
    w_market/(w_model+w_market), handy for explaining which way it leaned.
    If p_market is unavailable, p_final falls back to p_model alone
    (w_market_share=0.0) -- no market data means no market vote."""
    if p_market is None or (isinstance(p_market, float) and np.isnan(p_market)):
        return p_model, None, 0.0
    p_mkt_c = corrected_p_market(p_market, bucket_adj_market)
    w_model, w_market = blend_weights(market_key, reliability)
    final = (w_model * p_model + w_market * p_mkt_c) / (w_model + w_market)
    return float(final), p_mkt_c, float(w_market / (w_model + w_market))


# ---------------------------------------------------------------------------
# Step 7 — edge / EV (the connection point; no Kelly/portfolio module exists
# in this codebase yet to feed further into)
# ---------------------------------------------------------------------------
def edge_and_ev(p_final_value: float, p_market_raw: float, american_odds: float) -> dict:
    """edge = p_final - p_market (the ORIGINAL, uncorrected p_market — the
    correction is already folded into p_final, so using p_market_corrected
    here would double-count it). EV is per $1 stake."""
    decimal_odds = american_to_decimal(american_odds)
    edge = p_final_value - p_market_raw
    ev = p_final_value * decimal_odds - 1.0
    return {"edge": round(edge, 4), "ev_per_1": round(ev, 4), "decimal_odds": round(decimal_odds, 4)}


# ---------------------------------------------------------------------------
# Step 8 — holdout verification
# ---------------------------------------------------------------------------
def holdout_check(df: pd.DataFrame, p_col: str = "p_market", test_frac: float = 0.3) -> dict:
    """Temporal split (train = earliest dates, test = most recent): fit
    bucket adjustments on train only, then compare mean squared calibration
    error on test with vs. without the correction. If the corrected MSE
    isn't actually lower out-of-sample, the correction is fitting bucket
    noise, not a real market bias -- don't trust it for sizing yet."""
    d = _valid(df, p_col).sort_values("date").reset_index(drop=True)
    n = len(d)
    if n < 30:
        return {"ok": False, "reason": f"only {n} usable ({p_col}) rows — too few for a holdout split"}
    split = int(n * (1 - test_frac))
    train, test = d.iloc[:split], d.iloc[split:]

    adj = build_bucket_adjustments(train, p_col)
    test = test.copy()
    test["p_corrected"] = test[p_col].apply(lambda p: corrected_p_market(p, adj))

    mse_raw = float(((test[p_col] - test["hit"]) ** 2).mean())
    mse_corrected = float(((test["p_corrected"] - test["hit"]) ** 2).mean())
    return {
        "ok": True,
        "n_train": len(train),
        "n_test": len(test),
        "test_date_range": [str(test["date"].min()), str(test["date"].max())],
        "mse_raw_p_market": round(mse_raw, 5),
        "mse_corrected_p_market": round(mse_corrected, 5),
        "improved": mse_corrected < mse_raw,
        "improvement": round(mse_raw - mse_corrected, 5),
    }


# ---------------------------------------------------------------------------
# Print / main
# ---------------------------------------------------------------------------
def print_summary(df, bucket_market, bucket_model, reliability, holdout) -> None:
    n_market = len(_valid(df, "p_market"))
    n_model = len(_valid(df, "p_model"))
    print("=" * 96)
    print("  MARKET CALIBRATION WEIGHTING")
    print("=" * 96)
    print(f"  {len(df)} graded bets total   ·   {n_model} with usable p_model   ·   "
          f"{n_market} with usable p_market")
    print()

    print("  Market (p_market) reliability by band — pooled across markets:")
    print(f"  {'Band':<10} {'n':>5} {'avg p_market':>13} {'hit rate':>9} {'raw gap':>8} {'shrunk adj':>10}")
    for band, v in sorted(bucket_market.items()):
        print(f"  {band:<10} {v['n']:>5} {v['avg_p']:>13.3f} {v['hit_rate']:>9.3f} "
              f"{v['raw_gap']:>+8.3f} {v['adj']:>+10.3f}")

    print()
    print("  Model (p_model) reliability by band — pooled across markets:")
    print(f"  {'Band':<10} {'n':>5} {'avg p_model':>13} {'hit rate':>9} {'raw gap':>8} {'shrunk adj':>10}")
    for band, v in sorted(bucket_model.items()):
        print(f"  {band:<10} {v['n']:>5} {v['avg_p']:>13.3f} {v['hit_rate']:>9.3f} "
              f"{v['raw_gap']:>+8.3f} {v['adj']:>+10.3f}")

    print()
    print(f"  Pooled MSE — model: {reliability['pooled_model_mse']:.5f}   "
          f"market (corrected): {reliability['pooled_market_mse']}")
    print("  (lower MSE = more reliable source; this is what sets w_model/w_market per market)")

    if holdout.get("ok"):
        print()
        print(f"  Step 8 — holdout check (train/test split, {holdout['n_train']}/{holdout['n_test']}):")
        print(f"    test window: {holdout['test_date_range'][0]} .. {holdout['test_date_range'][1]}")
        print(f"    MSE raw p_market:       {holdout['mse_raw_p_market']:.5f}")
        print(f"    MSE corrected p_market: {holdout['mse_corrected_p_market']:.5f}")
        verdict = "IMPROVES calibration out-of-sample" if holdout["improved"] else \
                  "does NOT improve out-of-sample — likely fitting bucket noise, not signal"
        print(f"    -> correction {verdict} (Δ={holdout['improvement']:+.5f})")
    else:
        print(f"\n  Step 8 — holdout check skipped: {holdout.get('reason')}")
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--holdout", action="store_true", help="also run (and print) the step-8 check")
    ap.add_argument("--out", type=Path, default=CALIB_FILE)
    args = ap.parse_args()

    if not LOG_FILE.exists():
        print(f"bet_log.csv not found at {LOG_FILE}")
        return 1

    df = load_graded(LOG_FILE)

    bucket_market = build_bucket_adjustments(df, "p_market")
    bucket_market_by_mkt = build_bucket_adjustments_by_market(df, "p_market")
    bucket_model = build_bucket_adjustments(df, "p_model")
    reliability = build_model_market_reliability(df, bucket_market)
    holdout = holdout_check(df, "p_market") if (args.holdout or args.summary) else {"ok": False, "reason": "not requested"}

    if args.summary:
        print_summary(df, bucket_market, bucket_model, reliability, holdout)
    elif args.holdout:
        print(json.dumps(holdout, indent=2))

    # Merge into calibration.json rather than overwrite it -- this module
    # adds a section, it doesn't own the whole file (calibrate.py does).
    existing = {}
    if args.out.exists():
        with open(args.out) as fh:
            existing = json.load(fh)
    existing["market_calibration"] = {
        "n_graded": len(df),
        "n_with_p_market": len(_valid(df, "p_market")),
        "n_with_p_model": len(_valid(df, "p_model")),
        "market_calib_adj": bucket_market,
        "market_calib_adj_by_market": bucket_market_by_mkt,
        "model_calib_adj": bucket_model,
        "reliability": reliability,
        "holdout_check": holdout if holdout.get("ok") else None,
    }
    with open(args.out, "w") as fh:
        json.dump(existing, fh, indent=2)
    print(f"\n  Wrote market_calibration section -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
