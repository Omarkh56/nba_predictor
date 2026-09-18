"""
verify_kelly_sizing.py — Step 7: verify fractional-Kelly stake sizing
against history before trusting it live.

playerlinepredictor.compute_stake_pct() has never run against a real pick
(stake_pct/realized_profit are brand new columns in bet_log.csv with no
historical values — see checkresults.py's _migrate_stake_columns()). This
retroactively applies the formula to every graded historical bet that has
a usable p_model + decimal_odds (backfilled from odds_tracker.csv), using
each row's Flags to reconstruct an approximate total_penalty — the same
flag vocabulary and calibrated pp values already used to dock confidence,
via fit_flag_penalties.PENALTY_MAP and calibration.json's flag_penalties.

Checks:
  1. Every negative-EV historical bet receives exactly zero stake.
  2. No historical bet would have been sized above MAX_STAKE_PCT.
  3. Discount ordering: a thin-book / high-mins-volatility pick's stake
     should come out smaller than a well-covered pick's, for similar raw
     Kelly fraction — i.e. the risk discount is actually doing its job,
     not just leaving stakes unchanged regardless of flags.

Usage:
    python3 verify_kelly_sizing.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import fit_flag_penalties as ffp
import playerlinepredictor as props_model

_HERE = Path(__file__).parent
LOG_FILE = _HERE / "bet_log.csv"
CALIB_FILE = _HERE / "calibration.json"


def _approx_total_penalty(flags: str, pick: str, market: str, flag_penalties: dict) -> float:
    """Reconstruct an approximate total_penalty for a historical row from
    its Flags string, using the SAME matchers _flag_penalty() effectively
    inverts live (fit_flag_penalties.PENALTY_MAP), and the calibrated pp
    (falling back to the hand-tuned constant when calibration.json has no
    entry / insufficient data for that flag)."""
    row = {"pick": pick, "market": market}
    total = 0.0
    for name, (matcher, hand_pp, _source) in ffp.PENALTY_MAP.items():
        if not matcher(flags, row):
            continue
        entry = flag_penalties.get(name)
        pp = entry["pp"] if entry else hand_pp
        if pp is None:
            continue
        total += -pp  # pp is "confidence points gained"; penalty is its negation
    return total


def load_history() -> pd.DataFrame:
    df = pd.read_csv(LOG_FILE)
    df["flags"] = df["flags"].fillna("")
    df["market"] = df["market"].astype(str).str.strip()
    df["pick"] = df["pick"].astype(str).str.strip().str.upper()
    df["result"] = df["result"].astype(str).str.strip().str.upper()
    df = df[df["result"].isin(("HIT", "MISS"))].copy()
    for col in ("p_model", "decimal_odds"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["p_model"].notna() & df["decimal_odds"].notna() & (df["decimal_odds"] > 1.0)].copy()


def main() -> int:
    if not LOG_FILE.exists():
        print(f"bet_log.csv not found at {LOG_FILE}")
        return 1

    flag_penalties = {}
    if CALIB_FILE.exists():
        with open(CALIB_FILE) as fh:
            flag_penalties = json.load(fh).get("flag_penalties", {})

    df = load_history()
    n = len(df)
    print("=" * 96)
    print("  KELLY STAKE SIZING — retroactive verification against history")
    print("=" * 96)
    print(f"  {n} historical graded bets with usable p_model + decimal_odds "
          f"(of {pd.read_csv(LOG_FILE)['result'].isin(['HIT', 'MISS']).sum()} graded total)")
    if n == 0:
        print("  Nothing to verify.")
        return 0

    df["total_penalty_approx"] = [
        _approx_total_penalty(f, p, m, flag_penalties)
        for f, p, m in zip(df["flags"], df["pick"], df["market"])
    ]
    df["ev"] = df["p_model"] * df["decimal_odds"] - 1.0
    df["stake_pct"] = [
        props_model.compute_stake_pct(p, d, tp)
        for p, d, tp in zip(df["p_model"], df["decimal_odds"], df["total_penalty_approx"])
    ]
    # Pre-discount stake (f* * KELLY_FRACTION only) -- used to isolate what
    # the risk discount itself is doing, separate from the raw Kelly math.
    df["f_star"] = np.maximum(0.0, df["ev"] / (df["decimal_odds"] - 1.0))
    df["stake_pre_discount"] = df["f_star"] * props_model.KELLY_FRACTION

    # ── Check 1: negative EV -> zero stake ──────────────────────────────
    neg_ev = df[df["ev"] < 0]
    bad_neg_ev = neg_ev[neg_ev["stake_pct"] > 0]
    check1 = bad_neg_ev.empty
    print("\n  Check 1 — negative-EV bets get zero stake:")
    print(f"    {len(neg_ev)} negative-EV historical bets, {len(bad_neg_ev)} of them got stake > 0")
    print(f"    -> {'PASS' if check1 else 'FAIL'}")

    # ── Check 2: hard cap respected ──────────────────────────────────────
    over_cap = df[df["stake_pct"] > props_model.MAX_STAKE_PCT + 1e-9]
    check2 = over_cap.empty
    print(f"\n  Check 2 — no bet sized above MAX_STAKE_PCT ({props_model.MAX_STAKE_PCT:.2%}):")
    print(f"    max stake_pct observed: {df['stake_pct'].max():.4f}")
    print(f"    -> {'PASS' if check2 else 'FAIL'} ({len(over_cap)} violation(s))")

    # ── Check 3: discount ordering vs. risk flags ────────────────────────
    thin_book = df["flags"].str.contains("1-BOOK|2-BOOK", regex=True)
    high_vol = df["flags"].str.contains("MINS-VOL", regex=False)
    any_risk = thin_book | high_vol

    def _discount_ratio(sub):
        # stake_pct / stake_pre_discount = the discount factor actually
        # applied (1.0 = no shrinkage, 0.0 = fully zeroed). Only meaningful
        # where there was something to discount (stake_pre_discount > 0).
        s = sub[sub["stake_pre_discount"] > 1e-9]
        return float((s["stake_pct"] / s["stake_pre_discount"]).mean()) if len(s) else float("nan")

    ratio_risky = _discount_ratio(df[any_risk])
    ratio_clean = _discount_ratio(df[~any_risk])
    check3 = (not np.isnan(ratio_risky) and not np.isnan(ratio_clean) and ratio_risky < ratio_clean)
    print("\n  Check 3 — flagged picks get MORE shrinkage than clean picks:")
    print(f"    mean discount factor, 1-BOOK/2-BOOK/MINS-VOL flagged (n={any_risk.sum()}): "
          f"{ratio_risky:.3f}" if not np.isnan(ratio_risky) else "    (no flagged rows with a pre-discount stake)")
    print(f"    mean discount factor, unflagged (n={(~any_risk).sum()}): "
          f"{ratio_clean:.3f}" if not np.isnan(ratio_clean) else "    (no unflagged rows with a pre-discount stake)")
    print(f"    -> {'PASS' if check3 else 'FAIL / INCONCLUSIVE'}"
          f"{'' if check3 else ' — consider tuning STAKE_PENALTY_DIVISOR'}")

    print(f"\n  Overall stake_pct distribution (n={n}):")
    print(f"    mean={df['stake_pct'].mean():.4f}  median={df['stake_pct'].median():.4f}  "
          f"max={df['stake_pct'].max():.4f}  "
          f"n(stake>0)={int((df['stake_pct'] > 0).sum())}")
    print("=" * 96)

    return 0 if (check1 and check2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
