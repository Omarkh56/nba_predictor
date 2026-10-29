"""Replay metrics: exact-market bias table, chronological splits, paired clustered bootstrap.

Residual convention everywhere: residual = projection - actual  (positive = the model over-projects).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from replay_config import BOOTSTRAP_REPS, BOOTSTRAP_SEED, MARKET_ORDER, SPLIT_FRACTIONS, SPLITS_FILE


def market_metrics(df: pd.DataFrame, proj_col: str = "projection") -> Dict[str, float]:
    if df.empty:
        return {"n": 0}
    res = df[proj_col].values - df["actual"].values
    mean_actual = float(df["actual"].mean())
    return {
        "n": int(len(df)),
        "mean_actual": mean_actual,
        "mean_projection": float(df[proj_col].mean()),
        "mean_residual": float(res.mean()),
        "pct_bias": float(res.mean() / mean_actual) if mean_actual else float("nan"),
        "median_residual": float(np.median(res)),
        "mae": float(np.abs(res).mean()),
        "rmse": float(np.sqrt((res ** 2).mean())),
        "p_actual_gt_projection": float((df["actual"].values > df[proj_col].values).mean()),
    }


def metrics_table(df: pd.DataFrame, proj_col: str = "projection") -> pd.DataFrame:
    """One row per exact market (fixed order) plus an ALL row pooled over every market."""
    rows = []
    for m in MARKET_ORDER:
        sub = df[df["market"] == m]
        if len(sub):
            rows.append(dict(market=m, **market_metrics(sub, proj_col)))
    rows.append(dict(market="ALL", **market_metrics(df, proj_col)))
    return pd.DataFrame(rows).set_index("market")


def macro_pct_bias(table: pd.DataFrame) -> float:
    """Unweighted mean of per-market percentage biases (ALL row excluded)."""
    t = table.drop(index="ALL", errors="ignore")
    return float(t["pct_bias"].mean()) if len(t) else float("nan")


# ---------------------------------------------------------------------------------------------- splits
def compute_splits(df: pd.DataFrame, fractions: Sequence[float] = SPLIT_FRACTIONS) -> Dict[str, Dict[str, str]]:
    """Chronological train/validation/test boundaries per season type, by observation count over distinct dates."""
    out: Dict[str, Dict[str, str]] = {}
    for st, g in df.groupby("season_type"):
        per_date = g.groupby("date").size().sort_index()
        cum = per_date.cumsum() / per_date.sum()
        train_end = cum.index[(cum >= fractions[0]).values.argmax()]
        val_end = cum.index[(cum >= fractions[0] + fractions[1]).values.argmax()]
        out[st] = {"train_end": str(train_end), "val_end": str(val_end)}
    return out


def load_or_create_splits(df: pd.DataFrame, season: str, path: str = SPLITS_FILE,
                          persist: bool = True) -> Dict[str, Dict[str, str]]:
    """Fixed chronological boundaries, created once per season/regime and then frozen so every later model
    version is evaluated on identical train/validation/untouched-test periods.

    A regime already present in the file is never recomputed. New regimes are added only when `persist` is True
    (pass False for partial-date-window runs, which must not define the boundaries)."""
    data: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
    known = dict(data.get(season, {}))
    fresh = compute_splits(df)
    changed = False
    for regime, bounds in fresh.items():
        if regime not in known:
            known[regime] = bounds
            changed = True
    if changed and persist:
        data[season] = known
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
    return known


def assign_split(df: pd.DataFrame, splits: Dict[str, Dict[str, str]]) -> pd.Series:
    def one(row_date: str, st: str) -> str:
        b = splits[st]
        return "train" if row_date <= b["train_end"] else ("val" if row_date <= b["val_end"] else "test")

    return pd.Series([one(d, s) for d, s in zip(df["date"], df["season_type"])], index=df.index)


# ---------------------------------------------------------------------------------------------- paired comparison
def paired_frame(a: pd.DataFrame, b: pd.DataFrame, keys: Sequence[str] = ("date", "player_id", "market"),
                 a_col: str = "projection", b_col: str = "projection") -> pd.DataFrame:
    left = a[list(keys) + ["actual", a_col]].rename(columns={a_col: "proj_a"})
    right = b[list(keys) + [b_col]].rename(columns={b_col: "proj_b"})
    return left.merge(right, on=list(keys), how="inner", validate="one_to_one")


def paired_bootstrap(paired: pd.DataFrame, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    """Change B - A in |mean residual|, MAE and RMSE with a 95% CI, resampling whole game DATES (a slate's
    observations are correlated). Negative = B is better."""
    if paired.empty:
        return {"n_paired": 0}
    ra = paired["proj_a"].values - paired["actual"].values
    rb = paired["proj_b"].values - paired["actual"].values
    tmp = pd.DataFrame({"date": paired["date"].values, "n": 1, "ra": ra, "rb": rb,
                        "aa": np.abs(ra), "ab": np.abs(rb), "sa": ra ** 2, "sb": rb ** 2})
    g = tmp.groupby("date").sum()
    S = g[["n", "ra", "rb", "aa", "ab", "sa", "sb"]].values
    D = len(S)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(D, np.full(D, 1.0 / D), size=reps).astype(float)
    tot = w @ S  # (reps, 7)
    n = tot[:, 0]

    def stat(t: np.ndarray, which: str) -> np.ndarray:
        if which == "absbias":
            return np.abs(t[:, 2] / n) - np.abs(t[:, 1] / n)
        if which == "mae":
            return t[:, 4] / n - t[:, 3] / n
        return np.sqrt(t[:, 6] / n) - np.sqrt(t[:, 5] / n)

    full = S.sum(axis=0)
    nn = full[0]
    point = {
        "abs_bias": float(abs(full[2] / nn) - abs(full[1] / nn)),
        "mae": float(full[4] / nn - full[3] / nn),
        "rmse": float(np.sqrt(full[6] / nn) - np.sqrt(full[5] / nn)),
    }
    out: Dict[str, Any] = {"n_paired": int(len(paired)), "n_dates": int(D)}
    for key, which in (("abs_bias", "absbias"), ("mae", "mae"), ("rmse", "rmse")):
        boot = stat(tot, which)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[key] = {"delta": point[key], "ci95": [float(lo), float(hi)], "significant": bool(lo > 0 or hi < 0)}
    return out
