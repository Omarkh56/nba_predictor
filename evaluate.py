"""
evaluate.py — Model Evaluation Framework
=========================================
Drop any model's predictions in; get back calibration, significance tests,
temporal stability, and an optional market comparison. Every run is appended
to eval_log.jsonl so you can compare across iterations.

Quick usage
-----------
    from evaluate import evaluate, print_report

    report = evaluate(
        y_true          = df_val["home_win"].values,
        probs           = my_model_probs,
        dates           = df_val["game_date"].values,
        model_name      = "xgboost_v1",
        season          = "2023-24",
        split           = "val",
        baseline_probs  = hand_tuned_probs,   # optional but recommended
    )
    print_report(report)

CLI usage
---------
    python3 evaluate.py                                 # last 15 logged runs
    python3 evaluate.py --all                           # all runs
    python3 evaluate.py --run <run_id>                  # detail for one run
    python3 evaluate.py --compare <run_id_a> <run_id_b> # side-by-side diff
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _sp_stats

_HERE    = Path(__file__).parent
LOG_PATH = _HERE / "eval_log.jsonl"

_EPS = 1e-12   # numerical floor for log computations

# =============================================================================
# SECTION 1 — CORE METRIC PRIMITIVES
# =============================================================================

def _acc(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean((p >= threshold).astype(int) == y))

def _ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def _bs(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


# =============================================================================
# SECTION 2 — CALIBRATION
# =============================================================================

def calibration_table(
    y_true: np.ndarray,
    probs:  np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Bucket predictions into n equal-width bins; report actual win rate per bin.
    Returns a DataFrame you can inspect or pass to print_report().

    Over-confidence: error > 0  (model predicts higher prob than observed)
    Under-confidence: error < 0
    """
    y, p = np.asarray(y_true, float), np.asarray(probs, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        pred = float(p[mask].mean())
        actual = float(y[mask].mean())
        rows.append({
            "bucket":        f"{int(lo*100)}-{int(hi*100)}%",
            "lo":            lo,
            "hi":            hi,
            "n":             n,
            "pred_prob":     round(pred, 4),
            "actual_rate":   round(actual, 4),
            "error":         round(pred - actual, 4),
        })
    return pd.DataFrame(rows)


# =============================================================================
# SECTION 3 — TEMPORAL STABILITY
# =============================================================================

def stability_by_month(
    y_true: np.ndarray,
    probs:  np.ndarray,
    dates:  Sequence,
    min_n:  int = 10,
) -> pd.DataFrame:
    """
    Per-month accuracy and log loss. Surfaces variance across the season —
    a model that 'wins' overall but collapses in March is not stable.
    """
    y, p = np.asarray(y_true, float), np.asarray(probs, float)
    months = pd.to_datetime(dates).to_period("M")
    rows = []
    for period in sorted(months.unique()):
        mask = np.asarray(months == period)
        n = int(mask.sum())
        if n < min_n:
            continue
        rows.append({
            "month":          str(period),
            "n":              n,
            "accuracy":       round(_acc(y[mask], p[mask]), 4),
            "log_loss":       round(_ll(y[mask],  p[mask]), 4),
            "brier":          round(_bs(y[mask],  p[mask]), 4),
            "home_win_rate":  round(float(y[mask].mean()), 4),
        })
    return pd.DataFrame(rows)


def stability_rolling(
    y_true:  np.ndarray,
    probs:   np.ndarray,
    dates:   Sequence,
    window:  int = 30,
) -> pd.DataFrame:
    """
    Rolling window accuracy and log loss (sorted by date).
    Useful for plotting — identifies when a model goes cold.
    """
    y, p = np.asarray(y_true, float), np.asarray(probs, float)
    order = np.argsort(pd.to_datetime(dates).values)
    y, p = y[order], p[order]
    dates_s = pd.to_datetime(dates).values[order]
    rows = []
    for i in range(window, len(y) + 1):
        yw, pw = y[i - window:i], p[i - window:i]
        rows.append({
            "end_game_idx": i,
            "end_date":     str(dates_s[i - 1])[:10],
            "accuracy":     round(_acc(yw, pw), 4),
            "log_loss":     round(_ll(yw,  pw), 4),
        })
    return pd.DataFrame(rows)


# =============================================================================
# SECTION 4 — STATISTICAL TESTS
# =============================================================================

def mcnemar_test(
    y_true:   np.ndarray,
    probs_a:  np.ndarray,
    probs_b:  np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    McNemar's test on win/loss calls: is B's error pattern significantly
    different from A's on the SAME games?

    Contingency table (each cell = number of games):
        A correct, B correct  |  A correct, B wrong  (c)
        A wrong,   B correct  |  A wrong,   B wrong
                              (b)

    Only the off-diagonal cells (b, c) carry information.
    Uses continuity-corrected chi-squared when b+c > 25, exact binomial otherwise.

    Positive c - b  →  B makes fewer errors  →  B is better.
    """
    y  = np.asarray(y_true, float)
    pa = (np.asarray(probs_a) >= threshold).astype(int)
    pb = (np.asarray(probs_b) >= threshold).astype(int)

    b = int(((pa == y) & (pb != y)).sum())   # A right, B wrong
    c = int(((pa != y) & (pb == y)).sum())   # A wrong,  B right
    nd = b + c

    if nd == 0:
        return {"chi2": None, "p_value": 1.0, "b": 0, "c": 0,
                "n_discordant": 0, "direction": "identical",
                "interpretation": "Models make identical calls on every game"}

    if nd > 25:
        chi2 = float((abs(b - c) - 1.0) ** 2 / nd)
        p    = float(1 - _sp_stats.chi2.cdf(chi2, df=1))
    else:
        chi2 = None
        k    = min(b, c)
        p    = float(2 * _sp_stats.binom.cdf(k, nd, 0.5))
        p    = min(p, 1.0)

    direction = "B_better" if c > b else ("A_better" if b > c else "tied")

    if p < 0.05:
        winner = "B" if c > b else "A"
        interp = f"Model {winner} significantly better (p={p:.3f})"
    else:
        interp = f"No significant difference in win/loss calls (p={p:.3f})"

    return {
        "chi2":          round(chi2, 4) if chi2 is not None else None,
        "p_value":       round(p, 4),
        "b_a_right_b_wrong": b,
        "c_a_wrong_b_right": c,
        "n_discordant":  nd,
        "direction":     direction,
        "interpretation": interp,
    }


def bootstrap_ci(
    y_true:     np.ndarray,
    probs:      np.ndarray,
    metric_fn:  Callable[[np.ndarray, np.ndarray], float],
    n_boot:     int   = 5_000,
    alpha:      float = 0.05,
    seed:       int   = 42,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for any scalar metric.
    Returns (lower_bound, upper_bound) at (1-alpha) confidence level.
    """
    y, p = np.asarray(y_true, float), np.asarray(probs, float)
    rng  = np.random.default_rng(seed)
    n    = len(y)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx     = rng.integers(0, n, n)
        boot[i] = metric_fn(y[idx], p[idx])
    lo = float(np.percentile(boot, alpha / 2 * 100))
    hi = float(np.percentile(boot, (1 - alpha / 2) * 100))
    return lo, hi


def bootstrap_diff_ci(
    y_true:     np.ndarray,
    probs_a:    np.ndarray,
    probs_b:    np.ndarray,
    metric_fn:  Callable[[np.ndarray, np.ndarray], float],
    n_boot:     int   = 5_000,
    alpha:      float = 0.05,
    seed:       int   = 42,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for metric(A) - metric(B).
    CI entirely above 0 → A strictly better; entirely below 0 → B strictly better.
    """
    y, pa, pb = (np.asarray(x, float) for x in (y_true, probs_a, probs_b))
    rng = np.random.default_rng(seed)
    n   = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx      = rng.integers(0, n, n)
        diffs[i] = metric_fn(y[idx], pa[idx]) - metric_fn(y[idx], pb[idx])
    lo = float(np.percentile(diffs, alpha / 2 * 100))
    hi = float(np.percentile(diffs, (1 - alpha / 2) * 100))
    return lo, hi


# =============================================================================
# SECTION 5 — MARKET COMPARISON
# =============================================================================

def market_comparison(
    y_true:       np.ndarray,
    probs_model:  np.ndarray,
    probs_market: np.ndarray,
) -> dict:
    """
    Compare model's implied probabilities against closing line implied probs.

    probs_market must be de-vigged (true probability, not raw implied).
    Pass None to skip — the main evaluate() function handles that guard.

    Key questions answered:
      1. Does the model beat the market on accuracy / log-loss / Brier?
      2. When model and market DISAGREE, who is right more often?
         (Disagreement accuracy > 50% → model has genuine edge vs line.)
      3. On games where the model is ≥5pp more confident than the market,
         does that extra confidence pay off?
    """
    y  = np.asarray(y_true,       float)
    pm = np.asarray(probs_model,  float)
    pk = np.asarray(probs_market, float)
    n  = len(y)

    model_m  = {"accuracy": _acc(y, pm), "log_loss": _ll(y, pm), "brier": _bs(y, pm)}
    market_m = {"accuracy": _acc(y, pk), "log_loss": _ll(y, pk), "brier": _bs(y, pk)}
    delta    = {k: round(model_m[k] - market_m[k], 4) for k in model_m}

    # Agreement / disagreement split
    agree_mask    = (pm >= 0.5) == (pk >= 0.5)
    disagree_mask = ~agree_mask
    n_agree    = int(agree_mask.sum())
    n_disagree = int(disagree_mask.sum())

    disagree_acc = None
    if n_disagree >= 10:
        disagree_acc = round(_acc(y[disagree_mask], pm[disagree_mask]), 4)

    # Games where model is notably more confident than market
    conf_edge = np.abs(pm - 0.5) - np.abs(pk - 0.5)
    edge_mask  = conf_edge > 0.05
    n_edge     = int(edge_mask.sum())
    edge_info  = None
    if n_edge >= 10:
        edge_info = {
            "n":              n_edge,
            "model_accuracy": round(_acc(y[edge_mask], pm[edge_mask]), 4),
            "market_accuracy": round(_acc(y[edge_mask], pk[edge_mask]), 4),
        }

    # McNemar: are the model's calls significantly different from market calls?
    mn = mcnemar_test(y, pm, pk)

    return {
        "n_games":         n,
        "model":           {k: round(v, 4) for k, v in model_m.items()},
        "market":          {k: round(v, 4) for k, v in market_m.items()},
        "delta":           delta,
        "n_agree":         n_agree,
        "n_disagree":      n_disagree,
        "disagree_model_accuracy": disagree_acc,
        "model_edge_games": edge_info,
        "mcnemar":         mn,
    }


# =============================================================================
# SECTION 6 — MAIN EVALUATE ENTRY POINT
# =============================================================================

def evaluate(
    y_true:         np.ndarray | Sequence,
    probs:          np.ndarray | Sequence,
    dates:          Sequence,
    model_name:     str,
    season:         str,
    split:          str,                    # "val" or "test"
    baseline_probs: Optional[np.ndarray | Sequence] = None,
    market_probs:   Optional[np.ndarray | Sequence] = None,
    n_boot:         int  = 5_000,
    save:           bool = True,
    log_path:       Path = LOG_PATH,
) -> dict:
    """
    Full evaluation of one model on one season/split.

    Parameters
    ----------
    y_true         : (N,) array — 1 = home win, 0 = home loss
    probs          : (N,) predicted home-win probability in [0, 1]
    dates          : (N,) game dates (anything pd.to_datetime accepts)
    model_name     : short identifier, e.g. "logreg_v1" or "xgb_depth4"
    season         : e.g. "2023-24"
    split          : "val" or "test"
    baseline_probs : optional — if supplied, adds McNemar + bootstrap CI
                     comparing model vs baseline (typically the hand-tuned model)
    market_probs   : optional — de-vigged closing line home-win probability
    n_boot         : bootstrap resamples.  5 000 is fast; use 10 000 for
                     tighter CIs before making a deployment decision
    save           : append result to log_path (JSONL)
    log_path       : path to the append-only evaluation log

    Returns
    -------
    dict with keys: run_id, model_name, season, split, timestamp, n_games,
    metrics, calibration, stability_monthly, vs_baseline (opt), vs_market (opt)
    """
    y  = np.asarray(y_true, float)
    p  = np.asarray(probs,  float)
    n  = len(y)

    if n == 0:
        raise ValueError("y_true is empty")
    if not np.isfinite(p).all():
        raise ValueError("probs contains NaN or Inf")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + model_name

    # ── Point estimates ──────────────────────────────────────────────────
    acc = _acc(y, p)
    ll  = _ll(y, p)
    bs  = _bs(y, p)

    # ── Bootstrap CIs (on the model alone) ───────────────────────────────
    acc_lo, acc_hi = bootstrap_ci(y, p, _acc, n_boot=n_boot)
    ll_lo,  ll_hi  = bootstrap_ci(y, p, _ll,  n_boot=n_boot)
    bs_lo,  bs_hi  = bootstrap_ci(y, p, _bs,  n_boot=n_boot)

    result: dict = {
        "run_id":        run_id,
        "model_name":    model_name,
        "season":        season,
        "split":         split,
        "timestamp":     datetime.now().isoformat(timespec="seconds"),
        "n_games":       n,
        "home_win_rate": round(float(y.mean()), 4),
        "metrics": {
            "accuracy":    round(acc, 4),
            "accuracy_ci": [round(acc_lo, 4), round(acc_hi, 4)],
            "log_loss":    round(ll, 4),
            "log_loss_ci": [round(ll_lo, 4), round(ll_hi, 4)],
            "brier":       round(bs, 4),
            "brier_ci":    [round(bs_lo, 4), round(bs_hi, 4)],
        },
        "calibration":        calibration_table(y, p).to_dict("records"),
        "stability_monthly":  stability_by_month(y, p, dates).to_dict("records"),
    }

    # ── vs baseline ───────────────────────────────────────────────────────
    if baseline_probs is not None:
        bp = np.asarray(baseline_probs, float)
        b_acc = _acc(y, bp)
        b_ll  = _ll(y, bp)
        b_bs  = _bs(y, bp)

        # Bootstrap CIs on the DIFFERENCES (model - baseline)
        # Positive acc delta = model better; negative ll delta = model better
        acc_d_lo, acc_d_hi = bootstrap_diff_ci(y, p, bp, _acc, n_boot=n_boot)
        ll_d_lo,  ll_d_hi  = bootstrap_diff_ci(y, p, bp, _ll,  n_boot=n_boot)
        bs_d_lo,  bs_d_hi  = bootstrap_diff_ci(y, p, bp, _bs,  n_boot=n_boot)

        result["vs_baseline"] = {
            "baseline_accuracy":    round(b_acc, 4),
            "baseline_log_loss":    round(b_ll,  4),
            "baseline_brier":       round(b_bs,  4),
            "accuracy_delta":       round(acc - b_acc, 4),
            "log_loss_delta":       round(ll  - b_ll,  4),
            "brier_delta":          round(bs  - b_bs,  4),
            # CI on difference; entirely one side of 0 = significant
            "accuracy_diff_ci_95":  [round(acc_d_lo, 4), round(acc_d_hi, 4)],
            "log_loss_diff_ci_95":  [round(ll_d_lo,  4), round(ll_d_hi,  4)],
            "brier_diff_ci_95":     [round(bs_d_lo,  4), round(bs_d_hi,  4)],
            # McNemar: game-by-game win/loss comparison
            "mcnemar":              mcnemar_test(y, p, bp),
            # Significance flags for the two core metrics
            "sig_accuracy":  acc_d_lo > 0,     # CI entirely above 0 → model better
            "sig_log_loss":  ll_d_hi  < 0,     # CI entirely below 0 → model lower LL
        }

    # ── vs market ─────────────────────────────────────────────────────────
    if market_probs is not None:
        result["vs_market"] = market_comparison(
            y, p, np.asarray(market_probs, float)
        )

    if save:
        _append_log(result, log_path)

    return result


# =============================================================================
# SECTION 7 — LOGGING & LOG QUERIES
# =============================================================================

def _append_log(result: dict, log_path: Path = LOG_PATH) -> None:
    """Append one result as a single JSON line (JSONL format)."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as fh:
        fh.write(json.dumps(result, default=str) + "\n")


def load_log(log_path: Path = LOG_PATH) -> pd.DataFrame:
    """
    Load all logged runs as a flat DataFrame (one row per run).
    Useful for a quick cross-run comparison table.

    Nested fields (calibration, stability_monthly, vs_baseline.*) are
    included as columns where unambiguous; the raw dict is always in the
    original log file if you need deeper access.
    """
    lp = Path(log_path)
    if not lp.exists() or lp.stat().st_size == 0:
        return pd.DataFrame()

    rows = []
    with open(lp) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = r.get("metrics", {})
            flat = {
                "run_id":     r.get("run_id"),
                "model_name": r.get("model_name"),
                "season":     r.get("season"),
                "split":      r.get("split"),
                "timestamp":  r.get("timestamp"),
                "n_games":    r.get("n_games"),
                "accuracy":   m.get("accuracy"),
                "acc_ci_lo":  (m.get("accuracy_ci") or [None])[0],
                "acc_ci_hi":  (m.get("accuracy_ci") or [None, None])[1],
                "log_loss":   m.get("log_loss"),
                "ll_ci_lo":   (m.get("log_loss_ci") or [None])[0],
                "ll_ci_hi":   (m.get("log_loss_ci") or [None, None])[1],
                "brier":      m.get("brier"),
            }
            if "vs_baseline" in r:
                vb = r["vs_baseline"]
                flat.update({
                    "vs_b_acc_delta":  vb.get("accuracy_delta"),
                    "vs_b_ll_delta":   vb.get("log_loss_delta"),
                    "vs_b_mcnemar_p":  (vb.get("mcnemar") or {}).get("p_value"),
                    "vs_b_sig_acc":    vb.get("sig_accuracy"),
                    "vs_b_sig_ll":     vb.get("sig_log_loss"),
                })
            rows.append(flat)

    return pd.DataFrame(rows)


def get_run(run_id: str, log_path: Path = LOG_PATH) -> Optional[dict]:
    """Return the raw result dict for a specific run_id, or None."""
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("run_id") == run_id:
                    return r
            except json.JSONDecodeError:
                continue
    return None


def compare_runs(
    run_id_a: str,
    run_id_b: str,
    log_path: Path = LOG_PATH,
) -> dict:
    """
    Side-by-side metric comparison of two logged runs using stored metrics.

    Note: for a proper game-by-game McNemar test you need both models' raw
    predictions on the same games — call evaluate() with baseline_probs set
    to the other model's predictions at training time instead.
    """
    a = get_run(run_id_a, log_path)
    b = get_run(run_id_b, log_path)
    if a is None:
        raise KeyError(f"run_id {run_id_a!r} not in log")
    if b is None:
        raise KeyError(f"run_id {run_id_b!r} not in log")

    if a["season"] != b["season"] or a["split"] != b["split"]:
        warnings.warn(
            f"Comparing runs on different season/split: "
            f"A={a['season']}/{a['split']}  B={b['season']}/{b['split']}",
            stacklevel=2,
        )

    am, bm = a["metrics"], b["metrics"]
    result = {
        "run_a":   {"run_id": run_id_a, "model": a["model_name"], **am},
        "run_b":   {"run_id": run_id_b, "model": b["model_name"], **bm},
        # delta = B - A; positive accuracy = B better; negative log_loss = B better
        "delta_accuracy":  round(bm["accuracy"] - am["accuracy"], 4),
        "delta_log_loss":  round(bm["log_loss"]  - am["log_loss"],  4),
        "delta_brier":     round(bm["brier"]      - am["brier"],     4),
        "note": (
            "delta = B minus A.  To check if CIs overlap: "
            "A acc CI = {}, B acc CI = {}.  "
            "For a proper per-game significance test re-evaluate B with "
            "baseline_probs = A's raw predictions."
        ).format(am.get("accuracy_ci"), bm.get("accuracy_ci")),
    }
    return result


# =============================================================================
# SECTION 8 — PRETTY PRINTING
# =============================================================================

_W = 68

def _hdr(title: str) -> None:
    print(f"\n  ── {title} {'─' * max(0, _W - len(title) - 6)}")

def print_report(result: dict) -> None:
    """Print a formatted evaluation report to stdout."""
    m = result["metrics"]
    n = result["n_games"]

    print(f"\n{'━' * _W}")
    print(f"  {result['model_name']}  ·  {result['season']}  "
          f"·  {result['split'].upper()}  ·  {n:,} games")
    print(f"  {result['run_id']}")
    print(f"{'━' * _W}")

    # ── Core metrics ──────────────────────────────────────────────────────
    print(f"\n  {'Metric':<14} {'Value':>8}  {'95% CI':>18}")
    print(f"  {'─' * 44}")
    print(f"  {'Accuracy':<14} {m['accuracy']:>7.1%}   "
          f"[{m['accuracy_ci'][0]:.1%}, {m['accuracy_ci'][1]:.1%}]")
    print(f"  {'Log Loss':<14} {m['log_loss']:>8.4f}  "
          f"[{m['log_loss_ci'][0]:.4f}, {m['log_loss_ci'][1]:.4f}]")
    print(f"  {'Brier':<14} {m['brier']:>8.4f}  "
          f"[{m['brier_ci'][0]:.4f}, {m['brier_ci'][1]:.4f}]")
    print(f"  Home win rate: {result['home_win_rate']:.1%}")

    # ── vs baseline ───────────────────────────────────────────────────────
    if "vs_baseline" in result:
        vb = result["vs_baseline"]
        _hdr("VS BASELINE")
        print(f"\n  {'Metric':<14} {'Baseline':>9}  {'Model':>8}  "
              f"{'Delta':>8}  {'95% CI diff':>18}  Sig")
        print(f"  {'─' * 64}")
        rows_vb = [
            ("Accuracy", "accuracy", "accuracy_diff_ci_95", "sig_accuracy", True),
            ("Log Loss", "log_loss", "log_loss_diff_ci_95", "sig_log_loss", False),
            ("Brier",    "brier",    "brier_diff_ci_95",    None,           False),
        ]
        for label, key, ci_key, sig_key, higher_better in rows_vb:
            bval  = vb[f"baseline_{key}"]
            mval  = m[key]
            delta = vb[f"{key}_delta"]
            ci    = vb.get(ci_key, [None, None])
            sig   = ("✓" if vb.get(sig_key) else "·") if sig_key else " "
            ci_s  = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci[0] is not None else "—"
            fmt = ".1%" if key == "accuracy" else ".4f"
            print(f"  {label:<14} {bval:>{9 if fmt == '.4f' else 9}{fmt}}  "
                  f"{mval:>{8 if fmt == '.4f' else 8}{fmt}}  "
                  f"{delta:>+8.4f}  {ci_s:>18}  {sig}")

        mn = vb["mcnemar"]
        print(f"\n  McNemar: p={mn['p_value']:.4f}  "
              f"(model better on {mn['c_a_wrong_b_right']} games, "
              f"baseline better on {mn['b_a_right_b_wrong']} games)")
        print(f"  → {mn['interpretation']}")
        print(f"  (✓ = 95% bootstrap CI excludes 0)")

    # ── Calibration ───────────────────────────────────────────────────────
    _hdr("CALIBRATION")
    print(f"\n  {'Bucket':<10} {'N':>5}  {'Predicted':>10}  {'Actual':>8}  {'Error':>7}")
    print(f"  {'─' * 46}")
    for row in result["calibration"]:
        err = row["error"]
        flag = "  ▲ over" if err > 0.05 else ("  ▼ under" if err < -0.05 else "")
        print(f"  {row['bucket']:<10} {row['n']:>5}  "
              f"{row['pred_prob']:>9.1%}  {row['actual_rate']:>7.1%}  {err:>+7.3f}{flag}")

    # ── Stability by month ────────────────────────────────────────────────
    _hdr("STABILITY BY MONTH")
    stable = result.get("stability_monthly", [])
    if stable:
        acc_vals = [r["accuracy"] for r in stable]
        print(f"\n  {'Month':<9} {'N':>5}  {'Acc':>6}  {'LogLoss':>8}")
        print(f"  {'─' * 34}")
        for row in stable:
            bar_len = int((row["accuracy"] - 0.45) / 0.30 * 20)
            bar = "█" * max(0, min(bar_len, 20))
            print(f"  {row['month']:<9} {row['n']:>5}  "
                  f"{row['accuracy']:>5.1%}  {row['log_loss']:>8.4f}  {bar}")
        print(f"\n  Range: {min(acc_vals):.1%} – {max(acc_vals):.1%}  "
              f"(std {np.std(acc_vals):.3f})")

    # ── vs market ─────────────────────────────────────────────────────────
    if "vs_market" in result:
        vm = result["vs_market"]
        _hdr("VS MARKET")
        print(f"\n  {'':18} {'Acc':>7}  {'LogLoss':>9}  {'Brier':>7}")
        print(f"  {'─' * 46}")
        print(f"  {'Model':<18} {vm['model']['accuracy']:>6.1%}  "
              f"{vm['model']['log_loss']:>9.4f}  {vm['model']['brier']:>7.4f}")
        print(f"  {'Market':<18} {vm['market']['accuracy']:>6.1%}  "
              f"{vm['market']['log_loss']:>9.4f}  {vm['market']['brier']:>7.4f}")
        d = vm["delta"]
        print(f"  {'Delta':<18} {d['accuracy']:>+6.4f}  {d['log_loss']:>+9.4f}  "
              f"{d['brier']:>+7.4f}")
        print(f"\n  Agree:    {vm['n_agree']:>4} games")
        print(f"  Disagree: {vm['n_disagree']:>4} games", end="")
        if vm.get("disagree_model_accuracy") is not None:
            print(f"  — model acc on disagreements: {vm['disagree_model_accuracy']:.1%}", end="")
        print()
        if vm.get("model_edge_games"):
            eg = vm["model_edge_games"]
            print(f"  Model ≥5pp more confident than market: {eg['n']} games  "
                  f"model {eg['model_accuracy']:.1%}  market {eg['market_accuracy']:.1%}")
        mn = vm["mcnemar"]
        print(f"  McNemar vs market: p={mn['p_value']:.4f}  →  {mn['interpretation']}")

    print(f"\n{'━' * _W}\n")


def print_log_table(log_path: Path = LOG_PATH, n: Optional[int] = 15) -> None:
    """Tabular summary of all logged runs."""
    df = load_log(log_path)
    if df.empty:
        print("No runs logged yet.")
        return
    if n is not None:
        df = df.tail(n)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df[["run_id", "model_name", "season", "split", "n_games",
              "accuracy", "log_loss", "brier"]].to_string(index=False))


# =============================================================================
# SECTION 9 — MARKET ODDS HELPERS
# =============================================================================

def american_to_devig_prob(over_odds: float, under_odds: float) -> float:
    """
    Convert a two-sided American odds pair to a de-vigged home-win probability.
    Assumes over_odds corresponds to the home team winning.
    Uses the additive method (equal vig removal).
    """
    def implied(o: float) -> float:
        return 100 / (100 + o) if o > 0 else abs(o) / (abs(o) + 100)

    raw_h = implied(over_odds)
    raw_a = implied(under_odds)
    total = raw_h + raw_a
    return float(raw_h / total)


def load_market_probs_from_csv(
    df_games: pd.DataFrame,
    odds_csv: str | Path,
    home_col:  str = "home_team",
    away_col:  str = "away_team",
    date_col:  str = "game_date",
    h_odds_col: str = "home_ml",
    a_odds_col: str = "away_ml",
) -> np.ndarray:
    """
    Attempt to match a CSV of closing moneylines to df_games rows.

    The CSV must have columns: date, home team abbreviation, away team
    abbreviation, and American odds (positive or negative integers) for
    each side.  Rows that cannot be matched are filled with NaN so you can
    filter them out before passing to evaluate().

    Example CSV columns:  date, home, away, home_ml, away_ml
    """
    odds = pd.read_csv(odds_csv)
    odds["_date"]  = pd.to_datetime(odds[date_col]).dt.date
    games = df_games.copy()
    games["_date"] = pd.to_datetime(games[date_col]).dt.date

    probs = np.full(len(games), np.nan)
    for i, row in games.iterrows():
        mask = (
            (odds["_date"] == row["_date"]) &
            (odds[home_col].str.upper() == str(row.get("home_team", "")).upper()) &
            (odds[away_col].str.upper() == str(row.get("away_team", "")).upper())
        )
        if mask.sum() == 1:
            o = odds[mask].iloc[0]
            try:
                probs[i] = american_to_devig_prob(
                    float(o[h_odds_col]), float(o[a_odds_col])
                )
            except Exception:
                pass
    return probs


# =============================================================================
# CLI
# =============================================================================

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="NBA model evaluation log viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--all",     action="store_true",
                        help="Show every logged run (default: last 15)")
    parser.add_argument("--run",     metavar="RUN_ID",
                        help="Print full report for one run")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
                        help="Side-by-side delta comparison of two runs")
    parser.add_argument("--log",     default=str(LOG_PATH),
                        help=f"Path to eval log (default: {LOG_PATH})")
    args = parser.parse_args()

    log = Path(args.log)

    if args.run:
        r = get_run(args.run, log)
        if r is None:
            print(f"run_id {args.run!r} not found in {log}")
        else:
            print_report(r)
        return

    if args.compare:
        comp = compare_runs(args.compare[0], args.compare[1], log)
        a, b = comp["run_a"], comp["run_b"]
        W = 64
        print(f"\n{'━' * W}")
        print(f"  {a['model']:20s} vs  {b['model']}")
        print(f"{'━' * W}")
        print(f"  {'Metric':<14} {'A':>8}  {'B':>8}  {'Δ (B-A)':>10}")
        print(f"  {'─' * 46}")
        for k in ("accuracy", "log_loss", "brier"):
            delta = comp[f"delta_{k.replace(' ', '_')}"]
            print(f"  {k.capitalize():<14} {a[k]:>8.4f}  {b[k]:>8.4f}  {delta:>+10.4f}")
        print(f"\n  {comp['note']}")
        print(f"{'━' * W}\n")
        return

    # Default: print summary table
    print_log_table(log, n=None if args.all else 15)


if __name__ == "__main__":
    _cli()
