#!/usr/bin/env python3
"""Diagnose held-out projection bias and test a regime-aware correction.

This is report-only. It does not change live parameters. The experiment selects
its correction form on chronological validation dates, then reports final-test
dates once. Outcome-based minutes buckets are diagnostic and are never inputs.

    python3 scripts/replay/diagnose_projection_bias.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from replay_metrics import assign_split  # noqa: E402

REGIMES = ("Regular Season", "Playoffs")
FORMS = ("additive", "multiplicative")
K_VALUES = (10, 50, 250, 1000, 5000)
SEED = 20260923


def metrics(frame: pd.DataFrame, projection: str) -> dict:
    residual = frame[projection].to_numpy(float) - frame.actual.to_numpy(float)
    actual = frame.actual.to_numpy(float)
    return {
        "n": int(len(frame)),
        "mean_projection": float(frame[projection].mean()),
        "mean_actual": float(actual.mean()),
        "mean_residual": float(residual.mean()),
        "pct_bias": float(residual.sum() / actual.sum()),
        "mae": float(np.abs(residual).mean()),
    }


def date_bootstrap_bias(frame: pd.DataFrame, projection: str, draws: int = 10000) -> list:
    """95% interval that resamples whole slates rather than correlated rows."""
    daily = frame.assign(_residual=frame[projection] - frame.actual).groupby("date").agg(
        residual=("_residual", "sum"), actual=("actual", "sum")
    )
    rng = np.random.default_rng(SEED)
    values = np.empty(draws)
    for index in range(draws):
        sampled = daily.iloc[rng.integers(0, len(daily), len(daily))]
        values[index] = sampled.residual.sum() / sampled.actual.sum()
    return [float(x) for x in np.quantile(values, (0.025, 0.975))]


def grouped(frame: pd.DataFrame, keys: list[str], projection: str) -> dict:
    result = {}
    key_arg = keys[0] if len(keys) == 1 else keys
    for key, group in frame.groupby(key_arg, observed=True, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        result[" / ".join(str(value) for value in key)] = metrics(group, projection)
    return result


def fit_static(train: pd.DataFrame, target: pd.DataFrame, form: str,
               k: int, buckets: int) -> np.ndarray:
    """Fit per-market corrections; optional quartile edges come from train only."""
    output = pd.Series(np.nan, index=target.index, dtype=float)
    for market, target_market in target.groupby("market"):
        train_market = train[train.market == market]
        if buckets == 1:
            ranges = [(-np.inf, np.inf)]
        else:
            edges = np.unique(np.quantile(train_market.projection, (0, .25, .5, .75, 1)))
            ranges = list(zip(np.r_[-np.inf, edges[1:-1]],
                              np.r_[edges[1:-1], np.inf]))
        for low, high in ranges:
            fit_rows = train_market[(train_market.projection >= low) &
                                    (train_market.projection < high)]
            use = target_market.index[(target_market.projection >= low) &
                                      (target_market.projection < high)]
            n = len(fit_rows)
            shrinkage = n / (n + k)
            if form == "additive":
                correction = shrinkage * (fit_rows.actual - fit_rows.projection).mean()
                prediction = target.loc[use, "projection"] + correction
            else:
                correction = shrinkage * (
                    (fit_rows.actual.sum() - fit_rows.projection.sum()) /
                    fit_rows.projection.sum()
                )
                prediction = target.loc[use, "projection"] * (1 + correction)
            output.loc[use] = np.maximum(prediction, 0)
    if output.isna().any():
        raise ValueError("candidate failed to price every target row")
    return output.to_numpy()


def candidate_experiment(raw: pd.DataFrame, corrected: pd.DataFrame) -> dict:
    """Select only on validation; refit through validation, then reveal test."""
    candidates = []
    for form in FORMS:
        for k in K_VALUES:
            for buckets in (1, 4):
                validation, objective = {}, 0.0
                for regime in REGIMES:
                    block = raw[raw.season_type == regime].copy()
                    train, val = block[block.split == "train"], block[block.split == "val"].copy()
                    val["candidate"] = fit_static(train, val, form, k, buckets)
                    summary = metrics(val, "candidate")
                    validation[regime] = summary
                    objective += summary["mae"] + abs(summary["mean_residual"])
                candidates.append({"form": form, "k": k, "buckets": buckets,
                                   "validation_objective": float(objective),
                                   "validation": validation})
    candidates.sort(key=lambda row: (row["validation_objective"], row["form"],
                                     row["k"], row["buckets"]))
    best = candidates[0]
    best["test"] = {}
    for regime in REGIMES:
        block = raw[raw.season_type == regime].copy()
        prior = block[block.split.isin(("train", "val"))]
        test = block[block.split == "test"].copy()
        test["candidate"] = fit_static(prior, test, best["form"], best["k"], best["buckets"])
        best["test"][regime] = {
            "overall": metrics(test, "candidate"),
            "by_market": grouped(test, ["market"], "candidate"),
        }

    current_validation, current_objective = {}, 0.0
    for regime in REGIMES:
        val = corrected[(corrected.season_type == regime) & (corrected.split == "val")]
        summary = metrics(val, "corrected_projection")
        current_validation[regime] = summary
        current_objective += summary["mae"] + abs(summary["mean_residual"])
    return {
        "selection_rule": "lowest RS+PO validation sum of MAE + absolute mean residual",
        "test_not_used_for_selection": True,
        "best": best,
        "current_validation_objective": float(current_objective),
        "current_validation": current_validation,
        "all_candidates": candidates,
        "deployment_status": "diagnostic_only",
        "deployment_blocker": (
            "playoff fit uses the current postseason's early outcomes; only one unseen "
            "postseason is available, so stability and playoff cold-start behavior are unproven"
        ),
    }


def build_report(raw_path: Path, corrected_path: Path, splits_path: Path) -> dict:
    raw = pd.read_csv(raw_path)
    corrected = pd.read_csv(corrected_path)
    splits = json.loads(splits_path.read_text())["2025-26"]
    raw["split"] = assign_split(raw, splits)
    corrected["split"] = assign_split(corrected, splits)
    metadata = raw[["date", "game_id", "player_id", "market", "proj_min", "actual_min",
                    "n_prior", "pos_group", "roll_min10"]]
    joined = corrected.merge(metadata, on=["date", "game_id", "player_id", "market"],
                             how="left", validate="one_to_one")
    test = joined[joined.split == "test"].copy()
    test["minute_error"] = test.proj_min - test.actual_min
    test["corrected_residual"] = test.corrected_projection - test.actual
    test["minute_bucket"] = pd.cut(
        test.minute_error, (-100, -5, -2, 2, 5, 100),
        labels=("under by >5", "under by 2-5", "within 2", "over by 2-5", "over by >5"),
    )
    test["history_bucket"] = pd.cut(
        test.n_prior, (0, 14, 29, 49, 1000), labels=("10-14", "15-29", "30-49", "50+")
    )
    projection_level = []
    for (_regime, _market), group in test.groupby(["season_type", "market"]):
        labels = pd.qcut(group.corrected_projection, 4,
                         labels=("low", "mid-low", "mid-high", "high"), duplicates="drop")
        projection_level.append(pd.Series(labels, index=group.index))
    test["projection_level"] = pd.concat(projection_level).sort_index()

    result = {
        "source": "2025-26 unconditional player-game replay; final chronological holdout only",
        "split_boundaries": splits,
        "aggregate": {},
        "by_market": {},
        "by_projection_level": {},
        "by_minute_error": {},
        "by_history_depth": {},
    }
    for regime in REGIMES:
        block = test[test.season_type == regime]
        raw_summary = metrics(block, "projection")
        corrected_summary = metrics(block, "corrected_projection")
        raw_summary["date_bootstrap_95_pct_bias"] = date_bootstrap_bias(block, "projection")
        corrected_summary["date_bootstrap_95_pct_bias"] = date_bootstrap_bias(
            block, "corrected_projection"
        )
        result["aggregate"][regime] = {"raw": raw_summary, "corrected": corrected_summary,
                                         "dates": int(block.date.nunique())}
        result["by_market"][regime] = grouped(block, ["market"], "corrected_projection")
        result["by_projection_level"][regime] = grouped(
            block, ["projection_level"], "corrected_projection"
        )
        result["by_minute_error"][regime] = grouped(
            block, ["minute_bucket"], "corrected_projection"
        )
        result["by_history_depth"][regime] = grouped(
            block, ["history_bucket"], "corrected_projection"
        )
    result["candidate_experiment"] = candidate_experiment(raw, corrected)
    return result


def pct(value: float) -> str:
    return f"{100 * value:+.2f}%"


def markdown(report: dict) -> str:
    lines = ["# Projection bias diagnostic", "",
             "Final chronological holdout from the 2025-26 replay. Confidence intervals",
             "resample whole game dates. Lower MAE is better; positive bias means the model",
             "projects too high.", "", "## Answer", ""]
    rs = report["aggregate"]["Regular Season"]
    po = report["aggregate"]["Playoffs"]
    lines += [
        f"- Raw projections are high: regular season {pct(rs['raw']['pct_bias'])} and "
        f"playoffs {pct(po['raw']['pct_bias'])}.",
        f"- The v9 chronological correction reduces this to regular season "
        f"{pct(rs['corrected']['pct_bias'])} and playoffs {pct(po['corrected']['pct_bias'])}.",
        f"- The regular-season corrected 95% interval is "
        f"{pct(rs['corrected']['date_bootstrap_95_pct_bias'][0])} to "
        f"{pct(rs['corrected']['date_bootstrap_95_pct_bias'][1])}; zero is plausible.",
        f"- The playoff corrected interval is "
        f"{pct(po['corrected']['date_bootstrap_95_pct_bias'][0])} to "
        f"{pct(po['corrected']['date_bootstrap_95_pct_bias'][1])}; the remaining upward "
        "bias is unlikely to be random in this holdout.", "",
        "| regime | rows | raw bias | corrected bias | corrected MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for regime in REGIMES:
        row = report["aggregate"][regime]
        lines.append(f"| {regime} | {row['corrected']['n']:,} | "
                     f"{pct(row['raw']['pct_bias'])} | {pct(row['corrected']['pct_bias'])} | "
                     f"{row['corrected']['mae']:.3f} |")
    lines += ["", "## Corrected bias by market", "",
              "| regime | market | bias | MAE |", "|---|---|---:|---:|"]
    for regime in REGIMES:
        for market, row in report["by_market"][regime].items():
            lines.append(f"| {regime} | {market} | {pct(row['pct_bias'])} | {row['mae']:.3f} |")
    lines += ["", "## Where the remaining error sits", "",
              "The corrected model is too low in the lowest projection quartile and too high",
              "in the highest quartile. That pattern explains why multiplicative correction",
              "continues to deserve testing. Minutes misses are even more important: when",
              "projected minutes land within two minutes of actual minutes, final-test bias is",
              "close to zero. Actual minutes are used only to diagnose the error after the game.", "",
              "| regime | projection level | bias | MAE |", "|---|---|---:|---:|"]
    for regime in REGIMES:
        levels = report["by_projection_level"][regime]
        for level in ("low", "mid-low", "mid-high", "high"):
            row = levels[level]
            lines.append(f"| {regime} | {level} | {pct(row['pct_bias'])} | {row['mae']:.3f} |")
    lines += ["", "| regime | minutes error | rows | bias | MAE |",
              "|---|---|---:|---:|---:|"]
    for regime in REGIMES:
        for bucket, row in report["by_minute_error"][regime].items():
            lines.append(f"| {regime} | {bucket} | {row['n']:,} | "
                         f"{pct(row['pct_bias'])} | {row['mae']:.3f} |")

    experiment = report["candidate_experiment"]
    best = experiment["best"]
    lines += ["", "## Improvement experiment", "",
              f"Validation selected **{best['form']}**, k={best['k']}, "
              f"{best['buckets']} projection bucket(s), fitted separately by regime and market.",
              f"Its validation objective was {best['validation_objective']:.3f}, versus "
              f"{experiment['current_validation_objective']:.3f} for the current correction.", "",
              "| regime | current test bias | candidate test bias | current MAE | candidate MAE |",
              "|---|---:|---:|---:|---:|"]
    for regime in REGIMES:
        current = report["aggregate"][regime]["corrected"]
        candidate = best["test"][regime]["overall"]
        lines.append(f"| {regime} | {pct(current['pct_bias'])} | "
                     f"{pct(candidate['pct_bias'])} | {current['mae']:.3f} | "
                     f"{candidate['mae']:.3f} |")
    blocker = experiment["deployment_blocker"]
    lines += ["", "The candidate is not deployed. " + blocker[0].upper() + blocker[1:] + ".",
              "The next model change should improve role/minutes forecasting and be validated",
              "on another unseen season or postseason before replacing the current correction.", ""]
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path,
                        default=HERE / "output" / "after-correction-raw_2025-26.csv.gz")
    parser.add_argument("--corrected", type=Path,
                        default=HERE / "output" / "corrected-chronological_2025-26.csv.gz")
    parser.add_argument("--splits", type=Path, default=HERE / "splits.json")
    parser.add_argument("--report", type=Path,
                        default=HERE / "reports" / "projection-bias-diagnostic.json")
    args = parser.parse_args(argv)
    report = build_report(args.raw, args.corrected, args.splits)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    args.report.with_suffix(".md").write_text(markdown(report))
    print("wrote", args.report, "and", args.report.with_suffix(".md"))


if __name__ == "__main__":
    main()
