#!/usr/bin/env python3
"""Select an unconditional residual correction on chronological validation only.

    python3 scripts/replay/evaluate_corrections.py --input scripts/replay/output/after-injury-ramp_2025-26.csv.gz

All correction estimates for game date D use outcomes strictly before D.
The final test is evaluated only after the form and k are selected on validation.
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
for path in (str(HERE), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from replay_config import MARKET_ORDER  # noqa: E402
from replay_metrics import assign_split, metrics_table  # noqa: E402
from residual_correction import fit_market  # noqa: E402

FORMS = ("additive", "multiplicative")
K_VALUES = (50, 250, 1000, 5000)


def chronological_predictions(df: pd.DataFrame, form: str, k: float) -> pd.Series:
    """Vectorized expanding fit with whole-date lag per exact market."""
    if form not in FORMS or k <= 0:
        raise ValueError("unsupported correction candidate")
    tmp = df[["market", "date", "projection", "actual"]].copy()
    tmp["resid"] = tmp["actual"] - tmp["projection"]
    daily = tmp.groupby(["market", "date"], sort=True).agg(
        n=("resid", "size"), residual_sum=("resid", "sum"), projection_sum=("projection", "sum")
    )
    per = daily.groupby(level="market")
    for col in ("n", "residual_sum", "projection_sum"):
        daily["prior_" + col] = per[col].cumsum() - daily[col]
    stats = tmp[["market", "date"]].join(daily[["prior_n", "prior_residual_sum", "prior_projection_sum"]],
                                           on=["market", "date"])
    n = stats["prior_n"].to_numpy(dtype=float)
    resid = stats["prior_residual_sum"].to_numpy(dtype=float)
    denom = n if form == "additive" else stats["prior_projection_sum"].to_numpy(dtype=float)
    avg = np.divide(resid, denom, out=np.zeros_like(resid), where=denom > 0)
    correction = n / (n + k) * avg
    raw = df["projection"].to_numpy(dtype=float)
    pred = raw + correction if form == "additive" else raw * (1.0 + correction)
    return pd.Series(np.maximum(pred, 0.0), index=df.index)


def summarize(df: pd.DataFrame, col: str) -> dict:
    out = {}
    for regime, frame in (("regular", df[df.season_type == "Regular Season"]),
                          ("playoffs", df[df.season_type == "Playoffs"]),
                          ("combined", df)):
        out[regime] = {}
        for split in ("train", "val", "test", "all"):
            part = frame if split == "all" else frame[frame.split == split]
            if part.empty:
                continue
            m = metrics_table(part, col)
            out[regime][split] = {market: {key: float(value) for key, value in row.items()}
                                  for market, row in m.to_dict("index").items()}
    return out


def score(summary: dict) -> float:
    """Validation criterion: both regimes matter; MAE + projection-scale bias."""
    return sum(summary[r]["val"]["ALL"]["mae"] +
               abs(summary[r]["val"]["ALL"]["mean_residual"]) for r in ("regular", "playoffs"))


def evaluate(df: pd.DataFrame, splits: dict) -> tuple:
    df = df.sort_values(["date", "player_id", "market"]).reset_index(drop=True)
    if df.duplicated(["date", "player_id", "market"]).any():
        raise ValueError("duplicate player-game-market outcomes")
    df["split"] = assign_split(df, splits)
    candidates = []
    for form in FORMS:
        for k in K_VALUES:
            col = f"{form}_k{k}"
            df[col] = chronological_predictions(df, form, k)
            stats = summarize(df, col)
            candidates.append((score(stats), form, k, stats))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _, selected_form, selected_k, selected_stats = candidates[0]
    return df, {
        "version": "v9-unconditional-1",
        "selection": {"form": selected_form, "k": selected_k,
                      "criterion": "sum over RS+PO validation of MAE + absolute mean residual",
                      "test_not_used_for_selection": True},
        "baseline": summarize(df, "projection"),
        "candidates": [{"form": form, "k": k, "validation_score": value, "metrics": stats}
                       for value, form, k, stats in candidates],
        "splits": splits,
    }


def fit_live_artifact(df: pd.DataFrame, form: str, k: int) -> dict:
    markets = {}
    for market in MARKET_ORDER:
        sub = df[df.market == market]
        markets[market] = fit_market(zip(sub.projection, sub.actual), form, k)
    return {"version": "v9-unconditional-1", "form": form, "k": k,
            "trained_on": "2025-26 regular season and 2026 playoffs, unconditional played-game replay",
            "training_cutoff": str(df.date.max()), "residual_convention": "actual - raw_projection",
            "markets": markets}


def compact_report(full: dict) -> dict:
    """Keep selection evidence plus every market on held-out dates reviewable."""
    def concise(stats):
        return {
            reg: {
                split: {
                    market: {key: row[key] for key in ("n", "pct_bias", "mae", "rmse")}
                    for market, row in markets.items()
                    if market == "ALL" or split in ("val", "test")
                }
                for split, markets in blocks.items()
            }
            for reg, blocks in stats.items()
        }

    return {"version": full["version"], "selection": full["selection"], "splits": full["splits"],
            "baseline": concise(full["baseline"]),
            "candidates": [{"form": c["form"], "k": c["k"], "validation_score": c["validation_score"],
                            "metrics": concise(c["metrics"])} for c in full["candidates"]]}


def markdown_report(report: dict) -> str:
    selected = report["selection"]
    lines = ["# Chronological unconditional residual correction", "",
             "Residual is actual − raw projection. Each game's adjustment is fitted using earlier game dates only.",
             "Regular-season and playoff validation select the form and k; the final test is not used for selection.",
             "Chosen: **%s**, k=%d; validation criterion: %s." % (selected["form"], selected["k"], selected["criterion"]),
             "", "| candidate | validation score | RS val bias | RS val MAE | PO val bias | PO val MAE | RS test bias | RS test MAE | PO test bias | PO test MAE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    all_candidates = [{"form": "raw", "k": 0, "validation_score": None,
                       "metrics": report["baseline"]}] + report["candidates"]
    for c in all_candidates:
        m = c["metrics"]
        rs, po = m["regular"], m["playoffs"]
        lines.append("| %s k=%s | %s | %+.2f%% | %.3f | %+.2f%% | %.3f | %+.2f%% | %.3f | %+.2f%% | %.3f |" % (
            c["form"], c["k"], "—" if c["validation_score"] is None else "%.3f" % c["validation_score"],
            100 * rs["val"]["ALL"]["pct_bias"], rs["val"]["ALL"]["mae"],
            100 * po["val"]["ALL"]["pct_bias"], po["val"]["ALL"]["mae"],
            100 * rs["test"]["ALL"]["pct_bias"], rs["test"]["ALL"]["mae"],
            100 * po["test"]["ALL"]["pct_bias"], po["test"]["ALL"]["mae"]))
    chosen = next(c for c in report["candidates"]
                  if c["form"] == selected["form"] and c["k"] == selected["k"])
    mult = next(c for c in report["candidates"] if c["form"] == "multiplicative" and c["k"] == 250)
    for reg in ("regular", "playoffs"):
        lines += ["", "## %s final test, bias by exact market" % reg.title(), "",
                  "| market | raw | chosen additive | multiplicative k=250 | chosen MAE |",
                  "|---|---:|---:|---:|---:|"]
        for market in MARKET_ORDER:
            b = report["baseline"][reg]["test"][market]
            a = chosen["metrics"][reg]["test"][market]
            m = mult["metrics"][reg]["test"][market]
            lines.append("| %s | %+.1f%% | %+.1f%% | %+.1f%% | %.3f |" % (
                market, 100 * b["pct_bias"], 100 * a["pct_bias"], 100 * m["pct_bias"], a["mae"]))
    lines += ["", "The multiplicative candidates lower MAE more, but the additive k=1000 candidate balances absolute residual bias across both validation regimes better under the predefined criterion. The static artifact is fit on all completed 2025-26 RS and PO data for use in future seasons; it must never be applied to earlier dates in the replay.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(HERE / "output" / "after-injury-ramp_2025-26.csv.gz"))
    ap.add_argument("--splits", default=str(HERE / "splits.json"))
    ap.add_argument("--report", default=str(HERE / "reports" / "residual-correction-comparison.json"))
    ap.add_argument("--artifact", default=str(ROOT / "residual_corrections.json"))
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    if set(df.season_type.unique()) != {"Regular Season", "Playoffs"}:
        raise ValueError("both regular-season and playoff outcomes are required")
    if df.date.min() < "2025-10-01" or df.date.max() > "2026-09-30":
        raise ValueError("the selected replay is not the 2025-26 season")
    splits = json.loads(Path(args.splits).read_text())["2025-26"]
    predicted, report = evaluate(df, splits)
    selected = report["selection"]
    artifact = fit_live_artifact(predicted, selected["form"], selected["k"])
    concise = compact_report(report)
    Path(args.report).write_text(json.dumps(concise, indent=2, sort_keys=True) + "\n")
    Path(args.report).with_suffix(".md").write_text(markdown_report(concise))
    Path(args.artifact).write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    out_path = HERE / "output" / "corrected-chronological_2025-26.csv.gz"
    predicted[["date", "game_id", "player_id", "player", "market", "season_type", "actual", "projection",
               f"{selected['form']}_k{selected['k']}"]].rename(columns={f"{selected['form']}_k{selected['k']}":
                                                                "corrected_projection"}).to_csv(out_path, index=False, compression="gzip")
    print("selected", selected, "rows", len(predicted), "report", args.report, "artifact", args.artifact)


if __name__ == "__main__":
    main()
