"""Summaries and readable reports for a replay run (small JSON + markdown; per-row outputs stay gitignored)."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from replay_metrics import (
    assign_split,
    macro_pct_bias,
    market_metrics,
    metrics_table,
    paired_bootstrap,
    paired_frame,
)
from replay_config import (
    CORE_MINUTES_THRESHOLD,
    LOW_MINUTES_THRESHOLD,
    MARKET_ORDER,
    SEASON_TYPE_PO,
    SEASON_TYPE_RS,
)

REGIMES = (("regular", SEASON_TYPE_RS), ("playoffs", SEASON_TYPE_PO))
METRIC_COLS = ["n", "mean_actual", "mean_projection", "mean_residual", "pct_bias", "median_residual", "mae", "rmse",
               "p_actual_gt_projection"]


def _regime_frames(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {name: df[df["season_type"] == st] for name, st in REGIMES}
    out["combined"] = df
    return {k: v for k, v in out.items() if len(v)}


def role_stratum(df: pd.DataFrame) -> pd.Series:
    """Pre-game role from the rolling average of the last 10 played games' minutes (reporting only).
    low: RS < 15, PO < 10 (the former minute-filter thresholds); core: >= 20; mid: in between."""
    low = df["season_type"].map(LOW_MINUTES_THRESHOLD)
    return pd.Series(np.where(df["roll_min10"] < low, "low", np.where(df["roll_min10"] >= CORE_MINUTES_THRESHOLD, "core", "mid")),
                     index=df.index)


def _table_json(t: pd.DataFrame) -> Dict[str, Any]:
    return {m: {k: (None if pd.isna(v) else v) for k, v in row.items()} for m, row in t[METRIC_COLS].to_dict("index").items()}


def build_summary(label: str, df: pd.DataFrame, info: Dict[str, Any], prov: Dict[str, Any], season: str,
                  season_types, bundle_meta: Dict[str, Any], cache: Dict[str, Any], runtime: Dict[str, Any],
                  splits: Dict[str, Any], prev: Optional[pd.DataFrame], prev_label: Optional[str],
                  proj_col: str = "projection") -> Dict[str, Any]:
    df = df.copy()
    df["split"] = assign_split(df, splits)
    frames = _regime_frames(df)
    metrics: Dict[str, Any] = {}
    macro: Dict[str, Any] = {}
    for name, f in frames.items():
        t_all = metrics_table(f, proj_col)
        metrics[name] = {"all": _table_json(t_all)}
        macro[name] = macro_pct_bias(t_all)
        for sp in ("train", "val", "test"):
            sub = f[f["split"] == sp]
            if len(sub):
                metrics[name][sp] = _table_json(metrics_table(sub, proj_col))

    strata: Dict[str, Any] = {}
    if "roll_min10" in df.columns:
        df["role"] = role_stratum(df)
        for name, f in _regime_frames(df).items():
            strata[name] = {}
            for role in ("low", "mid", "core"):
                sub = f[f["role"] == role]
                if len(sub):
                    t = metrics_table(sub, proj_col)
                    strata[name][role] = {
                        "player_games": int(sub[["date", "player_id"]].drop_duplicates().shape[0]),
                        "pooled": _table_json(t)["ALL"],
                        "pct_bias_by_market": {m: float(t.loc[m, "pct_bias"]) for m in t.index if m != "ALL"},
                    }

    summary: Dict[str, Any] = {
        "label": label,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "season": season,
        "season_types": list(season_types),
        "provenance": prov,
        "replay_range": {n: [f["date"].min(), f["date"].max()] for n, f in frames.items()},
        "chronological_splits": splits,
        "training_cutoff": ("no fitted residual correction is applied in this run; the model's learned-parameter "
                            "artifact was trained on %s" % (prov.get("artifacts", {}).get("learned_params.json", {}).get("trained_on"))),
        "sample": {
            "rows": int(len(df)),
            "player_games": int(df[["date", "player_id"]].drop_duplicates().shape[0]),
            "by_regime": {n: {"rows": int(len(f)), "player_games": int(f[["date", "player_id"]].drop_duplicates().shape[0])}
                          for n, f in frames.items()},
        },
        "population": {"counts": info["counts"], "reconstruction_failures": info["failures"], "bundle": bundle_meta},
        "cache": cache,
        "runtime": runtime,
        "overrides": info.get("overrides", {}),
        "internal_stat_to_market": {"FG3M": "3PM"},
        "metrics": metrics,
        "macro_pct_bias": macro,
        "role_strata": strata,
    }
    if prev is not None:
        comp: Dict[str, Any] = {"previous_label": prev_label, "by_regime": {}}
        for name, f in frames.items():
            pf = prev if name == "combined" else prev[prev["season_type"] == dict(REGIMES).get(name)]
            paired = paired_frame(pf, f)
            ta, tb = metrics_table(pf), metrics_table(f)
            rows = {}
            for m in list(MARKET_ORDER) + ["ALL"]:
                if m in ta.index and m in tb.index:
                    rows[m] = {
                        "d_pct_bias": float(tb.loc[m, "pct_bias"] - ta.loc[m, "pct_bias"]),
                        "d_mae": float(tb.loc[m, "mae"] - ta.loc[m, "mae"]),
                        "d_rmse": float(tb.loc[m, "rmse"] - ta.loc[m, "rmse"]),
                    }
            comp["by_regime"][name] = {
                "markets": rows,
                "rows_prev": int(len(pf)), "rows_now": int(len(f)), "rows_paired": int(len(paired)),
                "paired_bootstrap_all_markets": paired_bootstrap(paired),
            }
        summary["comparison"] = comp
    return summary


def _md_table(t: Dict[str, Any]) -> str:
    head = "| market | n | mean actual | mean proj | mean resid | % bias | median resid | MAE | RMSE | P(actual>proj) |\n"
    head += "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    rows = []
    for m in list(MARKET_ORDER) + ["ALL"]:
        if m in t:
            r = t[m]
            rows.append("| %s | %d | %.3f | %.3f | %+.3f | %+.1f%% | %+.2f | %.3f | %.3f | %.3f |" % (
                m, r["n"], r["mean_actual"], r["mean_projection"], r["mean_residual"], 100 * r["pct_bias"],
                r["median_residual"], r["mae"], r["rmse"], r["p_actual_gt_projection"]))
    return head + "\n".join(rows)


def render_markdown(s: Dict[str, Any]) -> str:
    p = s["provenance"]
    out = ["# Replay report: `%s`" % s["label"], ""]
    out += ["- season: **%s**, season types: %s" % (s["season"], ", ".join(s["season_types"])),
            "- code commit: `%s` on `%s` (working tree dirty at run time: %s), source fingerprint `%s`" % (
                p["git"]["commit"][:10], p["git"]["branch"], p["git"]["working_tree_dirty"], p["code_fingerprint"][:16]),
            "- replay config: `%s` (eligibility `%s`), cache schema `%s`" % (
                p["replay_config_version"], p["eligibility_version"], p["cache_schema_version"]),
            "- replay date range: %s" % "; ".join("%s %s..%s" % (k, v[0], v[1]) for k, v in s["replay_range"].items()),
            "- training cutoff: %s" % s["training_cutoff"],
            "- sample: **%d rows** over **%d player-games** (%s)" % (
                s["sample"]["rows"], s["sample"]["player_games"],
                ", ".join("%s %d rows" % (k, v["rows"]) for k, v in s["sample"]["by_regime"].items())),
            "- market naming: internal stat column `FG3M` is displayed/priced as market `3PM` (same quantity).",
            "- residual = projection - actual; % bias = mean(projection - actual) / mean(actual); "
            "P(actual>proj) = share of observations where the actual exceeded the projection.", ""]
    if s["overrides"]:
        out += ["- model overrides for this run: `%s`" % json.dumps(s["overrides"]), ""]
    c, rt = s["cache"], s["runtime"]
    out += ["## Runtime and cache", "",
            "- cache state: **%s** (entries before run: %d), dir `%s`" % (c["state"], c["entries_before"], c["dir"]),
            "- API: %d requests, %d retries, %d failures, %d cache hits, %d misses, %d writes, "
            "rate-limit delay %.1fs, backoff delay %.1fs" % (
                c["stats"]["requests"], c["stats"]["retries"], c["stats"]["failures"], c["stats"]["cache_hits"],
                c["stats"]["cache_misses"], c["stats"]["cache_writes"], c["stats"]["rate_limit_delay_s"],
                c["stats"]["backoff_delay_s"]),
            "- runtime: data load %.1fs + replay %.1fs (%d worker(s), %.1fs cumulative worker time) = **%.1fs total**" % (
                rt["load_s"], rt["replay_wall_s"], rt["workers"], rt["worker_busy_s"], rt["total_s"]), ""]
    pop = s["population"]
    out += ["## Population and exclusions", "", "```", json.dumps(pop["counts"], indent=1),
            "reconstruction failures: " + json.dumps(pop["reconstruction_failures"]),
            "bundle: " + json.dumps(pop["bundle"], default=str), "```", ""]
    for name in ("regular", "playoffs", "combined"):
        if name not in s["metrics"]:
            continue
        out += ["## %s: all replayed observations" % name.upper(), "", _md_table(s["metrics"][name]["all"]), ""]
        out += ["Chronological split (pooled over markets):", "", "| split | n | % bias | MAE | RMSE |", "|---|---:|---:|---:|---:|"]
        for sp in ("train", "val", "test"):
            if sp in s["metrics"][name]:
                r = s["metrics"][name][sp]["ALL"]
                out.append("| %s | %d | %+.1f%% | %.3f | %.3f |" % (sp, r["n"], 100 * r["pct_bias"], r["mae"], r["rmse"]))
        out += ["", "Macro-average of per-market %% bias: **%+.1f%%**" % (100 * s["macro_pct_bias"][name]), ""]
    if s.get("role_strata"):
        out += ["## By pre-game role stratum (rolling minutes of the last 10 played games; reporting only, never an exclusion)", "",
                "low = RS < 15 / PO < 10; core >= 20; mid = in between", ""]
        for name, blk in s["role_strata"].items():
            out += ["### %s" % name.upper(), "", "| stratum | player-games | rows | % bias | MAE | RMSE | " + " | ".join(MARKET_ORDER) + " |",
                    "|---|---:|---:|---:|---:|---:|" + "---:|" * len(MARKET_ORDER)]
            for role in ("low", "mid", "core"):
                if role in blk:
                    r, pm = blk[role]["pooled"], blk[role]["pct_bias_by_market"]
                    out.append("| %s | %d | %d | %+.1f%% | %.3f | %.3f | %s |" % (
                        role, blk[role]["player_games"], r["n"], 100 * r["pct_bias"], r["mae"], r["rmse"],
                        " | ".join("%+.0f%%" % (100 * pm[m]) if m in pm else "-" for m in MARKET_ORDER)))
            out.append("")
    if "comparison" in s:
        cmp_ = s["comparison"]
        out += ["## Change from previous run `%s`" % cmp_["previous_label"], ""]
        for name, blk in cmp_["by_regime"].items():
            out += ["### %s (paired rows %d of %d previous / %d current)" % (name, blk["rows_paired"], blk["rows_prev"], blk["rows_now"]), "",
                    "| market | Δ % bias (pts) | Δ MAE | Δ RMSE |", "|---|---:|---:|---:|"]
            for m, r in blk["markets"].items():
                out.append("| %s | %+.2f | %+.4f | %+.4f |" % (m, 100 * r["d_pct_bias"], r["d_mae"], r["d_rmse"]))
            pb = blk["paired_bootstrap_all_markets"]
            if pb.get("n_paired"):
                out += ["", "Paired bootstrap over game dates (all markets, negative = current is better): "
                        + "; ".join("%s %+.4f [%+.4f, %+.4f]%s" % (k, pb[k]["delta"], pb[k]["ci95"][0], pb[k]["ci95"][1],
                                                                   "" if pb[k]["significant"] else " (n.s.)")
                                    for k in ("abs_bias", "mae", "rmse")), ""]
    return "\n".join(out) + "\n"


def write_reports(summary: Dict[str, Any], reports_dir: str) -> Dict[str, str]:
    os.makedirs(reports_dir, exist_ok=True)
    jp = os.path.join(reports_dir, "%s.json" % summary["label"])
    mp_ = os.path.join(reports_dir, "%s.md" % summary["label"])
    with open(jp, "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True, default=str)
    with open(mp_, "w") as fh:
        fh.write(render_markdown(summary))
    return {"json": jp, "markdown": mp_}
