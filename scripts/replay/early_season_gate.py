#!/usr/bin/env python3
"""v10 evidence gate for EARLY_SEASON_FALLBACK_ENABLED (regular season, < 10 current-season games).

    python3 scripts/replay/early_season_gate.py            # run every replay (cache only), then decide
    python3 scripts/replay/early_season_gate.py --reuse    # re-evaluate saved replay outputs only

For each of 2024-25 and 2025-26 it replays the first 30 days of the regular season (opening night included,
every played player-game eligible) with the rule OFF and ON, and the full 2025-26 RS+PO population (>= 10 prior
games) with the rule OFF and ON. The decision is "on" only if EVERY predeclared criterion holds in BOTH seasons:

  1. opening-night coverage with the rule > 0, and week-1 coverage strictly higher than without it
  2. paired rows (projected by both): upper end of the 95% date-clustered bootstrap CI on the change in
     residual-corrected MAE <= 0 (not worse)
  3. rows only the rule can project: corrected MAE <= 1.10 x the paired rows' corrected MAE without the rule,
     and |corrected mean bias| <= 10%
  4. rule-active rows (fewer than 10 prior games): share with actual > corrected projection in [0.45, 0.55]
  5. full-season population: identical with the rule on and off (same rows, max |delta projection| == 0)

"Corrected" applies the live residual_corrections.json exactly as project_prop does at 2026-27 time. The criteria
were fixed before any of these replays were run. Writes reports/early_season_fallback_gate.{json,md}.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import replay_provenance as provenance  # noqa: E402
from replay_cache import NBACache  # noqa: E402
from replay_config import BOOTSTRAP_REPS, BOOTSTRAP_SEED, DEFAULT_CACHE_DIR, DEFAULT_OUT_DIR  # noqa: E402
from replay_data import load_season_bundle  # noqa: E402

FLAG = "EARLY_SEASON_FALLBACK_ENABLED"
WINDOW_DAYS = 30
MAX_GAMES = 10
SEASONS = {"2024-25": {"rosters": False}, "2025-26": {"rosters": True}}  # rosters only where cached
FULL_SEASON = "2025-26"
REPORT = os.path.join(ROOT, "reports", "early_season_fallback_gate")
SCRATCH_REPORTS = os.path.join(DEFAULT_OUT_DIR, "early_season_gate_runs")  # gitignored


def offline(*_a, **_k):
    raise OSError("network disabled for the early-season gate")


def bundle_for(season: str):
    with contextlib.redirect_stdout(io.StringIO()):
        return load_season_bundle(NBACache(DEFAULT_CACHE_DIR, fetcher=offline, retries=1), season, want_rosters=False)


def replay(label: str, season: str, flag: bool, extra: list) -> None:
    cmd = [sys.executable, "-u", os.path.join(HERE, "replay_props.py"), "run", "--label", label, "--season", season,
           "--cache-only", "--reports-dir", SCRATCH_REPORTS, "--set", f"{FLAG}={flag}", *extra]
    print("[gate] " + " ".join(cmd[3:]), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"replay {label} failed:\n{proc.stderr[-3000:]}")


def rows_of(label: str, season: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DEFAULT_OUT_DIR, f"{label}_{season}.csv.gz"))


def corrected(df: pd.DataFrame, corr: dict) -> pd.Series:
    return np.maximum(0.0, df.projection + df.market.map(corr).fillna(0.0))


def bias_mae(proj: pd.Series, actual: pd.Series) -> dict:
    r = proj - actual
    return {"n": int(len(r)), "bias_pct": float(100 * r.mean() / actual.mean()) if len(r) else None,
            "median_residual": float(np.median(r)) if len(r) else None,
            "mae": float(r.abs().mean()) if len(r) else None,
            "p_actual_gt_projection": float((actual > proj).mean()) if len(r) else None}


def date_bootstrap(diff: pd.Series, dates: pd.Series) -> list:
    s = pd.DataFrame({"d": diff.values, "date": dates.values}).groupby("date").d.agg(["sum", "size"]).values
    w = np.random.default_rng(BOOTSTRAP_SEED).multinomial(len(s), np.full(len(s), 1 / len(s)),
                                                         size=BOOTSTRAP_REPS).astype(float)
    boot = (w @ s[:, 0]) / (w @ s[:, 1])
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def coverage(rows: pd.DataFrame, frame: pd.DataFrame) -> dict:
    got = set(zip(rows.date, rows.player_id))
    hit = sum((d, p) in got for d, p in zip(frame.date, frame.PLAYER_ID))
    return {"projected": int(hit), "played": int(len(frame)), "share": float(hit / len(frame)) if len(frame) else None}


def evaluate_season(season: str, off: pd.DataFrame, on: pd.DataFrame, corr: dict, window) -> dict:
    played = bundle_for(season).players
    played = played[(played.SEASON_TYPE == "Regular Season") & (played.GAME_DATE >= window[0])
                    & (played.GAME_DATE <= window[1])].copy()
    played["date"] = played.GAME_DATE.dt.strftime("%Y-%m-%d")
    opening = played[played.date == window[0]]
    week1 = played[played.GAME_DATE < pd.Timestamp(window[0]) + pd.Timedelta(days=7)]
    for df in (off, on):
        df["corrected"] = corrected(df, corr)
    k = ["date", "player_id", "market"]
    paired = off[k + ["actual", "corrected"]].merge(on[k + ["corrected"]], on=k, suffixes=("_off", "_on"))
    diff = (paired.corrected_on - paired.actual).abs() - (paired.corrected_off - paired.actual).abs()
    only = on.merge(off[k], on=k, how="left", indicator=True)
    only = only[only._merge == "left_only"]
    active = on[on.n_prior < MAX_GAMES]
    m = {
        "window": list(window),
        "coverage": {"off": {"opening_night": coverage(off, opening), "week1": coverage(off, week1),
                             "window": coverage(off, played)},
                     "on": {"opening_night": coverage(on, opening), "week1": coverage(on, week1),
                            "window": coverage(on, played)}},
        "paired": {"n": int(len(paired)), "off": bias_mae(paired.corrected_off, paired.actual),
                   "on": bias_mae(paired.corrected_on, paired.actual),
                   "corrected_mae_change": float(diff.mean()), "ci95": date_bootstrap(diff, paired.date)},
        "rule_only_rows": bias_mae(only.corrected, only.actual),
        "rule_active_rows": bias_mae(active.corrected, active.actual),
    }
    c = m["coverage"]
    m["checks"] = {
        "1_coverage": c["on"]["opening_night"]["projected"] > 0
        and c["on"]["week1"]["share"] > c["off"]["week1"]["share"],
        "2_paired_mae_not_worse": m["paired"]["ci95"][1] <= 0,
        "3_rule_only_quality": (m["rule_only_rows"]["n"] == 0 or (
            m["rule_only_rows"]["mae"] <= 1.10 * m["paired"]["off"]["mae"]
            and abs(m["rule_only_rows"]["bias_pct"]) <= 10)),
        "4_no_directional_lean": 0.45 <= m["rule_active_rows"]["p_actual_gt_projection"] <= 0.55,
    }
    return m


def evaluate_full(off: pd.DataFrame, on: pd.DataFrame) -> dict:
    k = ["date", "player_id", "market"]
    j = off[k + ["projection"]].merge(on[k + ["projection"]], on=k, how="outer", suffixes=("_off", "_on"),
                                      indicator=True)
    both = j[j._merge == "both"]
    max_delta = float((both.projection_on - both.projection_off).abs().max()) if len(both) else None
    return {"rows_off": int(len(off)), "rows_on": int(len(on)), "rows_paired": int(len(both)),
            "rows_unpaired": int((j._merge != "both").sum()), "max_abs_projection_delta": max_delta,
            "checks": {"5_full_season_identical": len(both) == len(off) == len(on) and max_delta == 0.0}}


def markdown(rep: dict) -> str:
    out = ["# Early-season fallback gate", "",
           f"Decision: **{rep['decision']}** (generated {rep['generated_utc']}, commit `{rep['git']['commit'][:10]}`, "
           f"working tree dirty: {rep['git']['working_tree_dirty']})", "",
           "Rule: regular season and fewer than 10 current-season games -> project from last season "
           "(minutes_forecast + prior-season rate shrinkage). Every criterion was fixed before running.", "",
           "| season | opening night off -> on | week 1 off -> on | paired corrected MAE off -> on (change, 95% CI) "
           "| rule-only rows: n, bias, MAE | rule-active: actual > proj | checks |",
           "|---|---|---|---|---|---|---|"]
    for season, m in rep["seasons"].items():
        c, p, o, a = m["coverage"], m["paired"], m["rule_only_rows"], m["rule_active_rows"]
        out.append("| %s | %d/%d -> %d/%d | %.0f%% -> %.0f%% | %.3f -> %.3f (%+.3f, [%+.3f, %+.3f]) | %d, %+.1f%%, %.3f "
                   "| %.3f | %s |" % (
                       season, c["off"]["opening_night"]["projected"], c["off"]["opening_night"]["played"],
                       c["on"]["opening_night"]["projected"], c["on"]["opening_night"]["played"],
                       100 * c["off"]["week1"]["share"], 100 * c["on"]["week1"]["share"],
                       p["off"]["mae"], p["on"]["mae"], p["corrected_mae_change"], p["ci95"][0], p["ci95"][1],
                       o["n"], o["bias_pct"] or 0.0, o["mae"] or 0.0, a["p_actual_gt_projection"],
                       ", ".join(f"{k}={'pass' if v else 'FAIL'}" for k, v in m["checks"].items())))
    f = rep["full_season"]
    out += ["", f"Full {FULL_SEASON} population (>= 10 prior games, RS+PO): {f['rows_paired']} paired rows, "
            f"{f['rows_unpaired']} unpaired, max |delta projection| = {f['max_abs_projection_delta']} -> "
            f"{'pass' if f['checks']['5_full_season_identical'] else 'FAIL'}.", "",
            "Mean bias uses the full-season residual correction, which over-corrects the early-season mean; the median "
            "residual and the actual > projection share are reported so that skew stays visible.", "",
            "Limits: 2025-26 is partly in-sample (the minutes_forecast constants were hand-set while studying it); "
            "2024-25 is the cleaner out-of-sample season. 2024-25 replays use manual positions only (its rosters are "
            "not cached). Live team assignment for 0-game players (current rosters) is not replayable historically "
            "and is covered by unit tests and a live smoke test instead.", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reuse", action="store_true", help="evaluate saved replay outputs without re-running")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    t0 = time.perf_counter()
    corr = {m: v["correction"] for m, v in json.load(open(os.path.join(ROOT, "residual_corrections.json")))["markets"].items()}
    seasons = {}
    for season, opt in SEASONS.items():
        first = bundle_for(season).players
        first = first[first.SEASON_TYPE == "Regular Season"].GAME_DATE.min()
        window = (first.strftime("%Y-%m-%d"), (first + pd.Timedelta(days=WINDOW_DAYS - 1)).strftime("%Y-%m-%d"))
        extra = ["--season-type", "regular", "--min-prior-games", "0", "--date-from", window[0],
                 "--date-to", window[1], "--workers", str(a.workers)] + ([] if opt["rosters"] else ["--skip-rosters"])
        labels = {flag: f"esg-early-{'on' if flag else 'off'}" for flag in (False, True)}
        if not a.reuse:
            for flag, label in labels.items():
                replay(label, season, flag, extra)
        seasons[season] = evaluate_season(season, rows_of(labels[False], season), rows_of(labels[True], season),
                                          corr, window)
    full_labels = {flag: f"esg-full-{'on' if flag else 'off'}" for flag in (False, True)}
    if not a.reuse:
        for flag, label in full_labels.items():
            replay(label, FULL_SEASON, flag, ["--season-type", "both", "--workers", str(a.workers)])
    full = evaluate_full(rows_of(full_labels[False], FULL_SEASON), rows_of(full_labels[True], FULL_SEASON))
    passed = all(all(m["checks"].values()) for m in seasons.values()) and all(full["checks"].values())
    rep = {"decision": "on" if passed else "off", "flag": FLAG, "max_current_games": MAX_GAMES,
           "window_days": WINDOW_DAYS, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "git": provenance.git_info(ROOT), "code_fingerprint": provenance.code_fingerprint(ROOT),
           "artifacts": provenance.artifact_info(ROOT), "seasons": seasons, "full_season": full,
           "runtime_s": round(time.perf_counter() - t0, 1)}
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT + ".json", "w") as fh:
        json.dump(rep, fh, indent=1, sort_keys=True, default=str)
    with open(REPORT + ".md", "w") as fh:
        fh.write(markdown(rep))
    print(markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
