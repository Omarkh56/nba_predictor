#!/usr/bin/env python3
"""Compare coherent proxy-fitted Platt calibration with identity on real held-out lines.

    python3 scripts/replay/calibrate_probabilities.py

Proxy inputs are synthetic lines for calibration diagnostics only. No odds,
ROI, edge, EV or profit are inferred from proxy lines. No simulated row goes
into bet_log.csv.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (str(HERE), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ["ODDS_API_KEY"] = "replay-dummy-key"

from backtest_bets import PAPER_BETS, pregame_probability  # noqa: E402
from replay_config import MARKET_ORDER, SEASON_TYPE_PO, SEASON_TYPE_RS  # noqa: E402
from residual_correction import load_artifact as load_residual  # noqa: E402

RAW_REPLAY = HERE / "output" / "after-correction-raw_2025-26.csv.gz"
CHRONO = HERE / "output" / "corrected-chronological_2025-26.csv.gz"
REAL_OUT = HERE / "output" / "real-lines-regenerated.csv.gz"
PROXY_OUT = ROOT / "backtest_bets.csv"
REPORT_JSON = HERE / "reports" / "probability-calibration.json"
REPORT_MD = HERE / "reports" / "probability-calibration.md"
ARTIFACT = ROOT / "probability_calibration.json"
ALIASES = {"robertwilliams": "robertwilliamsiii"}
EPS = 1e-6


def _name(value):
    ascii_name = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    key = re.sub("[^a-z0-9]", "", ascii_name)
    return ALIASES.get(key, key)


def _logit(values):
    p = np.clip(np.asarray(values, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_platt(data: pd.DataFrame) -> dict:
    """One OVER calibrator per exact market, with UNDER its complement."""
    result = {}
    for market in MARKET_ORDER:
        block = data[(data.market == market) & (data.graded_status == "GRADED")]
        if len(block) < 100 or block.y_over.nunique() < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
        clf.fit(_logit(block.p_over_uncalibrated), block.y_over.astype(int))
        result[market] = {"intercept": float(clf.intercept_[0]),
                          "slope": float(clf.coef_[0][0]), "n": int(len(block)),
                          "as_of_last_date": str(block.date.max())}
    return result


def apply_platt(raw, markets, fitted):
    raw = np.clip(np.asarray(raw, dtype=float), EPS, 1 - EPS)
    result = raw.copy()
    mk = np.asarray(markets)
    for market, p in fitted.items():
        select = mk == market
        z = p["intercept"] + p["slope"] * _logit(raw[select]).ravel()
        result[select] = 1 / (1 + np.exp(-np.clip(z, -40, 40)))
    return np.clip(result, EPS, 1 - EPS)


def scoring(df, column, y_col="y_over"):
    p = np.clip(df[column].to_numpy(dtype=float), EPS, 1 - EPS)
    y = df[y_col].to_numpy(dtype=float)
    return {"n": int(len(df)), "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "mean_probability": float(np.mean(p)), "realized_rate": float(np.mean(y)),
            "gap": float(np.mean(p) - np.mean(y)),
            "range": [float(np.min(p)), float(np.max(p))]}


def reliability(df, p_col, y_col):
    p = df[p_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=int)
    bins = np.minimum(9, np.floor(10 * p).astype(int))
    table, ece = [], 0.0
    for i in range(10):
        take = bins == i
        n = int(take.sum())
        avg = float(p[take].mean()) if n else None
        real = float(y[take].mean()) if n else None
        if n:
            ece += n / len(df) * abs(avg - real)
        table.append({"bin": f"{i/10:.1f}-{(i+1)/10:.1f}", "n": n,
                      "mean_probability": avg, "realized": real,
                      "min_probability": float(p[take].min()) if n else None,
                      "max_probability": float(p[take].max()) if n else None})
    return {"ece": ece, "bins": table, "nonempty_bins": int(sum(b["n"] > 0 for b in table))}


def detailed_scores(data: pd.DataFrame) -> dict:
    """Separate direction reports over all opportunities, never only selected picks."""
    result = {}
    for market in ["ALL"] + MARKET_ORDER:
        group = data if market == "ALL" else data[data.market == market]
        if group.empty:
            continue
        result[market] = {}
        for direction in ("OVER", "UNDER"):
            result[market][direction] = {}
            for method in ("identity", "platt"):
                p_col = "p_over_uncalibrated" if method == "identity" else "p_over_platt"
                y_col = "y_over"
                g = group.copy()
                if direction == "UNDER":
                    g["p_direction"] = 1 - g[p_col]
                    g["y_direction"] = 1 - g[y_col]
                    p_col, y_col = "p_direction", "y_direction"
                summary = scoring(g, p_col, y_col)
                summary["reliability"] = reliability(g, p_col, y_col)
                result[market][direction][method] = summary
    return result


def chronological_proxy_predictions(data: pd.DataFrame) -> pd.DataFrame:
    """Weekly expanding fits; each week's outcomes are unavailable until later weeks."""
    data = data.copy()
    data["y_over"] = (data.actual > data.proxy_line).astype(int)
    data["p_over_platt"] = np.nan
    data["calibration_asof_date"] = ""
    data["week_start"] = pd.to_datetime(data.date).dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")
    graded = data[data.graded_status == "GRADED"]
    for start, part in data.groupby("week_start", sort=True):
        past = graded[graded.date < start]
        params = fit_platt(past)
        assert not params or max(v["as_of_last_date"] for v in params.values()) < start
        data.loc[part.index, "p_over_platt"] = apply_platt(part.p_over_uncalibrated, part.market, params)
        data.loc[part.index, "calibration_asof_date"] = str(past.date.max()) if len(past) else ""
    return data.drop(columns="week_start")


def proxy_segments(proxy: pd.DataFrame, splits: dict) -> dict:
    """Expanding as-of fits for every row, including rows not used in scoring."""
    sources = {}
    for line_source in ("proxy_l10", "proxy_l5"):
        data = chronological_proxy_predictions(proxy[proxy.line_source == line_source])
        data["segment"] = "training_only"
        for st, label in ((SEASON_TYPE_RS, "regular"), (SEASON_TYPE_PO, "playoffs")):
            subset = data[data.season_type == st]
            for split, mask in (
                ("val", (subset.date > splits[st]["train_end"]) & (subset.date <= splits[st]["val_end"])),
                ("test", subset.date > splits[st]["val_end"]),
            ):
                target = subset[mask]
                if target.empty:
                    continue
                data.loc[target.index, "segment"] = f"{label}_{split}"
        sources[line_source] = data
    return sources


def prepare_real_lines(bets: pd.DataFrame, replay: pd.DataFrame, chrono: pd.DataFrame) -> tuple:
    """Reproject at each logged real line with strictly pre-game model outputs."""
    eligible = bets[(bets.result.isin(("HIT", "MISS"))) & bets.actual.notna()].copy()
    eligible["name_key"] = eligible.player.map(_name)
    replay["name_key"] = replay.player.map(_name)
    chrono["name_key"] = chrono.player.map(_name)
    keys = ["date", "name_key", "market"]
    if replay.duplicated(keys).any() or chrono.duplicated(keys).any():
        raise ValueError("ambiguous player/date/market replay join")
    joined = eligible.merge(replay[keys + ["sd", "projection", "actual"]], on=keys, how="left", validate="many_to_one",
                            suffixes=("_logged", "_replay"))
    joined = joined.merge(chrono[keys + ["corrected_projection"]], on=keys, how="left", validate="many_to_one")
    matched = joined[joined.corrected_projection.notna() & joined.sd.notna()].copy()
    invalid_actual = matched.actual_logged != matched.actual_replay
    if invalid_actual.any():
        raise ValueError("logged actual disagrees with replay outcome on %d rows" % invalid_actual.sum())
    matched["corrected_sd"] = matched.sd * np.sqrt(matched.corrected_projection / matched.projection_replay)
    matched["p_over_uncalibrated"], matched["anchored_projection"], matched["probability_sd"] = pregame_probability(
        matched.corrected_projection, matched.corrected_sd, matched.line, matched.market)
    matched["y_over"] = (matched.actual_logged > matched.line).astype(int)
    matched["graded_status"] = "GRADED"
    matched["line_source"] = "real_logged"
    return matched, {"logged_rows": int(len(bets)), "void_or_ungraded_excluded": int(len(bets) - len(eligible)),
                     "valid_graded": int(len(eligible)), "replay_matched": int(len(matched)),
                     "unmatched": int(len(eligible) - len(matched))}


def real_chronological_predictions(real: pd.DataFrame, proxy: pd.DataFrame) -> pd.DataFrame:
    out = real.copy()
    out["p_over_platt"] = np.nan
    for d, part in out.groupby("date"):
        past = proxy[proxy.date < d]
        params = fit_platt(past)
        assert not params or max(v["as_of_last_date"] for v in params.values()) < d
        out.loc[part.index, "p_over_platt"] = apply_platt(part.p_over_uncalibrated, part.market, params)
    return out


def bootstrap_noninferiority(real: pd.DataFrame, margin_brier=.005, margin_ll=.01, reps=2000) -> dict:
    """Paired whole-game-date bootstrap; upper 95% CI must clear both margins."""
    p0 = np.clip(real.p_over_uncalibrated.to_numpy(dtype=float), EPS, 1 - EPS)
    p1 = np.clip(real.p_over_platt.to_numpy(dtype=float), EPS, 1 - EPS)
    y = real.y_over.to_numpy(dtype=float)
    delta_brier = (p1-y)**2 - (p0-y)**2
    delta_ll = -(y*np.log(p1)+(1-y)*np.log(1-p1)) + (y*np.log(p0)+(1-y)*np.log(1-p0))
    totals = pd.DataFrame({"date": real.date, "n": 1, "brier": delta_brier, "log_loss": delta_ll}).groupby("date").sum()
    if len(totals) < 5:
        return {"passed": False, "reason": "fewer than five held-out real-line dates", "n_dates": len(totals)}
    values = totals[["n", "brier", "log_loss"]].to_numpy(dtype=float)
    rng = np.random.default_rng(20260921)
    counts = rng.multinomial(len(values), [1/len(values)] * len(values), size=reps)
    boot = counts @ values
    deltas = boot[:, 1:] / boot[:, [0]]
    actual = values.sum(axis=0)[1:] / values.sum(axis=0)[0]
    upper = np.percentile(deltas, 97.5, axis=0)
    return {"n_dates": int(len(values)), "n_rows": int(len(real)),
            "delta_brier": float(actual[0]), "delta_log_loss": float(actual[1]),
            "ci95_upper_brier": float(upper[0]), "ci95_upper_log_loss": float(upper[1]),
            "margin_brier": margin_brier, "margin_log_loss": margin_ll,
            "passed": bool(upper[0] <= margin_brier and upper[1] <= margin_ll)}


def selection_comparison(real: pd.DataFrame, method: str) -> dict:
    """Logged-line opportunity comparison; no invented sportsbook opportunities."""
    from playerlinepredictor import OVER_EDGE_PCT_FLOOR, UNDER_EDGE_PCT_FLOOR

    p = real.p_over_platt if method == "platt" else real.p_over_uncalibrated
    replay_pick = np.where(p > .5, "OVER", "UNDER")
    edge = (abs(real.anchored_projection - real.line) / real.line * 100).replace([np.inf, -np.inf], np.nan)
    floor = np.where(replay_pick == "OVER", OVER_EDGE_PCT_FLOOR, UNDER_EDGE_PCT_FLOOR)
    passes = (edge >= floor).fillna(False).to_numpy()
    correct = np.where(replay_pick == "OVER", real.y_over == 1, real.y_over == 0)
    logged_pts_over = (real.market == "PTS") & (real.pick == "OVER")

    def summarize(mask, picks, outcomes):
        chosen = np.asarray(mask, dtype=bool)
        out = {"n": int(chosen.sum())}
        for direction in ("OVER", "UNDER"):
            which = chosen & (np.asarray(picks) == direction)
            out[direction] = {"n": int(which.sum()),
                              "share": float(which.sum() / chosen.sum()) if chosen.any() else None,
                              "hit_rate": float(np.asarray(outcomes)[which].mean()) if which.any() else None}
        return out

    logged_correct = (real.result == "HIT").to_numpy()
    result = {
        "population": "only historical logged real lines; no estimate of all available sportsbook markets",
        "logged": summarize(np.ones(len(real), dtype=bool), real.pick, logged_correct),
        "replayed_all": summarize(np.ones(len(real), dtype=bool), replay_pick, correct),
        "replayed_over_edge_floor": summarize(passes, replay_pick, correct),
        "removed_by_edge_floor": int((~passes).sum()),
        "pts_logged_over": {"n": int(logged_pts_over.sum()),
                            "flips_to_under": int((logged_pts_over & (replay_pick == "UNDER")).sum()),
                            "stays_over_below_edge_floor": int((logged_pts_over &
                                (replay_pick == "OVER") & ~passes).sum())},
    }
    result["by_market"] = {m: {"logged": summarize(real.market == m, real.pick, logged_correct),
                                "replayed_floor": summarize((real.market == m) & passes,
                                                             replay_pick, correct)}
                           for m in MARKET_ORDER}
    return result


def write_readable_report(report: dict) -> None:
    lines = ["# v9 chronological probability calibration", "",
             "Synthetic rolling-average lines are **calibration diagnostics only**. They are not",
             "bookmaker prices; no ROI, edge or profit is inferred from them.", "",
             f"- Real logged-line coverage: {report['real_coverage']['replay_matched']}/{report['real_coverage']['valid_graded']}; "
             f"void/ungraded excluded before fitting: {report['real_coverage']['void_or_ungraded_excluded']}.",
             f"- Held-out real dates after {report['real_holdout_after']}: "
             f"{report['real_gate']['n_rows']} rows on {report['real_gate']['n_dates']} dates.",
             f"- Paired date-bootstrap, Platt − identity: Brier {report['real_gate']['delta_brier']:+.4f} "
             f"(95% upper {report['real_gate']['ci95_upper_brier']:+.4f}, margin +0.005); "
             f"log loss {report['real_gate']['delta_log_loss']:+.4f} "
             f"(upper {report['real_gate']['ci95_upper_log_loss']:+.4f}, margin +0.010).",
             f"- Gate {'passed' if report['real_gate']['passed'] else 'failed'}; "
             f"2026-27 artifact: **{report['deployed_method']}**. Runtime: {report['runtime_s']:.1f}s.", "",
             "| regime / line | sample | identity Brier | Platt Brier | identity log loss | Platt log loss |", 
             "|---|---:|---:|---:|---:|---:|"]
    for source, segments in report["proxy"].items():
        for name, block in sorted(segments.items()):
            a, b = block["ALL"]["OVER"]["identity"], block["ALL"]["OVER"]["platt"]
            lines.append(f"| {source} {name} | {a['n']} | {a['brier']:.4f} | {b['brier']:.4f} | "
                         f"{a['log_loss']:.4f} | {b['log_loss']:.4f} |")
    for label, block in report["real"].items():
        a, b = block["ALL"]["OVER"]["identity"], block["ALL"]["OVER"]["platt"]
        lines.append(f"| real logged {label} | {a['n']} | {a['brier']:.4f} | {b['brier']:.4f} | "
                     f"{a['log_loss']:.4f} | {b['log_loss']:.4f} |")
    lines.extend(["", "L5 and L10 cover different eligible players and use different lines; their",
                  "absolute scores are not a paired comparison. Reliability bins by exact market",
                  "and both directions are in `probability-calibration.json`.", "",
                  "## Logged bets: model probability versus observed hit rate", "",
                  "| logged pick | n | old logged p_model | fixed replay raw p_model | fixed replay calibrated p_model | realized |",
                  "|---|---:|---:|---:|---:|---:|"])
    for direction in ("OVER", "UNDER"):
        old = report["logged_p_model_baseline"][direction]
        raw = report["replayed_p_model_by_logged_pick"][direction]
        new = report["calibrated_p_model_by_logged_pick"][direction]
        lines.append(f"| {direction} | {old['n']} | {old['mean_p_model']:.3f} | "
                     f"{raw['mean_p_model']:.3f} | {new['mean_p_model']:.3f} | "
                     f"{old['realized_hit_rate']:.3f} |")
    selection = report["real_logged_line_selection"]
    lines.extend(["", "## Pick replay at the same historical real lines", "",
                  "This reselects a side at each of the *logged* opportunities, not all sportsbook",
                  "opportunities. Confidence/bookmaker/market-cap filters cannot be reconstructed",
                  "from this log; the displayed filter applies only the direction-specific edge floor.", "",
                  "| population | n | OVER | UNDER | remaining hit rate |", "|---|---:|---:|---:|---:|"])
    for label in ("logged", "replayed_all", "replayed_over_edge_floor"):
        block = selection[label]
        hit = sum(block[d]["n"] * (block[d]["hit_rate"] or 0) for d in ("OVER", "UNDER")) / block["n"]
        lines.append(f"| {label} | {block['n']} | {block['OVER']['n']} | {block['UNDER']['n']} | {hit:.3f} |")
    pts = selection["pts_logged_over"]
    lines.extend(["", f"Among {pts['n']} logged PTS OVER bets, "
                  f"{pts['flips_to_under']} switch to UNDER; "
                  f"{pts['stays_over_below_edge_floor']} remain OVER but fall below its edge floor.",
                  f"Overall, {selection['removed_by_edge_floor']} of 956 logged-line opportunities fail the edge floor.",
                  "This replay does not show an improved selected-pick hit rate; probability score",
                  "improvement does not establish bet profitability or justify a staking claim.", "",
                  "## Candidate/void coverage", "",
                  "A pre-game team-inferred roster is evaluated even if the player does not play;",
                  "such rows are VOID and are excluded from fit and score. Current rosters are not",
                  "a complete historical injury or transaction archive. Proxy opportunities and",
                  "void counts, early-season/low-minute exclusions and runtime are in",
                  "`opportunity-coverage.md`. Large row files are gitignored.", ""])
    REPORT_MD.write_text("\n".join(lines))


def main():
    start = time.perf_counter()
    proxy = pd.read_csv(PAPER_BETS)
    replay = pd.read_csv(RAW_REPLAY)
    chrono = pd.read_csv(CHRONO)
    bets = pd.read_csv(ROOT / "bet_log.csv")
    splits = json.loads((HERE / "splits.json").read_text())["2025-26"]
    sources = proxy_segments(proxy, splits)
    real, coverage = prepare_real_lines(bets, replay, chrono)
    real = real_chronological_predictions(real, sources["proxy_l10"])
    dates = sorted(real.date.unique())
    cutoff = dates[int(len(dates) * .7) - 1]
    real["heldout"] = real.date > cutoff
    holdout = real[real.heldout]
    gate = bootstrap_noninferiority(holdout)
    chosen_method = "platt" if gate["passed"] else "identity"
    paper = pd.concat(sources.values(), ignore_index=True)
    paper["p_over_calibrated"] = (paper.p_over_platt if chosen_method == "platt"
                                   else paper.p_over_uncalibrated)
    paper["p_under_calibrated"] = 1.0 - paper.p_over_calibrated
    paper["selected_pick"] = np.where(paper.p_over_calibrated > 0.5, "OVER", "UNDER")
    paper["graded_result"] = np.where(paper.graded_status == "DNP_VOID", "VOID",
                                       np.where(paper.selected_pick == paper.over_outcome, "HIT", "MISS"))
    paper["calibration_version"] = np.where(
        paper.calibration_asof_date != "", "v9-probability-1-weekly-asof", "identity-no-prior")
    paper.drop(columns=["y_over", "p_over_platt", "segment"]).sort_values(
        ["date", "player_id", "market", "line_source"]
    ).to_csv(PROXY_OUT, index=False)
    full_proxy_l10 = sources["proxy_l10"]
    final_fit = fit_platt(full_proxy_l10)
    artifact = {"version": "v9-probability-1", "method": chosen_method,
                "candidate_method": "platt", "training_line_source": "proxy_l10",
                "training_cutoff": str(full_proxy_l10.date.max()),
                "real_line_heldout_gate": gate,
                "review_after": "first four weeks of 2026-27 real bets" if chosen_method == "identity" else None,
                "markets": final_fit if chosen_method == "platt" else {}}
    ARTIFACT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    reports = {"version": "v9-probability-1", "label": "CALIBRATION/BACKTEST DIAGNOSTIC — SYNTHETIC PROXY LINES",
               "runtime_s": None, "real_coverage": coverage, "real_holdout_after": cutoff,
               "real_gate": gate, "deployed_method": chosen_method, "proxy": {}, "real": {}}
    reports["real_logged_line_selection"] = selection_comparison(real, chosen_method)
    reports["proxy_selection"] = {}
    for source, subset in paper[paper.graded_status == "GRADED"].groupby("line_source"):
        reports["proxy_selection"][source] = {}
        for season_type, block in subset.groupby("season_type"):
            reports["proxy_selection"][source][season_type] = {
                "n": len(block), "over": int((block.selected_pick == "OVER").sum()),
                "under": int((block.selected_pick == "UNDER").sum()),
                "hit_rate": float((block.graded_result == "HIT").mean())}
    for label, data in sources.items():
        data = data[(data.segment != "training_only") & (data.graded_status == "GRADED")]
        reports["proxy"][label] = {}
        for segment, g in data.groupby("segment"):
            reports["proxy"][label][segment] = detailed_scores(g)
    for label, data in (("all", real), ("heldout", holdout)):
        reports["real"][label] = detailed_scores(data)
    reports["logged_p_model_baseline"] = {}
    reports["replayed_p_model_by_logged_pick"] = {}
    reports["calibrated_p_model_by_logged_pick"] = {}
    for pick in ("OVER", "UNDER"):
        g = real[real.pick == pick]
        logged = g.p_model.to_numpy(dtype=float)
        replayed = np.where(pick == "OVER", g.p_over_uncalibrated, 1 - g.p_over_uncalibrated)
        calibrated = np.where(pick == "OVER", g.p_over_platt, 1 - g.p_over_platt)
        hit = (g.result == "HIT").to_numpy(dtype=float)
        reports["logged_p_model_baseline"][pick] = {"n": len(g), "mean_p_model": float(np.nanmean(logged)),
                                                      "realized_hit_rate": float(hit.mean())}
        reports["replayed_p_model_by_logged_pick"][pick] = {"n": len(g), "mean_p_model": float(np.mean(replayed)),
                                                              "realized_hit_rate": float(hit.mean())}
        reports["calibrated_p_model_by_logged_pick"][pick] = {"n": len(g),
            "mean_p_model": float(np.mean(calibrated)), "realized_hit_rate": float(hit.mean())}
    reports["runtime_s"] = time.perf_counter() - start
    REPORT_JSON.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
    write_readable_report(reports)
    real.to_csv(REAL_OUT, index=False, compression="gzip")
    print("[calibration] real matched", coverage, "gate", gate, "deployed", chosen_method,
          "runtime %.1fs" % reports["runtime_s"])


if __name__ == "__main__":
    main()
