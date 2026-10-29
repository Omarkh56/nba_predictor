#!/usr/bin/env python3
"""Produce pre-game candidate projections (including DNP/voids) and L10/L5 proxy-line diagnostics.

    python3 scripts/replay/backtest_bets.py opportunities --workers 8
    python3 scripts/replay/backtest_bets.py build

The large per-row files are gitignored. No synthetic bet is appended to bet_log.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (str(HERE), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ["ODDS_API_KEY"] = "replay-dummy-key"

from replay_cache import NBACache  # noqa: E402
from replay_config import MARKETS, LOW_MINUTES_THRESHOLD, SEASON_TYPE_PO, SEASON_TYPE_RS  # noqa: E402
from replay_data import load_season_bundle  # noqa: E402
from replay_engine import run_replay  # noqa: E402
from replay_provenance import code_fingerprint  # noqa: E402
from playerlinepredictor import COMBO_UNDER_SD_MULT, MARKET_ANCHOR_WEIGHT  # noqa: E402

OPPORTUNITIES = HERE / "output" / "opportunities_l5_2025-26.csv.gz"
PAPER_BETS = ROOT / "backtest_bets.csv"
OPPORTUNITY_INFO = HERE / "output" / "opportunity-info.json"
OPPORTUNITY_REPORT = HERE / "reports" / "opportunity-coverage.md"
CORRECTION_SOURCE = HERE / "output" / "after-correction-raw_2025-26.csv.gz"


def run_opportunities(args) -> None:
    start = time.perf_counter()
    cache = NBACache(str(args.cache_dir), min_interval_s=args.min_interval)
    bundle = load_season_bundle(cache, "2025-26")
    df, info = run_replay(bundle, [SEASON_TYPE_RS, SEASON_TYPE_PO], args.workers, {}, str(ROOT),
                          log=lambda m: print("[proxy] " + m, flush=True),
                          min_prior_games=5, include_voids=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, compression="gzip")
    OPPORTUNITY_INFO.write_text(json.dumps({"runtime_s": time.perf_counter() - start,
                                            "cache": cache.stats.as_dict(), "info": info,
                                            "bundle": bundle.meta}, indent=2, default=str) + "\n")
    write_opportunity_report()
    print("[proxy] wrote %s (%d market rows, %d void, %.1fs elapsed, %d API requests)" % (
        args.output, len(df), int((df.graded_status == "DNP_VOID").sum()),
        time.perf_counter() - start, cache.stats.requests))
    print("[proxy] counts=%s failures=%s" % (info["counts"], info["failures"]))


def write_opportunity_report() -> None:
    info = json.loads(OPPORTUNITY_INFO.read_text())
    counts = info["info"]["counts"]
    failures = info["info"]["failures"]
    lines = ["# 2025-26 RS and 2026 PO pre-game proxy coverage", "",
             "Synthetic L10/L5 lines are calibration diagnostics only, never ROI/edge/profit evidence.",
             "A prior played team plus that day's schedule identifies the pre-game candidate.",
             "A player not observed for that team that day is a void. Trades/long absences",
             "and missing injury records limit historical roster reconstruction.", "",
             f"- Replay dates: {info['info']['dates']}; eligible candidate player-games (L5): "
             f"{counts.get('eligible_player_games', 0)}; projected market rows: "
             f"{counts.get('projection_rows', 0)}.",
             f"- DNP/ambiguous candidate player-games: {counts.get('dnp_void_player_games', 0)}; "
             f"historical same-day changed-team ambiguity: {failures.get('changed_team_unobservable', 0)}; "
             f"stale prior-team candidates excluded: {failures.get('stale_prior_team', 0)}.",
             f"- Reconstruction failures: {failures.get('project_prop_returned_none', 0)} "
             "market projections unavailable; these are never fabricated as bets.",
             f"- Warm local API cache: {info['cache'].get('requests', 'unknown')} live requests; "
             f"wall runtime {info['runtime_s']:.1f}s.", "",
             "| proxy prerequisite | excluded for too few played games | of these, Oct–Nov | of these, low recent minutes | eligible low-minute roles |",
             "|---|---:|---:|---:|---:|",]
    for n in (5, 10):
        lines.append(f"| L{n} | {counts.get(f'excluded_l{n}_prior_games', 0)} | "
                     f"{counts.get(f'excluded_l{n}_early_season', 0)} | "
                     f"{counts.get(f'excluded_l{n}_low_minutes', 0)} | "
                     f"{counts.get(f'eligible_l{n}_low_minutes', 0)} |")
    lines.extend(["", "Early season and low-minute categories overlap. Low-minute status uses",
                  "the prior 10 played games and is **reported**, not used to reject a candidate.",
                  "L10 is primary; L5 is sensitivity. Only graded rows enter calibration.", ""])
    OPPORTUNITY_REPORT.write_text("\n".join(lines))


def proxy_history(players: pd.DataFrame, n: int) -> pd.DataFrame:
    """Per-market last n PLAYED games; a target day's game is not included."""
    wide = players.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()
    markets = []
    for label, _, columns in MARKETS:
        market = wide[["PLAYER_ID", "GAME_DATE", "MIN"]].copy()
        market["market"] = label
        market["stat"] = wide[columns].astype(float).sum(axis=1)
        markets.append(market)
    long = pd.concat(markets, ignore_index=True).sort_values(["PLAYER_ID", "market", "GAME_DATE"])
    grp = long.groupby(["PLAYER_ID", "market"], sort=False)
    long["rolling_average"] = grp["stat"].transform(lambda s: s.rolling(n, min_periods=n).mean())
    long["sample_count"] = grp.cumcount() + 1
    return long[["PLAYER_ID", "GAME_DATE", "market", "rolling_average", "sample_count"]]


def join_proxy_history(opps: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    left = opps.rename(columns={"date": "prediction_date"}).copy()
    left["cutoff"] = pd.to_datetime(left["prediction_date"])
    right = history.rename(columns={"PLAYER_ID": "player_id", "GAME_DATE": "previous_game_date"})
    joined = pd.merge_asof(left.sort_values("cutoff"), right.sort_values("previous_game_date"),
                           left_on="cutoff", right_on="previous_game_date",
                           by=["player_id", "market"], allow_exact_matches=False, direction="backward")
    if not joined["previous_game_date"].dropna().lt(joined.loc[joined.previous_game_date.notna(), "cutoff"]).all():
        raise AssertionError("proxy history used same-day or future game")
    return joined


def online_corrections(opps: pd.DataFrame, replay: pd.DataFrame, k: int = 1000) -> pd.DataFrame:
    """Same additive k selected in Commit 6, using prior-date unconditional outcomes."""
    source = replay[["date", "market", "projection", "actual"]].copy()
    source["residual"] = source["actual"] - source["projection"]
    per_date = source.groupby(["date", "market"]).agg(n=("actual", "size"),
                                                        residual_sum=("residual", "sum")).reset_index()
    per_date = per_date.sort_values(["market", "date"])
    for key in ("n", "residual_sum"):
        per_date["cum_" + key] = per_date.groupby("market")[key].cumsum()
    left = opps.copy()
    left["cutoff"] = pd.to_datetime(left["date"])
    per_date["source_date"] = pd.to_datetime(per_date["date"])
    joined = pd.merge_asof(left.sort_values("cutoff"),
                           per_date[["source_date", "market", "cum_n", "cum_residual_sum"]].sort_values("source_date"),
                           left_on="cutoff", right_on="source_date", by="market",
                           allow_exact_matches=False, direction="backward")
    n = joined.cum_n.fillna(0.0).to_numpy(dtype=float)
    total = joined.cum_residual_sum.fillna(0.0).to_numpy(dtype=float)
    adjustment = total / (n + k)  # == n/(n+k) * mean residual; n=0 -> 0
    raw = joined.projection.to_numpy(dtype=float)
    joined["uncorrected_projection"] = raw
    joined["final_corrected_projection"] = np.maximum(0.0, raw + adjustment)
    joined["correction_asof_date"] = joined.source_date
    joined["correction_n_prior"] = n.astype(int)
    joined["corrected_sd"] = joined.sd * np.sqrt(np.divide(joined.final_corrected_projection, raw,
                                              out=np.ones_like(raw), where=raw > 0))
    return joined


def probability_over(proj, sd, line, is_single):
    """Vectorized form of playerlinepredictor._nb_over_prob/normal_cdf."""
    mean = np.asarray(proj, dtype=float)
    standard = np.asarray(sd, dtype=float)
    point = np.asarray(line, dtype=float)
    out = np.full(len(mean), 0.5)
    valid = (mean > 0) & (standard > 0) & np.isfinite(point)
    gaussian = valid & ((standard ** 2 <= mean) | ~is_single)
    out[gaussian] = norm.cdf((mean[gaussian] - point[gaussian]) / standard[gaussian])
    nb = valid & ~gaussian
    variance = standard[nb] ** 2
    shape = mean[nb] ** 2 / (variance - mean[nb])
    success = mean[nb] / variance
    out[nb] = nbinom.sf(np.floor(point[nb]).clip(min=0), shape, success)
    return np.clip(out, 1e-6, 1 - 1e-6)


def pregame_probability(proj, sd, line, markets):
    """Recreate live line anchor and the combo-UNDER SD inflation before calibration."""
    point = np.asarray(line, dtype=float)
    anchor = (1.0 - MARKET_ANCHOR_WEIGHT) * np.asarray(proj, dtype=float) + MARKET_ANCHOR_WEIGHT * point
    single = pd.Series(markets).isin(("PTS", "REB", "AST", "3PM")).to_numpy(dtype=bool)
    first = probability_over(anchor, sd, point, single)
    key = pd.Series(markets)
    under_combo = (first <= 0.5) & key.isin(("PTS+AST", "PRA", "PTS+REB")).to_numpy(dtype=bool)
    adjusted_sd = np.asarray(sd, dtype=float) * np.where(under_combo, COMBO_UNDER_SD_MULT, 1.0)
    return probability_over(anchor, adjusted_sd, point, single), anchor, adjusted_sd


def build_paper_bets(args) -> None:
    start = time.perf_counter()
    cache = NBACache(str(args.cache_dir), min_interval_s=args.min_interval)
    bundle = load_season_bundle(cache, "2025-26")
    opps = pd.read_csv(args.input)
    replay = pd.read_csv(args.raw_replay)
    for n in (10, 5):
        history = proxy_history(bundle.players, n)
        aligned = join_proxy_history(opps, history)
        aligned = aligned[(aligned.n_prior >= n) & (aligned.sample_count >= n)].copy()
        aligned["proxy_line"] = np.floor(aligned.rolling_average) + 0.5
        aligned["line_source"] = f"proxy_l{n}"
        if n == 10:
            both = [aligned]
        else:
            both.append(aligned)
    joined = pd.concat(both, ignore_index=True).rename(columns={"prediction_date": "date"})
    joined = online_corrections(joined, replay)
    joined["p_over_raw_projection"], joined["anchored_raw_projection"], _ = pregame_probability(
        joined.uncorrected_projection, joined.sd, joined.proxy_line, joined.market)
    joined["p_over_uncalibrated"], joined["anchored_corrected_projection"], joined["probability_sd"] = pregame_probability(
        joined.final_corrected_projection, joined.corrected_sd, joined.proxy_line, joined.market)
    joined["p_under_raw_projection"] = 1.0 - joined.p_over_raw_projection
    joined["p_under_uncalibrated"] = 1.0 - joined.p_over_uncalibrated
    joined["p_over_calibrated"] = np.nan  # set after chronological calibration
    joined["p_under_calibrated"] = np.nan
    joined["selected_pick"] = np.where(joined.p_over_uncalibrated > 0.5, "OVER", "UNDER")
    joined["over_outcome"] = np.where(joined.graded_status == "DNP_VOID", "VOID",
                                      np.where(joined.actual > joined.proxy_line, "OVER", "UNDER"))
    joined["graded_result"] = np.where(joined.graded_status == "DNP_VOID", "VOID",
                                       np.where(joined.selected_pick == joined.over_outcome, "HIT", "MISS"))
    joined["as_of_cutoff"] = joined.cutoff.dt.strftime("%Y-%m-%dT00:00:00")
    joined["model_version"] = code_fingerprint(str(ROOT))[:16]
    joined["residual_correction_version"] = "v9-unconditional-1-online-asof"
    joined["calibration_version"] = "pending"
    joined["season"] = "2025-26"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cols = ["date", "game_id", "player_id", "player", "team", "opp", "market", "season", "season_type",
            "as_of_cutoff", "previous_game_date", "sample_count", "rolling_average", "proxy_line",
            "line_source", "uncorrected_projection", "final_corrected_projection", "sd", "corrected_sd",
            "anchored_raw_projection", "anchored_corrected_projection", "probability_sd",
            "p_over_raw_projection", "p_under_raw_projection", "p_over_uncalibrated", "p_under_uncalibrated",
            "p_over_calibrated", "p_under_calibrated", "selected_pick", "actual", "over_outcome", "graded_result",
            "graded_status", "actual_min", "roll_min10", "n_prior", "correction_asof_date", "correction_n_prior",
            "model_version", "residual_correction_version", "calibration_version"]
    joined[cols].sort_values(["date", "player_id", "market", "line_source"]).to_csv(args.output, index=False)
    print("[proxy] wrote %s (%d rows, %d void, %.1fs, %d API requests)" % (
        args.output, len(joined), int((joined.graded_result == "VOID").sum()),
        time.perf_counter() - start, cache.stats.requests))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("opportunities")
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--cache-dir", type=Path, default=ROOT / "replay-cache")
    run.add_argument("--min-interval", type=float, default=1.2)
    run.add_argument("--output", type=Path, default=OPPORTUNITIES)
    run.set_defaults(fn=run_opportunities)
    build = sub.add_parser("build")
    build.add_argument("--input", type=Path, default=OPPORTUNITIES)
    build.add_argument("--raw-replay", type=Path, default=CORRECTION_SOURCE)
    build.add_argument("--cache-dir", type=Path, default=ROOT / "replay-cache")
    build.add_argument("--min-interval", type=float, default=1.2)
    build.add_argument("--output", type=Path, default=PAPER_BETS)
    build.set_defaults(fn=build_paper_bets)
    coverage = sub.add_parser("coverage")
    coverage.set_defaults(fn=lambda _: write_opportunity_report())
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
