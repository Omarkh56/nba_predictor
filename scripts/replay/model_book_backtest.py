#!/usr/bin/env python3
"""Backtest the model's own no-vig lines and two-sided fair odds.

Uses cached historical NBA outcomes, the saved pre-2025 game model, and the
strictly-as-of L10 prop replay. No sportsbook prices, ROI, EV or customer bet
behavior are inferred. A half-point prop line is set from the corrected
projection; Platt calibration is refitted using only earlier L10 proxy dates.

    python3 scripts/replay/model_book_backtest.py --season 2025-26
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for entry in (str(HERE), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)
os.environ["ODDS_API_KEY"] = "replay-dummy-key"

from calibrate_probabilities import apply_platt, fit_platt  # noqa: E402
from backtest_bets import pregame_probability  # noqa: E402
from game_features import build_game_dataset  # noqa: E402
from replay_cache import NBACache  # noqa: E402
from replay_config import DEFAULT_CACHE_DIR, MARKET_ORDER  # noqa: E402
from replay_data import load_season_bundle  # noqa: E402
import newnbapredictor as game_model  # noqa: E402

OUT = HERE / "output"
REPORTS = HERE / "reports"
EPS = 1e-6


def fair_odds(probability):
    """Return decimal and American fair odds; both sides have zero overround."""
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)
    decimal = 1.0 / p
    american = np.where(p >= 0.5, -100.0 * p / (1.0 - p),
                        100.0 * (1.0 - p) / p)
    return decimal, american


def score(p, y) -> dict:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    if not len(y):
        return {"n": 0}
    return {"n": int(len(y)), "mean_quote_probability": float(p.mean()),
            "realized_rate": float(y.mean()),
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1-y) * np.log(1-p))),
            "favorite_accuracy": float(np.mean((p >= .5) == y))}


def reliability(p, y) -> list:
    """Outcome rates by own quoted implied-probability bucket."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    edges = (0, .2, .3, .4, .5, .6, .7, .8, 1.000001)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        take = (p >= lo) & (p < hi)
        bins.append({"probability_band": f"{lo:.1f}-{min(hi, 1):.1f}",
                     "n": int(take.sum()),
                     "quoted": float(p[take].mean()) if take.any() else None,
                     "realized": float(y[take].mean()) if take.any() else None})
    return bins


def game_quotes(season: str) -> tuple[pd.DataFrame, dict]:
    def offline_fetch(*_args, **_kwargs):
        raise OSError("network disabled for the model-book replay")

    cache = NBACache(DEFAULT_CACHE_DIR, fetcher=offline_fetch, retries=1, base_delay_s=0)
    bundle = load_season_bundle(cache, season, want_rosters=False)
    if cache.stats.requests:
        raise AssertionError("historical model-book replay attempted a network request")
    team, player = bundle.teams.copy(), bundle.players.copy()
    team["season"] = season
    player["season"] = season
    dataset = build_game_dataset(team, player_logs=player)
    if dataset.empty:
        raise ValueError("shared game-feature builder produced no historical games")
    params = json.loads((ROOT / "learned_params.json").read_text())
    if season in params.get("trained_on", []) or season in (params.get("validated_on"), params.get("tested_on")):
        raise ValueError("refusing to treat an artifact's train/selection season as unseen holdout")
    if not game_model.USE_LEARNED_GAME_MODEL or game_model.USE_KALMAN_GAME_MODEL:
        raise ValueError("saved learned game model is not the active live game model")
    features = params["game_model"]["feature_names"]
    missing = [name for name in features if name not in dataset]
    if missing:
        raise ValueError("historical game features missing: " + ", ".join(missing))
    probs = [game_model._sigmoid_learned(row) for row in dataset[features].to_dict("records")]
    if any(p is None or not np.isfinite(p) for p in probs):
        raise ValueError("learned game model fell back or produced invalid probability")
    game_regime = team[["GAME_ID", "SEASON_TYPE"]].drop_duplicates().copy()
    game_regime["GAME_ID"] = game_regime.GAME_ID.astype(str)
    dataset["game_id"] = dataset.game_id.astype(str)
    dataset = dataset.merge(game_regime, left_on="game_id", right_on="GAME_ID",
                            how="left", validate="one_to_one")
    keep = [c for c in ("game_date", "game_id", "home_tid", "away_tid", "season",
                        "SEASON_TYPE", "home_win") if c in dataset]
    quoted = dataset[keep].copy()
    quoted = quoted.rename(columns={"SEASON_TYPE": "season_type"})
    team_abbr = (team[["TEAM_ID", "TEAM_ABBREVIATION"]].drop_duplicates("TEAM_ID")
                 .set_index("TEAM_ID")["TEAM_ABBREVIATION"].to_dict())
    quoted["home"] = quoted.home_tid.map(team_abbr)
    quoted["away"] = quoted.away_tid.map(team_abbr)
    quoted["p_home"] = np.asarray(probs, dtype=float)
    quoted["p_away"] = 1.0 - quoted.p_home
    quoted["home_decimal"], quoted["home_american"] = fair_odds(quoted.p_home)
    quoted["away_decimal"], quoted["away_american"] = fair_odds(quoted.p_away)
    quoted["quoted_before_game"] = True
    quoted["game_date"] = pd.to_datetime(quoted.game_date).dt.strftime("%Y-%m-%d")
    if quoted.game_id.duplicated().any():
        raise ValueError("duplicate historical game id")
    summary = {"population": f"{season} cached regular-season and playoff games with shared pre-game features",
               "training_seasons": params.get("trained_on"),
               "training_validation_test_seasons": [params.get("validated_on"), params.get("tested_on")],
               "feature_version": params["game_model"].get("feature_version"),
               "cache": cache.stats.as_dict(),
               "overall": score(quoted.p_home, quoted.home_win),
               "by_regime": {kind: score(g.p_home, g.home_win)
                             for kind, g in quoted.groupby("season_type")},
               "by_month": {month: score(g.p_home, g.home_win) for month, g in
                            quoted.groupby(quoted.game_date.str[:7])},
               "reliability": reliability(quoted.p_home, quoted.home_win),
               "examples": quoted.iloc[
                   [int(quoted.p_home.argmin()), int((quoted.p_home - .5).abs().argmin()),
                    int(quoted.p_home.argmax())]
               ][["game_date", "away", "home", "p_home", "home_american", "away_american",
                  "home_win"]].to_dict("records"),
               "overround_max_error": float(np.max(abs(1 / quoted.home_decimal +
                                                      1 / quoted.away_decimal - 1)))}
    return quoted, summary


def prop_quotes(season: str) -> tuple[pd.DataFrame, dict]:
    source = ROOT / "backtest_bets.csv"
    if not source.is_file():
        raise FileNotFoundError("backtest_bets.csv missing; run backtest_bets.py first")
    base = pd.read_csv(source)
    base = base[(base.line_source == "proxy_l10") & (base.season == season)].copy()
    if base.empty:
        raise ValueError("no L10 opportunities for season")
    base["house_line"] = np.floor(base.final_corrected_projection) + 0.5
    base["p_over_raw"], base["anchored_projection"], base["probability_sd"] = pregame_probability(
        base.final_corrected_projection, base.corrected_sd, base.house_line, base.market)
    base["p_over"] = np.nan
    base["calibration_last_prior_date"] = ""
    base["week_start"] = pd.to_datetime(base.date).dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")
    training = pd.read_csv(source, usecols=["date", "line_source", "market", "graded_status",
                                            "actual", "proxy_line", "p_over_uncalibrated"])
    training = training[(training.line_source == "proxy_l10") &
                        (training.graded_status == "GRADED")].copy()
    training["y_over"] = (training.actual > training.proxy_line).astype(int)
    for start, part in base.groupby("week_start", sort=True):
        past = training[training.date < start]
        fitted = fit_platt(past)
        if fitted and max(v["as_of_last_date"] for v in fitted.values()) >= start:
            raise AssertionError("same-week or future calibration outcome used")
        base.loc[part.index, "p_over"] = apply_platt(part.p_over_raw, part.market, fitted)
        base.loc[part.index, "calibration_last_prior_date"] = str(past.date.max()) if len(past) else ""
    base["p_under"] = 1.0 - base.p_over
    base["over_decimal"], base["over_american"] = fair_odds(base.p_over)
    base["under_decimal"], base["under_american"] = fair_odds(base.p_under)
    base["over_result"] = np.where(base.graded_status == "DNP_VOID", "VOID",
                                    np.where(base.actual > base.house_line, "HIT", "MISS"))
    grade = base[base.graded_status == "GRADED"]
    if grade.actual.isna().any():
        raise ValueError("graded prop has missing actual")
    by_market = {}
    for market in MARKET_ORDER:
        group = grade[grade.market == market]
        y = (group.over_result == "HIT").astype(int)
        by_market[market] = {"raw": score(group.p_over_raw, y),
                             "calibrated": score(group.p_over, y)}
    summary = {"population": "L10 eligible player-games; model sets floor(corrected projection)+0.5 line",
               "quote_price": "two-sided fair odds with no vig",
               "rows": int(len(base)), "graded": int(len(grade)),
               "void": int((base.graded_status == "DNP_VOID").sum()),
               "overall_raw": score(grade.p_over_raw, grade.over_result == "HIT"),
               "overall_calibrated": score(grade.p_over, grade.over_result == "HIT"),
               "by_market": by_market,
               "by_regime": {kind: score(g.p_over, g.over_result == "HIT")
                             for kind, g in grade.groupby("season_type")},
               "reliability": reliability(grade.p_over, grade.over_result == "HIT"),
               "overround_max_error": float(np.max(abs(1 / base.over_decimal +
                                                      1 / base.under_decimal - 1)))}
    cols = ["date", "game_id", "player_id", "player", "team", "opp", "market", "season_type",
            "house_line", "final_corrected_projection", "actual", "graded_status", "over_result",
            "p_over_raw", "p_over", "p_under", "over_decimal", "under_decimal",
            "over_american", "under_american", "calibration_last_prior_date"]
    return base[cols].copy(), summary


def write_report(season: str, games: dict, props: dict, runtime: dict) -> None:
    lines = [f"# Model-book historical replay: {season}", "",
             "All lines and odds below are generated by the model with **no vig**.",
             "No historical sportsbook prices or customer bets were used; this tests",
             "prediction/calibration, not betting ROI, EV or house revenue.", "",
             f"- Game model was trained on {', '.join(games['training_seasons'])}; "
             f"{season} is an unseen holdout.",
             f"- Games: {games['overall']['n']}; home Brier {games['overall']['brier']:.4f}, "
             f"log loss {games['overall']['log_loss']:.4f}, "
             f"most-likely-side accuracy {games['overall']['favorite_accuracy']:.1%}.",
             f"- Props: {props['graded']:,} graded market quotes and {props['void']:,} void quotes; "
             f"raw/calibrated OVER Brier {props['overall_raw']['brier']:.4f}/"
             f"{props['overall_calibrated']['brier']:.4f}; raw/calibrated log loss "
             f"{props['overall_raw']['log_loss']:.4f}/{props['overall_calibrated']['log_loss']:.4f}.",
             f"- Own-line prop quote: `floor(as-of corrected projection) + 0.5`; "
             "OVER/UNDER odds are priced from the predictive distribution and an",
             "earlier-week L10-fitted calibrator. A DNP is void.",
             f"- Max game/prop fair-odds overround error: "
             f"{games['overround_max_error']:.2g}/{props['overround_max_error']:.2g}.",
             f"- Runtime {runtime['total_s']:.1f}s; 0 historical odds requests.", "",
             "Game split:", ""]
    for regime, metrics in games["by_regime"].items():
        lines.append(f"- {regime}: {metrics['n']} games; Brier {metrics['brier']:.4f}, "
                     f"log loss {metrics['log_loss']:.4f}, "
                     f"most-likely-side accuracy {metrics['favorite_accuracy']:.1%}.")
    lines.extend(["",
             "Fixed examples: lowest home probability, nearest 50%, highest home probability:", ""])
    for game in games["examples"]:
        result = game["home"] if game["home_win"] else game["away"]
        lines.append(f"- {game['game_date']} {game['away']} @ {game['home']}: "
                     f"home {game['p_home']:.1%}, fair home/away American "
                     f"{game['home_american']:+.0f}/{game['away_american']:+.0f}; "
                     f"actual winner {result}.")
    lines.extend(["",
             "| prop market | n | raw Brier | calibrated Brier | raw log loss | calibrated log loss |",
             "|---|---:|---:|---:|---:|---:|"])
    for market in MARKET_ORDER:
        a, b = props["by_market"][market]["raw"], props["by_market"][market]["calibrated"]
        lines.append(f"| {market} | {a['n']} | {a['brier']:.4f} | {b['brier']:.4f} | "
                     f"{a['log_loss']:.4f} | {b['log_loss']:.4f} |")
    lines.extend(["", "The house line differs from the L10 training proxy line. This is a",
                  "cross-line calibration test, not a fitted-to-itself score. Exact row-level",
                  "quotes and game outcomes are saved locally under `scripts/replay/output/`.", "",
                  "## Reliability by model-quoted probability", "",
                  "The accompanying JSON contains game and prop outcome rates for each quoted",
                  "probability band, with regular-season/playoff game and prop breakdowns.", ""])
    (REPORTS / f"model-book-{season}.md").write_text("\n".join(lines))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="2025-26")
    args = ap.parse_args(argv)
    start = time.perf_counter()
    game_rows, game_summary = game_quotes(args.season)
    game_s = time.perf_counter() - start
    print("[model-book] quoted %d historical games in %.1fs" % (len(game_rows), game_s), flush=True)
    prop_rows, prop_summary = prop_quotes(args.season)
    runtime = {"game_s": game_s, "prop_s": time.perf_counter() - start - game_s,
               "total_s": time.perf_counter() - start}
    OUT.mkdir(exist_ok=True, parents=True)
    REPORTS.mkdir(exist_ok=True, parents=True)
    game_rows.to_csv(OUT / f"model_book_games_{args.season}.csv.gz", index=False, compression="gzip")
    prop_rows.to_csv(OUT / f"model_book_props_{args.season}.csv.gz", index=False, compression="gzip")
    result = {"season": args.season, "games": game_summary, "props": prop_summary, "runtime": runtime,
              "source": "model-own no-vig lines/odds; strictly historical; no sportsbook odds"}
    (REPORTS / f"model-book-{args.season}.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(args.season, game_summary, prop_summary, runtime)
    print("[model-book] quoted %d prop rows, %d graded, %d void in %.1fs" % (
        len(prop_rows), prop_summary["graded"], prop_summary["void"], runtime["total_s"]))


if __name__ == "__main__":
    main()
