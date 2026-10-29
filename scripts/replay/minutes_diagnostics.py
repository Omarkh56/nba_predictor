"""Minutes MAE, projection bias by minutes error, and first-three-week audit.

Player-game minutes are counted once. Bias is the unweighted mean of market
percentage biases, matching the deployed correction's reported benchmark.
Comparisons use shared player/game/market keys; coverage differences are explicit.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["date", "game_id", "player_id", "market"]
BUCKETS = ["<-10", "-10:-5", "-5:-2", "-2:2", "2:5", "5:10", ">10"]


def evaluate_gate(validation, heldout):
    """Conservative predeclared gate: both MAEs improve in each season/regime.

    The bias benchmark is the deployed correction's macro per-market bias.
    Missing/nonfinite evidence fails closed; equality does not count as better.
    """
    checks = []
    for split, report in (("validation", validation), ("heldout", heldout)):
        for regime in ("Regular Season", "Playoffs"):
            models = report.get("regimes", {}).get(regime, {})
            for metric in ("minutes_mae", "projection_mae"):
                candidate = models.get("candidate", {}).get(metric)
                baseline = models.get("baseline", {}).get(metric)
                valid = all(isinstance(v, (int, float)) and math.isfinite(v) for v in (candidate, baseline))
                checks.append({"split": split, "regime": regime, "metric": metric,
                               "candidate": candidate, "baseline": baseline,
                               "passed": bool(valid and candidate < baseline)})
    for regime, threshold in (("Regular Season", 2.3), ("Playoffs", 4.2)):
        bias = heldout.get("regimes", {}).get(regime, {}).get("candidate", {}).get("macro_bias_pct")
        passed = isinstance(bias, (int, float)) and math.isfinite(bias) and abs(bias) < threshold
        checks.append({"split": "heldout", "regime": regime, "metric": "absolute_macro_bias_pct",
                       "candidate": bias, "deployed_absolute_benchmark": threshold, "passed": bool(passed)})
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def metrics(frame):
    if frame.empty:
        return {"rows": 0, "player_games": 0, "minutes_mae": None, "macro_bias_pct": None,
                "projection_mae": None, "minutes_error_buckets": {}}
    frame = frame.copy()
    frame["minute_error"] = frame.proj_min - frame.actual_min
    minute_rows = frame.drop_duplicates(["date", "game_id", "player_id"])
    by_market = {}
    for market, group in frame.groupby("market"):
        denominator = group.actual.sum()
        by_market[market] = float(100 * (group.projection - group.actual).sum() / denominator) if denominator else None
    macro = [value for value in by_market.values() if value is not None]
    bucket = pd.cut(frame.minute_error, [-np.inf, -10, -5, -2, 2, 5, 10, np.inf], labels=BUCKETS)
    buckets = {}
    for label in BUCKETS:
        group = frame[bucket == label]
        denominator = group.actual.sum()
        buckets[label] = {"rows": len(group),
                          "bias_pct": float(100 * (group.projection - group.actual).sum() / denominator) if denominator else None}
    return {"rows": len(frame), "player_games": len(minute_rows),
            "minutes_mae": float(minute_rows.minute_error.abs().mean()),
            "macro_bias_pct": float(np.mean(macro)) if macro else None,
            "bias_pct_by_market": by_market,
            "projection_mae": float((frame.projection - frame.actual).abs().mean()),
            "minutes_error_buckets": buckets}


def diagnostics(candidate, baseline=None, season_start=None):
    candidate = candidate.dropna(subset=["actual", "actual_min", "proj_min", "projection"]).copy()
    def canonical_game(value):
        text = str(value)
        return str(int(text)) if text.isdigit() else text
    candidate["game_id"] = candidate.game_id.map(canonical_game)
    if baseline is not None:
        baseline = baseline.copy()
        baseline["game_id"] = baseline.game_id.map(canonical_game)
    if candidate.duplicated(KEYS).any():
        raise ValueError("duplicate replay keys")
    output = {"candidate_rows": len(candidate), "baseline_rows": len(baseline) if baseline is not None else None,
              "regimes": {}, "limitations": ["Historical starter flag uses a minutes proxy.",
                  "Historical book spreads are held at zero; blowout feature is unit-tested only.",
                  "Returns are scheduled-game absences, not inferred injury diagnoses.",
                  "Replay population is conditional on playing; unavailable projections are reported separately."]}
    if baseline is not None:
        if baseline.duplicated(KEYS).any():
            raise ValueError("duplicate baseline keys")
        shared = candidate[KEYS].merge(baseline[KEYS], on=KEYS, validate="one_to_one")
        output["paired_rows"] = len(shared)
        output["candidate_only_rows"] = len(candidate) - len(shared)
        output["baseline_only_rows"] = len(baseline) - len(shared)
        candidate = shared.merge(candidate, on=KEYS, validate="one_to_one")
        baseline = shared.merge(baseline, on=KEYS, validate="one_to_one")
    for regime, frame in candidate.groupby("season_type"):
        item = {"candidate": metrics(frame)}
        if baseline is not None:
            item["baseline"] = metrics(baseline[baseline.season_type == regime])
        output["regimes"][regime] = item
    if not candidate.empty:
        coverage = candidate.drop_duplicates(["date", "game_id", "player_id"])
        output["feature_coverage"] = {column: coverage[column].fillna("unavailable").value_counts().to_dict()
                                      for column in ("starter_source", "absence_source") if column in coverage}
        start = pd.Timestamp(season_start) if season_start else pd.to_datetime(candidate.date).min()
        end = start + pd.Timedelta(days=20)
        mask = (pd.to_datetime(candidate.date) <= end) & (candidate.n_prior < 10)
        output["first_three_weeks_low_games"] = {"from": str(start.date()), "through": str(end.date()),
                                                  "candidate": metrics(candidate[mask])}
        if baseline is not None:
            mask = (pd.to_datetime(baseline.date) <= end) & (baseline.n_prior < 10)
            output["first_three_weeks_low_games"]["baseline"] = metrics(baseline[mask])
    return output


def write_diagnostics(candidate, baseline, label, season, reports_dir, season_start=None):
    result = diagnostics(candidate, baseline, season_start)
    result.update(label=label, season=season)
    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (label + "_" + season + "_minutes.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Minutes replay: " + label + " / " + season, "",
             "Minutes MAE is counted once per player-game; bias is macro averaged over markets.", "",
             "| Regime | Model | Player-games | Minutes MAE | Bias | Projection MAE |",
             "|---|---|---:|---:|---:|---:|"]
    for regime, models in result["regimes"].items():
        for model, values in models.items():
            lines.append("| %s | %s | %d | %.4f | %+.4f%% | %.4f |" % (
                regime, model, values["player_games"], values["minutes_mae"],
                values["macro_bias_pct"], values["projection_mae"]))
    lines += ["", "Full minutes-error buckets and early-season results are in the adjacent JSON.", ""]
    lines += ["- " + limitation for limitation in result["limitations"]]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return result
