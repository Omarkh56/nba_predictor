"""Replay metrics: residual convention, exact-market table, frozen chronological splits, paired bootstrap."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "replay"))

from replay_metrics import (  # noqa: E402
    assign_split,
    compute_splits,
    load_or_create_splits,
    market_metrics,
    metrics_table,
    paired_bootstrap,
    paired_frame,
)


def test_market_metrics_hand_computed_and_residual_sign_is_projection_minus_actual():
    df = pd.DataFrame({"actual": [10.0, 20.0, 30.0, 40.0], "projection": [12.0, 18.0, 33.0, 41.0]})
    m = market_metrics(df)
    assert m["n"] == 4
    assert m["mean_actual"] == 25.0 and m["mean_projection"] == 26.0
    assert m["mean_residual"] == pytest.approx(1.0)  # projection - actual, positive = over-projection
    assert m["pct_bias"] == pytest.approx(0.04)  # mean(proj - actual) / mean(actual)
    assert m["median_residual"] == pytest.approx(1.5)
    assert m["mae"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx(np.sqrt(4.5))
    assert m["p_actual_gt_projection"] == pytest.approx(0.25)


def test_metrics_table_has_every_exact_market_and_a_pooled_all_row():
    rows = []
    for mk, act, proj in (("PTS", 20.0, 22.0), ("3PM", 2.0, 2.5), ("PRA", 30.0, 31.0)):
        rows += [{"market": mk, "actual": act, "projection": proj}] * 3
    t = metrics_table(pd.DataFrame(rows))
    assert list(t.index) == ["PTS", "3PM", "PRA", "ALL"] or set(t.index) == {"PTS", "3PM", "PRA", "ALL"}
    assert t.loc["3PM", "pct_bias"] == pytest.approx(0.25)
    assert t.loc["ALL", "n"] == 9


def _obs(dates_counts, st="Regular Season"):
    rows = []
    for d, n in dates_counts:
        rows += [{"date": d, "season_type": st, "player_id": i, "market": "PTS", "actual": 10.0, "projection": 11.0} for i in range(n)]
    return pd.DataFrame(rows)


def test_splits_are_chronological_by_observation_count_and_assign_correctly():
    df = _obs([("2026-01-01", 50), ("2026-01-02", 25), ("2026-01-03", 15), ("2026-01-04", 10)])
    sp = compute_splits(df)["Regular Season"]
    assert sp == {"train_end": "2026-01-01", "val_end": "2026-01-02"}
    labels = assign_split(df, {"Regular Season": sp})
    assert labels[df["date"] == "2026-01-01"].unique().tolist() == ["train"]
    assert labels[df["date"] == "2026-01-02"].unique().tolist() == ["val"]
    assert set(labels[df["date"] >= "2026-01-03"]) == {"test"}


def test_splits_are_frozen_once_written_and_partial_runs_never_define_them(tmp_path):
    path = str(tmp_path / "splits.json")
    full = _obs([("2026-01-01", 50), ("2026-01-02", 25), ("2026-01-03", 25)])
    first = load_or_create_splits(full, "2025-26", path=path)
    other = _obs([("2026-02-01", 10), ("2026-02-02", 10), ("2026-02-03", 10)])
    assert load_or_create_splits(other, "2025-26", path=path) == first  # frozen: later data cannot move the test period
    assert load_or_create_splits(_obs([("2026-05-01", 4)], st="Playoffs"), "2025-26", path=path, persist=False)["Regular Season"] == first["Regular Season"]
    assert "Playoffs" not in open(path).read()  # persist=False (partial window) wrote nothing


def _paired(a_proj, b_proj, actual, dates):
    a = pd.DataFrame({"date": dates, "player_id": range(len(actual)), "market": "PTS", "actual": actual, "projection": a_proj})
    b = a.assign(projection=b_proj)
    return paired_frame(a, b)


def test_paired_bootstrap_detects_a_real_improvement_and_reports_no_change_when_identical():
    rng = np.random.default_rng(0)
    n = 600
    actual = rng.normal(20, 5, n)
    dates = ["2026-01-%02d" % (1 + i % 30) for i in range(n)]
    same = paired_bootstrap(_paired(actual + 2, actual + 2, actual, dates), reps=400)
    assert same["abs_bias"]["delta"] == 0 and not same["abs_bias"]["significant"]
    better = paired_bootstrap(_paired(actual + 3, actual + 0.5, actual, dates), reps=400)
    assert better["abs_bias"]["delta"] < 0 and better["abs_bias"]["significant"]  # negative = second is better
    assert better["abs_bias"]["ci95"][1] < 0
    assert better["n_dates"] == 30


def test_paired_frame_requires_one_to_one_keys():
    a = pd.DataFrame({"date": ["d", "d"], "player_id": [1, 1], "market": ["PTS", "PTS"], "actual": [1.0, 1.0], "projection": [1.0, 1.0]})
    with pytest.raises(Exception):
        paired_frame(a, a)
