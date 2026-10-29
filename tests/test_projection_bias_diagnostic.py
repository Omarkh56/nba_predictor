"""Bias diagnostics preserve held-out outcomes and train-only correction fits."""

import pandas as pd
import pytest

from scripts.replay.diagnose_projection_bias import fit_static, metrics


def test_metrics_use_projection_minus_actual_as_positive_bias():
    frame = pd.DataFrame({"projection": [12.0, 10.0], "actual": [10.0, 10.0]})

    result = metrics(frame, "projection")

    assert result["mean_residual"] == 1.0
    assert result["pct_bias"] == pytest.approx(0.1)
    assert result["mae"] == 1.0


def test_static_candidate_never_uses_target_outcomes():
    train = pd.DataFrame({
        "market": ["PTS", "PTS"], "projection": [10.0, 20.0], "actual": [8.0, 16.0],
    })
    target = pd.DataFrame({
        "market": ["PTS", "PTS"], "projection": [15.0, 30.0], "actual": [0.0, 0.0],
    })

    expected = fit_static(train, target, "multiplicative", k=1, buckets=1)
    target["actual"] = [1000.0, 2000.0]

    assert fit_static(train, target, "multiplicative", k=1, buckets=1) == pytest.approx(expected)
