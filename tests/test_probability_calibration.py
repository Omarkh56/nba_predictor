"""v9 probability calibration is coherent and strictly pre-game."""
import numpy as np
import pandas as pd
import pytest

from probability_calibration import calibrate_over
from scripts.replay.calibrate_probabilities import chronological_proxy_predictions


def test_platt_changes_over_and_under_remains_its_complement(monkeypatch):
    import playerlinepredictor as model

    fit = {"version": "v9-probability-1", "method": "platt", "markets": {
        "PTS": {"intercept": -0.7, "slope": 0.8}}}
    monkeypatch.setattr(model, "PROBABILITY_CALIBRATION", fit)
    evaluated = model.evaluate_prop(20.5, 23.0, "player_points", sd=6.0, conf_penalty=8.0)
    base = model._nb_over_prob(23.0, 6.0, 20.5)
    calibrated = calibrate_over(base, "PTS", fit)
    assert evaluated["over_prob"] == pytest.approx(round(calibrated * 100, 1))
    assert evaluated["over_prob"] + evaluated["under_prob"] == pytest.approx(100.0)
    assert evaluated["p_model"] == pytest.approx(max(calibrated, 1-calibrated))
    assert evaluated["confidence"] <= evaluated["p_model"] * 100 - 7.9
    assert calibrate_over(base, "REB", fit) == pytest.approx(base)
    assert calibrate_over(base, "PTS", {"method": "identity"}) == pytest.approx(base)


def test_weekly_expanding_fit_cannot_see_same_week_or_future_outcomes():
    # At least 100 earlier PTS observations are needed before a fit is made.
    rows = [{"date": "2025-10-20", "market": "PTS", "actual": float(i % 2),
             "proxy_line": 0.5, "p_over_uncalibrated": .9,
             "graded_status": "GRADED"} for i in range(120)]
    rows += [{"date": "2025-10-27", "market": "PTS", "actual": 1.0,
              "proxy_line": .5, "p_over_uncalibrated": .9,
              "graded_status": "GRADED"},
             {"date": "2025-10-29", "market": "PTS", "actual": 1.0,
              "proxy_line": .5, "p_over_uncalibrated": .9,
              "graded_status": "GRADED"}]
    original = chronological_proxy_predictions(pd.DataFrame(rows))
    changed = pd.DataFrame(rows)
    changed.loc[changed.date == "2025-10-29", "actual"] = 0.0
    replayed = chronological_proxy_predictions(changed)
    assert original.iloc[-2].p_over_platt < .9
    assert original.iloc[-2].p_over_platt == pytest.approx(replayed.iloc[-2].p_over_platt)
    assert np.isfinite(original.p_over_platt).all()
    assert original.iloc[-2].calibration_asof_date == "2025-10-20"
