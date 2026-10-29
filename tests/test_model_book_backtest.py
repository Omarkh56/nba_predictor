"""The model-book quotes coherent, two-sided no-vig prices."""

import numpy as np
import pytest

from scripts.replay.model_book_backtest import fair_odds, reliability, score


def test_fair_odds_round_trip_and_zero_overround():
    probabilities = np.array([0.2, 0.5, 0.8])
    decimal, american = fair_odds(probabilities)

    assert decimal == pytest.approx([5.0, 2.0, 1.25])
    assert american == pytest.approx([400.0, -100.0, -400.0])
    assert 1 / decimal + 1 / fair_odds(1 - probabilities)[0] == pytest.approx(
        np.ones(3)
    )


def test_fair_odds_clip_impossible_prices_to_finite_values():
    decimal, american = fair_odds([0.0, 1.0])

    assert np.isfinite(decimal).all()
    assert np.isfinite(american).all()
    assert (decimal > 1).all()


def test_scoring_and_reliability_use_the_quoted_side_probability():
    metrics = score([0.8, 0.2], [1, 0])
    bands = reliability([0.8, 0.2], [1, 0])

    assert metrics["n"] == 2
    assert metrics["brier"] == pytest.approx(0.04)
    assert metrics["favorite_accuracy"] == 1.0
    assert sum(band["n"] for band in bands) == 2
