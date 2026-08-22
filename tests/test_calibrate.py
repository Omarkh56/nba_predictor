"""
Tests for pure/deterministic functions in calibrate.py.
No network access, no bet_log.csv required.
"""
import math
import pytest

# Import just the pure functions — calibrate.py doesn't hit the network at
# import time, so a direct import is safe.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from calibrate import (
    _laplace_smooth, _shrink_toward_prior,
    _calibrate_market_direction, _calibrate_margins, _calibrate_player_market,
    LAPLACE_K, SHRINKAGE_N, MIN_BETS_TO_ADJ,
)


# ===========================================================================
# _laplace_smooth
# ===========================================================================
class TestLaplaceSmooth:
    def test_zero_data_returns_prior(self):
        # With n=0, result is purely the prior (k pseudo-counts on each side)
        prior  = 0.50
        result = _laplace_smooth(hits=0, n=0, prior=prior, k=LAPLACE_K)
        # formula: (0 + k*prior) / (0 + k) = prior
        assert result == pytest.approx(prior, abs=1e-6)

    def test_many_hits_approaches_sample_rate(self):
        # With large n, Laplace smoothing should barely shift the empirical rate
        result = _laplace_smooth(hits=900, n=1000, prior=0.5, k=LAPLACE_K)
        assert result == pytest.approx(0.90, abs=0.01)

    def test_result_in_unit_interval(self):
        for hits, n, prior in [(0,0,0.5), (5,10,0.6), (100,100,0.4)]:
            r = _laplace_smooth(hits, n, prior, k=LAPLACE_K)
            assert 0.0 <= r <= 1.0, f"out of range: hits={hits} n={n} r={r}"

    def test_higher_k_pulls_harder_to_prior(self):
        r_low_k  = _laplace_smooth(hits=7, n=10, prior=0.5, k=1)
        r_high_k = _laplace_smooth(hits=7, n=10, prior=0.5, k=10)
        # High k should pull result closer to 0.5
        assert abs(r_high_k - 0.5) < abs(r_low_k - 0.5)

    def test_extreme_prior(self):
        # Prior=0.0 with no data → result should be 0
        assert _laplace_smooth(0, 0, 0.0, k=1.0) == pytest.approx(0.0)
        # Prior=1.0 with no data → result should be 1
        assert _laplace_smooth(0, 0, 1.0, k=1.0) == pytest.approx(1.0)


# ===========================================================================
# _shrink_toward_prior
# ===========================================================================
class TestShrinkTowardPrior:
    def test_zero_samples_returns_prior(self):
        result = _shrink_toward_prior(sample_rate=0.80, n=0, prior=0.50)
        assert result == pytest.approx(0.50, abs=1e-6)

    def test_large_n_approaches_sample_rate(self):
        # n >> SHRINKAGE_N → trust the sample nearly completely
        result = _shrink_toward_prior(sample_rate=0.80, n=10_000, prior=0.50)
        assert result == pytest.approx(0.80, abs=0.005)

    def test_half_trust_at_shrinkage_n(self):
        # At n == SHRINKAGE_N, weight = 0.5 exactly
        result = _shrink_toward_prior(0.80, n=SHRINKAGE_N, prior=0.50)
        expected = 0.5 * 0.80 + 0.5 * 0.50
        assert result == pytest.approx(expected, abs=1e-6)

    def test_weight_monotone_in_n(self):
        prior = 0.50
        sample = 0.70
        results = [_shrink_toward_prior(sample, n, prior) for n in [0, 5, 12, 50, 500]]
        # As n increases, result should move monotonically toward the sample rate
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]

    def test_result_between_prior_and_sample(self):
        for sr, pr in [(0.70, 0.50), (0.30, 0.50), (0.60, 0.40)]:
            r = _shrink_toward_prior(sr, n=20, prior=pr)
            lo, hi = min(sr, pr), max(sr, pr)
            assert lo <= r <= hi, f"sr={sr} pr={pr} r={r}"

    def test_custom_shrink_n(self):
        # Custom shrink_n changes where the 50/50 crossover happens
        r = _shrink_toward_prior(0.80, n=5, prior=0.50, shrink_n=5)
        assert r == pytest.approx(0.65, abs=1e-6)  # 0.5*0.80 + 0.5*0.50


# ===========================================================================
# _calibrate_market_direction (new helper extracted from calibrate())
# ===========================================================================
def _make_graded(rows):
    """Build minimal graded DataFrame as calibrate() expects."""
    df = pd.DataFrame(rows, columns=["result", "market", "pick", "hit"])
    df["hit"] = (df["result"] == "HIT").astype(int)
    return df


class TestCalibrateMarketDirection:
    def test_empty_returns_empty(self):
        graded = _make_graded([])
        result = _calibrate_market_direction(graded, prior=0.50)
        assert result == {}

    def test_single_group_keys(self):
        rows = [("HIT","PTS","OVER",1), ("HIT","PTS","OVER",1), ("MISS","PTS","OVER",0)]
        graded = _make_graded(rows)
        result = _calibrate_market_direction(graded, prior=0.50)
        assert "PTS_OVER" in result
        entry = result["PTS_OVER"]
        assert entry["n"] == 3
        assert entry["hits"] == 2
        assert 0 < entry["rate"] < 1
        assert "conf_adj" in entry

    def test_insufficient_data_zeroes_conf_adj(self):
        # n < MIN_BETS_TO_ADJ → conf_adj must be 0.0
        rows = [("HIT","REB","OVER",1)] * (MIN_BETS_TO_ADJ - 1)
        graded = _make_graded(rows)
        result = _calibrate_market_direction(graded, prior=0.50)
        assert result["REB_OVER"]["conf_adj"] == 0.0

    def test_perfect_hit_rate_positive_adj(self):
        rows = [("HIT","AST","OVER",1)] * 20
        graded = _make_graded(rows)
        result = _calibrate_market_direction(graded, prior=0.50)
        assert result["AST_OVER"]["conf_adj"] > 0

    def test_perfect_miss_rate_negative_adj(self):
        rows = [("MISS","AST","OVER",0)] * 20
        graded = _make_graded(rows)
        result = _calibrate_market_direction(graded, prior=0.50)
        assert result["AST_OVER"]["conf_adj"] < 0


class TestCalibrateMargins:
    def test_missing_margin_column_returns_empty(self):
        graded = _make_graded([("HIT","PTS","OVER",1)] * 5)
        result = _calibrate_margins(graded)
        assert result == {}

    def test_with_margin_column(self):
        rows = [("HIT","PTS","OVER",1)] * 5 + [("MISS","PTS","OVER",0)] * 5
        graded = _make_graded(rows)
        graded["margin"] = [1.5, 2.0, 0.5, 3.0, 1.0, -0.5, -1.5, -2.0, -0.3, -0.8]
        result = _calibrate_margins(graded)
        assert "PTS_OVER" in result
        entry = result["PTS_OVER"]
        assert "mean_margin" in entry
        assert "near_miss_rate" in entry
        assert 0 <= entry["near_miss_rate"] <= 1
