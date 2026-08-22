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

from calibrate import _laplace_smooth, _shrink_toward_prior, LAPLACE_K, SHRINKAGE_N


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
