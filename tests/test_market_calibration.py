"""
tests/test_market_calibration.py — Unit tests for market_calibration.py.

No network access required — pure pandas/numpy over bet_log.csv-shaped data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_proj = Path(__file__).parent.parent
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

import market_calibration as mc


def _mk_df(rows):
    df = pd.DataFrame(rows)
    df["market"] = df.get("market", "PTS")
    df["pick"] = df.get("pick", "OVER")
    df["date"] = df.get("date", "2026-01-01")
    df["hit"] = (df["result"] == "HIT").astype(int)
    return df


# ── bucket_of ────────────────────────────────────────────────────────────────

class TestBucketOf:
    def test_bucket_boundaries(self):
        assert mc.bucket_of(0.50) == "50-55"
        assert mc.bucket_of(0.549) == "50-55"
        assert mc.bucket_of(0.55) == "55-60"

    def test_low_and_high_extremes(self):
        assert mc.bucket_of(0.01) == "0-5"
        assert mc.bucket_of(0.99) == "95-100"


# ── shrink_gap ───────────────────────────────────────────────────────────────

class TestShrinkGap:
    def test_below_min_n_returns_zero(self):
        assert mc.shrink_gap(0.30, n=2) == 0.0

    def test_large_n_approaches_raw_gap(self):
        shrunk = mc.shrink_gap(0.10, n=10_000, k=30)
        assert shrunk == pytest.approx(0.10, abs=1e-3)

    def test_small_n_above_floor_is_heavily_shrunk(self):
        shrunk = mc.shrink_gap(0.30, n=5, k=30)
        # w = 5/35 ~ 0.143 -> shrunk gap should be far smaller than raw
        assert abs(shrunk) < abs(0.30) * 0.2


# ── reliability_table / build_bucket_adjustments ────────────────────────────

class TestReliabilityTable:
    def test_perfectly_calibrated_band_has_zero_gap(self):
        # p=0.60 for 100 bets, exactly 60 hits -> hit_rate matches avg_p
        rows = [{"result": "HIT" if i < 60 else "MISS", "p_market": 0.60} for i in range(100)]
        df = _mk_df(rows)
        table = mc.reliability_table(df, "p_market")
        row = table[table["_bucket"] == "60-65"].iloc[0]
        assert row["raw_gap"] == pytest.approx(0.0, abs=1e-6)

    def test_miscalibrated_band_shows_the_gap(self):
        # market says 80% but only 50% actually hit
        rows = [{"result": "HIT" if i < 50 else "MISS", "p_market": 0.80} for i in range(100)]
        df = _mk_df(rows)
        adj = mc.build_bucket_adjustments(df, "p_market")
        assert adj["80-85"]["raw_gap"] == pytest.approx(-0.30, abs=1e-6)
        assert adj["80-85"]["adj"] < 0  # correction should push the probability DOWN


# ── corrected_p_market ───────────────────────────────────────────────────────

class TestCorrectedPMarket:
    def test_no_bucket_data_returns_unchanged(self):
        assert mc.corrected_p_market(0.55, {}) == pytest.approx(0.55)

    def test_applies_bucket_adjustment(self):
        adj = {"55-60": {"adj": -0.10}}
        assert mc.corrected_p_market(0.57, adj) == pytest.approx(0.47)

    def test_clips_to_valid_range(self):
        adj = {"95-100": {"adj": -0.50}}
        result = mc.corrected_p_market(0.97, adj)
        assert result >= mc.P_CLIP[0]
        assert result == pytest.approx(mc.P_CLIP[0], abs=1e-6) or result < 0.97


# ── blend / p_final ──────────────────────────────────────────────────────────

class TestBlend:
    def test_more_reliable_source_dominates(self):
        # model MSE much lower than market MSE -> blend should sit close to p_model
        reliability = {
            "pooled_model_mse": 0.20,
            "pooled_market_mse": 0.20,
            "model_mse_by_market": {"PTS_OVER": {"n": 100, "mse": 0.05}},
            "market_mse_by_market": {"PTS_OVER": {"n": 100, "mse": 0.45}},
        }
        final, p_mkt_c, w_share = mc.p_final(
            p_model=0.70, p_market=0.50, market_key="PTS_OVER",
            reliability=reliability, bucket_adj_market={},
        )
        # w_model=20, w_market=2.22 (9:1) -> much closer to 0.70 than to 0.50
        assert abs(final - 0.70) < abs(final - 0.50)
        assert final > 0.65
        assert w_share < 0.15

    def test_missing_p_market_falls_back_to_p_model(self):
        reliability = {
            "pooled_model_mse": 0.20, "pooled_market_mse": 0.20,
            "model_mse_by_market": {}, "market_mse_by_market": {},
        }
        final, p_mkt_c, w_share = mc.p_final(
            p_model=0.65, p_market=None, market_key="PTS_OVER",
            reliability=reliability, bucket_adj_market={},
        )
        assert final == 0.65
        assert p_mkt_c is None
        assert w_share == 0.0

    def test_equal_reliability_is_roughly_midpoint(self):
        reliability = {
            "pooled_model_mse": 0.20, "pooled_market_mse": 0.20,
            "model_mse_by_market": {"PTS_OVER": {"n": 100, "mse": 0.20}},
            "market_mse_by_market": {"PTS_OVER": {"n": 100, "mse": 0.20}},
        }
        final, _, w_share = mc.p_final(
            p_model=0.60, p_market=0.50, market_key="PTS_OVER",
            reliability=reliability, bucket_adj_market={},
        )
        assert final == pytest.approx(0.55, abs=1e-6)
        assert w_share == pytest.approx(0.5, abs=1e-6)


# ── edge_and_ev ──────────────────────────────────────────────────────────────

class TestEdgeAndEv:
    def test_edge_uses_uncorrected_p_market(self):
        out = mc.edge_and_ev(p_final_value=0.60, p_market_raw=0.50, american_odds=-110)
        assert out["edge"] == pytest.approx(0.10, abs=1e-6)

    def test_ev_formula(self):
        # decimal odds for -110 ~= 1.909
        out = mc.edge_and_ev(p_final_value=0.55, p_market_raw=0.50, american_odds=-110)
        decimal_odds = mc.american_to_decimal(-110)  # unrounded, unlike out["decimal_odds"]
        expected_ev = 0.55 * decimal_odds - 1.0
        assert out["ev_per_1"] == pytest.approx(expected_ev, abs=1e-4)

    def test_positive_edge_can_still_be_negative_ev_at_bad_price(self):
        # small edge, but odds are so bad EV should still be negative
        out = mc.edge_and_ev(p_final_value=0.52, p_market_raw=0.50, american_odds=-500)
        assert out["ev_per_1"] < 0


# ── holdout_check ────────────────────────────────────────────────────────────

class TestHoldoutCheck:
    def test_too_few_rows_returns_not_ok(self):
        df = _mk_df([{"result": "HIT", "p_market": 0.5}] * 10)
        result = mc.holdout_check(df, "p_market")
        assert result["ok"] is False

    def test_real_signal_survives_holdout(self):
        # Plant a genuine, stable miscalibration: p_market=0.80 band always
        # hits at 50% -- same distribution across dates, so it should
        # generalize from train to test.
        rng = np.random.default_rng(0)
        rows = []
        for i in range(300):
            hit = "HIT" if rng.random() < 0.50 else "MISS"
            rows.append({
                "date": f"2026-01-{(i % 28) + 1:02d}",
                "result": hit, "p_market": 0.80,
            })
        df = _mk_df(rows).sort_values("date")
        result = mc.holdout_check(df, "p_market", test_frac=0.3)
        assert result["ok"] is True
        assert result["improved"] is True

    def test_pure_noise_bucket_does_not_reliably_improve(self):
        # Small sample, pure random noise -- the in-sample "gap" is noise and
        # should not be expected to help (or could even hurt) out-of-sample.
        # We don't assert a specific direction (that would be flaky by
        # construction) -- just that the function runs and returns a verdict.
        rng = np.random.default_rng(1)
        rows = []
        for i in range(60):
            hit = "HIT" if rng.random() < 0.50 else "MISS"
            rows.append({
                "date": f"2026-01-{(i % 28) + 1:02d}",
                "result": hit, "p_market": 0.50,
            })
        df = _mk_df(rows).sort_values("date")
        result = mc.holdout_check(df, "p_market", test_frac=0.3)
        assert result["ok"] is True
        assert "improved" in result
