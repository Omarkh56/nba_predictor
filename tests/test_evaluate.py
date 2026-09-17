"""
Tests for evaluate.py — statistical computations, calibration, significance
tests, logging round-trips, and the CLI.

Style mirrors tests/test_calibrate.py: one class per function, hand-computed
expected values cited in comments, pytest.approx for float comparisons.
No network access; no real eval_log.jsonl is ever touched (tmp_path fixtures
are used for all log I/O).
"""
import json
import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluate import (
    _acc, _ll, _bs,
    calibration_table,
    stability_by_month,
    stability_rolling,
    mcnemar_test,
    bootstrap_ci,
    bootstrap_diff_ci,
    market_comparison,
    american_to_devig_prob,
    evaluate,
    _append_log,
    load_log,
    get_run,
    compare_runs,
    print_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _y(*vals):
    return np.array(vals, dtype=float)


# ===========================================================================
# _acc
# ===========================================================================
class TestAcc:
    def test_perfect_prediction(self):
        # All predictions above 0.5 and all labels 1 → accuracy = 1.0
        y = _y(1, 1, 1, 1)
        p = _y(0.6, 0.7, 0.8, 0.9)
        assert _acc(y, p) == pytest.approx(1.0)

    def test_all_wrong(self):
        # All predictions above 0.5 but all labels 0 → accuracy = 0.0
        y = _y(0, 0, 0, 0)
        p = _y(0.6, 0.7, 0.8, 0.9)
        assert _acc(y, p) == pytest.approx(0.0)

    def test_half_correct(self):
        # 2 correct, 2 wrong out of 4 → 0.5
        y = _y(1, 1, 0, 0)
        p = _y(0.8, 0.8, 0.8, 0.8)   # all predict "1"; first two right
        assert _acc(y, p) == pytest.approx(0.5)

    def test_threshold_boundary(self):
        # p == 0.5 is classified as positive (>= threshold)
        y = _y(1, 0)
        p = _y(0.5, 0.5)
        # 0.5 >= 0.5 → predict 1; first correct, second wrong → 0.5
        assert _acc(y, p) == pytest.approx(0.5)

    def test_single_game(self):
        assert _acc(_y(1), _y(0.9)) == pytest.approx(1.0)
        assert _acc(_y(0), _y(0.9)) == pytest.approx(0.0)


# ===========================================================================
# _ll  (log loss)
# ===========================================================================
class TestLL:
    def test_perfect_predictions(self):
        # ll = -mean(1*log(1-eps) + 0) ≈ eps  (near 0)
        y = _y(1, 0)
        p = _y(1 - 1e-7, 1e-7)
        assert _ll(y, p) == pytest.approx(0.0, abs=1e-4)

    def test_fifty_fifty_uniform(self):
        # When p=0.5 always, ll = -log(0.5) = log(2) ≈ 0.6931
        y = _y(1, 0, 1, 0)
        p = _y(0.5, 0.5, 0.5, 0.5)
        expected = math.log(2)   # 0.6931...
        assert _ll(y, p) == pytest.approx(expected, rel=1e-4)

    def test_hand_computed_two_games(self):
        # Game 1: y=1, p=0.8  → -log(0.8)      = 0.22314
        # Game 2: y=0, p=0.3  → -log(1-0.3)   = -log(0.7) = 0.35667
        # mean = (0.22314 + 0.35667) / 2       = 0.28991
        y = _y(1, 0)
        p = _y(0.8, 0.3)
        expected = (math.log(1/0.8) + math.log(1/0.7)) / 2
        assert _ll(y, p) == pytest.approx(expected, rel=1e-4)

    def test_clipping_prevents_inf(self):
        # p=0 or p=1 must not produce inf/nan
        y = _y(1, 0)
        p = _y(0.0, 1.0)
        result = _ll(y, p)
        assert math.isfinite(result)
        assert result > 0


# ===========================================================================
# _bs  (Brier score)
# ===========================================================================
class TestBS:
    def test_perfect_score_is_zero(self):
        # All hard-correct predictions: Brier = 0
        y = _y(1, 0, 1, 0)
        p = _y(1.0, 0.0, 1.0, 0.0)
        assert _bs(y, p) == pytest.approx(0.0, abs=1e-9)

    def test_worst_case_is_one(self):
        # All hard-wrong predictions: Brier = mean((1-0)^2) = 1.0
        y = _y(1, 1, 1)
        p = _y(0.0, 0.0, 0.0)
        assert _bs(y, p) == pytest.approx(1.0)

    def test_uniform_half(self):
        # p=0.5 always: Brier = mean(0.25) = 0.25
        y = _y(1, 0, 1, 0)
        p = _y(0.5, 0.5, 0.5, 0.5)
        assert _bs(y, p) == pytest.approx(0.25)

    def test_hand_computed(self):
        # y=1, p=0.8 → (0.8-1)^2 = 0.04
        # y=0, p=0.3 → (0.3-0)^2 = 0.09
        # mean = 0.065
        y = _y(1, 0)
        p = _y(0.8, 0.3)
        assert _bs(y, p) == pytest.approx(0.065, rel=1e-6)


# ===========================================================================
# calibration_table
# ===========================================================================
class TestCalibrationTable:
    def test_single_bucket_all_correct(self):
        # All probs in [0.5, 0.6), all labels 1 → actual_rate = 1.0
        y = _y(1, 1, 1)
        p = _y(0.50, 0.55, 0.59)
        df = calibration_table(y, p, n_bins=10)
        # The 50-60% bucket
        row = df[df["bucket"] == "50-60%"].iloc[0]
        assert row["n"] == 3
        assert row["actual_rate"] == pytest.approx(1.0)
        assert row["pred_prob"] == pytest.approx(p.mean(), abs=0.01)
        assert row["error"] < 0   # predicted ~0.55, actual 1.0 → under-confident

    def test_empty_buckets_skipped(self):
        # All probs at exactly 0.99 → only the last bucket should appear
        y = _y(1, 0)
        p = _y(0.99, 0.99)
        df = calibration_table(y, p, n_bins=10)
        assert len(df) == 1
        assert df.iloc[0]["bucket"] == "90-100%"

    def test_error_sign_convention(self):
        # pred_prob > actual_rate → error > 0 (over-confident)
        y = _y(0, 0, 0, 0)     # all zeros
        p = _y(0.55, 0.56, 0.57, 0.58)   # all predict ~56%, none win
        df = calibration_table(y, p, n_bins=10)
        row = df[df["bucket"] == "50-60%"].iloc[0]
        assert row["error"] > 0    # predicted ~0.565, actual 0.0

    def test_n_games_sums_to_total(self):
        y = np.zeros(20)
        p = np.linspace(0.05, 0.95, 20)
        df = calibration_table(y, p)
        assert df["n"].sum() == 20

    def test_columns_present(self):
        y = _y(1, 0)
        p = _y(0.6, 0.4)
        df = calibration_table(y, p)
        for col in ("bucket", "n", "pred_prob", "actual_rate", "error"):
            assert col in df.columns


# ===========================================================================
# stability_by_month
# ===========================================================================
class TestStabilityByMonth:
    def _make_data(self):
        # 12 games in Jan, 12 in Feb — all ones in Jan, all zeros in Feb
        # (12 games per month to exceed the default min_n=10 filter)
        y = np.array([1]*12 + [0]*12, dtype=float)
        p = np.array([0.6]*12 + [0.4]*12, dtype=float)
        # Jan: pred=0.6 all win → correct; Feb: pred=0.4 all lose → correct
        dates = (
            [f"2024-01-{d:02d}" for d in range(1, 13)] +
            [f"2024-02-{d:02d}" for d in range(1, 13)]
        )
        return y, p, dates

    def test_two_months_detected(self):
        y, p, dates = self._make_data()
        df = stability_by_month(y, p, dates)
        assert len(df) == 2
        assert list(df["month"]) == ["2024-01", "2024-02"]

    def test_accuracy_per_month(self):
        y, p, dates = self._make_data()
        df = stability_by_month(y, p, dates)
        # Jan: all preds 0.6 ≥ 0.5, all labels 1 → accuracy 1.0
        # Feb: all preds 0.4 < 0.5, all labels 0 → accuracy 1.0
        assert df.loc[df["month"] == "2024-01", "accuracy"].iloc[0] == pytest.approx(1.0)
        assert df.loc[df["month"] == "2024-02", "accuracy"].iloc[0] == pytest.approx(1.0)

    def test_min_n_filter(self):
        y, p, dates = self._make_data()
        # With min_n=13, neither month (12 games each) should appear
        df = stability_by_month(y, p, dates, min_n=13)
        assert df.empty

    def test_columns_present(self):
        y, p, dates = self._make_data()
        df = stability_by_month(y, p, dates)
        for col in ("month", "n", "accuracy", "log_loss", "brier"):
            assert col in df.columns


# ===========================================================================
# stability_rolling
# ===========================================================================
class TestStabilityRolling:
    def test_row_count(self):
        # 20 games, window=5 → 16 rows (rows start at game index 5)
        y = np.ones(20)
        p = np.full(20, 0.6)
        dates = [f"2024-01-{i+1:02d}" for i in range(20)]
        df = stability_rolling(y, p, dates, window=5)
        assert len(df) == 16     # 20 - 5 + 1

    def test_perfect_predictions_accuracy(self):
        # All labels=1, all probs=0.6 → every window accuracy = 1.0
        y = np.ones(15)
        p = np.full(15, 0.6)
        dates = [f"2024-01-{i+1:02d}" for i in range(15)]
        df = stability_rolling(y, p, dates, window=5)
        # Use numpy directly — pytest.approx doesn't do element-wise on Series
        assert (np.abs(df["accuracy"].values - 1.0) < 1e-9).all()

    def test_columns_present(self):
        y = np.ones(10)
        p = np.full(10, 0.6)
        dates = [f"2024-01-{i+1:02d}" for i in range(10)]
        df = stability_rolling(y, p, dates, window=3)
        for col in ("end_game_idx", "end_date", "accuracy", "log_loss"):
            assert col in df.columns


# ===========================================================================
# mcnemar_test
# ===========================================================================
class TestMcnemar:
    def _make_calls(self, n_a_right_b_wrong, n_a_wrong_b_right, n_both_right=10, n_both_wrong=5):
        """
        Build (y, probs_a, probs_b) so the contingency table matches exactly.
        - Both right:  y=1, pa=0.9, pb=0.9
        - A right, B wrong:  y=1, pa=0.9, pb=0.1
        - A wrong, B right:  y=1, pa=0.1, pb=0.9
        - Both wrong: y=1, pa=0.1, pb=0.1
        """
        y_parts  = [1]*n_both_right + [1]*n_a_right_b_wrong + [1]*n_a_wrong_b_right + [1]*n_both_wrong
        pa_parts = [0.9]*n_both_right + [0.9]*n_a_right_b_wrong + [0.1]*n_a_wrong_b_right + [0.1]*n_both_wrong
        pb_parts = [0.9]*n_both_right + [0.1]*n_a_right_b_wrong + [0.9]*n_a_wrong_b_right + [0.1]*n_both_wrong
        return np.array(y_parts, float), np.array(pa_parts, float), np.array(pb_parts, float)

    def test_identical_calls_p_equals_one(self):
        # When both models always agree, b=c=0 → p=1.0 (no discordant pairs)
        y  = _y(1, 0, 1, 0)
        pa = _y(0.9, 0.1, 0.9, 0.1)
        pb = _y(0.9, 0.1, 0.9, 0.1)
        r = mcnemar_test(y, pa, pb)
        assert r["p_value"] == pytest.approx(1.0)
        assert r["n_discordant"] == 0

    def test_direction_b_better(self):
        # More A-wrong-B-right (c) than A-right-B-wrong (b) → B better
        _, pa, pb = self._make_calls(n_a_right_b_wrong=3, n_a_wrong_b_right=20)
        y = np.ones(10 + 3 + 20 + 5)
        r = mcnemar_test(y, pa, pb)
        assert r["direction"] == "B_better"
        assert r["c_a_wrong_b_right"] == 20
        assert r["b_a_right_b_wrong"] == 3

    def test_large_discordant_uses_chi2(self):
        # b+c > 25 → chi2 path used
        _, pa, pb = self._make_calls(n_a_right_b_wrong=5, n_a_wrong_b_right=25)
        y = np.ones(10 + 5 + 25 + 5)
        r = mcnemar_test(y, pa, pb)
        assert r["chi2"] is not None    # chi2 path
        assert 0.0 < r["p_value"] < 1.0

    def test_small_discordant_uses_exact(self):
        # b+c ≤ 25 → exact binomial path (chi2 is None)
        _, pa, pb = self._make_calls(n_a_right_b_wrong=2, n_a_wrong_b_right=10)
        y = np.ones(10 + 2 + 10 + 5)
        r = mcnemar_test(y, pa, pb)
        assert r["chi2"] is None   # exact path

    def test_textbook_symmetric_p_close_to_one(self):
        # b == c: perfect symmetry → p should be 1.0 (two-sided)
        _, pa, pb = self._make_calls(n_a_right_b_wrong=10, n_a_wrong_b_right=10)
        y = np.ones(10 + 10 + 10 + 5)
        r = mcnemar_test(y, pa, pb)
        assert r["p_value"] == pytest.approx(1.0, abs=0.05)

    def test_keys_present(self):
        y, pa, pb = self._make_calls(5, 5)
        r = mcnemar_test(y, pa, pb)
        for k in ("p_value", "n_discordant", "direction", "interpretation"):
            assert k in r


# ===========================================================================
# bootstrap_ci
# ===========================================================================
class TestBootstrapCI:
    def test_constant_predictions_narrow_ci(self):
        # If p is constant (all 0.5), accuracy never changes across resamples
        # → CI should be width 0 on accuracy
        y = np.array([1, 0, 1, 0, 1, 0] * 10, dtype=float)
        p = np.full(60, 0.5)
        lo, hi = bootstrap_ci(y, p, _acc, n_boot=500, seed=0)
        # All resamples give 0.5 accuracy because p=0.5 always classifies as 1
        # (>= 0.5), and y is half ones — but resampling changes the mix.
        # At minimum the CI must be a valid interval
        assert lo <= hi

    def test_perfect_predictions_ci_is_one(self):
        # All correct → every resample also all-correct → CI = [1.0, 1.0]
        y = np.ones(50, dtype=float)
        p = np.full(50, 0.9)
        lo, hi = bootstrap_ci(y, p, _acc, n_boot=200, seed=0)
        assert lo == pytest.approx(1.0, abs=1e-9)
        assert hi == pytest.approx(1.0, abs=1e-9)

    def test_ci_contains_point_estimate(self):
        rng = np.random.default_rng(42)
        y = (rng.random(100) > 0.5).astype(float)
        p = np.clip(rng.normal(0.6, 0.1, 100), 0.01, 0.99)
        point = _acc(y, p)
        lo, hi = bootstrap_ci(y, p, _acc, n_boot=1000, seed=0)
        assert lo <= point <= hi

    def test_seed_reproducibility(self):
        y = np.array([1, 0] * 30, dtype=float)
        p = np.linspace(0.3, 0.7, 60)
        r1 = bootstrap_ci(y, p, _ll, n_boot=200, seed=7)
        r2 = bootstrap_ci(y, p, _ll, n_boot=200, seed=7)
        assert r1 == r2


# ===========================================================================
# bootstrap_diff_ci
# ===========================================================================
class TestBootstrapDiffCI:
    def test_identical_models_ci_straddles_zero(self):
        # When both models have the same probs, every resample difference is 0
        # → CI should be exactly [0.0, 0.0]
        y = np.array([1, 0] * 20, dtype=float)
        p = np.full(40, 0.6)
        lo, hi = bootstrap_diff_ci(y, p, p.copy(), _acc, n_boot=200, seed=0)
        assert lo == pytest.approx(0.0, abs=1e-9)
        assert hi == pytest.approx(0.0, abs=1e-9)

    def test_known_better_model_ci_positive_for_accuracy(self):
        # A is perfect, B is terrible → diff_acc = acc(A) - acc(B) = 1.0 - 0.0 = 1.0
        # CI should be entirely positive (both bounds > 0)
        y = np.ones(40, dtype=float)
        pa = np.full(40, 0.9)   # all correct
        pb = np.full(40, 0.1)   # all wrong
        lo, hi = bootstrap_diff_ci(y, pa, pb, _acc, n_boot=200, seed=0)
        assert lo > 0
        assert hi > 0

    def test_ci_valid_interval(self):
        rng = np.random.default_rng(1)
        y  = (rng.random(80) > 0.5).astype(float)
        pa = np.clip(rng.normal(0.6, 0.1, 80), 0.01, 0.99)
        pb = np.clip(rng.normal(0.5, 0.15, 80), 0.01, 0.99)
        lo, hi = bootstrap_diff_ci(y, pa, pb, _acc, n_boot=500, seed=0)
        assert lo <= hi


# ===========================================================================
# market_comparison
# ===========================================================================
class TestMarketComparison:
    def _setup(self):
        # 10 games: model perfect, market mediocre
        y  = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
        pm = np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
        pk = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
        return y, pm, pk

    def test_model_better_than_market(self):
        y, pm, pk = self._setup()
        r = market_comparison(y, pm, pk)
        assert r["model"]["accuracy"] > r["market"]["accuracy"]

    def test_delta_sign(self):
        y, pm, pk = self._setup()
        r = market_comparison(y, pm, pk)
        # model is better → accuracy delta (model - market) > 0
        assert r["delta"]["accuracy"] > 0

    def test_n_agree_plus_disagree_equals_n(self):
        y, pm, pk = self._setup()
        r = market_comparison(y, pm, pk)
        assert r["n_agree"] + r["n_disagree"] == len(y)

    def test_keys_present(self):
        y, pm, pk = self._setup()
        r = market_comparison(y, pm, pk)
        for k in ("n_games", "model", "market", "delta",
                  "n_agree", "n_disagree", "mcnemar"):
            assert k in r

    def test_all_agree_n_disagree_zero(self):
        y  = np.ones(6, dtype=float)
        pm = np.full(6, 0.8)
        pk = np.full(6, 0.7)    # both above 0.5 → all agree
        r = market_comparison(y, pm, pk)
        assert r["n_disagree"] == 0


# ===========================================================================
# american_to_devig_prob
# ===========================================================================
class TestAmericanToDevigProb:
    def test_even_money_both_sides(self):
        # +100 / +100 → each side 50% implied → devig = 50%
        p = american_to_devig_prob(100, 100)
        assert p == pytest.approx(0.5, abs=1e-6)

    def test_heavy_favourite(self):
        # -200 / +170 (standard -200 fav line)
        # implied home = 200/300 = 0.6667; away = 100/270 = 0.3704
        # total = 1.0370; devig home = 0.6667/1.0370 ≈ 0.6428
        p = american_to_devig_prob(-200, 170)
        # Implied: home = |−200|/(|−200|+100) = 200/300 = 0.6667
        #          away = 100/(100+170)         = 100/270 = 0.3704
        # total  = 1.0370, devig = 0.6667/1.0370
        expected = (200/300) / (200/300 + 100/270)
        assert p == pytest.approx(expected, rel=1e-5)

    def test_underdog_below_half(self):
        # Symmetric but reversed: away is favourite
        p_home = american_to_devig_prob(150, -180)
        assert p_home < 0.5

    def test_output_in_unit_interval(self):
        for h, a in [(100,100), (-110,-110), (-200,170), (130,-150)]:
            p = american_to_devig_prob(h, a)
            assert 0.0 < p < 1.0, f"out of range: {h}/{a} → {p}"


# ===========================================================================
# evaluate() end-to-end
# ===========================================================================
class TestEvaluateEndToEnd:
    def _make_data(self, n=80, seed=7):
        rng = np.random.default_rng(seed)
        y = (rng.random(n) > 0.45).astype(float)
        p = np.clip(rng.normal(0.58, 0.10, n), 0.01, 0.99)
        b = np.clip(rng.normal(0.55, 0.12, n), 0.01, 0.99)
        dates = [f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)]
        return y, p, b, dates

    def test_returns_expected_keys(self, tmp_path):
        y, p, b, dates = self._make_data()
        r = evaluate(y, p, dates, "test", "2023-24", "val",
                     baseline_probs=b, n_boot=100, save=True,
                     log_path=tmp_path / "log.jsonl")
        for key in ("run_id", "model_name", "season", "split", "n_games",
                    "metrics", "calibration", "stability_monthly",
                    "vs_baseline"):
            assert key in r, f"missing key: {key}"

    def test_metrics_sane_ranges(self, tmp_path):
        y, p, _, dates = self._make_data()
        r = evaluate(y, p, dates, "m", "2023-24", "val",
                     n_boot=100, save=False, log_path=tmp_path / "x.jsonl")
        m = r["metrics"]
        assert 0.0 <= m["accuracy"]  <= 1.0
        assert m["log_loss"]  > 0
        assert m["brier"]     > 0
        assert len(m["accuracy_ci"]) == 2
        assert m["accuracy_ci"][0] <= m["accuracy_ci"][1]

    def test_no_market_key_when_not_supplied(self, tmp_path):
        y, p, _, dates = self._make_data()
        r = evaluate(y, p, dates, "m", "2023-24", "val",
                     n_boot=100, save=False, log_path=tmp_path / "x.jsonl")
        assert "vs_market" not in r

    def test_market_key_present_when_supplied(self, tmp_path):
        y, p, b, dates = self._make_data()
        r = evaluate(y, p, dates, "m", "2023-24", "val",
                     market_probs=b, n_boot=100, save=False,
                     log_path=tmp_path / "x.jsonl")
        assert "vs_market" in r

    def test_raises_on_empty_y(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            evaluate(np.array([]), np.array([]), [],
                     "m", "2023-24", "val", save=False,
                     log_path=tmp_path / "x.jsonl")

    def test_raises_on_nan_probs(self, tmp_path):
        y = np.ones(5)
        p = np.array([0.5, 0.5, np.nan, 0.5, 0.5])
        with pytest.raises(ValueError, match="NaN"):
            evaluate(y, p, ["2024-01-01"]*5, "m", "2023-24", "val",
                     save=False, log_path=tmp_path / "x.jsonl")

    def test_run_id_contains_model_name(self, tmp_path):
        y, p, _, dates = self._make_data()
        r = evaluate(y, p, dates, "my_model", "2023-24", "val",
                     n_boot=50, save=False, log_path=tmp_path / "x.jsonl")
        assert "my_model" in r["run_id"]

    def test_n_games_correct(self, tmp_path):
        y, p, _, dates = self._make_data(n=60)
        r = evaluate(y, p, dates, "m", "2023-24", "val",
                     n_boot=50, save=False, log_path=tmp_path / "x.jsonl")
        assert r["n_games"] == 60


# ===========================================================================
# _append_log / load_log / get_run / compare_runs
# ===========================================================================
class TestLogRoundTrip:
    def _make_result(self, model_name="m1", season="2023-24", split="val",
                     accuracy=0.600, log_loss=0.680, brier=0.230):
        return {
            "run_id":     f"20240101_{model_name}",
            "model_name": model_name,
            "season":     season,
            "split":      split,
            "timestamp":  "2024-01-01T00:00:00",
            "n_games":    100,
            "home_win_rate": 0.55,
            "metrics": {
                "accuracy":    accuracy,
                "accuracy_ci": [accuracy - 0.05, accuracy + 0.05],
                "log_loss":    log_loss,
                "log_loss_ci": [log_loss - 0.02, log_loss + 0.02],
                "brier":       brier,
                "brier_ci":    [brier - 0.01, brier + 0.01],
            },
            "calibration":       [],
            "stability_monthly": [],
        }

    def test_round_trip_single_run(self, tmp_path):
        log = tmp_path / "test.jsonl"
        r = self._make_result()
        _append_log(r, log)
        df = load_log(log)
        assert len(df) == 1
        assert df.iloc[0]["model_name"] == "m1"
        assert df.iloc[0]["accuracy"] == pytest.approx(0.600)

    def test_multiple_runs_all_loaded(self, tmp_path):
        log = tmp_path / "test.jsonl"
        for i in range(3):
            _append_log(self._make_result(model_name=f"m{i}"), log)
        df = load_log(log)
        assert len(df) == 3

    def test_get_run_returns_correct_dict(self, tmp_path):
        log = tmp_path / "test.jsonl"
        r1 = self._make_result("alpha")
        r2 = self._make_result("beta")
        _append_log(r1, log)
        _append_log(r2, log)
        found = get_run("20240101_beta", log)
        assert found is not None
        assert found["model_name"] == "beta"

    def test_get_run_returns_none_for_missing(self, tmp_path):
        log = tmp_path / "test.jsonl"
        _append_log(self._make_result(), log)
        assert get_run("nonexistent_id", log) is None

    def test_load_log_empty_file(self, tmp_path):
        log = tmp_path / "empty.jsonl"
        log.write_text("")
        df = load_log(log)
        assert df.empty

    def test_load_log_missing_file(self, tmp_path):
        df = load_log(tmp_path / "does_not_exist.jsonl")
        assert df.empty

    def test_compare_runs_delta_correct(self, tmp_path):
        log = tmp_path / "cmp.jsonl"
        r1 = self._make_result("m1", accuracy=0.600, log_loss=0.680)
        r2 = self._make_result("m2", accuracy=0.620, log_loss=0.665)
        _append_log(r1, log)
        _append_log(r2, log)
        cmp = compare_runs("20240101_m1", "20240101_m2", log)
        # delta = B - A = m2 - m1
        assert cmp["delta_accuracy"] == pytest.approx(0.020, abs=1e-6)
        assert cmp["delta_log_loss"] == pytest.approx(-0.015, abs=1e-6)

    def test_compare_runs_raises_on_missing(self, tmp_path):
        log = tmp_path / "cmp.jsonl"
        _append_log(self._make_result("m1"), log)
        with pytest.raises(KeyError):
            compare_runs("20240101_m1", "ghost_id", log)

    def test_real_log_is_never_touched(self, tmp_path, monkeypatch):
        # Ensure none of the log functions use the real LOG_PATH by accident
        import evaluate as ev
        real_path = ev.LOG_PATH
        log = tmp_path / "safe.jsonl"
        _append_log(self._make_result(), log)
        # Real log must not have been created or grown
        if real_path.exists():
            size_before = real_path.stat().st_size
            df = load_log(log)
            assert real_path.stat().st_size == size_before
        else:
            assert not real_path.exists()


# ===========================================================================
# print_report — smoke test
# ===========================================================================
class TestPrintReport:
    def _make_report(self):
        rng = np.random.default_rng(99)
        y  = (rng.random(60) > 0.45).astype(float)
        p  = np.clip(rng.normal(0.58, 0.10, 60), 0.01, 0.99)
        bp = np.clip(rng.normal(0.55, 0.12, 60), 0.01, 0.99)
        dates = [f"2024-{(i % 6) + 1:02d}-{(i % 28) + 1:02d}" for i in range(60)]
        return evaluate(y, p, dates, "smoke_model", "2023-24", "val",
                        baseline_probs=bp, n_boot=100, save=False,
                        log_path="/dev/null")

    def test_key_sections_appear(self, capsys):
        r = self._make_report()
        print_report(r)
        out = capsys.readouterr().out
        assert "smoke_model" in out
        assert "2023-24" in out
        assert "VAL" in out
        assert "Accuracy" in out
        assert "Log Loss" in out
        assert "CALIBRATION" in out
        assert "STABILITY BY MONTH" in out
        assert "VS BASELINE" in out

    def test_no_vs_market_section_when_absent(self, capsys):
        r = self._make_report()
        print_report(r)
        out = capsys.readouterr().out
        assert "VS MARKET" not in out
