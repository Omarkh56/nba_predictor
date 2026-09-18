"""
tests/test_fit_flag_penalties.py — Unit tests for fit_flag_penalties.py.

No network access required — this module is pure pandas/sklearn over
bet_log.csv-shaped data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_proj = Path(__file__).parent.parent
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

import fit_flag_penalties as ffp


def _mk_df(rows):
    df = pd.DataFrame(rows)
    df["flags"] = df["flags"].fillna("")
    df["market"] = df["market"].astype(str)
    df["pick"] = df["pick"].astype(str).str.upper()
    df["hit"] = (df["result"] == "HIT").astype(int)
    return df


# ── coef_to_pp ───────────────────────────────────────────────────────────────

class TestCoefToPp:
    def test_zero_coef_is_zero_pp(self):
        assert ffp.coef_to_pp(0.0, 0.5) == 0.0

    def test_matches_derivative_formula(self):
        # d(sigmoid)/d(logit) at p=0.5 is 0.25 -> coef=1.0 -> 25 points
        assert ffp.coef_to_pp(1.0, 0.5) == pytest.approx(25.0)

    def test_scales_with_base_rate_variance(self):
        # p(1-p) is maximized at p=0.5; further from 0.5 -> smaller pp for the same coef
        at_half = ffp.coef_to_pp(1.0, 0.5)
        at_extreme = ffp.coef_to_pp(1.0, 0.9)
        assert at_extreme < at_half


# ── PENALTY_MAP matchers ───────────────────────────────────────────────────────

class TestPenaltyMapMatchers:
    def test_under_pick_matches_only_under(self):
        matcher = ffp.PENALTY_MAP["UNDER_PICK"][0]
        assert matcher("", {"pick": "UNDER", "market": "PTS"}) is True
        assert matcher("", {"pick": "OVER", "market": "PTS"}) is False

    def test_pts_reb_over_requires_both_market_and_pick(self):
        matcher = ffp.PENALTY_MAP["PTS_REB_OVER"][0]
        assert matcher("", {"pick": "OVER", "market": "PTS+REB"}) is True
        assert matcher("", {"pick": "UNDER", "market": "PTS+REB"}) is False
        assert matcher("", {"pick": "OVER", "market": "PTS"}) is False

    def test_suspicious_edge_prefix_match_ignores_numeric_suffix(self):
        matcher = ffp.PENALTY_MAP["SUSPICIOUS_EDGE"][0]
        assert matcher("⚠SUSP-45% NO-PO", {}) is True
        assert matcher("⚠SUSP-100%", {}) is True
        assert matcher("NO-PO 1-BOOK", {}) is False

    def test_one_book_does_not_match_two_book(self):
        one_book = ffp.PENALTY_MAP["ONE_BOOK"][0]
        two_book = ffp.PENALTY_MAP["TWO_BOOK"][0]
        assert one_book("1-BOOK", {}) is True
        assert one_book("2-BOOK", {}) is False
        assert two_book("2-BOOK", {}) is True
        assert two_book("1-BOOK", {}) is False

    def test_trend_down_over_requires_pick_direction(self):
        matcher = ffp.PENALTY_MAP["TREND_DOWN_OVER"][0]
        assert matcher("TREND↓", {"pick": "OVER"}) is True
        assert matcher("TREND↓", {"pick": "UNDER"}) is False
        assert matcher("TREND↑", {"pick": "OVER"}) is False

    def test_dvp_prefix_matchers_are_disjoint(self):
        high = ffp.PENALTY_MAP["DVP_HIGH"][0]
        low = ffp.PENALTY_MAP["DVP_LOW"][0]
        assert high("DvP+1.15", {}) is True
        assert low("DvP+1.15", {}) is False
        assert low("DvP-0.78", {}) is True
        assert high("DvP-0.78", {}) is False


# ── build_features ──────────────────────────────────────────────────────────

class TestBuildFeatures:
    def test_flag_columns_match_matchers(self):
        df = _mk_df([
            {"market": "PTS", "pick": "OVER", "result": "HIT", "flags": "1-BOOK"},
            {"market": "PTS", "pick": "UNDER", "result": "MISS", "flags": ""},
        ])
        X, names = ffp.build_features(df)
        assert "ONE_BOOK" in names
        assert "UNDER_PICK" in names
        assert list(X["ONE_BOOK"]) == [1, 0]
        assert list(X["UNDER_PICK"]) == [0, 1]

    def test_small_markets_pooled_into_other(self):
        rows = [{"market": "PTS", "pick": "OVER", "result": "HIT", "flags": ""}] * 25
        rows += [{"market": "RARE", "pick": "OVER", "result": "HIT", "flags": ""}] * 3
        df = _mk_df(rows)
        X, names = ffp.build_features(df)
        assert "mkt_RARE" not in names
        # PTS is the reference level (alphabetically first / drop_first), so it
        # may not appear as its own dummy either -- just confirm RARE got pooled
        # and no column blew up to one-per-row.
        assert all(not n.startswith("mkt_RARE") for n in names)

    def test_all_penalty_map_flags_are_binary(self):
        rows = [
            {"market": "PTS", "pick": "OVER", "result": "HIT",
             "flags": "1-BOOK ⚠BENCH NO-PO LOW-LINE TREND↓ ★STRONG DvP+1.20"},
            {"market": "REB", "pick": "UNDER", "result": "MISS", "flags": ""},
        ]
        df = _mk_df(rows)
        X, names = ffp.build_features(df)
        flag_cols = [n for n in names if n in ffp.PENALTY_MAP]
        assert set(X[flag_cols].values.flatten()).issubset({0, 1})


# ── End-to-end: fit runs and produces sane output on synthetic data ──────────

class TestFitEndToEnd:
    def test_flag_with_planted_signal_is_detected(self):
        # Plant an obvious effect: UNDER_PICK strongly predicts a miss.
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(400):
            pick = "UNDER" if rng.random() < 0.5 else "OVER"
            p_hit = 0.25 if pick == "UNDER" else 0.65
            result = "HIT" if rng.random() < p_hit else "MISS"
            rows.append({"market": "PTS", "pick": pick, "result": result, "flags": ""})
        df = _mk_df(rows)
        X, names = ffp.build_features(df)
        y = df["hit"].values
        clf = ffp.fit_logistic(X, y)
        coefs = dict(zip(names, clf.coef_[0]))
        base_rate = float(y.mean())
        fitted_pp = ffp.coef_to_pp(coefs["UNDER_PICK"], base_rate)
        # Planted effect is large and unambiguous -- should recover a clearly
        # negative, double-digit-point penalty, not the current -2.0pp constant.
        assert fitted_pp < -10.0
