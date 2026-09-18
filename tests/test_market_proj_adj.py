"""PTS is included in MARKET_PROJ_ADJ (no SKIP_MARKET_ADJ exclusion)."""

from playerlinepredictor import _build_market_proj_adj


def test_pts_over_hits_floor_from_observed_hit_rate():
    adj = _build_market_proj_adj({
        "PTS_OVER": {"n": 113, "raw_rate": 0.389},
        "REB_OVER": {"n": 80, "raw_rate": 0.525},
        "AST_OVER": {"n": 80, "raw_rate": 0.537},
        "3PM_OVER": {"n": 80, "raw_rate": 0.517},
    })
    assert adj["PTS"] == 0.933
    assert adj["REB"] == 0.985
    assert adj["AST"] == 0.978
    assert adj["FG3M"] == 0.99


def test_combo_and_under_keys_are_ignored():
    adj = _build_market_proj_adj({
        "PTS_OVER": {"n": 113, "raw_rate": 0.389},
        "PTS_UNDER": {"n": 36, "raw_rate": 0.472},
        "PTS+AST_OVER": {"n": 106, "raw_rate": 0.50},
        "PRA_OVER": {"n": 90, "raw_rate": 0.40},
    })
    assert set(adj) == {"PTS"}


def test_thin_samples_skipped():
    assert _build_market_proj_adj({"PTS_OVER": {"n": 19, "raw_rate": 0.2}}) == {}
