"""
Tests for pure/deterministic functions in playerlinepredictor.py:
  - normal_cdf()
  - _nb_over_prob()
  - _avg_minutes()
No network access, no NBA API calls.
"""
import math
import numpy as np
import pandas as pd
import pytest
import sys, os, types, importlib, importlib.util, unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Safe import: stub every network/API dependency
# ---------------------------------------------------------------------------
def _import_player():
    nba_stubs = {}
    for name in [
        "nba_api", "nba_api.stats", "nba_api.stats.endpoints",
        "nba_api.stats.library", "nba_api.stats.library.http",
        "nba_api.stats.static",
        "nba_api.stats.endpoints.playergamelog",
        "nba_api.stats.endpoints.leaguegamefinder",
        "nba_api.stats.endpoints.leaguedashteamstats",
        "nba_api.stats.endpoints.leaguedashplayerbiostats",
        "nba_api.stats.endpoints.commonteamroster",
        "nba_api.stats.endpoints.leaguegamelog",
    ]:
        nba_stubs[name] = types.ModuleType(name)

    nba_stubs["nba_api.stats.library.http"].STATS_HEADERS = {}
    nba_stubs["nba_api.stats.library.http"].STATS_TIMEOUT = 60
    nba_stubs["nba_api.stats.static"].players = types.SimpleNamespace(
        find_players_by_full_name=lambda *a, **k: [],
        get_players=lambda: [],
    )
    nba_stubs["nba_api.stats.static"].teams = types.SimpleNamespace(
        find_teams_by_full_name=lambda *a, **k: [],
        get_teams=lambda: [],
    )

    req_stub = types.ModuleType("requests")
    req_stub.get  = mock.Mock()
    req_stub.RequestException = Exception
    exc_stub = types.ModuleType("requests.exceptions")
    exc_stub.RequestException = Exception
    exc_stub.Timeout = Exception
    exc_stub.ConnectionError = Exception
    req_stub.exceptions = exc_stub

    dvp_stub = types.ModuleType("dvp")
    dvp_stub.get_dvp_factor = mock.Mock(return_value=1.0)

    nba_utils_stub = types.ModuleType("nba_api_utils")
    nba_utils_stub.api_call_with_retry = mock.Mock(side_effect=lambda fn, **kw: fn())
    nba_utils_stub.safe_dataframe_call = mock.Mock(return_value=None)

    with mock.patch.dict("sys.modules", {
            **nba_stubs, "requests": req_stub, "requests.exceptions": exc_stub,
            "dvp": dvp_stub, "nba_api_utils": nba_utils_stub,
        }):
        spec = importlib.util.spec_from_file_location(
            "playerlinepredictor",
            os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "playerlinepredictor.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass
        return mod


_pl = _import_player()
normal_cdf   = _pl.normal_cdf
_nb_over_prob = _pl._nb_over_prob
_avg_minutes  = _pl._avg_minutes


# ===========================================================================
# normal_cdf
# ===========================================================================
class TestNormalCdf:
    def test_zero_gives_half(self):
        assert normal_cdf(0.0) == pytest.approx(0.5)

    def test_symmetry(self):
        assert normal_cdf(1.0) + normal_cdf(-1.0) == pytest.approx(1.0, abs=1e-9)

    def test_positive_above_half(self):
        assert normal_cdf(1.96) == pytest.approx(0.975, abs=0.001)

    def test_negative_below_half(self):
        assert normal_cdf(-1.96) == pytest.approx(0.025, abs=0.001)

    def test_extreme_positive(self):
        assert normal_cdf(10.0) > 0.9999

    def test_extreme_negative(self):
        assert normal_cdf(-10.0) < 0.0001

    def test_output_in_unit_interval(self):
        for x in [-5, -2, -1, 0, 1, 2, 5]:
            p = normal_cdf(x)
            assert 0 <= p <= 1, f"normal_cdf({x}) = {p}"


# ===========================================================================
# _nb_over_prob
# ===========================================================================
class TestNbOverProb:
    # ------ guard / degenerate inputs ------

    def test_zero_proj_returns_half(self):
        assert _nb_over_prob(0, 5, 20.5) == pytest.approx(0.5)

    def test_zero_sd_returns_half(self):
        assert _nb_over_prob(20, 0, 20.5) == pytest.approx(0.5)

    def test_negative_proj_returns_half(self):
        assert _nb_over_prob(-1, 5, 20.5) == pytest.approx(0.5)

    # ------ Normal-CDF fallback (variance <= mean) ------

    def test_poisson_regime_uses_normal_fallback(self):
        # var == mean → falls back to Normal CDF
        proj, line = 20.0, 18.5
        sd = math.sqrt(proj)          # sd^2 == proj → var == mean
        p = _nb_over_prob(proj, sd, line)
        # Should be strictly > 0.5 because proj > line
        assert p > 0.5

    def test_variance_exactly_equal_mean_boundary(self):
        # When var ≈ mean, falls back to Normal CDF.
        # proj > line → result should be > 0 (direction correct).
        proj = 15.0
        sd   = math.sqrt(proj)        # var == proj (≈ floating-point boundary)
        p_over  = _nb_over_prob(proj, sd, proj - 1.0)   # proj well above line
        p_under = _nb_over_prob(proj, sd, proj + 1.0)   # proj well below line
        assert p_over > p_under

    # ------ NB regime (variance > mean) ------

    def test_nb_regime_proj_well_above_line(self):
        # proj=25, line=22.5, wide variance → high OVER probability
        p = _nb_over_prob(25, 6, 22.5)
        assert p > 0.6

    def test_nb_regime_proj_well_below_line(self):
        # proj=15, line=22.5 → very low OVER probability
        p = _nb_over_prob(15, 5, 22.5)
        assert p < 0.15

    def test_proj_equals_line_gives_roughly_half(self):
        # When proj == line, probability should be near (but not exactly) 0.5
        p = _nb_over_prob(20, 5, 20)
        assert 0.3 < p < 0.7

    def test_output_always_in_unit_interval(self):
        cases = [
            (10, 3, 9.5), (25, 8, 22.5), (5, 2, 5.5),
            (30, 10, 35), (0.5, 0.3, 1.5),  # tiny values
        ]
        for proj, sd, line in cases:
            p = _nb_over_prob(proj, sd, line)
            assert 0.0 <= p <= 1.0, f"proj={proj} sd={sd} line={line} → {p}"

    def test_monotone_in_proj(self):
        # Fixing sd and line, higher proj → higher OVER probability
        sd, line = 5.0, 20.5
        probs = [_nb_over_prob(p, sd, line) for p in [15, 18, 20, 22, 25]]
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1], f"not monotone at i={i}: {probs}"


# ===========================================================================
# _avg_minutes (EWMA weight calculation)
# ===========================================================================
def _make_logs(minutes):
    """Build a minimal DataFrame with a MIN column."""
    return pd.DataFrame({"MIN": minutes})


class TestAvgMinutes:
    def test_empty_logs_returns_zero(self):
        assert _avg_minutes(pd.DataFrame()) == 0.0

    def test_single_game(self):
        logs = _make_logs([32])
        result = _avg_minutes(logs)
        assert result == pytest.approx(32.0, abs=0.01)

    def test_equal_minutes_returns_that_value(self):
        logs = _make_logs([36, 36, 36, 36])
        assert _avg_minutes(logs) == pytest.approx(36.0, abs=0.01)

    def test_recent_game_weighted_more(self):
        # Index 0 gets the highest weight (0.5^0 = 1.0); index 1 gets 0.5^(1/hl).
        # Two symmetric series: same values, big game at opposite ends.
        # The series with the big game at index 0 should produce a higher average.
        logs_big_recent = _make_logs([40, 10])  # big game first → high avg
        logs_big_old    = _make_logs([10, 40])  # big game last  → low avg
        assert _avg_minutes(logs_big_recent) > _avg_minutes(logs_big_old)

    def test_all_zeros(self):
        logs = _make_logs([0, 0, 0])
        assert _avg_minutes(logs) == pytest.approx(0.0)

    def test_half_life_effect(self):
        # Shorter half_life → more weight on first element → result closer to it
        logs = _make_logs([40, 10, 10, 10])
        avg_hl1 = _avg_minutes(logs, half_life=1)
        avg_hl8 = _avg_minutes(logs, half_life=8)
        assert avg_hl1 > avg_hl8  # shorter HL → more weight on the 40 → higher avg

    def test_missing_min_column_returns_zero(self):
        logs = pd.DataFrame({"PTS": [20, 25]})
        assert _avg_minutes(logs) == 0.0

    def test_returns_float(self):
        logs = _make_logs([30, 32, 34])
        assert isinstance(_avg_minutes(logs), float)
