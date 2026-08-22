"""
Tests for pure/deterministic functions in newnbapredictor.py.
No network access required.
"""
import math
import pytest
import sys, os, types, importlib, importlib.util, unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Safe import: stub out all network-requiring imports before loading the module
# ---------------------------------------------------------------------------
def _import_newnba():
    _stub = lambda name: types.ModuleType(name)
    stubs = {}
    for name in ["nba_api", "nba_api.stats", "nba_api.stats.endpoints",
                 "nba_api.stats.library", "nba_api.stats.library.http",
                 "nba_api.stats.static"]:
        stubs[name] = types.ModuleType(name)

    # Minimal attribute stubs
    stubs["nba_api.stats.library.http"].STATS_HEADERS = {}
    stubs["nba_api.stats.library.http"].STATS_TIMEOUT = 60

    # Stub endpoint classes
    for ep in ["leaguedashteamstats", "leaguedashteamclutch",
               "leaguegamefinder", "leaguedashplayerstats"]:
        m = types.ModuleType(f"nba_api.stats.endpoints.{ep}")
        stubs[f"nba_api.stats.endpoints.{ep}"] = m

    req_stub = types.ModuleType("requests")
    req_stub.get        = mock.Mock()
    req_stub.RequestException = Exception

    with mock.patch.dict("sys.modules", {**stubs, "requests": req_stub}):
        spec = importlib.util.spec_from_file_location(
            "newnbapredictor",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "newnbapredictor.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass
        return mod


_nb = _import_newnba()
_sigmoid = _nb._sigmoid
RTG_SCALE = _nb.RTG_SCALE


# ===========================================================================
# _sigmoid
# ===========================================================================
class TestSigmoid:
    def test_zero_input_gives_half(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_positive_input_above_half(self):
        assert _sigmoid(5.0) > 0.5

    def test_negative_input_below_half(self):
        assert _sigmoid(-5.0) < 0.5

    def test_symmetry(self):
        x = 7.3
        assert _sigmoid(x) + _sigmoid(-x) == pytest.approx(1.0, abs=1e-9)

    def test_output_range(self):
        for x in [-100, -10, -1, 0, 1, 10, 100]:
            p = _sigmoid(x)
            assert 0.0 < p < 1.0, f"sigmoid({x}) = {p}"

    def test_large_spread_near_certainty(self):
        # 30-point spread at RTG_SCALE=9.5 → very high probability
        assert _sigmoid(30.0) > 0.95

    def test_against_manual_formula(self):
        x = 9.5
        expected = 1.0 / (1.0 + math.exp(-x / RTG_SCALE))
        assert _sigmoid(x) == pytest.approx(expected, rel=1e-9)

    def test_three_point_spread(self):
        # 3-pt favourite at RTG_SCALE=9.5: 1/(1+exp(-3/9.5)) ≈ 0.578
        p = _sigmoid(3.0)
        assert 0.57 < p < 0.62

    def test_seven_point_spread(self):
        # 7-pt favourite → roughly 68-72% probability
        p = _sigmoid(7.0)
        assert 0.65 < p < 0.75
