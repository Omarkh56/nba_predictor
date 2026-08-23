"""
Tests for pure/deterministic functions in oddstracker.py.
No network access, no live data required.
"""
import math
import sys
import importlib
import types
import pytest


# ---------------------------------------------------------------------------
# Isolation: import only the pure functions without triggering the module-
# level requests / NBA-API calls that oddstracker.py makes at the bottom.
# We do this by temporarily stubbing requests before import.
# ---------------------------------------------------------------------------
def _import_oddstracker():
    """Import oddstracker, suppressing side-effect API calls."""
    import unittest.mock as mock
    stub_requests = types.ModuleType("requests")
    stub_requests.get  = mock.Mock(return_value=mock.Mock(json=lambda: [], raise_for_status=lambda: None))
    stub_requests.post = mock.Mock()
    stub_requests.RequestException = Exception
    exc_stub = types.ModuleType("requests.exceptions")
    exc_stub.RequestException = Exception
    exc_stub.Timeout = Exception
    exc_stub.ConnectionError = Exception
    stub_requests.exceptions = exc_stub

    stub_plp = types.ModuleType("playerlinepredictor")
    stub_plp.ODDS_API_KEY = ""
    stub_plp.BASE_URL     = "https://api.the-odds-api.com/v4"
    stub_plp.SPORT        = "basketball_nba"
    stub_plp.MARKETS      = []
    stub_plp.MARKET_LABELS = {}

    stub_nba_utils = types.ModuleType("nba_api_utils")
    stub_nba_utils.api_call_with_retry = mock.Mock(side_effect=lambda fn, **kw: fn())
    stub_nba_utils.safe_dataframe_call = mock.Mock(return_value=None)

    with mock.patch.dict("sys.modules", {
        "requests": stub_requests, "requests.exceptions": exc_stub,
        "playerlinepredictor": stub_plp, "nba_api_utils": stub_nba_utils,
    }):
        if "oddstracker" in sys.modules:
            mod = sys.modules["oddstracker"]
        else:
            spec = importlib.util.spec_from_file_location(
                "oddstracker",
                __file__.replace("tests/test_odds.py", "oddstracker.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception:
                pass  # side-effect failures are OK; we only need the pure fns
        return mod


import importlib.util
_ot = _import_oddstracker()
american_to_implied = _ot.american_to_implied
devig               = _ot.devig
devig_power         = _ot.devig_power
book_lean           = _ot.book_lean


# ===========================================================================
# american_to_implied
# ===========================================================================
class TestAmericanToImplied:
    def test_even_money(self):
        assert american_to_implied(100) == pytest.approx(0.5)
        assert american_to_implied(-100) == pytest.approx(0.5)

    def test_heavy_favourite(self):
        # -300 favourite: 300/400 = 0.75
        assert american_to_implied(-300) == pytest.approx(0.75)

    def test_heavy_underdog(self):
        # +300 underdog: 100/400 = 0.25
        assert american_to_implied(300) == pytest.approx(0.25)

    def test_result_in_unit_interval(self):
        for price in [-500, -200, -110, 100, 200, 500, 1000]:
            p = american_to_implied(price)
            assert 0.0 < p < 1.0, f"price={price} gave p={p}"

    def test_symmetry_around_110(self):
        # Standard -110 / +110 should be roughly symmetric about 0.5
        over  = american_to_implied(-110)
        under = american_to_implied(110)
        assert over + under == pytest.approx(1.0, abs=0.02)  # ~2% vig


# ===========================================================================
# devig
# ===========================================================================
class TestDevig:
    def test_fair_line_sums_to_one(self):
        # Perfectly fair -100/+100 market
        dv_o, dv_u, vig = devig(100, -100)
        assert dv_o + dv_u == pytest.approx(1.0, abs=1e-6)
        assert vig == pytest.approx(0.0, abs=0.01)

    def test_vigged_market_sums_to_one(self):
        # Standard -110/-110 juice
        dv_o, dv_u, vig = devig(-110, -110)
        assert dv_o + dv_u == pytest.approx(1.0, abs=1e-6)
        assert vig == pytest.approx(4.76, abs=0.1)

    def test_heavy_favourite(self):
        dv_o, dv_u, vig = devig(-300, 240)
        assert dv_o + dv_u == pytest.approx(1.0, abs=1e-6)
        assert dv_o > 0.5  # over (the -300 side) more likely

    def test_symmetric_returns_half_each(self):
        dv_o, dv_u, _ = devig(-110, -110)
        assert dv_o == pytest.approx(dv_u, abs=1e-4)


# ===========================================================================
# devig_power
# ===========================================================================
class TestDevigPower:
    def test_sums_to_one(self):
        dv_o, dv_u, vig, k = devig_power(-110, -110)
        assert dv_o + dv_u == pytest.approx(1.0, abs=1e-4)

    def test_k_greater_than_one_for_vigged(self):
        _, _, _, k = devig_power(-110, -110)
        assert k > 1.0

    def test_fair_line_k_near_one(self):
        # Fair market should need minimal exponent adjustment
        _, _, _, k = devig_power(100, -100)
        assert k == pytest.approx(1.0, abs=0.02)

    def test_result_range(self):
        for op, up in [(-200, 170), (-110, -110), (150, -180)]:
            dv_o, dv_u, _, _ = devig_power(op, up)
            assert 0 < dv_o < 1
            assert 0 < dv_u < 1

    def test_asymmetric_line(self):
        # Heavy favourite: de-vigged over should be >0.5
        dv_o, dv_u, _, _ = devig_power(-300, 240)
        assert dv_o > 0.5
        assert dv_o + dv_u == pytest.approx(1.0, abs=1e-4)


# ===========================================================================
# book_lean
# ===========================================================================
class TestBookLean:
    def test_over_lean(self):
        label, strength = book_lean(0.60)
        assert label == "OVER"
        assert strength == pytest.approx(0.10, abs=1e-6)

    def test_under_lean(self):
        label, _ = book_lean(0.40)
        assert label == "UNDER"

    def test_neutral_band(self):
        # Right at the 0.50 boundary should be NEUTRAL
        label, _ = book_lean(0.50)
        assert label == "NEUTRAL"

    def test_neutral_just_inside(self):
        label, _ = book_lean(0.53)  # within the 0.04 neutral band
        assert label == "NEUTRAL"

    def test_over_just_outside(self):
        label, _ = book_lean(0.55)  # just beyond the 0.04 band
        assert label == "OVER"

    def test_strength_is_absolute_deviation(self):
        _, s = book_lean(0.30)
        assert s == pytest.approx(0.20, abs=1e-6)
