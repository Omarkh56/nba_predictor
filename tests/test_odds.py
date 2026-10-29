"""
Tests for pure/deterministic functions in oddstracker.py.
No network access, no live data required.
"""
import math
import sys
import importlib
import types
import pandas as pd
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
    stub_nba_utils.dataframe_call_result = mock.Mock(return_value=None)

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
parse_props_to_rows = _ot.parse_props_to_rows
parse_props_to_snapshot_rows = _ot.parse_props_to_snapshot_rows
append_consensus_rows = _ot._append_rows
append_snapshot_rows = _ot._append_snapshot_rows


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


class TestConsensusPrice:
    @staticmethod
    def _payload(prices):
        return {"bookmakers": [{
            "key": f"book_{i}", "title": f"Book {i}", "markets": [{
                "key": "PTS", "outcomes": [
                    {"description": "Player A", "name": "Over", "point": 20.5, "price": over},
                    {"description": "Player A", "name": "Under", "point": 20.5, "price": under},
                ],
            }],
        } for i, (over, under) in enumerate(prices)]}

    def test_books_either_side_of_even_money_give_a_real_price(self):
        # A plain mean of -110 and +104 is -3: an impossible price that reads as a 3% chance.
        [row] = parse_props_to_rows(self._payload([(-110, -110), (104, -125)]), "AAA@BBB", "2026-10-24")
        assert row["avg_over_odds"] <= -100 or row["avg_over_odds"] >= 100
        assert row["implied_over"] == pytest.approx(
            (american_to_implied(-110) + american_to_implied(104)) / 2, abs=1e-3)
        assert 0.45 < row["devig_over"] < 0.55

    def test_average_is_taken_in_probability_space(self):
        assert _ot.average_american([-110, -110]) == pytest.approx(-110)
        assert _ot.average_american([150, 150]) == pytest.approx(150)
        assert _ot.average_american([-100, 100]) == pytest.approx(-100)
        for price in (-500, -150, -100, 100, 175, 900):
            assert _ot.implied_to_american(american_to_implied(price)) == pytest.approx(
                price if price != 100 else -100)


class TestTimestampedPropSnapshots:
    def test_parser_preserves_every_player_and_book_quote(self):
        payload = {"bookmakers": [{
            "key": "book_a", "title": "Book A", "markets": [{
                "key": "PTS", "outcomes": [
                    {"description": player, "name": side, "point": line, "price": -110}
                    for player, line in (("Player A", 20.5), ("Player B", 15.5))
                    for side in ("Over", "Under")
                ],
            }],
        }]}
        consensus = parse_props_to_rows(
            payload, "AAA@BBB", "2026-05-01",
            snapshot_at="2026-05-01T18:00:00+00:00",
            event_id="event-1", commence_time="2026-05-01T20:00:00+00:00",
        )
        snapshots = parse_props_to_snapshot_rows(
            payload, "AAA@BBB", "2026-05-01", "2026-05-01T18:00:00+00:00",
            "event-1", "2026-05-01T20:00:00+00:00",
        )
        assert {row["player"] for row in consensus} == {"Player A", "Player B"}
        assert len(snapshots) == 4
        assert all(row["event_id"] == "event-1" for row in snapshots)
        assert all(row["devig_probability"] == pytest.approx(0.5) for row in snapshots)

    def test_one_sided_quote_keeps_vig_and_devig_missing(self):
        payload = {"bookmakers": [{
            "key": "book_a", "title": "Book A", "markets": [{
                "key": "PTS", "outcomes": [
                    {"description": "Player A", "name": "Over", "point": 20.5, "price": -110}
                ],
            }],
        }]}
        snapshots = parse_props_to_snapshot_rows(
            payload, "AAA@BBB", "2026-05-01", "2026-05-01T18:00:00+00:00",
            "event-1", "2026-05-01T20:00:00+00:00",
        )
        assert len(snapshots) == 1
        assert snapshots[0]["vig_pct"] is None
        assert snapshots[0]["devig_probability"] is None

    def test_consensus_updates_but_book_snapshots_append(self, tmp_path):
        consensus_path = tmp_path / "odds_tracker.csv"
        snapshot_path = tmp_path / "prop_odds_snapshots.csv"
        base = {
            "date": "2026-05-01", "game": "AAA@BBB", "player": "Player A",
            "market": "PTS", "line": 20.5, "books_count": 1,
            "avg_over_odds": -110, "avg_under_odds": -110,
            "implied_over": .5238, "implied_under": .5238,
            "devig_over": .5, "devig_under": .5, "book_lean": "NEUTRAL",
            "lean_strength": 0, "vig_pct": 4.76, "k_value": 1.07,
            "snapshot_at": "2026-05-01T18:00:00+00:00", "event_id": "event-1",
            "commence_time": "2026-05-01T20:00:00+00:00",
            "odds_source": "odds_tracker_consensus", "result": "",
        }
        append_consensus_rows([base], "2026-05-01", path=str(consensus_path))
        append_consensus_rows([
            {**base, "avg_over_odds": -120, "snapshot_at": "2026-05-01T19:00:00+00:00"}
        ], "2026-05-01", path=str(consensus_path))
        saved = pd.read_csv(consensus_path)
        assert len(saved) == 1
        assert saved.loc[0, "avg_over_odds"] == -120

        snap = {
            "snapshot_at": "2026-05-01T18:00:00+00:00", "date": "2026-05-01",
            "event_id": "event-1", "commence_time": "2026-05-01T20:00:00+00:00",
            "game": "AAA@BBB", "book_key": "book_a", "book_title": "Book A",
            "player": "Player A", "market": "PTS", "line": 20.5, "side": "OVER",
            "american_odds": -110, "decimal_odds": 1.909, "raw_implied_probability": .5238,
            "opposing_american_odds": -110, "devig_probability": .5,
            "vig_pct": 4.76, "k_value": 1.07, "odds_source": "the_odds_api",
        }
        assert append_snapshot_rows([snap], path=str(snapshot_path)) == 1
        assert append_snapshot_rows([
            {**snap, "snapshot_at": "2026-05-01T19:00:00+00:00", "american_odds": -120}
        ], path=str(snapshot_path)) == 1
        assert len(pd.read_csv(snapshot_path)) == 2
