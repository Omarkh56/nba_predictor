"""
tests/test_nba_combined.py — Unit tests for nba_combined.py's market-edge
ranking metric.

playerlinepredictor.py's evaluate_prop() computes edge_pct as deviation from
the posted line, not from the market's de-vigged probability. This tests the
fix that makes _rank_score() / top-N selection use
model_probability - devig_market_probability (true edge vs. the market)
whenever a same-day odds_tracker.csv snapshot exists for (player, market),
falling back to edge_pct (and flagging NO-MKT-CHECK) otherwise.

No network access required.
"""

import importlib.util
import os
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("ODDS_API_KEY", "test-odds-api-key")
os.environ.setdefault("API_FOOTBALL_KEY", "test-api-football-key")

_proj = Path(__file__).parent.parent
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))


def _import_nba_combined():
    """Load nba_combined.py (and everything it imports: newnbapredictor,
    oddstracker, playerlinepredictor, predictions_db) with nba_api/dvp
    stubbed out so nothing touches the network at import time."""
    nba_stubs = {}
    for name in [
        "nba_api", "nba_api.stats", "nba_api.stats.endpoints",
        "nba_api.stats.library", "nba_api.stats.library.http",
        "nba_api.stats.static",
        "nba_api.stats.endpoints.leaguedashplayerstats",
        "nba_api.stats.endpoints.leaguedashteamclutch",
        "nba_api.stats.endpoints.leaguedashteamstats",
        "nba_api.stats.endpoints.leaguegamefinder",
        "nba_api.stats.endpoints.playergamelog",
        "nba_api.stats.endpoints.leaguedashplayerbiostats",
        "nba_api.stats.endpoints.commonteamroster",
        "nba_api.stats.endpoints.leaguegamelog",
    ]:
        nba_stubs[name] = types.ModuleType(name)
    nba_stubs["nba_api.stats.library.http"].STATS_HEADERS = {}
    nba_stubs["nba_api.stats.library.http"].STATS_TIMEOUT = 60

    static_stub = types.ModuleType("nba_api.stats.static")
    static_stub.players = types.SimpleNamespace(
        find_players_by_full_name=lambda *a, **k: [], get_players=lambda: []
    )
    static_stub.teams = types.SimpleNamespace(
        find_team_name_by_abbreviation=lambda *a, **k: None, get_teams=lambda: []
    )
    nba_stubs["nba_api.stats.static"] = static_stub

    req_stub = types.ModuleType("requests")
    req_stub.get = mock.Mock()
    req_stub.RequestException = Exception
    exc_stub = types.ModuleType("requests.exceptions")
    exc_stub.RequestException = Exception
    req_stub.exceptions = exc_stub

    dvp_stub = types.ModuleType("dvp")
    dvp_stub.get_dvp_factor = mock.Mock(return_value=1.0)

    with mock.patch.dict("sys.modules", {
        **nba_stubs, "requests": req_stub, "requests.exceptions": exc_stub, "dvp": dvp_stub,
    }):
        spec = importlib.util.spec_from_file_location(
            "nba_combined_stub", str(_proj / "nba_combined.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


_nc = _import_nba_combined()


def _row(player="Test Player", market="PTS", pick="OVER", line=20.5,
         edge_pct=8.0, confidence=60.0, flags=""):
    return {
        "Player": player, "Market": market, "Pick": pick, "Line": line,
        "Edge%": edge_pct, "Confidence": confidence, "Flags": flags,
    }


@pytest.fixture(autouse=True)
def _clear_lean():
    """Every test starts with an empty book-lean lookup."""
    _nc._TODAY_LEAN = {}
    yield
    _nc._TODAY_LEAN = {}


# ── _market_edge_pct ───────────────────────────────────────────────────────

class TestMarketEdgePct:
    def test_none_when_no_snapshot(self):
        assert _nc._market_edge_pct("nobody", "PTS", "OVER", 60.0) is None

    def test_over_uses_devig_over(self):
        _nc._TODAY_LEAN[("lebron james", "PTS")] = {
            "devig_over": 0.55, "devig_under": 0.45,
        }
        # model 65% on OVER vs. devig 55% -> +10.0 pp
        result = _nc._market_edge_pct("LeBron James", "PTS", "OVER", 65.0)
        assert result == pytest.approx(10.0)

    def test_under_uses_devig_under(self):
        _nc._TODAY_LEAN[("lebron james", "PTS")] = {
            "devig_over": 0.55, "devig_under": 0.45,
        }
        # model 60% on UNDER vs. devig 45% -> +15.0 pp
        result = _nc._market_edge_pct("LeBron James", "PTS", "UNDER", 60.0)
        assert result == pytest.approx(15.0)

    def test_player_lookup_is_case_and_whitespace_insensitive(self):
        _nc._TODAY_LEAN[("lebron james", "PTS")] = {
            "devig_over": 0.50, "devig_under": 0.50,
        }
        assert _nc._market_edge_pct("  LEBRON JAMES  ", "PTS", "OVER", 55.0) == pytest.approx(5.0)

    def test_negative_true_edge_is_a_bad_bet_despite_positive_edge_pct(self):
        # Market already prices this pick at 70% -- model's 55% confidence
        # means the "edge" is actually against us, even though edge_pct
        # (deviation from the line) would look fine in isolation.
        _nc._TODAY_LEAN[("jayson tatum", "PTS")] = {
            "devig_over": 0.70, "devig_under": 0.30,
        }
        result = _nc._market_edge_pct("Jayson Tatum", "PTS", "OVER", 55.0)
        assert result == pytest.approx(-15.0)
        assert result < 0


# ── _enrich_with_market_edge ────────────────────────────────────────────────

class TestEnrichWithMarketEdge:
    def test_matched_row_gets_mkt_edge_and_no_flag(self):
        _nc._TODAY_LEAN[("test player", "PTS")] = {"devig_over": 0.52, "devig_under": 0.48}
        df = pd.DataFrame([_row(confidence=62.0)])
        out = _nc._enrich_with_market_edge(df)
        assert out.loc[0, "MktEdgePct"] == pytest.approx(10.0)
        assert _nc.NO_MARKET_FLAG not in out.loc[0, "Flags"]

    def test_unmatched_row_flagged_no_market_check(self):
        df = pd.DataFrame([_row(player="Nobody Tracked")])
        out = _nc._enrich_with_market_edge(df)
        assert pd.isna(out.loc[0, "MktEdgePct"])
        assert _nc.NO_MARKET_FLAG in out.loc[0, "Flags"]

    def test_preserves_existing_flags(self):
        df = pd.DataFrame([_row(player="Nobody Tracked", flags="★STRONG ROAD-B2B")])
        out = _nc._enrich_with_market_edge(df)
        assert "★STRONG" in out.loc[0, "Flags"]
        assert "ROAD-B2B" in out.loc[0, "Flags"]
        assert _nc.NO_MARKET_FLAG in out.loc[0, "Flags"]

    def test_idempotent_does_not_duplicate_flag(self):
        df = pd.DataFrame([_row(player="Nobody Tracked")])
        once = _nc._enrich_with_market_edge(df)
        twice = _nc._enrich_with_market_edge(once)
        assert twice.loc[0, "Flags"].count(_nc.NO_MARKET_FLAG) == 1

    def test_empty_dataframe_gets_mkt_edge_column(self):
        out = _nc._enrich_with_market_edge(pd.DataFrame())
        assert "MktEdgePct" in out.columns
        assert out.empty


# ── _rank_score ─────────────────────────────────────────────────────────────

class TestRankScore:
    def test_uses_market_edge_when_available(self):
        # edge_pct says 25 (would rank very high); true market edge says -5
        # (a bad bet). The market figure must win.
        hi = _nc._rank_score(25.0, 55.0, "PTS", "OVER", 20.0, mkt_edge_pct=-5.0)
        lo = _nc._rank_score(2.0, 55.0, "PTS", "OVER", 20.0, mkt_edge_pct=8.0)
        assert lo > hi

    def test_falls_back_to_edge_pct_when_no_market_data(self):
        with_none = _nc._rank_score(12.0, 55.0, "PTS", "OVER", 20.0, mkt_edge_pct=None)
        assert with_none == pytest.approx(12.0)  # no market weight applied here

    def test_low_line_cap_applies_to_market_edge_too(self):
        capped = _nc._rank_score(2.0, 90.0, "3PM", "OVER", 1.5, mkt_edge_pct=50.0)
        # 3PM static weight 0.40, cap at 30 -> 30 * 0.40 = 12.0
        assert capped == pytest.approx(12.0)


# ── End-to-end: ranking actually flips vs. the old edge_pct-only order ──────

class TestRankingChangesBehavior:
    def test_high_edge_pct_bad_market_price_ranks_below_low_edge_pct_good_price(self):
        _nc._TODAY_LEAN[("player a", "PTS")] = {"devig_over": 0.80, "devig_under": 0.20}
        _nc._TODAY_LEAN[("player b", "PTS")] = {"devig_over": 0.45, "devig_under": 0.55}

        df = pd.DataFrame([
            # Big deviation from the line, but the market already prices it
            # at 80% -- model's 60% confidence is actually a bad bet (-20pp).
            _row(player="Player A", edge_pct=22.0, confidence=60.0),
            # Small deviation from the line, but the market is only at 45% --
            # model's 55% confidence is a genuine +10pp edge.
            _row(player="Player B", edge_pct=4.0, confidence=55.0),
        ])

        # Old behavior (what edge_pct alone would have ranked):
        assert df.loc[0, "Edge%"] > df.loc[1, "Edge%"]  # A > B on the old metric

        enriched = _nc._enrich_with_market_edge(df)
        enriched["_rscore"] = enriched.apply(
            lambda r: _nc._rank_score(
                r["Edge%"], r["Confidence"], r["Market"], r["Pick"], r["Line"], r["MktEdgePct"]
            ),
            axis=1,
        )
        ranked = enriched.sort_values("_rscore", ascending=False).reset_index(drop=True)

        # New behavior: B (genuine market edge) must outrank A (market already
        # agrees, so A's "edge" was an illusion from the line, not the price).
        assert ranked.loc[0, "Player"] == "Player B"
        assert ranked.loc[0, "MktEdgePct"] > ranked.loc[1, "MktEdgePct"]
