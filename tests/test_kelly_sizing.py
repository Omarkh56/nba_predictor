"""
tests/test_kelly_sizing.py — Unit tests for playerlinepredictor.compute_stake_pct()
and its wiring into process_game()'s row dict / nba_combined.py's _pick_to_dict().

No network access required.
"""
import importlib.util
import os
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pandas  # preload before patch.dict restores sys.modules after the stub import
import pytest

os.environ.setdefault("ODDS_API_KEY", "test-odds-api-key")
os.environ.setdefault("API_FOOTBALL_KEY", "test-api-football-key")

_proj = Path(__file__).parent.parent
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))


def _import_playerline():
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

    dvp_stub = types.ModuleType("dvp")
    dvp_stub.get_dvp_factor = mock.Mock(return_value=1.0)

    with mock.patch.dict("sys.modules", {**nba_stubs, "requests": req_stub, "dvp": dvp_stub}):
        spec = importlib.util.spec_from_file_location(
            "playerlinepredictor_stub", str(_proj / "playerlinepredictor.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


_pl = _import_playerline()


class TestComputeStakePct:
    def test_negative_ev_gets_zero_stake(self):
        # p=0.40, d=2.0 -> EV = 0.40*2.0 - 1 = -0.20 (negative)
        assert _pl.compute_stake_pct(p=0.40, decimal_odds=2.0, total_penalty=0.0) == 0.0

    def test_zero_ev_gets_zero_stake(self):
        # p=0.50, d=2.0 -> EV exactly 0
        assert _pl.compute_stake_pct(p=0.50, decimal_odds=2.0, total_penalty=0.0) == 0.0

    def test_positive_ev_gets_positive_stake(self):
        # p=.54, d=2 -> EV=.08, full Kelly=.08, quarter Kelly=.02.
        stake = _pl.compute_stake_pct(p=0.54, decimal_odds=2.0, total_penalty=0.0)
        assert _pl.KELLY_FRACTION == 0.25
        assert stake == pytest.approx(0.02)

    def test_none_p_returns_zero(self):
        assert _pl.compute_stake_pct(p=None, decimal_odds=2.0, total_penalty=0.0) == 0.0

    def test_none_decimal_odds_returns_zero(self):
        assert _pl.compute_stake_pct(p=0.70, decimal_odds=None, total_penalty=0.0) == 0.0

    def test_decimal_odds_at_or_below_one_returns_zero(self):
        assert _pl.compute_stake_pct(p=0.70, decimal_odds=1.0, total_penalty=0.0) == 0.0
        assert _pl.compute_stake_pct(p=0.70, decimal_odds=0.5, total_penalty=0.0) == 0.0

    def test_hard_cap_applies_on_a_huge_edge(self):
        # p=0.95, d=5.0 -> EV = 3.75, f* = 3.75/4 = 0.9375, way above any cap
        stake = _pl.compute_stake_pct(p=0.95, decimal_odds=5.0, total_penalty=0.0)
        assert stake == pytest.approx(_pl.MAX_STAKE_PCT)

    def test_insufficient_edge_is_zero_with_reason(self):
        decision = _pl.compute_stake_decision(0.515, 2.0)
        assert decision.expected_value > 0
        assert decision.stake_pct == 0.0
        assert decision.stake_skip_reason == "insufficient_model_edge"

    def test_unavailable_vig_does_not_invent_a_value_or_block_valid_edge(self):
        decision = _pl.compute_stake_decision(0.56, 2.0, p_market=None, vig_pct=None)
        assert decision.stake_pct > 0
        assert decision.stake_skip_reason == ""

    def test_paired_market_edge_must_exceed_vig(self):
        decision = _pl.compute_stake_decision(0.56, 2.0, p_market=0.53, vig_pct=4.0)
        assert decision.model_edge == pytest.approx(0.03)
        assert decision.stake_pct == 0.0
        assert decision.stake_skip_reason == "edge_not_above_market_vig"

    def test_stale_required_data_is_zero_with_reason(self):
        decision = _pl.compute_stake_decision(0.60, 2.0, data_is_fresh=False)
        assert decision.stake_pct == 0.0
        assert decision.stake_skip_reason == "stale_required_data"

    def test_legacy_penalty_argument_does_not_change_formal_policy(self):
        base = _pl.compute_stake_pct(0.54, 2.0, total_penalty=0.0)
        penalized = _pl.compute_stake_pct(0.54, 2.0, total_penalty=500.0)
        assert penalized == pytest.approx(base)
