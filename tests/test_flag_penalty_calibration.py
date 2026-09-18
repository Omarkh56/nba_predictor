"""
tests/test_flag_penalty_calibration.py — Unit tests for playerlinepredictor.py's
_flag_penalty() and its wiring into _compute_direction_penalty()/
_self_injury_check().

Verifies the data-driven replacement for hand-tuned penalty constants:
  - falls back to the exact original constant when calibration.json has no
    flag_penalties data yet (missing file, empty dict, or missing key) --
    this must reproduce today's behavior unchanged;
  - actually changes behavior once calibration.json has a fitted value.

No network access required.
"""
import importlib.util
import os
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest

os.environ.setdefault("ODDS_API_KEY", "test-odds-api-key")
os.environ.setdefault("API_FOOTBALL_KEY", "test-api-football-key")

_proj = Path(__file__).parent.parent
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))


def _import_playerline():
    """Load playerlinepredictor.py with all network deps stubbed."""
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

    with mock.patch.dict("sys.modules", {
        **nba_stubs, "requests": req_stub, "dvp": dvp_stub
    }):
        spec = importlib.util.spec_from_file_location(
            "playerlinepredictor_stub", str(_proj / "playerlinepredictor.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


_pl = _import_playerline()


def _reset_calib():
    _pl.CALIB = {}


# ── _flag_penalty ────────────────────────────────────────────────────────────

class TestFlagPenalty:
    def test_falls_back_to_default_when_calib_empty(self):
        _reset_calib()
        assert _pl._flag_penalty("UNDER_PICK", 2.0) == 2.0

    def test_falls_back_to_default_when_key_missing(self):
        _pl.CALIB = {"flag_penalties": {"SOME_OTHER_FLAG": {"pp": 5.0}}}
        assert _pl._flag_penalty("UNDER_PICK", 2.0) == 2.0

    def test_uses_fitted_value_negated(self):
        # pp = +9.25 (confidence gained) -> penalty = -9.25 (a bonus, not a penalty)
        _pl.CALIB = {"flag_penalties": {"UNDER_PICK": {"pp": 9.25}}}
        assert _pl._flag_penalty("UNDER_PICK", 2.0) == -9.25

    def test_negative_pp_still_yields_a_positive_penalty(self):
        _pl.CALIB = {"flag_penalties": {"NO_PO": {"pp": -3.25}}}
        assert _pl._flag_penalty("NO_PO", 8.0) == 3.25


# ── Wiring into _compute_direction_penalty ──────────────────────────────────

class TestDirectionPenaltyWiring:
    def test_default_behavior_unchanged_with_no_calibration(self):
        _reset_calib()
        meta = {"po_count": 1, "po_raw_count": 1}
        penalty, flag = _pl._compute_direction_penalty(
            "player_points", "PTS", "OVER", proj=20.0, line=18.5, book_count=1, meta=meta
        )
        # book_count==1 -> hardcoded 4.0 book penalty, nothing else triggers
        assert penalty == 4.0
        assert flag == "1-BOOK"

    def test_calibration_overrides_book_penalty(self):
        _pl.CALIB = {"flag_penalties": {"ONE_BOOK": {"pp": 8.31}}}
        meta = {"po_count": 1, "po_raw_count": 1}
        penalty, flag = _pl._compute_direction_penalty(
            "player_points", "PTS", "OVER", proj=20.0, line=18.5, book_count=1, meta=meta
        )
        # data says 1-book props hit MORE -> penalty becomes a bonus (negative)
        assert penalty == -8.31
        assert flag == "1-BOOK"
        _reset_calib()

    def test_under_pick_penalty_uses_calibration(self):
        _pl.CALIB = {"flag_penalties": {"UNDER_PICK": {"pp": 9.25}}}
        meta = {"po_count": 1, "po_raw_count": 1}
        penalty, _ = _pl._compute_direction_penalty(
            "player_rebounds", "REB", "UNDER", proj=8.0, line=9.5, book_count=0, meta=meta
        )
        assert penalty == pytest.approx(-9.25)
        _reset_calib()

    def test_unaffected_flags_keep_hardcoded_constant(self):
        # STAR_UNDER not present in calibration -> falls back to STAR_UNDER_PENALTY
        _pl.CALIB = {"flag_penalties": {"UNDER_PICK": {"pp": 0.0}}}
        meta = {"po_count": 1, "po_raw_count": 1}
        penalty, _ = _pl._compute_direction_penalty(
            "player_rebounds", "REB", "UNDER", proj=30.0, line=9.5, book_count=0, meta=meta
        )
        # UNDER_PICK calibrated to 0.0 -> no contribution from that; but
        # proj >= STAR_PROJ_THRESHOLD (28) still adds the hardcoded STAR_UNDER_PENALTY.
        assert penalty == pytest.approx(_pl.STAR_UNDER_PENALTY)
        _reset_calib()


# ── Wiring into _self_injury_check ──────────────────────────────────────────

class TestSelfInjuryWiring:
    def test_default_gtd_penalty(self):
        _reset_calib()
        penalty, flag, skip = _pl._self_injury_check(
            "Test Player", [{"player": "Test Player", "status": "Day-To-Day"}]
        )
        assert penalty == 5.0
        assert flag == "⚠GTD"
        assert skip is False

    def test_calibrated_gtd_penalty(self):
        _pl.CALIB = {"flag_penalties": {"GTD": {"pp": -2.0}}}
        penalty, flag, _ = _pl._self_injury_check(
            "Test Player", [{"player": "Test Player", "status": "Questionable"}]
        )
        assert penalty == 2.0
        assert flag == "⚠GTD"
        _reset_calib()
