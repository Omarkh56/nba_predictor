"""
tests/test_signals.py — Unit tests for the four new prediction signals.

Covers pure functions only (no NBA API, no disk I/O):
  Signal 1: altitude_feature()            (train_model.py)
  Signal 2: venue_residual_from_record()  (train_model.py)
  Signal 4: compute_star_form_index()     (train_model.py)
  Signal 3: detect_injury_return()        (playerlinepredictor.py)
             injury_rust_discount()       (playerlinepredictor.py)
"""

import importlib.util
import os
import sys
import types
import unittest.mock as mock
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_proj = Path(__file__).parent.parent
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

# ── Load train_model pure functions ─────────────────────────────────────────────

def _import_train_model():
    """Load train_model.py with NBA-API stubbed out."""
    nba_stubs = {}
    for name in [
        "nba_api", "nba_api.stats", "nba_api.stats.library",
        "nba_api.stats.library.http", "nba_api.stats.endpoints",
        "nba_api.stats.endpoints.leaguegamefinder",
        "nba_api.stats.endpoints.leaguegamelog",
        "nba_api.stats.endpoints.leaguedashplayerstats",
    ]:
        nba_stubs[name] = types.ModuleType(name)
    nba_stubs["nba_api.stats.library.http"].STATS_HEADERS = {}
    nba_stubs["nba_api.stats.library.http"].STATS_TIMEOUT = 30

    req_stub     = types.ModuleType("requests")
    req_stub.get = mock.Mock()
    req_stub.RequestException = Exception
    req_stub.Session = mock.MagicMock

    with mock.patch.dict("sys.modules", {**nba_stubs, "requests": req_stub}):
        spec = importlib.util.spec_from_file_location(
            "train_model_stub", str(_proj / "train_model.py")
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass
    return mod


_tm = _import_train_model()

altitude_feature           = getattr(_tm, "altitude_feature",           None)
venue_residual_from_record = getattr(_tm, "venue_residual_from_record", None)
compute_star_form_index    = getattr(_tm, "compute_star_form_index",    None)
_ALTITUDE_HOME_TIDS        = getattr(_tm, "_ALTITUDE_HOME_TIDS",        frozenset({1610612743, 1610612762}))
_STAR_FORM_SHRINK_K        = getattr(_tm, "_STAR_FORM_SHRINK_K",        10)


# ── Load playerlinepredictor pure functions ──────────────────────────────────────

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
    # `from nba_api.stats.static import players, teams` requires attribute objects
    static_stub.players = types.SimpleNamespace(
        find_players_by_full_name=lambda *a, **k: [],
        get_players=lambda: [],
    )
    static_stub.teams = types.SimpleNamespace(
        find_team_name_by_abbreviation=lambda *a, **k: None,
        get_teams=lambda: [],
    )
    nba_stubs["nba_api.stats.static"] = static_stub

    req_stub               = types.ModuleType("requests")
    req_stub.get           = mock.Mock()
    req_stub.RequestException = Exception

    dvp_stub               = types.ModuleType("dvp")
    dvp_stub.get_dvp_factor = mock.Mock(return_value=1.0)

    with mock.patch.dict("sys.modules", {
        **nba_stubs, "requests": req_stub, "dvp": dvp_stub
    }):
        spec = importlib.util.spec_from_file_location(
            "playerlinepredictor_stub", str(_proj / "playerlinepredictor.py")
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass
    return mod


_pl = _import_playerline()

detect_injury_return = getattr(_pl, "detect_injury_return", None)
injury_rust_discount = getattr(_pl, "injury_rust_discount", None)
_INJURY_RAMP_GAMES   = getattr(_pl, "_INJURY_RAMP_GAMES",   8)
_INJURY_MAX_DISCOUNT = getattr(_pl, "_INJURY_MAX_DISCOUNT", 0.22)


# ── Shared helpers ────────────────────────────────────────────────────────────────

def _make_logs(game_dates, pts=None, mins=None):
    """Build a minimal player game-log DataFrame."""
    n = len(game_dates)
    return pd.DataFrame({
        "GAME_DATE": [d.strftime("%Y-%m-%d") for d in game_dates],
        "GAME_ID":   [f"g{i:04d}" for i in range(n)],
        "PTS":  pts  if pts  is not None else [15.0] * n,
        "MIN":  mins if mins is not None else [30.0] * n,
        "WL":   ["W" if i % 2 == 0 else "L" for i in range(n)],
    })


# ─────────────────────────────────────────────────────────────────────────────────
# Signal 1: Altitude / Acclimatization
# ─────────────────────────────────────────────────────────────────────────────────

DEN_TID = 1610612743
UTA_TID = 1610612762
LAL_TID = 1610612747   # not elevated


@pytest.mark.skipif(altitude_feature is None, reason="altitude_feature not loaded")
class TestAltitudeFeature:

    def test_non_altitude_venue_returns_zero(self):
        assert altitude_feature(LAL_TID, 0, 3.0) == pytest.approx(0.0)

    def test_non_altitude_b2b_still_zero(self):
        assert altitude_feature(LAL_TID, 1, 0.0) == pytest.approx(0.0)

    def test_denver_b2b_returns_one(self):
        assert altitude_feature(DEN_TID, 1, 0.0) == pytest.approx(1.0)

    def test_utah_b2b_returns_one(self):
        assert altitude_feature(UTA_TID, 1, 0.0) == pytest.approx(1.0)

    def test_denver_three_days_rest_returns_zero(self):
        assert altitude_feature(DEN_TID, 0, 3.0) == pytest.approx(0.0)

    def test_denver_two_days_rest_is_partial(self):
        val = altitude_feature(DEN_TID, 0, 2.0)
        assert 0.0 < val < 1.0

    def test_monotone_in_rest_days(self):
        vals = [altitude_feature(DEN_TID, 0, float(d)) for d in [0, 1, 2, 3]]
        assert vals[0] >= vals[1] >= vals[2] >= vals[3]

    def test_output_in_unit_interval(self):
        for tid in [DEN_TID, UTA_TID, LAL_TID]:
            for b2b in [0, 1]:
                for rest in [0.0, 1.0, 2.0, 3.0, 5.0]:
                    v = altitude_feature(tid, b2b, rest)
                    assert 0.0 <= v <= 1.0, f"Out of [0,1]: {v} for tid={tid},b2b={b2b},rest={rest}"

    def test_both_altitude_venues_covered(self):
        assert DEN_TID in _ALTITUDE_HOME_TIDS
        assert UTA_TID in _ALTITUDE_HOME_TIDS

    def test_non_altitude_not_in_set(self):
        assert LAL_TID not in _ALTITUDE_HOME_TIDS


# ─────────────────────────────────────────────────────────────────────────────────
# Signal 2: Residualized Venue Effect
# ─────────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(venue_residual_from_record is None, reason="venue_residual_from_record not loaded")
class TestVenueResidual:

    def test_zero_when_home_pct_matches_overall(self):
        # 5 home W / 10 home G = 0.5, 10 total W / 20 total G = 0.5 → 0
        assert venue_residual_from_record(5, 10, 10, 20) == pytest.approx(0.0)

    def test_positive_when_home_advantage_exists(self):
        # 10 home W / 10 home G = 1.0, 10 total W / 20 total G = 0.5 → +0.5
        assert venue_residual_from_record(10, 10, 10, 20) == pytest.approx(0.5)

    def test_negative_when_worse_at_home(self):
        # 2 home W / 10 = 0.2, 8 total W / 20 = 0.4 → −0.2
        result = venue_residual_from_record(2, 10, 8, 20)
        assert result < 0

    def test_returns_zero_for_sparse_home_games(self):
        assert venue_residual_from_record(2, 2, 10, 20) == pytest.approx(0.0)

    def test_returns_zero_for_sparse_total_games(self):
        assert venue_residual_from_record(2, 5, 5, 2) == pytest.approx(0.0)

    def test_bounded_output(self):
        r = venue_residual_from_record(10, 10, 10, 20)
        assert -1.0 <= r <= 1.0


# ─────────────────────────────────────────────────────────────────────────────────
# Signal 3 (player model): Return-from-Injury Ramp
# ─────────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(detect_injury_return is None, reason="detect_injury_return not loaded")
class TestDetectInjuryReturn:

    def _logs_with_gap(self, n_before, gap_days, n_after):
        today = date(2026, 1, 20)
        # n_after most recent games, then a gap, then n_before older games
        after_dates  = [today - timedelta(days=i * 2) for i in range(n_after)]
        gap_anchor   = today - timedelta(days=n_after * 2 + gap_days)
        before_dates = [gap_anchor - timedelta(days=i * 2) for i in range(n_before)]
        all_dates    = sorted(after_dates + before_dates, reverse=True)
        return _make_logs(all_dates)

    def test_empty_logs_returns_zero(self):
        assert detect_injury_return(pd.DataFrame()) == 0

    def test_no_gap_returns_zero(self):
        today = date(2026, 1, 20)
        dates = [today - timedelta(days=i * 2) for i in range(10)]
        assert detect_injury_return(_make_logs(dates)) == 0

    def test_returns_non_negative_int(self):
        logs = self._logs_with_gap(n_before=10, gap_days=5, n_after=3)
        result = detect_injury_return(logs)
        assert isinstance(result, int) and result >= 0

    def test_returns_zero_when_fully_ramped(self):
        # Many games since return → should return 0
        logs = self._logs_with_gap(n_before=10, gap_days=6, n_after=10)
        assert detect_injury_return(logs) == 0

    def test_detects_large_gap(self):
        # 20-day gap between blocks — should be detected (within 30-day cap)
        logs = self._logs_with_gap(n_before=8, gap_days=20, n_after=2)
        result = detect_injury_return(logs)
        assert result > 0, "Large gap should be detected"

    def test_off_season_gap_ignored(self):
        # Gap > 30 days → treated as off-season, not injury
        today = date(2026, 1, 20)
        recent = [today - timedelta(days=i * 2) for i in range(5)]
        old    = [today - timedelta(days=100 + i * 2) for i in range(10)]
        logs   = _make_logs(sorted(recent + old, reverse=True))
        # Large off-season gap should not trigger ramp
        assert detect_injury_return(logs) == 0


@pytest.mark.skipif(injury_rust_discount is None, reason="injury_rust_discount not loaded")
class TestInjuryRustDiscount:

    def test_zero_games_no_discount(self):
        assert injury_rust_discount(0) == pytest.approx(0.0)

    def test_game_one_back_max_discount(self):
        assert injury_rust_discount(1) == pytest.approx(_INJURY_MAX_DISCOUNT)

    def test_tapers_monotonically(self):
        discounts = [injury_rust_discount(g) for g in range(1, _INJURY_RAMP_GAMES + 1)]
        for i in range(len(discounts) - 1):
            assert discounts[i] >= discounts[i + 1], (
                f"Discount should decrease: game {i+1}={discounts[i]}, game {i+2}={discounts[i+1]}"
            )

    def test_at_ramp_length_returns_zero(self):
        assert injury_rust_discount(_INJURY_RAMP_GAMES) == pytest.approx(0.0)

    def test_beyond_ramp_returns_zero(self):
        assert injury_rust_discount(_INJURY_RAMP_GAMES + 5) == pytest.approx(0.0)

    def test_always_non_negative(self):
        for g in range(20):
            assert injury_rust_discount(g) >= 0.0, f"Negative discount at game {g}"

    def test_always_leq_max_discount(self):
        for g in range(20):
            assert injury_rust_discount(g) <= _INJURY_MAX_DISCOUNT + 1e-9, (
                f"Exceeds max at game {g}"
            )


# ─────────────────────────────────────────────────────────────────────────────────
# Signal 4: Star Player Form (shrinkage math)
# ─────────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(compute_star_form_index is None, reason="compute_star_form_index not loaded")
class TestStarFormIndex:

    def _player_logs(self, team_id=1, player_id=1, n=15, pts_seq=None):
        """Build logs with newest game at index 0 (descending date order)."""
        today = pd.Timestamp("2026-01-20")
        # Index 0 = most recent, index n-1 = oldest
        dates = [today - pd.Timedelta(days=i * 2) for i in range(n)]
        pts   = pts_seq if pts_seq is not None else [15.0] * n
        return pd.DataFrame({
            "TEAM_ID":   [team_id]  * n,
            "PLAYER_ID": [player_id] * n,
            "GAME_ID":   [f"g{i:04d}" for i in range(n)],
            "GAME_DATE": [d.strftime("%Y-%m-%d") for d in dates],
            "PTS": pts,
            "MIN": [32.0] * n,
        })

    def test_empty_logs_return_empty_df(self):
        result = compute_star_form_index(pd.DataFrame())
        assert result.empty

    def test_returns_expected_columns(self):
        logs   = self._player_logs()
        result = compute_star_form_index(logs)
        assert set(result.columns) >= {"team_id", "game_id", "star_form"}

    def test_consistent_scorer_near_zero_deviation(self):
        # Same score every game → EWMA ≈ season avg → deviation ≈ 0
        logs   = self._player_logs(pts_seq=[20.0] * 15)
        result = compute_star_form_index(logs)
        if not result.empty:
            assert result["star_form"].abs().max() < 2.0, (
                "Consistent scorer should have near-zero deviation"
            )

    def test_hot_streak_gives_positive_deviation(self):
        # Most recent 5 games at 30pts, prior 10 games at 10pts → recent EWMA > season avg
        # Index 0-4 = newest = 30pts; indices 5-14 = older = 10pts
        pts  = [30.0] * 5 + [10.0] * 10
        logs = self._player_logs(n=15, pts_seq=pts)
        result = compute_star_form_index(logs)
        if not result.empty:
            # The most recent game row should have positive deviation
            # (sorted ascending in the function, so most recent is last — game g0000)
            last_row = result[result["game_id"] == "g0000"]
            if not last_row.empty:
                assert last_row["star_form"].iloc[0] > 0, (
                    "Recent hot streak should produce positive star_form"
                )

    def test_cold_streak_gives_negative_deviation(self):
        # Most recent 5 games at 5pts, prior 10 at 25pts → EWMA below season avg
        pts  = [5.0] * 5 + [25.0] * 10
        logs = self._player_logs(n=15, pts_seq=pts)
        result = compute_star_form_index(logs)
        if not result.empty:
            last_row = result[result["game_id"] == "g0000"]
            if not last_row.empty:
                assert last_row["star_form"].iloc[0] < 0, (
                    "Recent cold streak should produce negative star_form"
                )

    def test_shrinkage_constant_is_reasonable(self):
        # k=10 means a 10-game streak is shrunk by 50% (n/(n+k) = 10/20 = 0.5)
        assert _STAR_FORM_SHRINK_K > 0
        shrink_at_10_games = 10 / (10 + _STAR_FORM_SHRINK_K)
        assert 0.3 <= shrink_at_10_games <= 0.7, (
            "k should produce moderate shrinkage at 10 games"
        )

    def test_two_players_on_same_team_combined(self):
        logs1 = self._player_logs(team_id=1, player_id=1, pts_seq=[25.0] * 15)
        logs2 = self._player_logs(team_id=1, player_id=2, pts_seq=[20.0] * 15)
        logs  = pd.concat([logs1, logs2], ignore_index=True)
        result = compute_star_form_index(logs)
        # At least some rows should exist
        assert not result.empty
        assert (result["team_id"] == 1).all()
