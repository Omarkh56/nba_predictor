"""Replay data layer: strict as-of filtering, as-of usage, eligibility, chunking, failed-fetch handling."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "replay"))

from replay_cache import NBACache  # noqa: E402
from replay_data import ReplayDataError, asof_before, load_season_bundle, parse_minutes, usage_asof  # noqa: E402
from replay_engine import eligibility, split_dates  # noqa: E402


def test_asof_before_is_strict_and_drops_later_games_from_a_cached_response():
    df = pd.DataFrame({"GAME_DATE": pd.to_datetime(["2026-04-18", "2026-04-20", "2026-04-21", "2026-06-01"]), "V": range(4)})
    out = asof_before(df, pd.Timestamp("2026-04-21"))
    assert list(out["V"]) == [0, 1]  # the game ON the cutoff date and everything later are excluded
    assert asof_before(df, pd.Timestamp("2026-04-18")).empty
    assert asof_before(pd.DataFrame(), pd.Timestamp("2026-04-21")).empty


@pytest.mark.parametrize("raw,expected", [("34:30", 34.5), (28, 28.0), ("12", 12.0), (None, 0.0), (float("nan"), 0.0), ("--", 0.0)])
def test_parse_minutes(raw, expected):
    assert parse_minutes(raw) == expected


def _box(game_id, date, team, pid, name, mins, fga, fta, tov, st="Regular Season"):
    return {"GAME_ID": game_id, "GAME_DATE": pd.Timestamp(date), "TEAM_ID": team, "PLAYER_ID": pid, "PLAYER_NAME": name,
            "MIN": mins, "FGA": fga, "FTA": fta, "TOV": tov, "SEASON_TYPE": st}


def _usage_frame():
    rows = []
    # one team, two players, each game: both 30 min and 10 possessions used -> team_min=60, team_pu=20
    for g in range(7):  # 7 regular-season games
        pu_p1 = 10 if g != 5 else 40  # game index 5 is a usage spike for player 1
        d = "2026-01-%02d" % (g + 1)
        rows.append(_box("G%d" % g, d, 1, 101, "P1", 30, pu_p1, 0, 0))
        rows.append(_box("G%d" % g, d, 1, 102, "P2", 30, 10, 0, 0))
    rows.append(_box("PO1", "2026-04-20", 1, 101, "P1", 30, 10, 0, 0, st="Playoffs"))
    df = pd.DataFrame(rows).sort_values(["GAME_DATE", "PLAYER_ID"]).reset_index(drop=True)
    return df


def test_usage_asof_uses_only_prior_games_and_playoffs_use_completed_regular_season():
    df = _usage_frame()
    u = usage_asof(df, min_games=3)
    p1 = df[(df.PLAYER_ID == 101) & (df.SEASON_TYPE == "Regular Season")].index
    vals = [u.loc[i] for i in p1]
    assert all(np.isnan(v) for v in vals[:3])  # fewer than min_games prior games -> unknown
    assert vals[3] == pytest.approx(0.2) and vals[4] == pytest.approx(0.2) and vals[5] == pytest.approx(0.2)
    # game 5 is the spike: its own value must NOT be visible when predicting game 5, only from game 6 on
    assert vals[6] > 0.2
    po = df[df.SEASON_TYPE == "Playoffs"].index[0]
    # expected regular-season final for P1: sum(pu * team_min/5) / sum(min * team_pu); team_min is always 60,
    # team_pu is 20 except in the spike game where P1's 40 + P2's 10 = 50
    pus = [10, 10, 10, 10, 10, 40, 10]
    num = sum(pu * 60 / 5.0 for pu in pus)
    den = sum(30 * (50 if pu == 40 else 20) for pu in pus)
    assert u.loc[po] == pytest.approx(num / den)


def test_eligibility_uses_all_prior_played_games_only():
    assert eligibility(np.array([30.0] * 9)) == (False, "insufficient_prior_games")
    # no minutes floor: low-minute players are eligible (they are a reporting stratum, not an exclusion)
    assert eligibility(np.array([5.0] * 30)) == (True, "")
    assert eligibility(np.array([10.0] * 5 + [19.0] * 10)) == (True, "")


def test_split_dates_are_chronological_contiguous_and_balanced():
    dates = list(pd.date_range("2026-01-01", periods=20))
    chunks = split_dates(dates, 4)
    assert sum(len(c) for c in chunks) == 20
    flat = [d for c in chunks for d in c]
    assert flat == dates  # no reordering, no gaps
    assert max(len(c) for c in chunks) - min(len(c) for c in chunks) <= 2


# ------------------------------------------------------------------ failed vs empty fetch handling in the loader
LGF_HEADERS = ["SEASON_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE",
               "MATCHUP", "MIN", "PTS", "REB", "AST", "FG3M", "FGA", "FTA", "TOV"]
TEAM_HEADERS = ["TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "PTS"]


class LoaderFetcher:
    def __init__(self, po_mode):
        self.po_mode = po_mode  # "empty" | "fail"

    def __call__(self, endpoint, params, timeout):
        if endpoint == "commonteamroster":
            return {"resultSets": [{"name": "CommonTeamRoster", "headers": ["PLAYER", "POSITION"], "rowSet": []}]}
        who, st = params["player_or_team_abbreviation"], params["season_type_nullable"]
        if st == "Playoffs":
            if self.po_mode == "fail":
                raise requests.exceptions.Timeout("playoffs endpoint down")
            return {"resultSets": [{"name": "R", "headers": LGF_HEADERS if who == "P" else TEAM_HEADERS, "rowSet": []}]}
        if who == "P":
            rows = [[22025, 7, "A B", 1, "LAL", "G1", "2026-01-02", "LAL vs. BOS", "30:00", 10, 5, 3, 1, 8, 2, 1]]
            return {"resultSets": [{"name": "R", "headers": LGF_HEADERS, "rowSet": rows}]}
        return {"resultSets": [{"name": "R", "headers": TEAM_HEADERS, "rowSet": [[1, "G1", "2026-01-02", "LAL vs. BOS", 100]]}]}


def _client(tmp_path, po_mode):
    return NBACache(str(tmp_path), fetcher=LoaderFetcher(po_mode), sleep=lambda s: None, min_interval_s=0, retries=2, base_delay_s=0)


def test_loader_accepts_a_successful_empty_playoff_response(tmp_path):
    bundle = load_season_bundle(_client(tmp_path, "empty"), "2025-26", want_rosters=False)
    assert len(bundle.players) == 1 and bundle.meta["fetch_status"]["P/Playoffs"] == "ok_empty"


def test_loader_refuses_to_run_on_a_failed_required_fetch(tmp_path):
    with pytest.raises(ReplayDataError):
        load_season_bundle(_client(tmp_path, "fail"), "2025-26", want_rosters=False)
