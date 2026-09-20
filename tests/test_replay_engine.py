"""End-to-end lookahead guard: run the REAL model through the replay engine on a synthetic league and prove that
neither the predicted game's own outcome nor any later game can influence its projection.

Runs in a subprocess so the engine's monkeypatching of the model never leaks into the rest of the test session.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = r'''
import json, os, sys
os.environ["ODDS_API_KEY"] = "x"
ROOT = sys.argv[1]
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts", "replay"))
import pandas as pd
import replay_engine as eng
from replay_data import SeasonBundle

TEAMS = {"LAL": 1610612747, "BOS": 1610612738}
PLAYERS = [(1, "Test Star", "LAL"), (2, "Test Mate", "LAL"), (3, "Opp One", "BOS"), (4, "Opp Two", "BOS")]
COLS = ["SEASON_ID","PLAYER_ID","PLAYER_NAME","TEAM_ID","TEAM_ABBREVIATION","TEAM_NAME","GAME_ID","GAME_DATE","MATCHUP","WL",
        "MIN","FGM","FGA","FG3M","FG3A","FTM","FTA","OREB","DREB","REB","AST","STL","BLK","TOV","PF","PTS","PLUS_MINUS",
        "SEASON_TYPE","USG_ASOF"]

def build(prior_pts, target_pts, future_pts):
    dates = list(pd.date_range("2026-01-05", periods=17, freq="2D"))  # 15 prior games, target game (idx 15), 1 later game
    rows = []
    for gi, d in enumerate(dates):
        home = gi % 2 == 0
        for pid, name, tm in PLAYERS:
            opp = "BOS" if tm == "LAL" else "LAL"
            matchup = "%s vs. %s" % (tm, opp) if (home == (tm == "LAL")) else "%s @ %s" % (tm, opp)
            pts = 12
            if pid == 1:
                pts = prior_pts if gi < 15 else (target_pts if gi == 15 else future_pts)
            rows.append({"SEASON_ID": 22025, "PLAYER_ID": pid, "PLAYER_NAME": name, "TEAM_ID": TEAMS[tm], "TEAM_ABBREVIATION": tm,
                         "TEAM_NAME": tm, "GAME_ID": "G%02d" % gi, "GAME_DATE": d, "MATCHUP": matchup, "WL": "W", "MIN": 32.0,
                         "FGM": 6, "FGA": 12, "FG3M": 2, "FG3A": 5, "FTM": 2, "FTA": 3, "OREB": 1, "DREB": 4, "REB": 5, "AST": 4,
                         "STL": 1, "BLK": 0, "TOV": 2, "PF": 2, "PTS": pts, "PLUS_MINUS": 0, "SEASON_TYPE": "Regular Season",
                         "USG_ASOF": 0.22})
    players = pd.DataFrame(rows, columns=COLS).sort_values(["GAME_DATE", "PLAYER_ID"]).reset_index(drop=True)
    return SeasonBundle("2025-26", players, pd.DataFrame(), {n: "F" for _, n, _ in PLAYERS}, {n: p for p, n, _ in PLAYERS}, {}), str(dates[15].date())

def project(prior_pts, target_pts, future_pts):
    bundle, target_date = build(prior_pts, target_pts, future_pts)
    df, info = eng.run_replay(bundle, ["Regular Season"], 1, {}, ROOT, date_from=target_date, date_to=target_date)
    star = df[df["player"] == "Test Star"].set_index("market")
    return star["projection"].to_dict(), star["actual"].to_dict(), info["counts"]

a, act_a, counts = project(20, 20, 20)
b, act_b, _ = project(20, 55, 0)     # different outcome for the predicted game AND for a later game
c, _, _ = project(30, 20, 20)        # different PRIOR history (sanity: the model does read history)
print(json.dumps({"same": a, "other_future": b, "other_prior": c, "actual_a": act_a, "actual_b": act_b, "counts": counts}))
'''


def test_projection_ignores_the_games_own_outcome_and_later_games_but_reads_prior_history():
    proc = subprocess.run([sys.executable, "-c", SCRIPT, ROOT], capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["counts"]["eligible_player_games"] >= 1
    assert out["same"] == out["other_future"], "projection changed when only the game's own/later outcomes changed"
    assert out["actual_a"]["PTS"] == 20 and out["actual_b"]["PTS"] == 55  # the actuals really did differ
    assert out["same"]["PTS"] != out["other_prior"]["PTS"], "projection did not react to prior history"
    assert len(out["same"]) == 8  # all eight supported markets are projected
