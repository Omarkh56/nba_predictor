"""dashboard.py reads the pipeline's files correctly and only takes orders from its own page."""

import csv
import http.client
import json
import sqlite3
import threading
from datetime import date

import pytest

import dashboard

DAY = "2026-10-21"


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def src(tmp_path):
    db = sqlite3.connect(tmp_path / "predictions.db")
    db.execute(f"CREATE TABLE prop_predictions ({', '.join(dashboard.PROP_COLUMNS)})")
    db.executemany(f"INSERT INTO prop_predictions VALUES ({', '.join('?' * len(dashboard.PROP_COLUMNS))})", [
        (DAY, "ATL@ORL", "Probe Guard", "AST", 5.5, "UNDER", 4.1, 94.6, 49.6, 0.7, -155, "", None,
         "2026-10-21T23:00:00Z", None, None, ""),
        (DAY, "ATL@ORL", "Probe Wing", "REB", 6.5, "OVER", 7.9, 71.0, 12.0, 0.6, -110, "", None,
         "2026-10-21T23:00:00Z", None, None, "2-BOOK"),
        (DAY, "MIL@WAS", "Probe Big", "PTS", 20.5, "OVER", 23.0, 80.0, 10.0, 0.6, -115, "", None,
         "2026-10-21T23:00:00Z", None, None, ""),
    ])
    db.execute("CREATE TABLE game_predictions (date, home_abbr, away_abbr, winner_pick, spread, "
               "spread_pick, win_pct, total, home_prob)")
    db.execute("INSERT INTO game_predictions VALUES (?,?,?,?,?,?,?,?,?)",
               (DAY, "ORL", "ATL", "ATL", 0.2, "ATL", 0.504, 229.6, 0.4955))
    db.commit()
    db.close()
    _write_csv(tmp_path / "bet_log.csv", [
        {"date": DAY, "player": "Probe Guard", "market": "AST", "pick": "UNDER", "actual": "5.0", "result": "HIT", "stake_amount": "", "realized_profit": ""},
        {"date": DAY, "player": "Probe Wing", "market": "REB", "pick": "OVER", "actual": "3.0", "result": "MISS", "stake_amount": "", "realized_profit": ""},
        {"date": "2026-10-10", "player": "Old", "market": "AST", "pick": "OVER", "actual": "9", "result": "HIT", "stake_amount": "10", "realized_profit": "9.1"},
        {"date": "2026-04-18", "player": "Older", "market": "PTS", "pick": "UNDER", "actual": "9", "result": "PUSH", "stake_amount": "", "realized_profit": ""},
    ])
    _write_csv(tmp_path / "game_log.csv", [{"date": DAY, "game": "ATL@ORL", "home_score": "125", "away_score": "132",
                                            "winner_hit": "HIT", "spread_hit": "HIT"}])
    _write_csv(tmp_path / "game_odds_tracker.csv", [
        {"date": DAY, "snapshot_at": "2026-10-21T20:00:00+00:00", "game": "MIL@WAS", "market": "h2h",
         "books_count": "9", "devig_home": "0.40", "devig_away": "0.60"},
        {"date": DAY, "snapshot_at": "2026-10-21T21:30:00+00:00", "game": "MIL@WAS", "market": "h2h",
         "books_count": "9", "devig_home": "0.35", "devig_away": "0.65"},
    ])
    _write_csv(tmp_path / "data" / "injury_snapshots.csv", [
        {"snapshot_at_utc": "2026-10-21T14:00:00+00:00", "slate_date": DAY, "team": "ORL", "player": "Probe Out",
         "status": "Out", "position": "F", "injury_type": "Knee", "injury_detail": "", "return_date": "",
         "short_comment": "", "report_updated_at": ""},
        {"snapshot_at_utc": "2026-10-21T22:00:00+00:00", "slate_date": DAY, "team": "ORL", "player": "Probe Late",
         "status": "Day-To-Day", "position": "G", "injury_type": "Ankle", "injury_detail": "", "return_date": "",
         "short_comment": "", "report_updated_at": ""},
    ])
    _write_csv(tmp_path / "data_cache" / "team_games_202425.csv", [
        {"GAME_ID": "1", "GAME_DATE": "2025-01-05", "TEAM_ABBREVIATION": "ORL", "MATCHUP": "ORL vs. ATL", "WL": "W", "PTS": "110", "PLUS_MINUS": "6.0", "season": "2024-25"},
        {"GAME_ID": "1", "GAME_DATE": "2025-01-05", "TEAM_ABBREVIATION": "ATL", "MATCHUP": "ATL @ ORL", "WL": "L", "PTS": "104", "PLUS_MINUS": "-6.0", "season": "2024-25"},
        {"GAME_ID": "2", "GAME_DATE": "2025-03-01", "TEAM_ABBREVIATION": "ORL", "MATCHUP": "ORL @ ATL", "WL": "L", "PTS": "99", "PLUS_MINUS": "-11.0", "season": "2024-25"},
    ])
    return dashboard.Sources(tmp_path)


SCOREBOARD = {"events": [{
    "date": "2026-10-21T23:00Z", "status": {"type": {"state": "post", "completed": True, "shortDetail": "Final"}},
    "competitions": [{"date": "2026-10-21T23:00Z", "broadcasts": [{"names": ["NBA League Pass"]}],
                      "odds": [{"details": "ORL -2.5", "moneyline": {"home": {"close": {"odds": "-500"}},
                                                                    "away": {"close": {"odds": "+400"}}}}],
                      "competitors": [
                          {"homeAway": "home", "score": "125", "winner": False, "records": [{"type": "total", "summary": "0-1"}],
                           "team": {"abbreviation": "ORL", "displayName": "Orlando Magic", "shortDisplayName": "Magic", "logo": "o.png"}},
                          {"homeAway": "away", "score": "132", "winner": True, "records": [{"type": "total", "summary": "1-0"}],
                           "team": {"abbreviation": "ATL", "displayName": "Atlanta Hawks", "shortDisplayName": "Hawks", "logo": "a.png"}},
                      ]}],
}, {
    "date": "2026-10-21T23:30Z", "status": {"type": {"state": "pre", "shortDetail": "7:30 PM"}},
    "competitions": [{"competitors": [
        {"homeAway": "home", "score": "0", "team": {"abbreviation": "WSH", "displayName": "Washington Wizards"}},
        {"homeAway": "away", "score": "0", "team": {"abbreviation": "MIL", "displayName": "Milwaukee Bucks"}},
    ]}],
}]}


@pytest.fixture
def live(src):
    history = dashboard.TeamHistory(src.team_logs, fetch_current=lambda season: [
        {"GAME_ID": "3", "GAME_DATE": DAY, "TEAM_ABBREVIATION": "ORL", "MATCHUP": "ORL vs. ATL",
         "WL": "L", "PTS": "125", "PLUS_MINUS": "-7.0"}])
    return dashboard.Live(src, scoreboard=lambda day: dashboard.parse_scoreboard(SCOREBOARD), history=history)


def test_parse_scoreboard_uses_nba_codes_and_hides_pregame_scores():
    finished, upcoming = dashboard.parse_scoreboard(SCOREBOARD)
    assert finished["key"] == "ATL@ORL" and finished["away"]["score"] == "132"
    assert finished["start_utc"] == "2026-10-21T23:00:00+00:00" and finished["line"] == "ORL -2.5"
    assert upcoming["key"] == "MIL@WAS" and upcoming["home"]["score"] is None  # ESPN's WSH -> WAS
    assert "espn_moneyline" not in finished  # win % only ever comes from the pipeline's files


def test_games_take_win_probability_from_the_pipeline_files_only(src, live):
    games = {g["key"]: g for g in dashboard.games_for(src, DAY, live)["games"]}
    atl = games["ATL@ORL"]
    assert atl["win"] == atl["model"] and atl["win"]["source"] == "Model"
    assert (atl["win"]["home_prob"], atl["win"]["away_prob"]) == (0.4955, 0.5045)
    mil = games["MIL@WAS"]  # no model row: the newest odds-tracker snapshot, never ESPN's moneyline
    assert mil["model"] is None and mil["win"]["source"] == "Market (9 books)"
    assert (mil["win"]["home_prob"], mil["win"]["away_prob"]) == (0.35, 0.65)
    assert atl["props"]["count"] == 2 and atl["props"]["best"]["player"] == "Probe Guard"
    assert atl["props"]["record"]["hits"] == 1 and atl["final"]["winner_hit"] == "HIT"


def test_head_to_head_is_as_of_the_game_and_injuries_are_per_team(src, live):
    detail = dashboard.game_detail(src, DAY, "ATL@ORL", live)
    assert [m["date"] for m in detail["h2h"]] == ["2025-03-01", "2025-01-05"]  # this season's game is the game itself
    assert detail["h2h"][1] == {"date": "2025-01-05", "season": "2024-25", "home": "ORL", "away": "ATL",
                                "home_pts": 110, "away_pts": 104, "winner": "ORL"}
    assert detail["h2h"][0]["winner"] == "ATL" and detail["h2h"][0]["home"] == "ATL"
    assert [p["player"] for p in detail["injuries"]["home"]] == ["Probe Late"]  # the slate's last snapshot
    assert detail["injuries"]["away"] == []
    assert [p["player"] for p in detail["props"]] == ["Probe Guard", "Probe Wing"]  # by confidence
    later = live.history.meetings("ORL", "ATL", before="2026-10-22")
    assert later[0]["date"] == DAY  # this season's meetings come from the current-season fetch
    with pytest.raises(KeyError):
        dashboard.game_detail(src, DAY, "BOS@NYK", live)


def test_performance_ranges_and_paper_pnl(src):
    today = date(2026, 10, 23)
    season = dashboard.performance(src, "season", today)
    assert season["overall"] == {"hits": 2, "misses": 1, "pushes": 0, "graded": 3, "rate": 0.6667}
    assert [d["date"] for d in season["by_day"]] == ["2026-10-10", DAY]
    assert season["by_market"][0]["market"] == "AST" and season["by_market"][0]["graded"] == 2
    assert season["paper"] == {"staked": 10.0, "profit": 9.1, "roi": 0.91}
    assert dashboard.performance(src, "7d", today)["overall"]["graded"] == 2
    assert dashboard.performance(src, "all", today)["overall"]["pushes"] == 1


def test_job_runner_runs_a_script_and_refuses_a_second_copy(tmp_path, monkeypatch):
    (tmp_path / "job.py").write_text("import time, sys; print('job ran', sys.argv[1:]); time.sleep(0.5)\n")
    monkeypatch.setattr(dashboard, "JOBS", {"echo": ("job.py", ("date",))})
    runner = dashboard.JobRunner(tmp_path, tmp_path / "logs")
    assert runner.start("echo", {"date": "2026-10-21"})["state"] == "running"
    with pytest.raises(RuntimeError):
        runner.start("echo", {})
    runner.jobs["echo"]["proc"].wait(timeout=10)
    status = runner.status()["echo"]
    assert status["exit_code"] == 0 and "job ran ['2026-10-21']" in status["log_tail"]
    with pytest.raises(ValueError):
        runner.start("echo", {"date": "not-a-date"})
    with pytest.raises(KeyError):
        runner.start("rm", {})


@pytest.fixture
def server(src, live):
    srv = dashboard.serve(port=0, src=src, live=live)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def _request(port, method, path, headers=None, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(method, path, body=body, headers={"Host": f"127.0.0.1:{port}", **(headers or {})})
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, data


def test_http_serves_the_page_and_api_on_localhost_only(server):
    status, page = _request(server, "GET", "/")
    assert status == 200 and b"NBA Props Tracker" in page
    status, body = _request(server, "GET", f"/api/games?date={DAY}")
    assert status == 200 and len(json.loads(body)["games"]) == 2
    assert _request(server, "GET", f"/api/game?date={DAY}&key=XXX@YYY")[0] == 404
    assert _request(server, "GET", "/api/games?date=nope")[0] == 400
    assert _request(server, "GET", "/api/overview", {"Host": "evil.example"})[0] == 403


def test_http_runs_only_start_from_the_page(server):
    json_headers = {"Content-Type": "application/json", "X-NBA-Dashboard": "1"}
    assert _request(server, "POST", "/api/run/injuries", {"Content-Type": "application/json"}, b"{}")[0] == 403
    assert _request(server, "POST", "/api/run/injuries",
                    {**json_headers, "Origin": "https://evil.example"}, b"{}")[0] == 403
    assert _request(server, "POST", "/api/run/injuries", {"X-NBA-Dashboard": "1"}, b"{}")[0] == 403
    assert _request(server, "POST", "/api/run/rm", json_headers, b"{}")[0] == 404
    assert _request(server, "POST", "/api/run/grade", json_headers, b'{"date": "x"}')[0] == 400
