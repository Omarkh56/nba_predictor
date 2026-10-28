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


def test_top_bets_rank_the_slate_by_probability_confidence_or_edge(src):
    by_prob = dashboard.top_bets(src, DAY, "prob")
    assert by_prob["count"] == 3
    assert [b["player"] for b in by_prob["bets"]] == ["Probe Guard", "Probe Big", "Probe Wing"]  # 0.7, then 0.6s by conf
    assert [b["player"] for b in dashboard.top_bets(src, DAY, "conf")["bets"]] == ["Probe Guard", "Probe Big", "Probe Wing"]
    assert [b["player"] for b in dashboard.top_bets(src, DAY, "edge", limit=1)["bets"]] == ["Probe Guard"]
    assert by_prob["bets"][0]["result"] == "HIT"


def test_top_bets_leave_out_picks_the_model_flags_as_likely_errors(src):
    db = sqlite3.connect(src.db)
    db.execute(f"INSERT INTO prop_predictions VALUES ({', '.join('?' * len(dashboard.PROP_COLUMNS))})",
               (DAY, "ATL@ORL", "Probe Short Minutes", "PTS", 13.5, "UNDER", 2.2, 62.8, 83.0, 0.91, -110, "",
                None, "2026-10-21T23:00:00Z", None, None, "⚠SUSP-83% ⚠BENCH"))
    db.commit()
    db.close()
    default = dashboard.top_bets(src, DAY, "prob")
    assert "Probe Short Minutes" not in [b["player"] for b in default["bets"]]
    assert (default["count"], default["flagged"]) == (4, 1)
    shown = dashboard.top_bets(src, DAY, "prob", include_flagged=True)
    assert shown["bets"][0]["player"] == "Probe Short Minutes"  # highest probability once allowed


def test_percentile_ranks_share_ties_and_weighted_ranking_reduces_to_one_metric():
    assert dashboard._pct_ranks([3, 1, 2, 2]) == [1.0, 0.0, 0.5, 0.5]
    picks = [{"player": n, "p_model": p, "confidence": c, "edge_pct": e, "flags": ""}
             for n, p, c, e in (("a", 0.9, 10, 5), ("b", 0.5, 90, 1), ("c", 0.7, 50, 9))]
    def order(by, w=None):
        return [p["player"] for p in dashboard.rank_slate(picks, by, w)]

    assert order("weighted", (1, 0, 0)) == order("prob") == ["a", "c", "b"]
    assert order("weighted", (0, 1, 0)) == order("conf") == ["b", "c", "a"]
    assert order("weighted", (0, 0, 1)) == ["c", "a", "b"]


def _ranking_src(tmp_path):
    """Two slates where confidence picks the winners and probability is noise, plus an
    old slate with no stored probability (like last season's)."""
    db = sqlite3.connect(tmp_path / "predictions.db")
    db.execute(f"CREATE TABLE prop_predictions ({', '.join(dashboard.PROP_COLUMNS)})")
    db.execute("CREATE TABLE game_predictions (date, home_abbr, away_abbr, winner_pick, spread, "
               "spread_pick, win_pct, total, home_prob)")
    rows, results = [], []
    for day in ("2026-10-21", "2026-10-23"):
        for i in range(20):
            hit = i < 10  # the 10 most confident picks win; probability is unrelated
            rows.append((day, "A@B", f"P{i}", "PTS", 10.5, "OVER", 11.0, 100 - i, float((3 * i) % 11),
                         0.55 + ((7 * i) % 20) / 100, -110, "", None, None, None, None, ""))
            results.append({"date": day, "player": f"P{i}", "market": "PTS", "pick": "OVER",
                            "actual": "1", "result": "HIT" if hit else "MISS"})
    rows.append(("2026-04-18", "A@B", "Old", "PTS", 10.5, "OVER", 11.0, 80, 5.0, None, -110, "",
                 None, None, None, None, ""))
    results.append({"date": "2026-04-18", "player": "Old", "market": "PTS", "pick": "OVER", "actual": "1", "result": "HIT"})
    db.executemany(f"INSERT INTO prop_predictions VALUES ({', '.join('?' * len(dashboard.PROP_COLUMNS))})", rows)
    db.commit()
    db.close()
    _write_csv(tmp_path / "bet_log.csv", results)
    return dashboard.Sources(tmp_path)


def test_fitted_weights_find_the_ranking_that_wins_and_report_out_of_sample(tmp_path):
    src = _ranking_src(tmp_path)
    report = dashboard.ranking_report(src, "all", date(2026, 10, 24))
    m = report["methods"]
    assert (m["conf"]["hits"], m["conf"]["misses"], m["conf"]["slates"]) == (21, 10, 3)  # 10-5 twice + old slate
    assert (m["prob"]["hits"], m["prob"]["misses"], m["prob"]["slates"]) == (12, 18, 2)  # old slate has none
    weights = report["weights"]
    assert weights["conf"] == max(weights.values())  # ties go to the most even mix that does best
    assert (m["weighted"]["hits"], m["weighted"]["misses"], m["weighted"]["slates"]) == (20, 10, 2)
    assert m["weighted_oos"]["slates"] == 2 and m["weighted_oos"]["graded"] == 30
    fitted = dashboard.top_bets(src, "2026-10-23", "weighted")
    assert fitted["fitted_on"] == 2
    assert {f"P{i}" for i in range(10)} <= {b["player"] for b in fitted["bets"]}  # every winner kept
    mine = dashboard.top_bets(src, "2026-10-23", "weighted", weights=(1, 0, 0))
    assert mine["fitted_on"] is None and mine["bets"][0]["player"] == "P17"  # highest probability


def _naive_fit(slates: dict) -> tuple:
    """The first version: rank every slate again for every candidate weight, on every refit."""
    def fit(group):
        best = None
        for w in dashboard.WEIGHT_GRID:
            r = dashboard._record([p for ps in group.values()
                                   for p in dashboard.rank_slate(ps, "weighted", w)[:dashboard.TOP_N] if p["result"]])
            key = (r["hits"] - r["misses"], -max(w))
            if best is None or key > best[0]:
                best = (key, w, r)
        return best[1], best[2]

    weights, record = fit(slates)
    held_out = []
    for d in slates:
        w, _ = fit({k: v for k, v in slates.items() if k != d})
        held_out += [p for p in dashboard.rank_slate(slates[d], "weighted", w)[:dashboard.TOP_N] if p["result"]]
    return weights, record, dashboard._record(held_out)


def test_one_pass_fit_matches_refitting_every_slate(monkeypatch):
    """The leave-one-out used to refit from scratch for every slate (3.6 s at 20 slates,
    growing with the square); sums of one table must pick the same weights and records."""
    import random

    rng = random.Random(7)
    results = ["HIT", "MISS", "PUSH", None]
    slates = {f"2026-11-{d:02d}": [
        {"p_model": round(rng.uniform(0.5, 0.8), 2), "confidence": float(rng.randint(40, 95)),
         "edge_pct": float(rng.randint(0, 45)), "flags": "⚠SUSP" if rng.random() < 0.05 else "",
         "result": rng.choices(results, weights=(5, 5, 1, 1))[0]} for _ in range(rng.randint(20, 60))]
        for d in range(1, 9)}
    weights, in_sample, held_out, n = dashboard._weighted_fit(slates)
    naive_weights, naive_record, naive_held_out = _naive_fit(slates)
    assert n == 8 and weights == naive_weights
    assert dashboard._record_of(*in_sample) == naive_record
    assert dashboard._record_of(*held_out) == naive_held_out
    assert dashboard.fit_weights(slates)["weights"] == naive_weights


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
    assert _request(server, "GET", "/api/ranking?range=all")[0] == 200
    status, body = _request(server, "GET", f"/api/top?date={DAY}&by=weighted&w=2,1,1")
    assert status == 200 and json.loads(body)["weights"] == {"prob": 0.5, "conf": 0.25, "edge": 0.25}
    assert _request(server, "GET", f"/api/top?date={DAY}&by=weighted&w=1,2")[0] == 400
    assert _request(server, "GET", "/api/overview", {"Host": "evil.example"})[0] == 403


def test_http_runs_only_start_from_the_page(server):
    json_headers = {"Content-Type": "application/json", "X-NBA-Dashboard": "1"}
    assert _request(server, "POST", "/api/run/injuries", {"Content-Type": "application/json"}, b"{}")[0] == 403
    assert _request(server, "POST", "/api/run/injuries",
                    {**json_headers, "Origin": "https://evil.example"}, b"{}")[0] == 403
    assert _request(server, "POST", "/api/run/injuries", {"X-NBA-Dashboard": "1"}, b"{}")[0] == 403
    assert _request(server, "POST", "/api/run/rm", json_headers, b"{}")[0] == 404
    assert _request(server, "POST", "/api/run/grade", json_headers, b'{"date": "x"}')[0] == 400
