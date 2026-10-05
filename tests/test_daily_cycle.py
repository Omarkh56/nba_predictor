"""Offline paper cycle: all state lives in tmp_path; all HTTP is forbidden."""
import csv
import json
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import requests

import predictions_db as db
from clv import closing_for_bet
from run_log import RunLog, read_runs

DAY = date(2026, 10, 21)
ENTRY_TIME = datetime(2026, 10, 21, 22, 45, tzinfo=timezone.utc)
CLOSE_TIME = datetime(2026, 10, 21, 22, 50, tzinfo=timezone.utc)
EVENTS = [
    {"id": "g1", "home_team": "Denver Nuggets", "away_team": "Utah Jazz",
     "commence_time": "2026-10-21T23:00:00Z"},
    {"id": "g2", "home_team": "Los Angeles Clippers", "away_team": "Golden State Warriors",
     "commence_time": "2026-10-21T23:10:00Z"},
]

def proposal(player="P1", event="g1", **updates):
    return {"game": "UTA@DEN", "player": player, "market": "PTS", "line": 25.5,
            "pick": "OVER", "projection": 29.0, "confidence": 65.0, "p_model": .65,
            "model_version": "test-model-v1", "bet_id": "first-bet", "logged_at": ENTRY_TIME.isoformat(),
            "entry_decimal_odds": 2.0, "decimal_odds": 2.0, "entry_american_odds": 100,
            "odds_snapshot_at": ENTRY_TIME.isoformat(), "odds_event_id": event,
            "odds_source": "the_odds_api", "game_start_at": EVENTS[0]["commence_time"],
            "stake_pct": .02, "stake_amount": 2.0, **updates}

@pytest.fixture(autouse=True)
def no_http(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("Offline test attempted HTTP")
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)

@pytest.fixture
def modules(tmp_path, monkeypatch):
    path = tmp_path / "predictions.db"
    load_props, load_games = db.load_prop_predictions, db.load_game_predictions
    write_props, write_games = db.upsert_prop_predictions, db.upsert_game_predictions
    monkeypatch.setattr(db, "DB_PATH", path)
    monkeypatch.setattr(db, "load_prop_predictions", lambda d, **k: load_props(d, path=k.get("path", path)))
    monkeypatch.setattr(db, "load_game_predictions", lambda d, **k: load_games(d, path=k.get("path", path)))
    monkeypatch.setattr(db, "upsert_prop_predictions", lambda r, d, **k: write_props(r, d, path=path, **k))
    monkeypatch.setattr(db, "upsert_game_predictions", lambda r, d: write_games(r, d, path=path))
    monkeypatch.setattr(sys, "argv", ["checkresults.py"])
    monkeypatch.setenv("NBA_RUN_LOG", str(tmp_path / "runs.jsonl"))
    monkeypatch.setenv("NBA_STARTING_BANKROLL", "100")
    import oddstracker as ot
    import nba_combined as nc
    import checkresults as cr
    import calibrate as cal
    monkeypatch.setattr(nc.props_model, "LAST_PROPS_STATUS", {})
    for name, filename in [("TRACKER_FILE", "odds_tracker.csv"), ("SNAPSHOT_FILE", "prop_odds_snapshots.csv"),
                           ("BET_LOG_FILE", "bet_log.csv"), ("CALIB_FILE", "calibration.json")]:
        monkeypatch.setattr(ot, name, str(tmp_path / filename))
    monkeypatch.setattr(ot.time, "sleep", lambda _: None)
    monkeypatch.setattr(ot, "LAST_SNAPSHOT_STATUS", {})
    monkeypatch.setattr(cr, "LOG_FILE", str(tmp_path / "bet_log.csv"))
    monkeypatch.setattr(cr, "GAME_LOG_FILE", str(tmp_path / "game_log.csv"))
    monkeypatch.setattr(cal, "LOG_FILE", tmp_path / "bet_log.csv")
    monkeypatch.setattr(cal, "CALIB_FILE", tmp_path / "calibration.json")
    return ot, nc, cr, cal, path

def test_first_priced_entry_freezes_whole_pick_and_each_run_appends(tmp_path):
    path = tmp_path / "p.db"
    first = proposal()
    db.upsert_prop_predictions([first], DAY, path=path, run_id="entry-run")
    frozen = db.load_prop_predictions(DAY, path=path)
    rerun = proposal(line=26.5, pick="UNDER", entry_decimal_odds=1.8,
                     bet_id="changed", logged_at=CLOSE_TIME.isoformat(), model_version="v2", stake_amount=5)
    db.upsert_prop_predictions([rerun], DAY, path=path, run_id="later-run")
    assert db.load_prop_predictions(DAY, path=path) == frozen
    with db._connect(path) as conn:
        rows = conn.execute("SELECT run_id,line,pick,entry_decimal_odds,model_version,paper_stake "
                            "FROM pick_runs ORDER BY rowid").fetchall()
        assert rows == [("entry-run", 25.5, "OVER", 2.0, "test-model-v1", 2.0),
                        ("later-run", 26.5, "UNDER", 1.8, "v2", 5.0)]
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE pick_runs SET line=0")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM pick_runs")
    db.upsert_prop_predictions([proposal(line=99)], DAY, path=path, run_id="later-run")
    with db._connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM pick_runs").fetchone()[0] == 2

def test_unpriced_entry_can_gain_first_price_then_stays_fixed(tmp_path):
    path = tmp_path / "p.db"
    db.upsert_prop_predictions([proposal(entry_decimal_odds=None, decimal_odds=None)], DAY,
                              path=path, run_id="unpriced")
    db.upsert_prop_predictions([proposal(line=24.5)], DAY, path=path, run_id="priced")
    db.upsert_prop_predictions([proposal(line=99)], DAY, path=path, run_id="later")
    row = db.load_prop_predictions(DAY, path=path)[0]
    assert row["line"] == 24.5 and row["run_id"] == "priced"

def test_retry_of_unpriced_run_cannot_rewrite_its_audit_record(tmp_path):
    path = tmp_path / "p.db"
    row = proposal(entry_decimal_odds=None, decimal_odds=None)
    db.upsert_prop_predictions([row], DAY, path=path, run_id="same-run")
    before = db.load_prop_predictions(DAY, path=path)
    db.upsert_prop_predictions([{**row, "line": 99}], DAY, path=path, run_id="same-run")
    assert db.load_prop_predictions(DAY, path=path) == before

def test_direct_prediction_caller_gets_runlog_provenance(tmp_path):
    path = tmp_path / "p.db"
    db.upsert_prop_predictions([proposal()], DAY, path=path)
    row = db.load_prop_predictions(DAY, path=path)[0]
    assert row["run_id"] == read_runs(tmp_path / "run_log.jsonl")[0]["run_id"]

def props(event_id, *, price=-110):
    player = "P1" if event_id == "g1" else "P2"
    return {"id": event_id, "bookmakers": [{"key": "book", "title": "Book", "markets": [
        {"key": "player_points", "outcomes": [
            {"name": "Over", "description": player, "point": 25.5, "price": price},
            {"name": "Under", "description": player, "point": 25.5, "price": -110},
        ]}]}]}

def test_closing_only_quotes_picked_events_in_window_and_preserves_consensus(modules, tmp_path, monkeypatch):
    ot, nc, cr, cal, path = modules
    picks = [proposal(), proposal(player="P2"),
             proposal(event="late", game_start_at="2026-10-21T23:30:01Z"),
             proposal(event="started", game_start_at="2026-10-21T22:50:00Z"),
             proposal(event="other-date", game_start_at="2026-10-22T23:00:00Z"),
             proposal(event="boundary", game_start_at="2026-10-21T23:20:00Z")]
    calls = []
    monkeypatch.setattr(ot, "get_event_props", lambda e: calls.append(e) or props(e))
    Path(ot.TRACKER_FILE).write_bytes(b"existing consensus\n")
    assert ot.closing_snapshot(DAY, predictions=picks, now=CLOSE_TIME) == 0
    assert calls == ["g1", "boundary"]
    assert Path(ot.TRACKER_FILE).read_bytes() == b"existing consensus\n"
    before = Path(ot.SNAPSHOT_FILE).read_bytes()
    assert ot.closing_snapshot(DAY, predictions=picks, now=CLOSE_TIME) == 0
    assert Path(ot.SNAPSHOT_FILE).read_bytes() == before
    assert set(pd.read_csv(ot.SNAPSHOT_FILE).snapshot_kind) == {"closing"}

@pytest.mark.parametrize("response,code,status", [("empty", 0, "market_unavailable"),
                                                 ("failed", 2, "fetch_failed")])
def test_closing_statuses_and_no_eligible_events(modules, tmp_path, monkeypatch, response, code, status):
    ot, *_ = modules
    def fetch(e):
        if response == "failed":
            raise requests.ConnectionError("unreachable")
        return {"bookmakers": []}
    monkeypatch.setattr(ot, "get_event_props", fetch)
    assert ot.closing_snapshot(DAY, predictions=[proposal()], now=CLOSE_TIME) == code
    assert read_runs(tmp_path / "runs.jsonl")[-1]["stages"][-1]["status"] == status
    monkeypatch.setattr(ot, "get_event_props", lambda e: pytest.fail("No eligible events means no API calls"))
    assert ot.closing_snapshot(DAY, predictions=[], now=CLOSE_TIME) == 0
    assert read_runs(tmp_path / "runs.jsonl")[-1]["stages"][0]["status"] == "skipped"

def test_entry_quote_alone_is_not_a_closing_snapshot():
    quote = {"snapshot_at": ENTRY_TIME.isoformat(), "event_id": "g1", "player": "P1", "market": "PTS",
             "side": "OVER", "line": 25.5, "american_odds": -110, "snapshot_kind": "entry"}
    assert closing_for_bet(proposal(date=str(DAY)), pd.DataFrame([quote]))["clv_unmatched_reason"] == "no_closing_snapshot"

def test_closing_cli_forwards_date_and_window(modules, monkeypatch):
    ot, *_ = modules
    calls = []
    monkeypatch.setattr(ot, "closing_snapshot", lambda d, **k: calls.append((d, k)) or 0)
    assert ot.main(["--closing", str(DAY), "--minutes", "45"]) == 0
    assert calls == [(DAY, {"minutes": 45})]
    with pytest.raises(SystemExit) as exc:
        ot.main(["--closing", str(DAY), "--minutes", "0"])
    assert exc.value.code == 2

def test_closing_partially_failed_slate_preserves_successful_quotes(modules, tmp_path, monkeypatch):
    ot, *_ = modules
    def fetch(e):
        if e == "g2":
            raise requests.ConnectionError("unreachable")
        return props(e)
    monkeypatch.setattr(ot, "get_event_props", fetch)
    picks = [proposal(), proposal(player="P2", event="g2", game_start_at=EVENTS[1]["commence_time"])]
    assert ot.closing_snapshot(DAY, predictions=picks, now=CLOSE_TIME) == 3
    assert set(pd.read_csv(ot.SNAPSHOT_FILE).event_id) == {"g1"}
    assert read_runs(tmp_path / "runs.jsonl")[-1]["status"] == "partial"

def test_slow_closing_response_is_not_backdated_before_tip(modules, tmp_path, monkeypatch):
    ot, *_ = modules
    times = iter([CLOSE_TIME, datetime(2026, 10, 21, 23, 0, 1, tzinfo=timezone.utc)])
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return next(times)
    monkeypatch.setattr(ot, "datetime", Clock)
    monkeypatch.setattr(ot, "get_event_props", props)
    assert ot.closing_snapshot(DAY, predictions=[proposal()]) == 0
    assert not Path(ot.SNAPSHOT_FILE).exists()
    assert read_runs(tmp_path / "runs.jsonl")[-1]["stages"][-1]["status"] == "skipped"

def test_legacy_snapshot_header_expands_without_losing_rows(modules):
    ot, *_ = modules
    fields = [f for f in ot.SNAPSHOT_FIELDS if f != "snapshot_kind"]
    old = {"snapshot_at": ENTRY_TIME.isoformat(), "date": str(DAY), "event_id": "g1",
           "commence_time": EVENTS[0]["commence_time"], "player": "P1", "market": "PTS",
           "book_key": "book", "line": 25.5, "side": "OVER", "american_odds": -110}
    with open(ot.SNAPSHOT_FILE, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(old)
    new = {**old, "snapshot_at": CLOSE_TIME.isoformat(), "snapshot_kind": "closing"}
    assert ot._append_snapshot_rows([new]) == 1
    rows = list(csv.DictReader(open(ot.SNAPSHOT_FILE)))
    assert len(rows) == 2
    assert rows[0]["snapshot_kind"] == "" and rows[0]["american_odds"] == "-110"
    assert rows[1]["snapshot_kind"] == "closing"
    assert ot._append_snapshot_rows([new]) == 0

@pytest.mark.parametrize("collect_closing", [True, False])
def test_two_game_daily_cycle_is_traceable_and_byte_identical_on_rerun(modules, tmp_path, monkeypatch, collect_closing):
    ot, nc, cr, cal, path = modules
    monkeypatch.setattr(ot, "get_event_props", props)
    entry = ot.snapshot(DAY, events=EVENTS, now=ENTRY_TIME)
    assert {r["game"] for r in entry} == {"UTA@DEN", "GSW@LAC"}
    monkeypatch.setattr(nc, "_TODAY_LEAN", {(r["player"].lower(), r["market"]): r for r in entry})
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return ENTRY_TIME
    monkeypatch.setattr(nc, "datetime", Clock)
    monkeypatch.setattr(nc, "_init_calibration", lambda: None)
    matchups = [{"home_abbr": "DEN", "away_abbr": "UTA", "game_date": DAY},
                {"home_abbr": "LAC", "away_abbr": "GSW", "game_date": DAY}]
    monkeypatch.setattr(nc, "_fetch_matchups_and_lean", lambda *a, **k: (matchups, DAY))
    monkeypatch.setattr(nc, "_fetch_injuries", lambda: ({}, {}))
    team_preds = [{**m, "favorite": m["home_abbr"], "fav_prob": .65, "home_prob": .65,
                   "expected_margin": 2.0, "home_spread": -2.0, "away_spread": 2.0,
                   "predicted_total": 220.0} for m in matchups]
    monkeypatch.setattr(nc, "_run_team_model", lambda *a: (team_preds, {}))
    monkeypatch.setattr(nc, "_setup_odds_and_spreads", lambda *a, **k: EVENTS)
    for method in ("get_team_season_stats", "get_league_logs", "get_all_player_usage", "prefetch_rosters_for_games"):
        monkeypatch.setattr(nc.props_model, method, lambda *a: None)
    next_proposal = {"line": 25.5, "pick": "OVER", "version": "offline-v1"}
    def process(matchup, team_map, events, injuries, exports):
        player = "P1" if matchup["home_abbr"] == "DEN" else "P2"
        row = pd.Series({"Player": player, "Market": "PTS", "Line": next_proposal["line"], "Pick": next_proposal["pick"],
                         "Projection": 29.0, "Confidence": 65.0, "P_Model": .65, "Edge%": 13.7,
                         "Flags": "", "meta": {"model_version": next_proposal["version"]}})
        exports.append(nc._pick_to_dict(row, matchup["away_abbr"]+"@"+matchup["home_abbr"], None, DAY))
        nc.props_model._set_props_status(EVENTS[0 if player == "P1" else 1], "ok", lines=1)
        return [], pd.DataFrame()
    monkeypatch.setattr(nc, "_process_single_game", process)
    assert nc.main(["--date", str(DAY)]) == 0
    prediction_run = read_runs(tmp_path / "runs.jsonl")[-1]["run_id"]
    frozen = db.load_prop_predictions(DAY)
    assert {r["run_id"] for r in frozen} == {prediction_run}
    monkeypatch.setattr(ot, "get_event_props", lambda e: props(e, price=-120))
    if collect_closing:
        assert ot.closing_snapshot(DAY, db_path=path, now=CLOSE_TIME) == 0
    actual = {1: {"PTS": 28, "MIN": 30}, 2: {"PTS": 20, "MIN": 30}}
    monkeypatch.setattr(cr, "get_player_id", lambda p: 1 if p == "P1" else 2)
    monkeypatch.setattr(cr, "build_actual_stats", lambda d: actual)
    game_results = {(m["home_abbr"], m["away_abbr"]): {"home": m["home_abbr"], "away": m["away_abbr"],
                    "home_score": 110, "away_score": 100, "margin": 10, "winner": m["home_abbr"]}
                    for m in matchups}
    monkeypatch.setattr(cr, "build_game_results", lambda d: game_results)
    assert cr.main(DAY) == 0
    cal.calibrate()
    assert json.loads((tmp_path / "calibration.json").read_text())["total_bets"] == 2
    bets = list(csv.DictReader((tmp_path / "bet_log.csv").open()))
    quotes = pd.read_csv(ot.SNAPSHOT_FILE)
    for bet in bets:
        assert bet["run_id"] == prediction_run and bet["model_version"] == "offline-v1"
        matching = quotes[(quotes.event_id == bet["odds_event_id"]) & (quotes.player == bet["player"])]
        assert pd.to_datetime(bet["odds_snapshot_at"], utc=True) in set(pd.to_datetime(matching.snapshot_at, utc=True))
        if collect_closing:
            assert bet["closing_snapshot_at"] and bet["clv_unmatched_reason"] == ""
        else:
            assert not bet["closing_snapshot_at"] and bet["clv_unmatched_reason"] == "no_closing_snapshot"
    assert json.loads((tmp_path / "reports/clv_summary.json").read_text())["closing_snapshot_rows"] == (2 if collect_closing else 0)
    assert json.loads((tmp_path / "reports/finance_summary.json").read_text())["finance_eligible_rows"] == 2
    csv_before = {name: (tmp_path / name).read_bytes() for name in ("bet_log.csv", "game_log.csv")}
    # A later prediction proposes different prices, line, direction and model.
    next_proposal.update(line=29.5, pick="UNDER", version="offline-v2")
    monkeypatch.setattr(nc, "_TODAY_LEAN", {(r["player"].lower(), r["market"]):
                       {**r, "line": 29.5, "avg_under_odds": -130} for r in entry})
    assert nc.main(["--date", str(DAY)]) == 0
    assert db.load_prop_predictions(DAY) == frozen
    assert cr.main(DAY) == 0
    for name, before in csv_before.items():
        assert (tmp_path / name).read_bytes() == before
    with db._connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM pick_runs").fetchone()[0] == 4
        for bet in bets:
            assert conn.execute("SELECT odds_event_id,odds_snapshot_at FROM pick_runs "
                                "WHERE run_id=? AND date=? AND player=? AND market=?",
                                (bet["run_id"], bet["date"], bet["player"], bet["market"])).fetchone() == (
                                    bet["odds_event_id"], bet["odds_snapshot_at"])
