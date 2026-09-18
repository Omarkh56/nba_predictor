"""Paired NBA moneyline tracking, quote validation, joins and persistence."""

import csv
import sqlite3
from datetime import date, datetime, timedelta, timezone
from unittest import mock

import pytest
import requests

import game_odds_tracker as tracker
import nba_combined as combined
import predictions_db as db

NOW = datetime(2026, 1, 15, 18, tzinfo=timezone.utc)
DAY = date(2026, 1, 15)
EVENT = {"id": "bos-lal", "home_team": "Boston Celtics", "away_team": "Los Angeles Lakers",
         "commence_time": "2026-01-16T00:30:00Z"}


def payload(home=-400, away=300):
    return {**EVENT, "bookmakers": [
        {"key": "book-a", "markets": [{"key": "h2h", "outcomes": [
            {"name": EVENT["away_team"], "price": away},
            {"name": EVENT["home_team"], "price": home},
        ]}]},
    ]}


def quote(home=-400, away=300, observed=NOW):
    return tracker.parse_game_odds_to_row(payload(home, away), EVENT, observed)


def test_heavy_favorite_and_eastern_date():
    row = quote()
    assert row["date"] == DAY.isoformat()  # UTC game starts the following day.
    assert row["game"] == "LAL@BOS"
    assert row["books_count"] == 1
    assert row["devig_home"] > 0.75
    assert row["devig_home"] + row["devig_away"] == pytest.approx(1, abs=0.0001)
    assert row["vig_pct"] == 5
    assert row["book_lean"] == "HOME"
    assert tracker.usable_snapshot(row, DAY, NOW)


def test_away_favorite_even_money_and_la_aliases():
    row = quote(300, -400)
    assert row["devig_away"] > 0.75
    assert row["book_lean"] == "AWAY"
    assert quote(100, -100)["devig_home"] == 0.5
    assert tracker.normalize_team("Los Angeles Clippers") == "LAC"
    assert tracker.normalize_team("Los Angeles Lakers") == "LAL"
    assert tracker.normalize_team("GS") == "GSW"
    assert tracker.normalize_team("NY") == "NYK"
    assert tracker.normalize_team("unknown") == ""


def test_only_paired_unique_books_count_and_mixed_sign_prices_stay_valid():
    data = payload(-105, -115)
    data["bookmakers"].append(payload(105, -125)["bookmakers"][0] | {"key": "book-b"})
    data["bookmakers"].append(data["bookmakers"][0])  # Duplicate doesn't increase count.
    data["bookmakers"].append({"key": "incomplete", "markets": [{"key": "h2h", "outcomes": [
        {"name": EVENT["home_team"], "price": -500},
    ]}]})
    data["bookmakers"].append({"key": "props-only", "markets": [{"key": "player_points"}]})
    row = tracker.parse_game_odds_to_row(data, EVENT, NOW)
    assert row["books_count"] == 2
    assert abs(row["avg_home_odds"]) >= 100  # Arithmetic American mean would be zero.
    assert row["home_decimal_odds"] == pytest.approx((1 + 100 / 105 + 2.05) / 2)
    assert tracker.usable_snapshot(row, DAY, NOW)


def test_consensus_averages_fair_book_probabilities_even_when_books_disagree():
    data = payload()
    data["bookmakers"].append(payload(300, -400)["bookmakers"][0] | {"key": "book-b"})
    row = tracker.parse_game_odds_to_row(data, EVENT, NOW)
    assert row["devig_home"] == pytest.approx(0.5)
    assert row["devig_away"] == pytest.approx(0.5)
    assert tracker.usable_snapshot(row, DAY, NOW)


@pytest.mark.parametrize("home,away", [(0, 100), (-2, 100), (float("nan"), 100),
                                       (float("inf"), 100), (300, 300)])
def test_bad_prices_and_underround_are_rejected(home, away):
    assert tracker.parse_game_odds_to_row(payload(home, away), EVENT, NOW) == {}


@pytest.mark.parametrize("change", [{"id": "another-game"}, {"home_team": EVENT["away_team"]}])
def test_response_identity_must_match_event(change):
    assert tracker.parse_game_odds_to_row(payload() | change, EVENT, NOW) == {}


@pytest.mark.parametrize("change", [
    {"market": "spreads"}, {"books_count": 0}, {"devig_home": float("nan")},
    {"devig_away": 0.5}, {"home_decimal_odds": 100}, {"away_abbr": "BOS"},
    {"commence_time": "2026-01-15T17:00:00Z"},
    {"snapshot_at": (NOW - timedelta(hours=1)).isoformat()},
    {"snapshot_at": (NOW + timedelta(minutes=1)).isoformat()},
    {"snapshot_at": "2026-01-15T18:00:00"},
    {"home_team": float("nan")}, {"event_id": float("nan")}, {"k_value": 0},
])
def test_invalid_stale_future_and_started_quotes_are_not_usable(change):
    assert not tracker.usable_snapshot(quote() | change, DAY, NOW)


def test_api_fetches_h2h_on_event_endpoint_and_widens_regions(monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return mock.Mock(json=lambda: {} if len(calls) == 1 else payload(), raise_for_status=lambda: None)

    monkeypatch.setattr(tracker.requests, "get", get)
    assert tracker.get_event_game_odds(EVENT["id"])["bookmakers"]
    assert len(calls) == 2
    assert calls[0][0].endswith(f'/events/{EVENT["id"]}/odds')
    assert calls[0][1]["params"]["markets"] == "h2h"
    assert calls[1][1]["params"]["regions"] == "us,us2,uk,eu,au"


def test_snapshots_preserve_movement_skip_other_dates_and_redact_failures(tmp_path, monkeypatch, caplog):
    destination = tmp_path / "game.csv"
    get = mock.Mock(return_value=payload())
    monkeypatch.setattr(tracker, "get_event_game_odds", get)
    events = [EVENT, EVENT | {"id": "past", "commence_time": "2026-01-15T17:00:00Z"},
              EVENT | {"id": "tomorrow", "commence_time": "2026-01-17T00:30:00Z"}]
    tracker.snapshot(DAY, events=events, path=destination, now=NOW)
    get.return_value = payload(-450, 350)
    tracker.snapshot(DAY, events=events, path=destination, now=NOW + timedelta(minutes=1))
    with destination.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert get.call_count == 2
    assert rows[0]["devig_home"] != rows[1]["devig_home"]
    get.side_effect = requests.ConnectionError("https://example.test?apiKey=never-print-me")
    assert tracker.snapshot(DAY, events=[EVENT], path=destination, now=NOW) == []
    assert "never-print-me" not in caplog.text


@pytest.fixture
def live_clock(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW if tz else NOW.replace(tzinfo=None)

    monkeypatch.setattr(combined, "datetime", Clock)
    monkeypatch.setattr(combined, "_TODAY_GAME_ODDS", {})


def write_quotes(path, rows):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=tracker.TRACKER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_lookup_selects_latest_observation_not_csv_order_and_no_network(tmp_path, monkeypatch, live_clock):
    destination = tmp_path / "game.csv"
    latest = quote(-450, 350)
    older = quote(observed=NOW - timedelta(minutes=1))
    write_quotes(destination, [latest, older, quote(observed=NOW - timedelta(hours=1))])
    monkeypatch.setattr(tracker, "TRACKER_FILE", destination)
    fetch = mock.Mock(side_effect=AssertionError("Existing fresh quote must avoid network"))
    monkeypatch.setattr(tracker, "snapshot", fetch)
    loaded = combined._load_today_game_odds(DAY)
    assert loaded[("BOS", "LAL")]["avg_home_odds"] == -450
    fetch.assert_not_called()


def test_missing_snapshot_fetches_then_loads_actual_written_rows(tmp_path, monkeypatch, live_clock):
    destination = tmp_path / "game.csv"
    monkeypatch.setattr(tracker, "TRACKER_FILE", destination)
    monkeypatch.setattr(tracker.oddstracker, "get_all_events", lambda: [EVENT])
    monkeypatch.setattr(tracker, "get_event_game_odds", lambda _: payload())
    with mock.patch.object(tracker, "datetime", combined.datetime):
        loaded = combined._load_today_game_odds(DAY)
    assert loaded[("BOS", "LAL")]["devig_home"] > 0.75


@pytest.mark.parametrize("home_prob", [0.83, 0.2])
def test_model_probability_join_both_sides_and_selected_side_roundtrip(home_prob, tmp_path, monkeypatch, live_clock):
    row = quote()
    monkeypatch.setattr(combined, "_TODAY_GAME_ODDS", {("BOS", "LAL"): row})
    pred = {"home_abbr": "BOS", "away_abbr": "LAL", "game_date": DAY,
            "home_prob": home_prob, "favorite": "BOS" if home_prob >= 0.5 else "LAL",
            "fav_prob": max(home_prob, 1 - home_prob), "expected_margin": 9.0,
            "home_spread": -9, "away_spread": 9, "predicted_total": 220}
    combined._attach_game_market(pred)
    assert pred["moneyline"]["home"]["edge"] == pytest.approx(home_prob - row["devig_home"])
    assert pred["moneyline"]["away"]["edge"] == pytest.approx((1 - home_prob) - row["devig_away"])
    assert pred["moneyline"]["home"]["ev"] == pytest.approx(home_prob * 1.25 - 1)
    assert pred["moneyline"]["away"]["ev"] == pytest.approx((1 - home_prob) * 4 - 1)
    picked = "home" if home_prob >= 0.5 else "away"
    assert pred["p_market"] == row[f"devig_{picked}"]
    assert pred["ev"] == pred["moneyline"][picked]["ev"]
    assert pred["home_prob"] == home_prob
    destination = tmp_path / "predictions.db"
    original = db.upsert_game_predictions
    monkeypatch.setattr(db, "upsert_game_predictions", lambda rows, day: original(rows, day, path=destination))
    combined.export_game_picks_db([pred], DAY)
    stored = db.load_game_predictions(DAY, path=destination)[0]
    assert stored["p_market"] == pred["p_market"]
    assert stored["ev"] == pred["ev"]
    assert stored["home_prob"] == home_prob
    assert stored["odds_event_id"] == EVENT["id"]


def test_missing_wrong_orientation_or_stale_quote_does_not_fabricate_market(monkeypatch, live_clock):
    pred = {"home_abbr": "BOS", "away_abbr": "LAL", "game_date": DAY,
            "home_prob": 0.7, "favorite": "BOS", "p_market": 0.1, "ev": 20}
    for quotes in ({}, {("LAL", "BOS"): quote()},
                   {("BOS", "LAL"): quote(observed=NOW - timedelta(hours=1))}):
        monkeypatch.setattr(combined, "_TODAY_GAME_ODDS", quotes)
        combined._attach_game_market(pred)
        assert "p_market" not in pred and "ev" not in pred


def test_team_pipeline_joins_final_probability(monkeypatch, live_clock):
    pred = {"home_abbr": "NY", "away_abbr": "GS", "home_prob": 0.68,
            "favorite": "NY", "game_date": DAY}
    event = EVENT | {"home_team": "New York Knicks", "away_team": "Golden State Warriors"}
    data = event | {"bookmakers": [{"key": "book", "markets": [{"key": "h2h", "outcomes": [
        {"name": event["home_team"], "price": -150}, {"name": event["away_team"], "price": 130},
    ]}]}]}
    row = tracker.parse_game_odds_to_row(data, event, NOW)
    monkeypatch.setattr(combined, "_TODAY_GAME_ODDS", {("NYK", "GSW"): row})
    monkeypatch.setattr(combined.team_model, "predict_games_data", lambda *_: [pred])
    predictions, _ = combined._run_team_model([], {})
    assert predictions[0]["market_edge"] == pytest.approx(0.68 - row["devig_home"])


def test_existing_database_migrates_without_erasing_game_rows(tmp_path):
    destination = tmp_path / "old.db"
    with sqlite3.connect(destination) as conn:
        conn.execute(db._CREATE_GAME)
        conn.execute("INSERT INTO game_predictions (date, home_abbr, away_abbr, winner_pick) VALUES (?,?,?,?)",
                     (DAY.isoformat(), "BOS", "LAL", "BOS"))
    old = db.load_game_predictions(DAY, path=destination)[0]
    assert old["winner_pick"] == "BOS" and old["p_market"] is None and old["ev"] is None
    with sqlite3.connect(destination) as conn:
        assert "odds_snapshot_at" in {r[1] for r in conn.execute("PRAGMA table_info(game_predictions)")}
