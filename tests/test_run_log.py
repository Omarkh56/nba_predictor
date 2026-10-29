"""run_log: failed fetches must be distinguishable from quiet days."""

import json
import os

import pytest

import run_log


def test_required_fetch_failure_fails_the_run(tmp_path):
    log = run_log.RunLog("checkresults", "2026-10-21", path=tmp_path / "runs.jsonl")
    log.stage("box_scores", "fetch_failed", required=True, error="timeout")
    record = log.finish()
    assert record["status"] == "fetch_failed"
    assert log.exit_code() == 2


def test_optional_fetch_failure_is_partial(tmp_path):
    log = run_log.RunLog("nba_combined", "2026-10-21", path=tmp_path / "runs.jsonl")
    log.stage("schedule", "ok", required=True, games=3)
    log.stage("props:NYK@BOS", "fetch_failed", error="HTTPError")
    log.stage("props:PHI@DET", "ok", props=40)
    assert log.finish()["status"] == "partial"
    assert log.exit_code() == 3


def test_empty_schedule_is_no_games_not_failure(tmp_path):
    log = run_log.RunLog("nba_combined", "2026-07-15", path=tmp_path / "runs.jsonl")
    log.stage("schedule", "no_games", required=True)
    assert log.finish()["status"] == "no_games"
    assert log.exit_code() == 0


def test_market_unavailable_is_ok(tmp_path):
    log = run_log.RunLog("nba_combined", "2026-10-05", path=tmp_path / "runs.jsonl")
    log.stage("schedule", "ok", required=True, games=1)
    log.stage("props:NYK@BOS", "market_unavailable", reason="no bookmakers")
    assert log.finish()["status"] == "ok"


def test_unknown_status_is_rejected(tmp_path):
    log = run_log.RunLog("x", path=tmp_path / "runs.jsonl")
    with pytest.raises(ValueError):
        log.stage("schedule", "empty")


def test_finish_appends_exactly_one_line(tmp_path):
    path = tmp_path / "runs.jsonl"
    first = run_log.RunLog("checkresults", "2026-10-21", path=path)
    first.stage("box_scores", "ok", required=True, games=2)
    first.finish()
    first.finish()  # second call is a no-op
    second = run_log.RunLog("checkresults", "2026-10-21", path=path)
    second.stage("box_scores", "ok", required=True, games=2)
    second.finish()
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["run_id"] != parsed[1]["run_id"]
    assert run_log.read_runs(path)[1]["stages"][0]["games"] == 2


def test_env_var_overrides_default_path(tmp_path, monkeypatch):
    target = tmp_path / "custom.jsonl"
    monkeypatch.setenv("NBA_RUN_LOG", str(target))
    log = run_log.RunLog("nba_combined")
    log.stage("schedule", "no_games", required=True)
    log.finish()
    assert target.is_file()


def test_a_second_copy_of_a_run_is_refused_until_the_first_ends(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_RUN_LOG", str(tmp_path / "runs.jsonl"))
    assert run_log.lock_holder("nba_combined") is None
    with run_log.exclusive("nba_combined") as first:
        assert first is True
        assert run_log.lock_holder("nba_combined") == os.getpid()
        with run_log.exclusive("nba_combined") as second:  # flock is per open file
            assert second is False
    assert run_log.lock_holder("nba_combined") is None
    with run_log.exclusive("nba_combined") as again:
        assert again is True
