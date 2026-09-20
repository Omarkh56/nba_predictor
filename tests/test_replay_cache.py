"""Replay API cache: only successful responses are cached; failures are never turned into empty data."""
import json
import os
import sys

import pandas as pd
import pytest
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "replay"))

from replay_cache import NBACache, cache_key  # noqa: E402


def payload(rows, headers=("A", "B")):
    return {"resultSets": [{"name": "X", "headers": list(headers), "rowSet": rows}]}


class Scripted:
    """Fetcher returning/raising the next scripted item on every call."""

    def __init__(self, items):
        self.items = list(items)
        self.calls = 0

    def __call__(self, endpoint, params, timeout):
        self.calls += 1
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make(tmp_path, items, **kw):
    t = [0.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        t[0] += s

    fetcher = Scripted(items)
    cache = NBACache(str(tmp_path), fetcher=fetcher, sleep=sleep, clock=lambda: t[0], **kw)
    return cache, fetcher, sleeps


P = {"player_or_team_abbreviation": "P"}


def fetch(cache, **over):
    args = dict(endpoint="leaguegamefinder", params=P, season="2025-26", season_type="Playoffs")
    args.update(over)
    return cache.fetch(**args)


def test_successful_nonempty_response_is_cached(tmp_path):
    cache, f, _ = make(tmp_path, [payload([[1, 2], [3, 4]])])
    first, second = fetch(cache), fetch(cache)
    assert first.status == "ok" and not first.from_cache
    assert second.status == "ok" and second.from_cache
    assert f.calls == 1
    assert cache.stats.cache_hits == 1 and cache.stats.cache_misses == 1 and cache.stats.cache_writes == 1
    pd.testing.assert_frame_equal(first.frame, second.frame)


def test_successful_empty_response_is_cached_and_distinguishable(tmp_path):
    cache, f, _ = make(tmp_path, [payload([])])
    first, second = fetch(cache), fetch(cache)
    assert first.status == "ok_empty" and first.ok and first.frame is not None and first.frame.empty
    assert second.status == "ok_empty" and second.from_cache
    assert f.calls == 1


def test_transient_failure_then_success_retries_and_caches(tmp_path):
    cache, f, sleeps = make(tmp_path, [requests.exceptions.Timeout("slow"), payload([[1, 2]])], base_delay_s=5.0)
    res = fetch(cache)
    assert res.status == "ok" and f.calls == 2
    assert cache.stats.requests == 2 and cache.stats.retries == 1 and cache.stats.failures == 0
    assert 5.0 in sleeps  # exponential backoff after the first failure
    assert fetch(cache).from_cache


def test_permanent_failure_is_reported_and_never_cached(tmp_path):
    err = requests.exceptions.ConnectionError("down")
    cache, f, _ = make(tmp_path, [err, err, err], retries=3)
    res = fetch(cache)
    assert res.status == "failed" and not res.ok and res.frame is None and "ConnectionError" in res.error
    assert cache.stats.failures == 1 and cache.stats.requests == 3 and cache.stats.retries == 2
    assert cache.entry_count() == 0 and cache.stats.cache_writes == 0


def test_malformed_response_is_a_failure_not_empty_data(tmp_path):
    cache, _f, _ = make(tmp_path, [{"unexpected": True}] * 3, retries=3)
    res = fetch(cache)
    assert res.status == "failed" and res.frame is None
    assert cache.entry_count() == 0


def test_failed_result_is_not_cached_so_a_later_success_is_fetched(tmp_path):
    err = requests.exceptions.Timeout("t")
    cache, f, _ = make(tmp_path, [err, err, err, payload([[9, 9]])], retries=3)
    assert fetch(cache).status == "failed"
    later = fetch(cache)
    assert later.status == "ok" and not later.from_cache and later.frame.iloc[0, 0] == 9
    assert f.calls == 4


def test_cache_key_covers_endpoint_params_season_type_bound_and_schema():
    base = cache_key("leaguegamefinder", P, "2025-26", "Playoffs", "open")
    assert base != cache_key("commonteamroster", P, "2025-26", "Playoffs", "open")
    assert base != cache_key("leaguegamefinder", {"player_or_team_abbreviation": "T"}, "2025-26", "Playoffs", "open")
    assert base != cache_key("leaguegamefinder", P, "2024-25", "Playoffs", "open")
    assert base != cache_key("leaguegamefinder", P, "2025-26", "Regular Season", "open")
    assert base != cache_key("leaguegamefinder", P, "2025-26", "Playoffs", "2026-04-20")
    assert base == cache_key("leaguegamefinder", dict(P), "2025-26", "Playoffs", "open")


def test_corrupt_cache_entry_is_discarded_and_refetched(tmp_path):
    cache, f, _ = make(tmp_path, [payload([[1, 2]]), payload([[7, 8]])])
    first = fetch(cache)
    path = cache._path("leaguegamefinder", first.key)
    with open(path, "wb") as fh:
        fh.write(b"not gzip")
    again = fetch(cache)
    assert again.status == "ok" and not again.from_cache and again.frame.iloc[0, 0] == 7
    assert cache.stats.corrupt_cache_entries == 1 and f.calls == 2


def test_live_requests_are_rate_limited(tmp_path):
    cache, _f, sleeps = make(tmp_path, [payload([[1, 2]]), payload([[3, 4]])], min_interval_s=1.2)
    fetch(cache, params={"x": 1})
    fetch(cache, params={"x": 2})
    assert cache.stats.rate_limit_delay_s == pytest.approx(1.2)
    assert sleeps == [pytest.approx(1.2)]


def test_cached_payload_is_raw_and_must_be_filtered_by_the_caller(tmp_path):
    """The cache stores whatever the API returned (including games after the replay date); as-of filtering is the
    caller's job and is exercised in test_replay_data."""
    rows = [["2026-04-20", 1], ["2026-06-01", 2]]
    cache, _f, _ = make(tmp_path, [payload(rows, headers=("GAME_DATE", "V"))])
    res = fetch(cache)
    assert list(res.frame["GAME_DATE"]) == ["2026-04-20", "2026-06-01"]
