"""Local cache for NBA API responses used by the replay engine.

Rules (each one is covered by tests/test_replay_cache.py):
  * Only SUCCESSFUL responses are cached. A timeout, exception, malformed response or retry exhaustion is
    reported as status="failed" and is NEVER written to disk and NEVER converted into an empty DataFrame.
  * A successful response with zero rows is a legitimate result (status="ok_empty") and may be cached;
    it stays distinguishable from a failure.
  * The cache key covers: endpoint, full request parameters, season, season type, the request's date bound
    ("fetch_through"), and a schema/version identifier.
  * A cached response is a raw API payload. It is NEVER trusted to be as-of: callers must still filter to the
    replay's as-of date (see replay_data.asof_before).
  * Live requests are rate limited (minimum interval) with bounded retries and exponential backoff; every
    request, retry, hit, miss and delay is counted in ApiStats.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

import pandas as pd
import requests

from replay_config import CACHE_SCHEMA_VERSION

# Exceptions that mean "the request/response was bad or unavailable" -> retry, then report failure.
TRANSIENT_ERRORS = (requests.exceptions.RequestException, json.JSONDecodeError, ValueError, KeyError, OSError)


@dataclass
class ApiStats:
    requests: int = 0  # live HTTP attempts (each retry is one more request)
    retries: int = 0
    failures: int = 0  # fetches that exhausted retries
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
    empty_success: int = 0
    corrupt_cache_entries: int = 0
    rate_limit_delay_s: float = 0.0
    backoff_delay_s: float = 0.0
    live_call_s: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("rate_limit_delay_s", "backoff_delay_s", "live_call_s"):
            d[k] = round(d[k], 3)
        return d


@dataclass
class FetchResult:
    status: str  # "ok" | "ok_empty" | "failed"
    frame: Optional[pd.DataFrame]
    from_cache: bool
    error: str = ""
    key: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "ok_empty")


def cache_key(endpoint: str, params: Dict[str, Any], season: str, season_type: str, fetch_through: str) -> str:
    blob = json.dumps(
        {
            "endpoint": endpoint,
            "params": params,
            "season": season,
            "season_type": season_type,
            "fetch_through": fetch_through,
            "schema": CACHE_SCHEMA_VERSION,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _validate_payload(payload: Any, result_set: Optional[str]) -> Dict[str, Any]:
    """Return {"headers": [...], "rows": [...]} or raise ValueError for a malformed response."""
    if not isinstance(payload, dict):
        raise ValueError("payload is not a dict")
    sets = payload.get("resultSets")
    if sets is None and "resultSet" in payload:
        sets = [payload["resultSet"]]
    if not isinstance(sets, list) or not sets:
        raise ValueError("payload has no resultSets")
    chosen = None
    for rs in sets:
        if result_set is None or rs.get("name") == result_set:
            chosen = rs
            break
    if chosen is None:
        raise ValueError("requested result set missing: %s" % result_set)
    headers, rows = chosen.get("headers"), chosen.get("rowSet")
    if not isinstance(headers, list) or not headers or not isinstance(rows, list):
        raise ValueError("result set missing headers/rowSet")
    if any((not isinstance(r, list)) or len(r) != len(headers) for r in rows[:50]):
        raise ValueError("row width does not match headers")
    return {"headers": headers, "rows": rows}


def default_fetcher(endpoint: str, params: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    """Real nba_api call. Imported lazily so tests never need the network stack."""
    from nba_api.stats import endpoints as ep

    classes = {"leaguegamefinder": ep.LeagueGameFinder, "commonteamroster": ep.CommonTeamRoster}
    if endpoint not in classes:
        raise ValueError("unsupported endpoint %s" % endpoint)
    return classes[endpoint](**params, timeout=timeout_s).get_dict()


class NBACache:
    def __init__(
        self,
        cache_dir: str,
        fetcher: Optional[Callable[[str, Dict[str, Any], float], Dict[str, Any]]] = None,
        min_interval_s: float = 1.2,
        retries: int = 3,
        base_delay_s: float = 5.0,
        timeout_s: float = 90.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cache_dir = cache_dir
        self.fetcher = fetcher or default_fetcher
        self.min_interval_s = min_interval_s
        self.retries = retries
        self.base_delay_s = base_delay_s
        self.timeout_s = timeout_s
        self._sleep = sleep
        self._clock = clock
        self._last_call: Optional[float] = None
        self.stats = ApiStats()

    # ------------------------------------------------------------------ paths / io
    def _path(self, endpoint: str, key: str) -> str:
        return os.path.join(self.cache_dir, endpoint, key[:2], key + ".json.gz")

    def entry_count(self) -> int:
        n = 0
        for _root, _dirs, files in os.walk(self.cache_dir):
            n += sum(1 for f in files if f.endswith(".json.gz"))
        return n

    def _read(self, path: str, key: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(path):
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                blob = json.load(fh)
            if blob["meta"]["schema"] != CACHE_SCHEMA_VERSION or blob["meta"]["key"] != key:
                raise ValueError("cache entry schema/key mismatch")
            _validate_payload({"resultSets": [{"headers": blob["headers"], "rowSet": blob["rows"]}]}, None)
            return blob
        except (OSError, ValueError, KeyError, json.JSONDecodeError, EOFError):
            self.stats.corrupt_cache_entries += 1
            try:
                os.remove(path)  # a corrupt entry is discarded, never trusted
            except OSError:
                pass
            return None

    def _write(self, path: str, blob: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        os.close(fd)
        try:
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump(blob, fh)
            os.replace(tmp, path)  # atomic: a crash never leaves a half-written valid entry
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        self.stats.cache_writes += 1

    # ------------------------------------------------------------------ live call
    def _throttle(self) -> None:
        if self._last_call is not None:
            wait = self.min_interval_s - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
                self.stats.rate_limit_delay_s += wait

    def _live(self, endpoint: str, params: Dict[str, Any], result_set: Optional[str]):
        last_err = ""
        for attempt in range(self.retries):
            self._throttle()
            self.stats.requests += 1
            t0 = self._clock()
            try:
                payload = self.fetcher(endpoint, params, self.timeout_s)
                data = _validate_payload(payload, result_set)
                self.stats.live_call_s += self._clock() - t0
                self._last_call = self._clock()
                return data, ""
            except TRANSIENT_ERRORS as exc:
                self.stats.live_call_s += self._clock() - t0
                self._last_call = self._clock()
                last_err = "%s: %s" % (type(exc).__name__, exc)
                if attempt < self.retries - 1:
                    self.stats.retries += 1
                    delay = self.base_delay_s * (2 ** attempt)
                    self._sleep(delay)
                    self.stats.backoff_delay_s += delay
        self.stats.failures += 1
        return None, last_err

    # ------------------------------------------------------------------ public
    def fetch(
        self,
        endpoint: str,
        params: Dict[str, Any],
        season: str,
        season_type: str,
        fetch_through: str = "open",
        result_set: Optional[str] = None,
    ) -> FetchResult:
        key = cache_key(endpoint, params, season, season_type, fetch_through)
        path = self._path(endpoint, key)
        blob = self._read(path, key)
        if blob is not None:
            self.stats.cache_hits += 1
            frame = pd.DataFrame(blob["rows"], columns=blob["headers"])
            return FetchResult("ok_empty" if frame.empty else "ok", frame, True, key=key)

        self.stats.cache_misses += 1
        data, err = self._live(endpoint, params, result_set)
        if data is None:
            # Failure: nothing is cached and no empty frame is fabricated.
            return FetchResult("failed", None, False, error=err, key=key)

        frame = pd.DataFrame(data["rows"], columns=data["headers"])
        meta = {
            "endpoint": endpoint,
            "params": params,
            "season": season,
            "season_type": season_type,
            "fetch_through": fetch_through,
            "schema": CACHE_SCHEMA_VERSION,
            "key": key,
            "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_rows": len(frame),
        }
        self._write(path, {"meta": meta, "headers": data["headers"], "rows": data["rows"]})
        if frame.empty:
            self.stats.empty_success += 1
        return FetchResult("ok_empty" if frame.empty else "ok", frame, False, key=key)
