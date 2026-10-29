"""raw_archive.py — save every external response before the pipeline uses it.

Live runs refetch everything and keep nothing, so a past day can't be replayed
and ESPN injury reports (not available historically) are lost. The fetch layers
call save() with each parsed response: odds_api.get_json(), nba_api's session
(nba_api_utils.ArchivingSession) and the ESPN fetches. Failed fetches are saved
too, with status="fetch_failed" and no payload, so gaps are visible.

Layout:
    data/raw/{source}/{endpoint}/dt={YYYY-MM-DD}/{fetched_at}_{params_hash}.json.gz
where dt is the US Eastern slate date of the fetch (team_codes.today_slate_date).
Each file is one gzipped JSON envelope: source, endpoint, params (credentials
removed), params_hash, fetched_at_utc, run_id, status, http_status (when known),
error (failures only) and payload.

save() never raises: an archive problem is logged as a warning and the caller
carries on with its data unchanged. NBA_RAW_ARCHIVE=0 turns archiving off;
NBA_RAW_ARCHIVE_DIR moves the root (tests point it at tmp_path). RunLog sets the
run id that files carry unless a caller passes its own.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import team_codes

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path(__file__).with_name("data") / "raw"
_SECRET_PARAM = re.compile(r"api[_-]?key|token|secret|password", re.IGNORECASE)
_UNSAFE_PATH = re.compile(r"[^A-Za-z0-9._-]+")
_run_id = None
_warned = set()


def enabled() -> bool:
    return os.environ.get("NBA_RAW_ARCHIVE", "1").strip().lower() not in ("0", "false", "off", "no")


def root() -> Path:
    override = os.environ.get("NBA_RAW_ARCHIVE_DIR")
    return Path(override) if override else DEFAULT_ROOT


def set_run_id(run_id) -> None:
    """Tag files saved from now on with this run's id (RunLog calls this)."""
    global _run_id
    _run_id = run_id


def strip_secrets(params) -> dict:
    """Params as a plain dict, without credentials such as the Odds API apiKey."""
    items = params.items() if isinstance(params, dict) else (params or [])
    return {str(k): v for k, v in items if not _SECRET_PARAM.search(str(k))}


def params_hash(params: dict) -> str:
    text = json.dumps(params, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def http_status_of(obj):
    """Status code of a Response, or of the response an HTTPError carries; else None."""
    response = getattr(obj, "response", obj)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _segment(value) -> str:
    return _UNSAFE_PATH.sub("_", str(value)).strip("._") or "unknown"


def save(source, endpoint, params, payload, *, run_id=None, status="ok",
         http_status=None, error=None):
    """Archive one response. Returns the file written, or None (disabled or failed)."""
    if not enabled():
        return None
    try:
        now = datetime.now(timezone.utc)
        clean = strip_secrets(params)
        digest = params_hash(clean)
        envelope = {
            "source": source,
            "endpoint": endpoint,
            "params": clean,
            "params_hash": digest,
            "fetched_at_utc": now.isoformat(timespec="microseconds"),
            "run_id": run_id if run_id is not None else _run_id,
            "status": status,
        }
        if isinstance(http_status, int):
            envelope["http_status"] = http_status
        if error:
            envelope["error"] = str(error)
        envelope["payload"] = payload
        body = json.dumps(envelope, default=str, separators=(",", ":")).encode("utf-8")

        folder = (root() / _segment(source) / _segment(endpoint)
                  / f"dt={team_codes.today_slate_date(now).isoformat()}")
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}_{digest}"
        path, n = folder / f"{stem}.json.gz", 1
        while path.exists():
            path, n = folder / f"{stem}-{n}.json.gz", n + 1
        partial = path.with_name(path.name + ".part")
        try:
            with gzip.open(partial, "wb") as fh:
                fh.write(body)
            os.replace(partial, path)  # readers never see a half-written file
        finally:
            if partial.exists():
                partial.unlink()
        return path
    except Exception as exc:  # the archive must never break the caller's fetch
        key = (source, type(exc).__name__)
        if key not in _warned:  # once per source and error, not once per request
            _warned.add(key)
            logger.warning("raw archive: could not save %s/%s (%s: %s); continuing without it",
                           source, endpoint, type(exc).__name__, exc)
        return None


def load(path) -> dict:
    """Read one archived envelope back."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)
