"""dashboard.py — local web dashboard for tracking the props pipeline.

    python3 dashboard.py                      # http://127.0.0.1:8765
    NBA_DASHBOARD_PORT=9000 python3 dashboard.py

It reads the pipeline's own files on every request (predictions.db, bet_log.csv,
game_log.csv, run_log.jsonl, data/injury_snapshots.csv and the installed launchd
agents), so the page is always current, and it never writes to them. The page's
"Run now" buttons start the same scripts the schedules run, logging to
~/Library/Logs/nba-props/web-<job>.log.

It listens on 127.0.0.1 only. Every request must name a localhost Host (so
DNS-rebinding pages can't read it), and POSTs must also send the
X-NBA-Dashboard header as JSON (so another site can't trigger a run).
"""

from __future__ import annotations

import csv
import getpass
import json
import os
import plistlib
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

import requests

import raw_archive
import run_log
import team_codes
from config import current_season

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "web" / "index.html"
LOG_DIR = Path.home() / "Library" / "Logs" / "nba-props"
LAUNCH_AGENTS = Path.home() / "Library" / "LaunchAgents"
AGENT_PREFIX = f"com.{getpass.getuser()}.nba-props."
ET = ZoneInfo("America/New_York")
BREAK_EVEN = 0.5238  # hit rate needed at -110 on both sides

# Jobs the page can start: name -> (script, arguments allowed from the page)
JOBS = {
    "predict": ("nba_combined.py", ()),          # today's slate
    "grade": ("checkresults.py", ("date",)),     # yesterday by default
    "injuries": ("snapshot_injuries.py", ()),
}
# Commands as run_log.jsonl records them, for the status tiles.
JOB_COMMANDS = {"predict": "nba_combined", "grade": "checkresults",
                "injuries": "snapshot_injuries", "closing": "oddstracker --closing"}
PROP_COLUMNS = ("date", "game", "player", "market", "line", "pick", "projection", "confidence",
                "edge_pct", "p_model", "entry_american_odds", "odds_book", "stake_amount",
                "game_start_at", "closing_line", "line_clv", "flags")


class Sources:
    """Where the pipeline's files live (tests point this at a temp dir)."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.db = self.root / "predictions.db"
        self.bet_log = self.root / "bet_log.csv"
        self.game_log = self.root / "game_log.csv"
        self.game_odds = self.root / "game_odds_tracker.csv"
        self.injuries = self.root / "data" / "injury_snapshots.csv"
        self.team_logs = self.root / "data_cache"
        self.run_log = run_log.default_path() if self.root == ROOT else self.root / "run_log.jsonl"


# ── data ──────────────────────────────────────────────────────────────────────
def _read_csv(path: Path) -> list:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _query(db: Path, sql: str, params=()) -> list:
    if not db.is_file():
        return []
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


def _counts(rows) -> tuple:
    return (sum(r["result"] == "HIT" for r in rows), sum(r["result"] == "MISS" for r in rows),
            sum(r["result"] == "PUSH" for r in rows))


def _record(rows) -> dict:
    """W-L-P for graded rows; hit rate over hits + misses (pushes excluded)."""
    return _record_of(*_counts(rows))


def _record_of(hits: int, misses: int, pushes: int) -> dict:
    graded = hits + misses
    return {"hits": hits, "misses": misses, "pushes": pushes, "graded": graded,
            "rate": round(hits / graded, 4) if graded else None}


def picks_for(src: Sources, day: str) -> dict:
    """Prop and game picks for one slate date, with results where graded."""
    props = _query(src.db, f"SELECT {', '.join(PROP_COLUMNS)} FROM prop_predictions "
                           "WHERE date = ? ORDER BY game, confidence DESC", (day,))
    graded = {(r["player"], r["market"]): r for r in _read_csv(src.bet_log) if r["date"] == day}
    for p in props:
        g = graded.get((p["player"], p["market"]), {})
        p["result"] = g.get("result") or None
        p["actual"] = _num(g.get("actual"))
    games = _query(src.db, "SELECT date, home_abbr, away_abbr, winner_pick, spread, spread_pick, "
                           "win_pct FROM game_predictions WHERE date = ? ORDER BY away_abbr", (day,))
    finals = {r["game"]: r for r in _read_csv(src.game_log) if r["date"] == day}
    for g in games:
        f = finals.get(f"{g['away_abbr']}@{g['home_abbr']}", {})
        g["score"] = f"{f['away_score']}-{f['home_score']}" if f.get("home_score") else None
        g["winner_hit"], g["spread_hit"] = f.get("winner_hit") or None, f.get("spread_hit") or None
    return {"date": day, "props": props, "games": games, "record": _record(
        [p for p in props if p["result"]])}


# ── Top 15: rankings, weights fitted on graded slates, and how each one has done ──
TOP_N = 15
RANK_METRICS = {"prob": "p_model", "conf": "confidence", "edge": "edge_pct"}
TOP_SORTS = {**RANK_METRICS, "weighted": None}
# (prob, conf, edge) weights in steps of 0.1 that sum to 1: 66 candidates.
WEIGHT_GRID = [(a / 10, b / 10, round((10 - a - b) / 10, 1)) for a in range(11) for b in range(11 - a)]
EQUAL_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)


def _flagged(p: dict) -> bool:
    """The model's own ⚠SUSP flag: edge over 40%, "real markets don't misprice by >40%"."""
    return "SUSP" in (p.get("flags") or "")


def _pct_ranks(values: list) -> list:
    """Percentile rank of each value within its slate, 0 (worst) to 1 (best); ties share."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks, i = [1.0] * len(values), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = ((i + j) / 2) / (len(values) - 1) if len(values) > 1 else 1.0
        i = j + 1
    return ranks


def rank_slate(props: list, by: str = "prob", weights=None, include_flagged: bool = False) -> list:
    """One slate's picks best first. "weighted" scores each pick by its weighted percentile
    ranks in probability, confidence and edge within the slate, so the three scales mix
    fairly. ⚠SUSP picks are left out unless include_flagged: their probability isn't
    docked the way their confidence is, so they would otherwise float to the top."""
    pool = [p for p in props if include_flagged or not _flagged(p)]
    if by != "weighted":
        column = RANK_METRICS[by]
        return sorted(pool, key=lambda p: (p.get(column) is None, -(p.get(column) or 0),
                                           -(p.get("confidence") or 0)))
    return _weighted_order(pool, _metric_ranks(pool), weights or EQUAL_WEIGHTS)


def _metric_ranks(pool: list) -> list:
    """Each pick's percentile rank in probability, confidence and edge. They don't depend on
    the weights, so fitting computes them once per slate, not once per candidate."""
    return [_pct_ranks([p.get(c) if p.get(c) is not None else float("-inf") for p in pool])
            for c in RANK_METRICS.values()]


def _weighted_order(pool: list, ranks: list, weights) -> list:
    scores = [sum(w * r[i] for w, r in zip(weights, ranks)) for i in range(len(pool))]
    order = sorted(range(len(pool)), key=lambda i: (-scores[i], -(pool[i].get("confidence") or 0)))
    return [pool[i] for i in order]


def _graded_slates(src: Sources, start: str = None) -> dict:
    """{date: every pick on that slate, with results} for slates that have graded picks."""
    sql = f"SELECT {', '.join(PROP_COLUMNS)} FROM prop_predictions"
    props = _query(src.db, sql + (" WHERE date >= ?" if start else ""), (start,) if start else ())
    graded = {(r["date"], r["player"], r["market"]): r for r in _read_csv(src.bet_log)
              if r.get("result") in ("HIT", "MISS", "PUSH")}
    slates = {}
    for p in props:
        g = graded.get((p["date"], p["player"], p["market"]), {})
        p["result"], p["actual"] = g.get("result") or None, _num(g.get("actual"))
        slates.setdefault(p["date"], []).append(p)
    return {d: ps for d, ps in sorted(slates.items()) if any(p["result"] for p in ps)}


def _has(props: list, column: str) -> bool:
    """A slate can be ranked by a metric only if (almost) every pick carries it; the
    pipeline didn't store probability before the 2026-27 season."""
    return bool(props) and sum(p.get(column) is not None for p in props) >= 0.9 * len(props)


def _top_results(slates: dict, by: str, weights=None) -> list:
    return [p for ps in slates.values() for p in rank_slate(ps, by, weights)[:TOP_N] if p["result"]]


def _fully_ranked(slates: dict) -> dict:
    return {d: ps for d, ps in slates.items() if all(_has(ps, c) for c in RANK_METRICS.values())}


def _grid_counts(slates: dict) -> dict:
    """{date: [(hits, misses, pushes) of that slate's weighted Top 15, for each WEIGHT_GRID
    entry]}. Fitting and leave-one-out both sum rows of this table, so each slate is ranked
    66 times in all instead of 66 times per fit (the refits made the leave-one-out grow
    with the square of the slate count: 3.6 s at 20 slates)."""
    table = {}
    for d, ps in slates.items():
        pool = [p for p in ps if not _flagged(p)]
        ranks = _metric_ranks(pool)
        table[d] = [_counts([p for p in _weighted_order(pool, ranks, w)[:TOP_N] if p["result"]])
                    for w in WEIGHT_GRID]
    return table


def _grid_totals(rows) -> list:
    """Sum (hits, misses, pushes) per WEIGHT_GRID entry over several slates' table rows."""
    return [tuple(map(sum, zip(*column))) for column in zip(*rows)]


def _best_weight(totals: list) -> int:
    """Index of the weights whose Top 15 did best: most hits minus misses, ties going to the
    most even mix (the least overfit choice), then to the earlier grid entry."""
    best = None
    for i, (w, (hits, misses, _)) in enumerate(zip(WEIGHT_GRID, totals)):
        key = (hits - misses, -max(w))
        if best is None or key > best[0]:
            best = (key, i)
    return best[1]


def fit_weights(slates: dict) -> dict:
    """The (prob, conf, edge) weights whose Top 15 did best on these slates (_best_weight)."""
    usable = _fully_ranked(slates)
    if not usable:
        return {"weights": EQUAL_WEIGHTS, "slates": 0, "record": _record([]), "fitted": False}
    totals = _grid_totals(_grid_counts(usable).values())
    i = _best_weight(totals)
    return {"weights": WEIGHT_GRID[i], "slates": len(usable), "record": _record_of(*totals[i]),
            "fitted": True}


def _weighted_fit(slates: dict) -> tuple:
    """(fitted weights, their in-sample counts, out-of-sample counts, slates used). Out of
    sample scores each slate with weights fitted on the others; needs two slates."""
    table = _grid_counts(_fully_ranked(slates))
    if not table:
        return EQUAL_WEIGHTS, (0, 0, 0), (0, 0, 0), 0
    totals = _grid_totals(table.values())
    best = _best_weight(totals)
    held_out = (0, 0, 0)
    if len(table) >= 2:
        for row in table.values():  # refit without this slate: the totals minus its row
            without = [tuple(t - x for t, x in zip(total, own)) for total, own in zip(totals, row)]
            held_out = tuple(a + b for a, b in zip(held_out, row[_best_weight(without)]))
    return WEIGHT_GRID[best], totals[best], held_out, len(table)


def ranking_report(src: Sources, range_key: str = "season", today: date = None) -> dict:
    """How each Top 15 ranking has done on graded slates, plus the weighted ranking both
    in-sample and out of sample (each slate scored with weights fitted without it)."""
    today = today or team_codes.today_slate_date()
    slates = _graded_slates(src, _range_start(range_key, today))
    methods = {}
    for by, column in RANK_METRICS.items():
        usable = {d: ps for d, ps in slates.items() if _has(ps, column)}
        methods[by] = {**_record(_top_results(usable, by)), "slates": len(usable)}
    weights, in_sample, held_out, n = _weighted_fit(slates)
    methods["weighted"] = {**_record_of(*in_sample), "slates": n}
    methods["weighted_oos"] = {**_record_of(*held_out), "slates": n if n >= 2 else 0}
    return {"range": range_key, "top_n": TOP_N, "methods": methods,
            "weights": dict(zip(RANK_METRICS, weights)), "break_even": BREAK_EVEN}


def top_bets(src: Sources, day: str, by: str = "prob", limit: int = TOP_N,
             include_flagged: bool = False, weights=None) -> dict:
    """The slate's best picks by one ranking. "weighted" uses `weights` if given, else the
    weights fitted on every graded slate so far."""
    props = picks_for(src, day)["props"]
    fit = None
    if by == "weighted" and weights is None:
        fit = fit_weights(_graded_slates(src))
        weights = fit["weights"]
    ranked = rank_slate(props, by, weights, include_flagged)
    return {"date": day, "by": by, "count": len(props), "flagged": sum(map(_flagged, props)),
            "include_flagged": include_flagged, "bets": ranked[:limit],
            "weights": dict(zip(RANK_METRICS, weights)) if by == "weighted" else None,
            "fitted_on": fit["slates"] if fit else None}


def _range_start(range_key: str, today: date):
    if range_key == "7d":
        return (today - timedelta(days=6)).isoformat()
    if range_key == "30d":
        return (today - timedelta(days=29)).isoformat()
    if range_key == "season":
        return f"{current_season(today)[:4]}-10-01"
    return None  # "all"


def performance(src: Sources, range_key: str = "season", today: date = None) -> dict:
    """Graded prop results over a date range: overall, per day, per market, per side."""
    today = today or team_codes.today_slate_date()
    start = _range_start(range_key, today)
    rows = [r for r in _read_csv(src.bet_log)
            if r.get("result") in ("HIT", "MISS", "PUSH") and (start is None or r["date"] >= start)]

    def grouped(key):
        groups = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r)
        return groups

    stakes = [_num(r.get("stake_amount")) or 0.0 for r in rows]
    profits = [_num(r.get("realized_profit")) for r in rows]
    staked = sum(s for s in stakes if s > 0)
    profit = sum(p for p in profits if p is not None)
    by_day = [{"date": d, **_record(rs)} for d, rs in sorted(grouped("date").items())]
    by_market = sorted(({"market": m, **_record(rs)} for m, rs in grouped("market").items()),
                       key=lambda x: -x["graded"])
    by_side = sorted(({"side": s, **_record(rs)} for s, rs in grouped("pick").items()),
                     key=lambda x: -x["graded"])
    return {"range": range_key, "start": start, "overall": _record(rows), "by_day": by_day,
            "by_market": by_market, "by_side": by_side, "break_even": BREAK_EVEN,
            "paper": {"staked": round(staked, 2), "profit": round(profit, 2),
                      "roi": round(profit / staked, 4) if staked else None}}


def runs(src: Sources, limit: int = 40) -> list:
    """Most recent runs first, with any failed stage named."""
    out = []
    for r in reversed(run_log.read_runs(src.run_log)[-limit:]):
        stages = r.get("stages", [])
        out.append({
            "run_id": r.get("run_id"), "command": r.get("command"),
            "slate_date": r.get("slate_date"), "started_at": r.get("started_at"),
            "finished_at": r.get("finished_at"), "status": r.get("status"),
            "picks": r.get("picks"), "props_analyzed": r.get("props_analyzed"),
            "stages": len(stages),
            "failed": [{"stage": s["stage"], "required": s.get("required"),
                        "detail": s.get("error") or s.get("reason") or ""}
                       for s in stages if s.get("status") == "fetch_failed"],
        })
    return out


def latest_runs(src: Sources) -> dict:
    latest = {}
    for r in run_log.read_runs(src.run_log):
        for job, command in JOB_COMMANDS.items():
            if r.get("command") == command:
                latest[job] = r
    return {job: {"started_at": r.get("started_at"), "status": r.get("status"),
                  "slate_date": r.get("slate_date"), "picks": r.get("picks"),
                  "stages": [{"stage": s["stage"], "status": s["status"]} for s in r.get("stages", [])]}
            for job, r in latest.items()}


def injuries(src: Sources) -> dict:
    rows = _read_csv(src.injuries)
    if not rows:
        return {"snapshot_at_utc": None, "rows": []}
    latest = max(r["snapshot_at_utc"] for r in rows)
    keep = ("team", "player", "status", "position", "injury_type", "injury_detail",
            "return_date", "short_comment", "report_updated_at")
    return {"snapshot_at_utc": latest,
            "rows": sorted(({k: r.get(k, "") for k in keep} for r in rows
                            if r["snapshot_at_utc"] == latest), key=lambda r: (r["team"], r["player"]))}


def _next_fire(times, now_local: datetime):
    """Next StartCalendarInterval time; an entry without "Hour" fires every hour."""
    upcoming = []
    for t in times:
        if "Hour" in t:
            candidate = now_local.replace(hour=t["Hour"], minute=t.get("Minute", 0), second=0, microsecond=0)
            step = timedelta(days=1)
        else:
            candidate = now_local.replace(minute=t.get("Minute", 0), second=0, microsecond=0)
            step = timedelta(hours=1)
        if candidate <= now_local:
            candidate += step
        upcoming.append(candidate)
    return min(upcoming) if upcoming else None


def _describe_times(times) -> list:
    daily = [f"{t['Hour']:02d}:{t.get('Minute', 0):02d}" for t in times if "Hour" in t]
    hourly = [f":{t.get('Minute', 0):02d}" for t in times if "Hour" not in t]
    return daily + ([f"hourly at {', '.join(hourly)}"] if hourly else [])


def agents(now: datetime = None) -> list:
    """Installed nba-props launchd agents: schedule, next run (local and ET), last exit."""
    now = (now or datetime.now(timezone.utc)).astimezone()  # this Mac's local time
    try:
        listed = subprocess.run(["launchctl", "list"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        listed = ""
    loaded = {}
    for line in listed.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) == 3 and parts[2].startswith(AGENT_PREFIX):
            loaded[parts[2]] = {"pid": None if parts[0] == "-" else int(parts[0]),
                                "last_exit": int(parts[1]) if parts[1].lstrip("-").isdigit() else None}
    out = []
    for path in sorted(LAUNCH_AGENTS.glob(f"{AGENT_PREFIX}*.plist")):
        try:
            with open(path, "rb") as fh:
                plist = plistlib.load(fh)
        except (OSError, plistlib.InvalidFileException):
            continue
        times = plist.get("StartCalendarInterval") or []
        times = [times] if isinstance(times, dict) else times
        nxt = _next_fire(times, now)
        label = plist.get("Label", path.stem)
        out.append({
            "name": label[len(AGENT_PREFIX):], "label": label,
            "keep_alive": bool(plist.get("KeepAlive")),
            "local_times": _describe_times(times),
            "next_local": nxt.isoformat(timespec="minutes") if nxt else None,
            "next_et": nxt.astimezone(ET).isoformat(timespec="minutes") if nxt else None,
            "loaded": label in loaded, **loaded.get(label, {"pid": None, "last_exit": None}),
        })
    return out


# ── games: ESPN scoreboard + model + market + head-to-head + injuries ─────────
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_HEADERS = {"Accept": "application/json"}  # ESPN's CDN rejects browser-like User-Agents


def _iso_utc(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def parse_scoreboard(data) -> list:
    """ESPN scoreboard payload -> one dict per game, team codes in NBA form. Only the
    schedule, teams and scores are used; win probabilities come from the pipeline's files."""
    games = []
    for event in (data or {}).get("events") or []:
        try:
            comp = event["competitions"][0]
            sides = {c["homeAway"]: c for c in comp["competitors"]}
            away_c, home_c = sides["away"], sides["home"]
        except (KeyError, IndexError, TypeError):
            continue
        status = (event.get("status") or {}).get("type") or {}
        state = status.get("state") or "pre"

        def team(c):
            t = c.get("team") or {}
            raw = t.get("abbreviation") or ""
            record = next((r.get("summary") for r in c.get("records") or []
                           if r.get("type") == "total"), None)
            return {"abbr": team_codes.normalize_team(raw) or raw, "name": t.get("displayName") or raw,
                    "short": t.get("shortDisplayName") or t.get("name") or raw, "record": record,
                    "score": c.get("score") if state != "pre" else None,
                    "winner": c.get("winner"), "logo": t.get("logo"), "logo_dark": t.get("logoDark")}

        away, home = team(away_c), team(home_c)
        odds = (comp.get("odds") or [{}])[0]
        games.append({
            "key": f"{away['abbr']}@{home['abbr']}",
            "start_utc": _iso_utc(comp.get("date") or event.get("date")),
            "state": state, "status_text": status.get("shortDetail"), "completed": bool(status.get("completed")),
            "network": ", ".join(n for b in comp.get("broadcasts") or [] for n in b.get("names") or []) or None,
            "line": odds.get("details"), "away": away, "home": home,
        })
    return games


def fetch_scoreboard(day: str):
    """Games for one US Eastern date from ESPN (archived like every other fetch); None if down."""
    params = {"dates": day.replace("-", "")}
    try:
        response = requests.get(SCOREBOARD_URL, params=params, headers=ESPN_HEADERS, timeout=12)
        response.raise_for_status()
        data = response.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        raw_archive.save("espn", "scoreboard", params, None, status="fetch_failed",
                         http_status=raw_archive.http_status_of(exc), error=type(exc).__name__)
        return None
    raw_archive.save("espn", "scoreboard", params, data, http_status=raw_archive.http_status_of(response))
    return parse_scoreboard(data)


def fetch_current_team_games(season: str) -> list:
    """This season's team game logs from stats.nba.com (one call; archived by nba_api_utils)."""
    from nba_api.stats.endpoints import leaguegamefinder

    import nba_api_utils

    nba_api_utils.configure_stats_http()
    frame = leaguegamefinder.LeagueGameFinder(
        season_nullable=season, league_id_nullable="00", season_type_nullable="Regular Season",
        player_or_team_abbreviation="T", timeout=30).get_data_frames()[0]
    return frame.to_dict("records")


class TeamHistory:
    """Regular-season meetings between two teams: data_cache/team_games_*.csv for past
    seasons (written by train_model.py) plus this season from stats.nba.com, cached."""

    def __init__(self, cache_dir: Path, fetch_current=fetch_current_team_games, ttl: float = 6 * 3600):
        self.cache_dir, self.fetch_current, self.ttl = Path(cache_dir), fetch_current, ttl
        self._past = None
        self._current, self._current_at = [], None
        self._lock = threading.Lock()

    @staticmethod
    def _normalize(row: dict):
        try:
            matchup, pts = str(row["MATCHUP"]), float(row["PTS"])
            return {"game_id": str(row["GAME_ID"]), "date": str(row["GAME_DATE"])[:10],
                    "season": row.get("season") or row.get("SEASON_ID"), "team": row["TEAM_ABBREVIATION"],
                    "opp": matchup.split()[-1], "home": " vs. " in matchup, "won": row["WL"] == "W",
                    "pts": int(pts), "opp_pts": int(round(pts - float(row["PLUS_MINUS"])))}
        except (KeyError, TypeError, ValueError, IndexError):
            return None

    def _past_rows(self) -> list:
        if self._past is None:
            current = current_season()
            rows = []
            for path in sorted(self.cache_dir.glob("team_games_*.csv")):
                for row in _read_csv(path):
                    if row.get("season") != current:
                        normalized = self._normalize(row)
                        if normalized:
                            rows.append(normalized)
            self._past = rows
        return self._past

    def _current_rows(self) -> list:
        if self.fetch_current and (self._current_at is None or time.monotonic() - self._current_at > self.ttl):
            try:
                rows = [r for r in map(self._normalize, self.fetch_current(current_season())) if r]
                self._current = [dict(r, season=current_season()) for r in rows]
            except Exception:  # this season's meetings are a bonus; past seasons still answer
                pass
            self._current_at = time.monotonic()
        return self._current

    def warm(self) -> None:
        self.meetings("", "")

    def meetings(self, a: str, b: str, limit: int = 10, before: str = None) -> list:
        """Most recent meetings first; `before` (YYYY-MM-DD) keeps only earlier games."""
        with self._lock:
            rows = self._past_rows() + self._current_rows()
        seen, out = set(), []
        mine = (r for r in rows if r["team"] == a and r["opp"] == b and (before is None or r["date"] < before))
        for r in sorted(mine, key=lambda r: r["date"], reverse=True):
            if r["game_id"] in seen:
                continue
            seen.add(r["game_id"])
            home, away = (a, b) if r["home"] else (b, a)
            home_pts, away_pts = (r["pts"], r["opp_pts"]) if r["home"] else (r["opp_pts"], r["pts"])
            out.append({"date": r["date"], "season": r["season"], "home": home, "away": away,
                        "home_pts": home_pts, "away_pts": away_pts, "winner": a if r["won"] else b})
            if len(out) >= limit:
                break
        return out


class Live:
    """The dashboard's own fetches: the scoreboard (cached a minute) and team history."""

    def __init__(self, src: Sources, scoreboard=fetch_scoreboard, history: TeamHistory = None):
        self.scoreboard_fn = scoreboard
        self.history = history or TeamHistory(src.team_logs)
        self._boards = {}
        self._lock = threading.Lock()

    def scoreboard(self, day: str):
        with self._lock:
            hit = self._boards.get(day)
            if hit and time.monotonic() - hit[0] < 60:
                return hit[1]
        board = self.scoreboard_fn(day)
        with self._lock:
            self._boards[day] = (time.monotonic(), board)
        return board


def _h2h_summary(meetings: list, away: str, home: str) -> dict:
    recent = meetings[:5]
    wins = {away: sum(m["winner"] == away for m in recent), home: sum(m["winner"] == home for m in recent)}
    return {"count": len(recent), "wins": wins, "last": meetings[0] if meetings else None}


def _latest_injuries(src: Sources, day: str) -> dict:
    """The last injury snapshot taken for this slate (today: the newest one)."""
    rows = _read_csv(src.injuries)
    for_day = [r for r in rows if r.get("slate_date") == day]
    if not for_day and day >= team_codes.today_slate_date().isoformat():
        for_day = rows
    rows = for_day
    if not rows:
        return {"snapshot_at_utc": None, "by_team": {}}
    latest = max(r["snapshot_at_utc"] for r in rows)
    by_team = {}
    for r in rows:
        if r["snapshot_at_utc"] == latest:
            by_team.setdefault(r["team"], []).append({k: r.get(k, "") for k in (
                "player", "status", "position", "injury_type", "injury_detail", "return_date",
                "short_comment", "report_updated_at")})
    for players in by_team.values():
        players.sort(key=lambda p: (p["status"] != "Out", p["player"]))
    return {"snapshot_at_utc": latest, "by_team": by_team}


def games_for(src: Sources, day: str, live: Live) -> dict:
    """Every game on a slate with the best win probability available, the model's picks,
    head-to-head, injury counts and the top prop pick."""
    board = live.scoreboard(day)
    model = {}
    for g in _query(src.db, "SELECT * FROM game_predictions WHERE date = ?", (day,)):
        home_prob = g.get("home_prob")
        if home_prob is None and g.get("win_pct") is not None and g.get("winner_pick"):
            home_prob = g["win_pct"] if g["winner_pick"] == g["home_abbr"] else 1 - g["win_pct"]
        model[f"{g['away_abbr']}@{g['home_abbr']}"] = {
            "source": "Model", "home_prob": None if home_prob is None else round(home_prob, 4),
            "away_prob": None if home_prob is None else round(1 - home_prob, 4),
            "winner_pick": g.get("winner_pick"), "spread": g.get("spread"),
            "spread_pick": g.get("spread_pick"), "total": g.get("total")}
    market = {}
    for r in _read_csv(src.game_odds):
        if r.get("date") == day and r.get("market") == "h2h":
            if r["game"] not in market or r["snapshot_at"] > market[r["game"]]["snapshot_at"]:
                market[r["game"]] = {"source": f"Market ({r.get('books_count') or '?'} books)",
                                     "home_prob": _num(r.get("devig_home")), "away_prob": _num(r.get("devig_away")),
                                     "snapshot_at": r["snapshot_at"]}
    picks = picks_for(src, day)
    props = {}
    for p in picks["props"]:
        props.setdefault(p["game"], []).append(p)
    finals = {g["away_abbr"] + "@" + g["home_abbr"]: g for g in picks["games"]}
    injuries = _latest_injuries(src, day)

    games = list(board or [])
    known = {g["key"] for g in games}
    for key in sorted((set(model) | set(props) | set(market)) - known):  # no scoreboard row
        if "@" not in key:
            continue
        away, home = key.split("@", 1)
        start = next((p["game_start_at"] for p in props.get(key, []) if p.get("game_start_at")), None)
        games.append({"key": key, "start_utc": _iso_utc(start), "state": "pre", "status_text": None,
                      "completed": False, "network": None, "line": None,
                      "away": {"abbr": away, "name": away, "short": away, "record": None, "score": None},
                      "home": {"abbr": home, "name": home, "short": home, "record": None, "score": None}})

    for g in games:
        key, away, home = g["key"], g["away"]["abbr"], g["home"]["abbr"]
        g["model"] = model.get(key) if model.get(key, {}).get("home_prob") is not None else None
        g["market"] = market.get(key) if market.get(key, {}).get("home_prob") is not None else None
        g["win"] = g["model"] or g["market"]  # predictions.db first, then your odds tracker
        g["h2h"] = _h2h_summary(live.history.meetings(home, away, limit=5, before=day), away, home)
        game_props = props.get(key, [])
        g["props"] = {"count": len(game_props), "record": _record([p for p in game_props if p["result"]]),
                      "best": max(game_props, key=lambda p: p.get("confidence") or 0, default=None)}
        g["final"] = finals.get(key)
        for side, abbr in (("away", away), ("home", home)):
            listed = injuries["by_team"].get(abbr, [])
            g[side]["injuries"] = {"out": sum(p["status"] == "Out" for p in listed), "listed": len(listed)}
    games.sort(key=lambda g: (g["start_utc"] or "9999", g["key"]))
    return {"date": day, "scoreboard_ok": board is not None, "games": games,
            "injuries_snapshot_at": injuries["snapshot_at_utc"]}


def game_detail(src: Sources, day: str, key: str, live: Live) -> dict:
    slate = games_for(src, day, live)
    game = next((g for g in slate["games"] if g["key"] == key), None)
    if game is None:
        raise KeyError(key)
    away, home = game["away"]["abbr"], game["home"]["abbr"]
    injuries = _latest_injuries(src, day)
    props = sorted((p for p in picks_for(src, day)["props"] if p["game"] == key),
                   key=lambda p: -(p.get("confidence") or 0))
    return {"date": day, "game": game, "props": props,
            "h2h": live.history.meetings(home, away, limit=10, before=day),
            "injuries": {"snapshot_at_utc": injuries["snapshot_at_utc"],
                         "away": injuries["by_team"].get(away, []), "home": injuries["by_team"].get(home, [])}}


# ── jobs started from the page ────────────────────────────────────────────────
class JobRunner:
    def __init__(self, root: Path = ROOT, log_dir: Path = LOG_DIR):
        self.root, self.log_dir = Path(root), Path(log_dir)
        self.jobs = {}
        self.lock = threading.Lock()

    def start(self, name: str, options: dict) -> dict:
        if name not in JOBS:
            raise KeyError(name)
        script, allowed = JOBS[name]
        args = []
        if "date" in allowed and options.get("date"):
            args.append(date.fromisoformat(str(options["date"])).isoformat())  # ValueError if bad
        with self.lock:
            current = self.jobs.get(name)
            if current and current["proc"].poll() is None:
                raise RuntimeError(f"{name} is already running")
            scheduled = next((a for a in agents() if a["name"] == name and a["pid"]), None)
            if scheduled:
                raise RuntimeError(f"the scheduled {name} run is in progress (pid {scheduled['pid']})")
            holder = run_log.lock_holder(JOB_COMMANDS.get(name, name))  # e.g. one started in a terminal
            if holder:
                raise RuntimeError(f"a {name} run is already in progress (pid {holder})")
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / f"web-{name}.log"
            log = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-u", str(self.root / script), *args],
                                    cwd=self.root, stdout=log, stderr=subprocess.STDOUT)
            log.close()  # the child holds its own handle
            self.jobs[name] = {"proc": proc, "args": args, "log": log_path,
                               "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        return self.status()[name]

    def status(self) -> dict:
        out = {}
        for name in JOBS:
            job = self.jobs.get(name)
            if not job:
                out[name] = {"state": "idle"}
                continue
            code = job["proc"].poll()
            out[name] = {"state": "running" if code is None else "finished", "exit_code": code,
                         "started_at": job["started_at"], "args": job["args"],
                         "log_tail": _tail(job["log"])}
        return out


def _tail(path: Path, lines: int = 40) -> list:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    noise = ("NotOpenSSLWarning", "warnings.warn(")
    return [line for line in text if not any(n in line for n in noise)][-lines:]


# ── HTTP ──────────────────────────────────────────────────────────────────────
def make_handler(src: Sources, runner: JobRunner, port: int, live: Live):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    allowed_origins = {f"http://{h}" for h in allowed_hosts}

    class Handler(BaseHTTPRequestHandler):
        server_version = "nba-props-dashboard"

        def log_message(self, fmt, *args):  # keep launchd logs quiet
            pass

        def _send(self, status: int, body, content_type="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            try:
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the page went away mid-response (a reload); nobody is left to answer

        def _host_ok(self) -> bool:
            return self.headers.get("Host", "") in allowed_hosts

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, {"error": "dashboard only answers on localhost"})
            url = urlsplit(self.path)
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                today = team_codes.today_slate_date()
                if url.path == "/api/overview":
                    return self._send(200, {
                        "today_et": today.isoformat(),
                        "now_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "latest": latest_runs(src), "agents": agents(), "jobs": runner.status(),
                        "predict_running": run_log.lock_holder("nba_combined")})
                day = date.fromisoformat(query.get("date") or today.isoformat()).isoformat()
                if url.path == "/api/games":
                    return self._send(200, games_for(src, day, live))
                if url.path == "/api/game":
                    try:
                        return self._send(200, game_detail(src, day, query.get("key", ""), live))
                    except KeyError:
                        return self._send(404, {"error": f"no game {query.get('key')!r} on {day}"})
                if url.path == "/api/top":
                    by = query.get("by", "prob")
                    if by not in TOP_SORTS:
                        return self._send(400, {"error": f"by must be one of {sorted(TOP_SORTS)}"})
                    weights = None
                    if query.get("w"):  # custom weights "prob,conf,edge"; scaled to sum to 1
                        parts = [max(0.0, float(x)) for x in query["w"].split(",")]
                        if len(parts) != 3 or not sum(parts):
                            raise ValueError("w must be three non-negative numbers: prob,conf,edge")
                        weights = tuple(x / sum(parts) for x in parts)
                    return self._send(200, top_bets(src, day, by, min(int(query.get("limit", TOP_N)), 100),
                                                    include_flagged=query.get("flagged") == "1",
                                                    weights=weights))
                if url.path == "/api/ranking":
                    key = query.get("range", "season")
                    if key not in ("7d", "30d", "season", "all"):
                        return self._send(400, {"error": "range must be 7d, 30d, season or all"})
                    return self._send(200, ranking_report(src, key, today))
                if url.path == "/api/picks":
                    return self._send(200, picks_for(src, day))
                if url.path == "/api/performance":
                    key = query.get("range", "season")
                    if key not in ("7d", "30d", "season", "all"):
                        return self._send(400, {"error": "range must be 7d, 30d, season or all"})
                    return self._send(200, performance(src, key, today))
                if url.path == "/api/runs":
                    return self._send(200, runs(src, min(int(query.get("limit", 40)), 200)))
                if url.path == "/api/injuries":
                    return self._send(200, injuries(src))
                if url.path == "/api/jobs":
                    return self._send(200, runner.status())
                return self._send(404, {"error": "not found"})
            except ValueError as exc:
                return self._send(400, {"error": str(exc)})
            except Exception as exc:  # a broken file must show as an error, not a dropped connection
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self):
            origin = self.headers.get("Origin")
            if (not self._host_ok() or (origin and origin not in allowed_origins)
                    or self.headers.get("X-NBA-Dashboard") != "1"
                    or not self.headers.get("Content-Type", "").startswith("application/json")):
                return self._send(403, {"error": "runs can only be started from the dashboard page"})
            url = urlsplit(self.path)
            if not url.path.startswith("/api/run/"):
                return self._send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
                options = json.loads(self.rfile.read(length) or b"{}") if length else {}
                if not isinstance(options, dict):
                    raise ValueError("body must be a JSON object")
                return self._send(202, runner.start(url.path[len("/api/run/"):], options))
            except KeyError:
                return self._send(404, {"error": f"unknown job; choose from {sorted(JOBS)}"})
            except RuntimeError as exc:
                return self._send(409, {"error": str(exc)})
            except ValueError as exc:
                return self._send(400, {"error": str(exc)})

    return Handler


def serve(port: int = None, src: Sources = None, live: Live = None) -> ThreadingHTTPServer:
    port = port if port is not None else int(os.environ.get("NBA_DASHBOARD_PORT", "8765"))
    src = src or Sources()
    live = live or Live(src)
    server = ThreadingHTTPServer(("127.0.0.1", port), None)
    server.RequestHandlerClass = make_handler(src, JobRunner(src.root), server.server_address[1], live)
    return server


def main() -> int:
    src = Sources()
    live = Live(src)
    server = serve(src=src, live=live)
    threading.Thread(target=live.history.warm, daemon=True).start()  # first page load stays fast
    print(f"  NBA props dashboard on http://127.0.0.1:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
